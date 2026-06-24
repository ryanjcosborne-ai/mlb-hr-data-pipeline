import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast


TIMEZONE = ZoneInfo("America/Toronto")
SEASON = datetime.now(TIMEZONE).year
DATA_DIR = "data"


def now_str() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def safe_number(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value


def write_csv(filename, df):
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def sample_quality(pa_or_bf, bbe):
    if pa_or_bf >= 40 and bbe >= 25:
        return "Good"
    if pa_or_bf >= 20 and bbe >= 10:
        return "Medium"
    return "Thin"


def get_people_map(player_ids):
    player_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})
    people = {}

    for i in range(0, len(player_ids), 100):
        chunk = player_ids[i:i + 100]
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": ",".join(map(str, chunk))},
            timeout=30,
        )
        response.raise_for_status()

        for person in response.json().get("people", []):
            mlbam_id = person.get("id")
            people[mlbam_id] = {
                "name": person.get("fullName", ""),
                "team": (person.get("currentTeam") or {}).get("abbreviation", ""),
            }

    return people


def pull_statcast_30_days():
    end_date = datetime.now(TIMEZONE).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    print(f"Pulling Statcast data from {start_date} to {end_date}...")
    df = statcast(start_dt=str(start_date), end_dt=str(end_date), verbose=False)

    if df.empty:
        raise RuntimeError("Statcast returned no data.")

    print(f"Pulled {len(df)} Statcast rows.")
    return df


def aggregate_player_window(df, window_days, role, people_map):
    pull_ts = now_str()
    end_date = datetime.now(TIMEZONE).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=window_days - 1)

    window = df.copy()
    window["game_date"] = pd.to_datetime(window["game_date"]).dt.date
    window = window[
        (window["game_date"] >= start_date) &
        (window["game_date"] <= end_date)
    ]

    id_col = "batter" if role == "batter" else "pitcher"

    pa_rows = window[window["events"].notna()].copy()
    bbe = window[window["launch_speed"].notna()].copy()

    rows = []

    for mlbam_id, group in pa_rows.groupby(id_col):
        mlbam_id = int(mlbam_id)
        people = people_map.get(mlbam_id, {})
        bbe_group = bbe[bbe[id_col] == mlbam_id]

        pa_or_bf = len(group)
        batted_balls = len(bbe_group)
        hr = int((group["events"] == "home_run").sum())

        avg_ev = bbe_group["launch_speed"].mean() if batted_balls else ""
        max_ev = bbe_group["launch_speed"].max() if batted_balls else ""

        hard_hit_count = (
            int((bbe_group["launch_speed"] >= 95).sum())
            if batted_balls
            else ""
        )
        hard_hit_pct = (
            hard_hit_count / batted_balls
            if batted_balls and hard_hit_count != ""
            else ""
        )

        if "launch_speed_angle" in bbe_group.columns and batted_balls:
            barrels = int((bbe_group["launch_speed_angle"] == 6).sum())
        else:
            barrels = ""

        barrel_pct = (
            barrels / batted_balls
            if batted_balls and barrels != ""
            else ""
        )

        barrel_per_pa = (
            barrels / pa_or_bf
            if pa_or_bf and barrels != ""
            else ""
        )

        xwoba = (
            bbe_group["estimated_woba_using_speedangle"].mean()
            if "estimated_woba_using_speedangle" in bbe_group.columns and batted_balls
            else ""
        )

        avg_launch_angle = (
            bbe_group["launch_angle"].mean()
            if batted_balls
            else ""
        )

        fly_balls = (
            int((bbe_group["bb_type"] == "fly_ball").sum())
            if "bb_type" in bbe_group.columns and batted_balls
            else 0
        )

        fb_pct = fly_balls / batted_balls if batted_balls else ""
        hr_per_fb = hr / fly_balls if fly_balls else ""

        rows.append({
            "pull_timestamp": pull_ts,
            "window_days": window_days,
            "player_name": people.get("name", ""),
            "team": people.get("team", ""),
            "mlbam_id": mlbam_id,
            "role": role,
            "pa_or_bf": pa_or_bf,
            "batted_balls": batted_balls,
            "hr": hr,
            "avg_ev": safe_number(avg_ev),
            "max_ev": safe_number(max_ev),
            "hard_hit_pct": safe_number(hard_hit_pct),
            "barrels": barrels,
            "barrel_pct": safe_number(barrel_pct),
            "barrel_per_pa": safe_number(barrel_per_pa),
            "xslg": "",
            "xwoba": safe_number(xwoba),
            "avg_launch_angle": safe_number(avg_launch_angle),
            "fb_pct": safe_number(fb_pct),
            "pull_pct": "",
            "sample_quality": sample_quality(pa_or_bf, batted_balls),
            "source": "pybaseball.statcast",
            "notes": "Rolling Statcast sample; no FanGraphs dependency.",
            "reserved_1": "",
            "reserved_2": "",
            "reserved_3": "",
            "reserved_4": "",
            "reserved_5": "",
            "hr_per_fb": safe_number(hr_per_fb),
        })

    return rows


def build_rolling_windows(df, people_map):
    rows = []

    for window_days in [7, 14, 30]:
        rows.extend(aggregate_player_window(df, window_days, "batter", people_map))
        rows.extend(aggregate_player_window(df, window_days, "pitcher", people_map))

    rolling_df = pd.DataFrame(rows)

    rolling_headers = [
        "pull_timestamp",
        "window_days",
        "player_name",
        "team",
        "mlbam_id",
        "role",
        "pa_or_bf",
        "batted_balls",
        "hr",
        "avg_ev",
        "max_ev",
        "hard_hit_pct",
        "barrels",
        "barrel_pct",
        "barrel_per_pa",
        "xslg",
        "xwoba",
        "avg_launch_angle",
        "fb_pct",
        "pull_pct",
        "sample_quality",
        "source",
        "notes",
        "reserved_1",
        "reserved_2",
        "reserved_3",
        "reserved_4",
        "reserved_5",
        "hr_per_fb",
    ]

    return rolling_df[rolling_headers]


def build_batter_daily_from_rolling(rolling_df):
    batter_30 = rolling_df[
        (rolling_df["role"] == "batter") &
        (rolling_df["window_days"] == 30)
    ].copy()

    rows = []

    for _, row in batter_30.iterrows():
        rows.append({
            "pull_timestamp": row["pull_timestamp"],
            "season": SEASON,
            "player_name": row["player_name"],
            "team": row["team"],
            "mlbam_id": row["mlbam_id"],
            "pa": row["pa_or_bf"],
            "batted_balls": row["batted_balls"],
            "hr": row["hr"],
            "xslg": "",
            "xwoba": row["xwoba"],
            "xba": "",
            "avg_ev": row["avg_ev"],
            "max_ev": row["max_ev"],
            "hard_hit_pct": row["hard_hit_pct"],
            "barrels": row["barrels"],
            "barrel_pct": row["barrel_pct"],
            "barrel_per_pa": row["barrel_per_pa"],
            "avg_launch_angle": row["avg_launch_angle"],
            "sweet_spot_pct": "",
            "fb_pct": row["fb_pct"],
            "pull_pct": "",
            "hr_per_fb": row["hr_per_fb"],
            "source": "pybaseball.statcast rolling_30",
            "notes": "30-day Statcast rolling proxy; FanGraphs blocked on GitHub runner.",
        })

    return pd.DataFrame(rows)


def build_pitcher_daily_from_rolling(rolling_df):
    pitcher_30 = rolling_df[
        (rolling_df["role"] == "pitcher") &
        (rolling_df["window_days"] == 30)
    ].copy()

    rows = []

    for _, row in pitcher_30.iterrows():
        rows.append({
            "pull_timestamp": row["pull_timestamp"],
            "season": SEASON,
            "player_name": row["player_name"],
            "team": row["team"],
            "mlbam_id": row["mlbam_id"],
            "batters_faced": row["pa_or_bf"],
            "batted_balls_allowed": row["batted_balls"],
            "hr_allowed": row["hr"],
            "xera": "",
            "xslg_allowed": "",
            "xwoba_allowed": row["xwoba"],
            "avg_ev_allowed": row["avg_ev"],
            "max_ev_allowed": row["max_ev"],
            "hard_hit_pct_allowed": row["hard_hit_pct"],
            "barrels_allowed": row["barrels"],
            "barrel_pct_allowed": row["barrel_pct"],
            "barrel_per_pa_allowed": row["barrel_per_pa"],
            "avg_launch_angle_allowed": row["avg_launch_angle"],
            "fb_pct_allowed": row["fb_pct"],
            "pull_pct_allowed": "",
            "hr_per_fb_allowed": row["hr_per_fb"],
            "pitch_hand": "",
            "source": "pybaseball.statcast rolling_30",
            "notes": "30-day Statcast rolling proxy; FanGraphs blocked on GitHub runner.",
        })

    return pd.DataFrame(rows)


def build_status_df(batter_df, pitcher_df, rolling_df):
    return pd.DataFrame([{
        "pull_timestamp": now_str(),
        "season": SEASON,
        "batter_rows": len(batter_df),
        "pitcher_rows": len(pitcher_df),
        "rolling_rows": len(rolling_df),
        "status": "OK",
        "notes": "Statcast-only pipeline completed. No Google Cloud. No FanGraphs.",
    }])


def main():
    ensure_data_dir()

    statcast_df = pull_statcast_30_days()

    player_ids = set(statcast_df["batter"].dropna().astype(int).unique())
    player_ids.update(set(statcast_df["pitcher"].dropna().astype(int).unique()))

    print(f"Fetching MLB people metadata for {len(player_ids)} players...")
    people_map = get_people_map(player_ids)

    rolling_df = build_rolling_windows(statcast_df, people_map)
    batter_df = build_batter_daily_from_rolling(rolling_df)
    pitcher_df = build_pitcher_daily_from_rolling(rolling_df)
    status_df = build_status_df(batter_df, pitcher_df, rolling_df)

    write_csv("batter_statcast_daily.csv", batter_df)
    write_csv("pitcher_statcast_daily.csv", pitcher_df)
    write_csv("statcast_rolling_windows.csv", rolling_df)
    write_csv("statcast_status.csv", status_df)

    print("Statcast CSV refresh complete.")


if __name__ == "__main__":
    main()
