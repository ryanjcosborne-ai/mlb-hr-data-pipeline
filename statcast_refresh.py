import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pybaseball import batting_stats, pitching_stats, statcast


TIMEZONE = ZoneInfo("America/Toronto")
SEASON = datetime.now(TIMEZONE).year
DATA_DIR = "data"


def now_str() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def clean_value(value):
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


def normalize_percent(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, str):
        value = value.replace("%", "").strip()
        if value == "":
            return ""
        try:
            return float(value) / 100
        except ValueError:
            return ""

    try:
        value = float(value)
    except Exception:
        return ""

    if value > 1:
        return value / 100

    return value


def pick(row, names, default=""):
    for name in names:
        if name in row.index:
            return clean_value(row[name])
    return default


def pick_percent(row, names, default=""):
    return normalize_percent(pick(row, names, default))


def write_csv(filename, df):
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def pull_batter_daily():
    print("Pulling season-to-date batter data...")
    source = batting_stats(SEASON, qual=0)
    pull_ts = now_str()

    rows = []

    for _, row in source.iterrows():
        pa = pick(row, ["PA"])
        barrels = pick(row, ["Barrels"])
        barrel_per_pa = ""

        try:
            if pa != "" and float(pa) > 0 and barrels != "":
                barrel_per_pa = float(barrels) / float(pa)
        except Exception:
            barrel_per_pa = ""

        rows.append({
            "pull_timestamp": pull_ts,
            "season": SEASON,
            "player_name": pick(row, ["Name"]),
            "team": pick(row, ["Team"]),
            "mlbam_id": "",
            "pa": pa,
            "batted_balls": pick(row, ["Events", "Batted Balls", "BBE"]),
            "hr": pick(row, ["HR"]),
            "xslg": pick(row, ["xSLG", "xSLG+"]),
            "xwoba": pick(row, ["xwOBA", "xWOBA"]),
            "xba": pick(row, ["xBA"]),
            "avg_ev": pick(row, ["EV", "Avg EV", "avgEV"]),
            "max_ev": pick(row, ["maxEV", "Max EV"]),
            "hard_hit_pct": pick_percent(row, ["HardHit%", "Hard Hit %", "HardHit%"]),
            "barrels": barrels,
            "barrel_pct": pick_percent(row, ["Barrel%", "Barrel %"]),
            "barrel_per_pa": barrel_per_pa,
            "avg_launch_angle": pick(row, ["LA", "Launch Angle", "avgLA"]),
            "sweet_spot_pct": pick_percent(row, ["SweetSpot%", "Sweet Spot %"]),
            "fb_pct": pick_percent(row, ["FB%", "FB %"]),
            "pull_pct": pick_percent(row, ["Pull%", "Pull %"]),
            "hr_per_fb": pick_percent(row, ["HR/FB", "HR/FB%"]),
            "source": "pybaseball.batting_stats",
            "notes": "",
        })

    return pd.DataFrame(rows)


def pull_pitcher_daily():
    print("Pulling season-to-date pitcher data...")
    source = pitching_stats(SEASON, qual=0)
    pull_ts = now_str()

    rows = []

    for _, row in source.iterrows():
        bf = pick(row, ["TBF", "BF"])
        barrels = pick(row, ["Barrels"])
        barrel_per_pa = ""

        try:
            if bf != "" and float(bf) > 0 and barrels != "":
                barrel_per_pa = float(barrels) / float(bf)
        except Exception:
            barrel_per_pa = ""

        rows.append({
            "pull_timestamp": pull_ts,
            "season": SEASON,
            "player_name": pick(row, ["Name"]),
            "team": pick(row, ["Team"]),
            "mlbam_id": "",
            "batters_faced": bf,
            "batted_balls_allowed": pick(row, ["Events", "Batted Balls", "BBE"]),
            "hr_allowed": pick(row, ["HR"]),
            "xera": pick(row, ["xERA"]),
            "xslg_allowed": pick(row, ["xSLG", "xSLG+"]),
            "xwoba_allowed": pick(row, ["xwOBA", "xWOBA"]),
            "avg_ev_allowed": pick(row, ["EV", "Avg EV", "avgEV"]),
            "max_ev_allowed": pick(row, ["maxEV", "Max EV"]),
            "hard_hit_pct_allowed": pick_percent(row, ["HardHit%", "Hard Hit %", "HardHit%"]),
            "barrels_allowed": barrels,
            "barrel_pct_allowed": pick_percent(row, ["Barrel%", "Barrel %"]),
            "barrel_per_pa_allowed": barrel_per_pa,
            "avg_launch_angle_allowed": pick(row, ["LA", "Launch Angle", "avgLA"]),
            "fb_pct_allowed": pick_percent(row, ["FB%", "FB %"]),
            "pull_pct_allowed": pick_percent(row, ["Pull%", "Pull %"]),
            "hr_per_fb_allowed": pick_percent(row, ["HR/FB", "HR/FB%"]),
            "pitch_hand": "",
            "source": "pybaseball.pitching_stats",
            "notes": "",
        })

    return pd.DataFrame(rows)


def sample_quality(pa_or_bf, bbe):
    if pa_or_bf >= 40 and bbe >= 25:
        return "Good"
    if pa_or_bf >= 20 and bbe >= 10:
        return "Medium"
    return "Thin"


def aggregate_window(df, window_days, role):
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
    pa_rows = window[window["events"].notna()]
    bbe = window[window["launch_speed"].notna()]

    rows = []

    for mlbam_id, group in pa_rows.groupby(id_col):
        bbe_group = bbe[bbe[id_col] == mlbam_id]

        pa_or_bf = len(group)
        batted_balls = len(bbe_group)
        hr = int((group["events"] == "home_run").sum())

        avg_ev = bbe_group["launch_speed"].mean() if batted_balls else ""
        max_ev = bbe_group["launch_speed"].max() if batted_balls else ""
        hard_hit = int((bbe_group["launch_speed"] >= 95).sum()) if batted_balls else ""
        hard_hit_pct = hard_hit / batted_balls if batted_balls else ""

        if "launch_speed_angle" in bbe_group.columns and batted_balls:
            barrels = int((bbe_group["launch_speed_angle"] == 6).sum())
        else:
            barrels = ""

        barrel_pct = barrels / batted_balls if batted_balls and barrels != "" else ""
        barrel_per_pa = barrels / pa_or_bf if pa_or_bf and barrels != "" else ""

        xwoba = (
            bbe_group["estimated_woba_using_speedangle"].mean()
            if "estimated_woba_using_speedangle" in bbe_group.columns and batted_balls
            else ""
        )

        avg_la = bbe_group["launch_angle"].mean() if batted_balls else ""

        fb_pct = (
            (bbe_group["bb_type"] == "fly_ball").sum() / batted_balls
            if batted_balls and "bb_type" in bbe_group.columns
            else ""
        )

        player_name = ""
        team = ""

        if role == "batter":
            player_rows = group[group["player_name"].notna()]
            if len(player_rows):
                player_name = player_rows.iloc[0]["player_name"]
            if "home_team" in group.columns and "away_team" in group.columns:
                team = ""
        else:
            player_name = ""

        rows.append({
            "pull_timestamp": pull_ts,
            "window_days": window_days,
            "player_name": player_name,
            "team": team,
            "mlbam_id": int(mlbam_id),
            "role": role,
            "pa_or_bf": pa_or_bf,
            "batted_balls": batted_balls,
            "hr": hr,
            "avg_ev": avg_ev,
            "max_ev": max_ev,
            "hard_hit_pct": hard_hit_pct,
            "barrels": barrels,
            "barrel_pct": barrel_pct,
            "barrel_per_pa": barrel_per_pa,
            "xslg": "",
            "xwoba": xwoba,
            "avg_launch_angle": avg_la,
            "fb_pct": fb_pct,
            "pull_pct": "",
            "sample_quality": sample_quality(pa_or_bf, batted_balls),
            "source": "pybaseball.statcast",
            "notes": "",
            "reserved_1": "",
            "reserved_2": "",
            "reserved_3": "",
            "reserved_4": "",
            "reserved_5": "",
        })

    return rows


def pull_rolling_windows():
    pull_ts = now_str()
    end_date = datetime.now(TIMEZONE).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

    print(f"Pulling 30-day Statcast data from {start_date} to {end_date}...")
    df = statcast(start_dt=str(start_date), end_dt=str(end_date), verbose=False)

    if df.empty:
        print("Statcast returned no rows.")
        return pd.DataFrame()

    rows = []

    for window_days in [7, 14, 30]:
        rows.extend(aggregate_window(df, window_days, "batter"))
        rows.extend(aggregate_window(df, window_days, "pitcher"))

    return pd.DataFrame(rows)


def main():
    ensure_data_dir()

    batter_df = pull_batter_daily()
    pitcher_df = pull_pitcher_daily()
    rolling_df = pull_rolling_windows()

    status_df = pd.DataFrame([{
        "pull_timestamp": now_str(),
        "season": SEASON,
        "batter_rows": len(batter_df),
        "pitcher_rows": len(pitcher_df),
        "rolling_rows": len(rolling_df),
        "status": "OK",
        "notes": "Generated by GitHub Actions. To be imported into Google Sheets by Apps Script."
    }])

    write_csv("batter_statcast_daily.csv", batter_df)
    write_csv("pitcher_statcast_daily.csv", pitcher_df)
    write_csv("statcast_rolling_windows.csv", rolling_df)
    write_csv("statcast_status.csv", status_df)

    print("Statcast CSV refresh complete.")


if __name__ == "__main__":
    main()
