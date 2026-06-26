#!/usr/bin/env python3
"""
MLB HR Prop Predictor — Phase 3 Statcast Feature Refresh

Output CSVs for Apps Script import:
- statcast_batter_rolling.csv
- statcast_pitcher_rolling.csv
- pitch_type_batter_damage.csv
- pitch_type_pitcher_vulnerability.csv
- bullpen_hr_vulnerability.csv
- manifest.csv
- refresh_summary.json

Env:
- RAW_BASE_URL = https://raw.githubusercontent.com/OWNER/REPO/main/data/exports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pybaseball import playerid_reverse_lookup, statcast


@dataclass
class Config:
    slate_date: date
    timezone: str
    output_dir: Path
    raw_base_url: str
    windows: tuple[int, ...]
    lookback_days: int


def now_iso(tz: str) -> str:
    return datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds")


def parse_date(value: str, tz: str) -> date:
    if value.lower() == "today":
        return datetime.now(ZoneInfo(tz)).date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def with_stamp(df: pd.DataFrame, cfg: Config, note: str) -> pd.DataFrame:
    out = df.copy()
    out["slate_date"] = cfg.slate_date.isoformat()
    out["pull_timestamp"] = now_iso(cfg.timezone)
    out["data_notes"] = note
    return out


def most_common(s: pd.Series) -> str:
    if s is None or len(s) == 0:
        return ""
    vc = s.dropna().astype(str)
    vc = vc[vc.ne("")]
    return "" if vc.empty else str(vc.value_counts().index[0])


def rate(num: float, den: float, digits: int = 4):
    return "" if den in (0, "", None) or pd.isna(den) else round(float(num) / float(den), digits)


def fetch_statcast_chunks(cfg: Config) -> pd.DataFrame:
    end = cfg.slate_date - timedelta(days=1)
    start = cfg.slate_date - timedelta(days=cfg.lookback_days)
    chunks = []
    cur = start

    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        print(f"Fetching Statcast {cur} to {chunk_end}", flush=True)
        df = statcast(start_dt=cur.isoformat(), end_dt=chunk_end.isoformat())
        if df is not None and not df.empty:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not chunks:
        raise RuntimeError("No Statcast rows returned.")
    out = pd.concat(chunks, ignore_index=True)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def add_batter_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "batter" not in out.columns:
        out["batter_name"] = ""
        return out

    ids = pd.to_numeric(out["batter"], errors="coerce").dropna().astype(int).unique().tolist()
    out["batter"] = out["batter"].astype(str)
    out["batter_name"] = ""

    if not ids:
        return out

    try:
        lookup = playerid_reverse_lookup(ids, key_type="mlbam")
        lookup["batter"] = lookup["key_mlbam"].astype(str)
        lookup["batter_name"] = (
            lookup["name_first"].fillna("") + " " + lookup["name_last"].fillna("")
        ).str.strip()

        out = out.drop(columns=["batter_name"]).merge(
            lookup[["batter", "batter_name"]].drop_duplicates(),
            on="batter",
            how="left"
        )
        out["batter_name"] = out["batter_name"].fillna("")
    except Exception as exc:
        print(f"WARNING: playerid_reverse_lookup failed: {exc}", file=sys.stderr)

    return out


def prepare_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_batter_names(raw)

    for col in ["launch_speed", "launch_angle", "hc_x", "release_speed", "inning"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    for col in [
        "events",
        "description",
        "bb_type",
        "stand",
        "p_throws",
        "home_team",
        "away_team",
        "inning_topbot",
        "pitch_type",
        "pitch_name",
        "player_name",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df["is_pa_event"] = df["events"].notna() & df["events"].astype(str).ne("")
    df["pa_key"] = (
        df.get("game_pk", "").astype(str)
        + "_"
        + df.get("at_bat_number", "").astype(str)
        + "_"
        + df["batter"].astype(str)
    )

    df["is_bbe"] = df["launch_speed"].notna() & df["launch_angle"].notna()
    df["is_hr"] = df["events"].astype(str).eq("home_run")
    df["is_air"] = df["is_bbe"] & (
        (df["launch_angle"] >= 10)
        | df["bb_type"].astype(str).isin(["fly_ball", "line_drive", "popup"])
    )
    df["is_hard_hit"] = df["is_bbe"] & (df["launch_speed"] >= 95)
    df["is_hard_air"] = df["is_air"] & (df["launch_speed"] >= 95)
    df["is_blast"] = (
        df["is_bbe"]
        & (df["launch_speed"] >= 100)
        & (df["launch_angle"].between(20, 35))
    )
    df["is_sweet_spot"] = df["is_bbe"] & df["launch_angle"].between(8, 32)

    if "launch_speed_angle" in df.columns:
        df["is_barrel"] = pd.to_numeric(df["launch_speed_angle"], errors="coerce").eq(6)
    else:
        df["is_barrel"] = (
            df["is_bbe"]
            & (df["launch_speed"] >= 98)
            & df["launch_angle"].between(24, 32)
        )

    # Pull-air proxy from Savant hit coordinates.
    # Approximation: RHB pull side tends lower hc_x; LHB pull side tends higher hc_x.
    df["is_pulled_proxy"] = np.where(
        df["stand"].astype(str).eq("R"),
        df["hc_x"] < 125,
        np.where(df["stand"].astype(str).eq("L"), df["hc_x"] > 125, False),
    )
    df["is_pulled_air_proxy"] = df["is_air"] & pd.Series(df["is_pulled_proxy"]).fillna(False)

    df["batter_team"] = np.where(
        df["inning_topbot"].astype(str).eq("Top"),
        df["away_team"],
        np.where(df["inning_topbot"].astype(str).eq("Bot"), df["home_team"], ""),
    )
    df["pitching_team"] = np.where(
        df["inning_topbot"].astype(str).eq("Top"),
        df["home_team"],
        np.where(df["inning_topbot"].astype(str).eq("Bot"), df["away_team"], ""),
    )
    return df


def summarize(g: pd.DataFrame) -> dict:
    pa = int(g.loc[g["is_pa_event"], "pa_key"].nunique())
    bbe = int(g["is_bbe"].sum())
    air = int(g["is_air"].sum())
    barrels = int(g["is_barrel"].sum())
    hard_air = int(g["is_hard_air"].sum())
    blasts = int(g["is_blast"].sum())
    pulled_air = int(g["is_pulled_air_proxy"].sum())
    sweet = int(g["is_sweet_spot"].sum())
    hr = int(g["is_hr"].sum())

    return {
        "pa": pa,
        "bbe": bbe,
        "air_bbe": air,
        "hr": hr,
        "barrels": barrels,
        "hard_air_bbe": hard_air,
        "blast_bbe": blasts,
        "pulled_air_proxy_bbe": pulled_air,
        "sweet_spot_bbe": sweet,
        "max_ev": round(float(g["launch_speed"].max()), 2) if bbe else "",
        "avg_ev": round(float(g.loc[g["is_bbe"], "launch_speed"].mean()), 2) if bbe else "",
        "avg_la": round(float(g.loc[g["is_bbe"], "launch_angle"].mean()), 2) if bbe else "",
        "hr_per_pa": rate(hr, pa),
        "barrel_per_pa": rate(barrels, pa),
        "barrel_per_bbe": rate(barrels, bbe),
        "hard_air_per_pa": rate(hard_air, pa),
        "blast_per_pa": rate(blasts, pa),
        "pulled_air_per_bbe": rate(pulled_air, bbe),
        "sweet_spot_per_bbe": rate(sweet, bbe),
    }


def batter_rolling(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for w in cfg.windows:
        d = df[df["game_date_dt"] >= cfg.slate_date - timedelta(days=w)]
        for (bid, name), g in d.groupby(["batter", "batter_name"], dropna=False):
            rec = {
                "window_days": w,
                "batter_id": str(bid),
                "player": name,
                "team_recent": most_common(g["batter_team"]),
                "stand_recent": most_common(g["stand"]),
            }
            rec.update(summarize(g))
            rows.append(rec)

    return with_stamp(
        pd.DataFrame(rows),
        cfg,
        "Batter rolling HR contact-shape metrics from Statcast.",
    )


def pitcher_rolling(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for w in cfg.windows:
        d = df[df["game_date_dt"] >= cfg.slate_date - timedelta(days=w)]
        for (pid, pname, throws), g in d.groupby(["pitcher", "player_name", "p_throws"], dropna=False):
            rec = {
                "window_days": w,
                "pitcher_id": str(pid),
                "pitcher": pname,
                "p_throws": throws,
                "avg_release_speed": round(float(g["release_speed"].mean()), 2)
                if g["release_speed"].notna().any()
                else "",
            }
            rec.update(summarize(g))
            rows.append(rec)

    return with_stamp(
        pd.DataFrame(rows),
        cfg,
        "Pitcher rolling HR/contact vulnerability metrics from Statcast.",
    )


def batter_pitch_type(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df[df["game_date_dt"] >= cfg.slate_date - timedelta(days=max(cfg.windows))]
    rows = []

    for (bid, name, stand, pitch), g in d.groupby(
        ["batter", "batter_name", "stand", "pitch_type"], dropna=False
    ):
        if not str(pitch).strip() or str(pitch) == "nan":
            continue

        rec = {
            "batter_id": str(bid),
            "player": name,
            "stand": stand,
            "pitch_type": pitch,
            "pitch_name": most_common(g["pitch_name"]),
        }
        rec.update(summarize(g))
        rows.append(rec)

    return with_stamp(pd.DataFrame(rows), cfg, "Batter damage by pitch type.")


def pitcher_pitch_type(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df[df["game_date_dt"] >= cfg.slate_date - timedelta(days=max(cfg.windows))]

    total = d.groupby("pitcher").size().rename("total_pitches").reset_index()
    total["pitcher"] = total["pitcher"].astype(str)

    rows = []
    for (pid, pname, throws, pitch), g in d.groupby(
        ["pitcher", "player_name", "p_throws", "pitch_type"], dropna=False
    ):
        if not str(pitch).strip() or str(pitch) == "nan":
            continue

        rec = {
            "pitcher_id": str(pid),
            "pitcher": pname,
            "p_throws": throws,
            "pitch_type": pitch,
            "pitch_name": most_common(g["pitch_name"]),
            "pitch_count": int(len(g)),
        }
        rec.update(summarize(g))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.merge(
            total,
            left_on="pitcher_id",
            right_on="pitcher",
            how="left",
            suffixes=("", "_drop"),
        )
        out["pitch_mix_pct"] = (
            pd.to_numeric(out["pitch_count"], errors="coerce")
            / pd.to_numeric(out["total_pitches"], errors="coerce")
        ).round(4)
        out = out.drop(columns=[c for c in ["pitcher_drop"] if c in out.columns], errors="ignore")

    return with_stamp(out, cfg, "Pitcher pitch mix and damage allowed by pitch type.")


def bullpen_vulnerability(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df[df["game_date_dt"] >= cfg.slate_date - timedelta(days=30)].copy()
    if d.empty:
        return with_stamp(pd.DataFrame(), cfg, "No rows.")

    first_inning = (
        d.groupby(["game_pk", "pitcher"])["inning"]
        .min()
        .reset_index()
        .rename(columns={"inning": "first_inning"})
    )
    d = d.merge(first_inning, on=["game_pk", "pitcher"], how="left")

    # Relief proxy: pitchers whose first inning was after the 1st, plus later-inning rows.
    rel = d[(d["first_inning"] > 1) | (d["inning"] >= 5)]

    rows = []
    for team, g in rel.groupby("pitching_team", dropna=False):
        if not str(team).strip():
            continue

        rec = {
            "team": team,
            "window_days": 30,
            "relief_pitchers_seen": int(g["pitcher"].nunique()),
        }
        rec.update(summarize(g))
        rows.append(rec)

    return with_stamp(
        pd.DataFrame(rows),
        cfg,
        "Bullpen HR vulnerability proxy from recent Statcast relief appearances.",
    )


def manifest(cfg: Config) -> pd.DataFrame:
    base = cfg.raw_base_url.rstrip("/")

    files = [
        ("Statcast Rolling Windows", "statcast_batter_rolling.csv", "pull_timestamp", "TRUE"),
        ("Pitcher Statcast Rolling", "statcast_pitcher_rolling.csv", "pull_timestamp", "TRUE"),
        ("Pitch Type Batter Damage", "pitch_type_batter_damage.csv", "pull_timestamp", "TRUE"),
        ("Pitch Type Pitcher Vulnerability", "pitch_type_pitcher_vulnerability.csv", "pull_timestamp", "TRUE"),
        ("Bullpen HR Vulnerability", "bullpen_hr_vulnerability.csv", "pull_timestamp", "FALSE"),
    ]

    return pd.DataFrame(
        [
            {
                "sheet_name": sheet,
                "url": f"{base}/{filename}" if base else filename,
                "date_column": date_col,
                "required": required,
                "import_mode": "replace",
            }
            for sheet, filename, date_col, required in files
        ]
    )


def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_statcast_chunks(cfg)
    df = prepare_statcast(raw)

    outputs = {
        "statcast_batter_rolling.csv": batter_rolling(df, cfg),
        "statcast_pitcher_rolling.csv": pitcher_rolling(df, cfg),
        "pitch_type_batter_damage.csv": batter_pitch_type(df, cfg),
        "pitch_type_pitcher_vulnerability.csv": pitcher_pitch_type(df, cfg),
        "bullpen_hr_vulnerability.csv": bullpen_vulnerability(df, cfg),
        "manifest.csv": manifest(cfg),
    }

    summary = {
        "slate_date": cfg.slate_date.isoformat(),
        "generated_at": now_iso(cfg.timezone),
        "lookback_days": cfg.lookback_days,
        "windows": list(cfg.windows),
        "outputs": {},
    }

    for filename, out in outputs.items():
        write_csv(out, cfg.output_dir / filename)
        summary["outputs"][filename] = int(len(out))

    with open(cfg.output_dir / "refresh_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    ap.add_argument("--timezone", default=os.getenv("RUN_DATE_TZ", "America/Toronto"))
    ap.add_argument("--output-dir", default="data/exports")
    ap.add_argument("--windows", default="7,14,30")
    ap.add_argument("--lookback-days", type=int, default=35)
    ap.add_argument("--raw-base-url", default=os.getenv("RAW_BASE_URL", ""))
    args = ap.parse_args()

    cfg = Config(
        slate_date=parse_date(args.date, args.timezone),
        timezone=args.timezone,
        output_dir=Path(args.output_dir),
        raw_base_url=args.raw_base_url,
        windows=tuple(int(x.strip()) for x in args.windows.split(",") if x.strip()),
        lookback_days=args.lookback_days,
    )
    run(cfg)


if __name__ == "__main__":
    main()
