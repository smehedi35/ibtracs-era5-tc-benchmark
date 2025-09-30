"""
build_dataset.py

Upgrade v1 -> v2 (benchmark-ready):
- Normalize columns (storm_id, iso_time, lat, lon, wmo_wind, …)
- Add multi-horizon targets (wind + track for 6/12/24h)
- Add masks, RI (24h), lifecycle labels (dissipates/ET), splits, weights
- Write: model_ready_2000_2024_v2.parquet
- Emit slim task views to outputs/tasks/
- Emit data dictionary (CSV + Markdown)

Run:  python scripts/build_dataset.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATASET_VERSION = "2.0.0"
H6, H12, H24, H48 = 1, 2, 4, 8  # 6h cadence

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
TASKS = OUT / "tasks"
TASKS.mkdir(parents=True, exist_ok=True)

SRC_V1   = OUT / "model_ready_2000_2024.parquet"           # your v1
MASTER_V2 = OUT / "model_ready_2000_2024_v2.parquet"
DICT_CSV  = OUT / "model_ready_2000_2024_v2_dictionary.csv"
DICT_MD   = OUT / "DATA_DICTIONARY.md"

# ---------- helpers ----------
def pick(df, *cands):
    cn = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cn: return cn[c.lower()]
    return None

def gshift(df, col, n):
    return df.groupby("storm_id", sort=False)[col].shift(-n)

def ensure_basics(df):
    # rename common aliases
    ren = {}
    for target, cands in {
        "storm_id": ("storm_id","sid","serial_num","track_id","id"),
        "iso_time": ("iso_time","time","datetime","timestamp","date_time","iso_datetime"),
        "lat": ("lat","latitude","lat_deg","Latitude"),
        "lon": ("lon","longitude","lon_deg","Longitude"),
        "wmo_wind": ("wmo_wind","wind_wmo","vmax","usa_wind","wind"),
        "season": ("season","year"),
        "name": ("name","storm_name","tc_name"),
        "basin": ("basin","ocean_basin"),
    }.items():
        f = pick(df, *cands)
        if f and f != target: ren[f] = target
    if ren: df = df.rename(columns=ren)

    if "storm_id" not in df:
        if "season" in df and "name" in df:
            df["storm_id"] = df["season"].astype(str) + "_" + df["name"].astype(str).str.strip().str.replace(r"\s+","_", regex=True)
        else:
            raise ValueError("Need storm_id or (season+name) to build groups.")

    if "iso_time" not in df:
        tcol = pick(df, "iso_time","time","datetime","timestamp","date_time","iso_datetime")
        if not tcol: raise ValueError("Missing time column (iso_time/time/...).")
        df["iso_time"] = df[tcol]

    # parse + sort
    df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce", utc=True)
    if df["iso_time"].isna().any():
        raise ValueError("Unparsable timestamps in iso_time.")
    df = df.sort_values(["storm_id","iso_time"]).reset_index(drop=True)

    # normalize lon
    if df["lon"].max() > 180:
        df["lon"] = ((df["lon"] + 180) % 360) - 180

    # ensure season
    if "season" not in df:
        df["season"] = df["iso_time"].dt.year.astype(int)
    return df

def write_dictionary(df):
    hints = {
        # IDs
        "storm_id":"Unique storm identifier (e.g., season + name)",
        "season":"Storm season year (UTC)",
        "name":"Storm name (if available)",
        "basin":"Ocean basin code",
        "iso_time":"Advisory time (UTC), 6-hourly",
        # features
        "lat":"Storm latitude (°N)", "lon":"Storm longitude (°E, −180..180)",
        "wmo_wind":"Max sustained wind (kt, WMO)", "wmo_pres":"Central pressure (hPa)",
        "sst_c":"Sea surface temperature (°C)", "t2m_c":"2-m air temperature (°C)",
        "msl_hpa":"Mean sea-level pressure (hPa)", "u10":"10-m u-wind (m/s)",
        "v10":"10-m v-wind (m/s)", "wind10":"10-m wind speed (m/s)", "tp":"Total precipitation (m)",
        # targets
        "wmo_wind_next6":"Wind (kt) 6h ahead", "wmo_wind_next12":"Wind (kt) 12h ahead",
        "wmo_wind_next24":"Wind (kt) 24h ahead",
        "lat_next6":"Latitude 6h ahead","lon_next6":"Longitude 6h ahead",
        "lat_next12":"Latitude 12h ahead","lon_next12":"Longitude 12h ahead",
        "lat_next24":"Latitude 24h ahead","lon_next24":"Longitude 24h ahead",
        # masks
        "has_next6":"Target exists at +6h","has_next12":"Target exists at +12h",
        "has_next24":"Target exists at +24h","has_next48":"Target exists at +48h",
        # lifecycle
        "ri_next24":"Rapid Intensification (≥30 kt/24h)",
        "dissipates_within48":"1 if no advisory at +48h",
        "et_within48":"1 if extratropical within 48h",
        # infra
        "split":"train/val/test (storm-level, year-based)",
        "sample_weight":"Inverse rows per storm",
        "dataset_version":"Semantic version (e.g., 2.0.0)",
    }
    dd = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "hint": [hints.get(c,"") for c in df.columns],
    })
    dd.to_csv(DICT_CSV, index=False)
    # Pretty Markdown, sectioned (short)
    sections = {
        "### Core identifiers": ["storm_id","season","name","basin","iso_time"],
        "### Core features (inputs)": ["lat","lon","wmo_wind","wmo_pres","sst_c","t2m_c","msl_hpa","u10","v10","wind10","tp"],
        "### Intensity targets": ["wmo_wind_next6","wmo_wind_next12","wmo_wind_next24"],
        "### Track targets": ["lat_next6","lon_next6","lat_next12","lon_next12","lat_next24","lon_next24"],
        "### Horizon masks": ["has_next6","has_next12","has_next24","has_next48"],
        "### Lifecycle": ["ri_next24","dissipates_within48","et_within48"],
        "### Infrastructure & metadata": ["split","sample_weight","dataset_version"],
    }
    with open(DICT_MD, "w", encoding="utf-8") as f:
        f.write("## Data Dictionary\n\n")
        for title, cols in sections.items():
            sub = dd[dd["column"].isin(cols)].copy()
            if sub.empty: continue
            sub["__o"] = sub["column"].apply(cols.index)
            sub = sub.sort_values("__o").drop(columns="__o")
            f.write(title + "\n\n" + sub.to_markdown(index=False) + "\n\n")

def main():
    assert SRC_V1.exists(), f"Missing source: {SRC_V1}"
    df = pd.read_parquet(SRC_V1)
    df = ensure_basics(df)

    # targets (wind + track)
    df["wmo_wind_next6"]  = gshift(df, "wmo_wind", H6)
    df["wmo_wind_next12"] = gshift(df, "wmo_wind", H12)
    df["wmo_wind_next24"] = gshift(df, "wmo_wind", H24)

    for h, lab in [(H6,"6"),(H12,"12"),(H24,"24")]:
        df[f"lat_next{lab}"] = gshift(df, "lat", h)
        df[f"lon_next{lab}"] = gshift(df, "lon", h)

    # masks
    df["has_next6"]  = gshift(df, "iso_time", H6).notna()
    df["has_next12"] = gshift(df, "iso_time", H12).notna()
    df["has_next24"] = gshift(df, "iso_time", H24).notna()
    df["has_next48"] = gshift(df, "iso_time", H48).notna()

    # RI 24h
    delta24 = df["wmo_wind_next24"] - df["wmo_wind"]
    df["ri_next24"] = ((df["has_next24"]) & (delta24 >= 30.0)).astype("int8")

    # lifecycle: dissipates + ET (if status present)
    df["dissipates_within48"] = (~df["has_next48"]).astype("int8")
    status_cols = [c for c in ["storm_status","nature","type","status"] if c in df]
    et_future_any = False
    if status_cols:
        is_et = pd.Series(False, index=df.index)
        for c in status_cols:
            s = df[c].astype(str).str.upper()
            is_et = is_et | s.str.contains("EXTRATROP", na=False) | s.str.contains(r"\bET\b", na=False)
        et = pd.Series(False, index=df.index)
        for k in range(1, H48+1):
            et = et | df.groupby("storm_id", sort=False)[is_et].shift(-k).fillna(False)
        et_future_any = et
    df["et_within48"] = (et_future_any if isinstance(et_future_any, pd.Series) else pd.Series(False, index=df.index)).astype("int8")

    # splits (storm first year)
    first_year = df.groupby("storm_id")["season"].transform("min")
    def split_from_year(y):
        if 2000 <= y <= 2016: return "train"
        if 2017 <= y <= 2020: return "val"
        if 2021 <= y <= 2024: return "test"
        return "other"
    df["split"] = first_year.apply(split_from_year).astype("category")

    # sample weights
    rows_per_storm = df.groupby("storm_id")["storm_id"].transform("count").astype("int32")
    df["sample_weight"] = (1.0 / rows_per_storm).astype("float32")

    # metadata
    df["dataset_version"] = DATASET_VERSION

    # save master
    df.to_parquet(MASTER_V2, index=False)
    print(f"✔ wrote {MASTER_V2.name}  rows={len(df)}  cols={df.shape[1]}")

    # slim task views
    INT_TGT = ["wmo_wind_next6","wmo_wind_next12","wmo_wind_next24"]
    TRK_TGT = ["lat_next6","lon_next6","lat_next12","lon_next12","lat_next24","lon_next24"]
    df.dropna(subset=INT_TGT, how="all").to_parquet(TASKS/"tc_intensity_forecast_v1.parquet", index=False)
    df.dropna(subset=TRK_TGT,  how="all").to_parquet(TASKS/"tc_track_forecast_v1.parquet", index=False)
    df.to_parquet(TASKS/"tc_ri_classification_v1.parquet", index=False)
    df.to_parquet(TASKS/"tc_lifecycle_v1.parquet", index=False)
    print("✔ wrote task views to outputs/tasks/")

    # dictionary
    write_dictionary(df)
    print("✔ wrote dictionary CSV + MD")

if __name__ == "__main__":
    main()
