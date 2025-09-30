"""
add_targets.py

Append high-impact extras to v2:
- rw_next24 (rapid weakening)
- max_wmo_wind_next48 (helper regression)
- major_within48 (Cat 3+ in 48h)
- landfall_within24 / landfall_within48 (if GeoPandas available)

Also refreshes the dictionary.

Run:  python scripts/add_targets.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
MASTER = OUT / "model_ready_2000_2024_v2.parquet"
DICT_CSV = OUT / "model_ready_2000_2024_v2_dictionary.csv"
DICT_MD  = OUT / "DATA_DICTIONARY.md"

H6, H12, H24, H48 = 1, 2, 4, 8

def add_rw_major(df: pd.DataFrame) -> pd.DataFrame:
    # rw_next24
    if "has_next24" not in df:
        df["has_next24"] = df.groupby("storm_id")["iso_time"].shift(-H24).notna()
    delta24 = df["wmo_wind_next24"] - df["wmo_wind"]
    df["rw_next24"] = ((df["has_next24"]) & (delta24 <= -30.0)).astype("int8")

    # max_wmo_wind_next48
    tmp_cols = []
    for k in range(1, H48+1):
        c = f"__wmo_tplus_{6*k}h"
        df[c] = df.groupby("storm_id")["wmo_wind"].shift(-k)
        tmp_cols.append(c)
    df["max_wmo_wind_next48"] = df[tmp_cols].max(axis=1).astype("float32")
    df.drop(columns=tmp_cols, inplace=True)

    # major_within48
    df["major_within48"] = ((df["max_wmo_wind_next48"].notna()) & (df["max_wmo_wind_next48"] >= 96.0)).astype("int8")
    return df

def add_landfall(df: pd.DataFrame) -> pd.DataFrame:
    # optional: needs geopandas + shapely (and geodatasets or URL fallback)
    try:
        import geopandas as gpd
        import shapely
        from shapely.ops import unary_union
        try:
            import geodatasets as gd
            land_path = gd.get_path("naturalearth.land")
            world = gpd.read_file(land_path).to_crs(epsg=4326)
        except Exception:
            url = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
            world = gpd.read_file(url).to_crs(epsg=4326)
        if "continent" in world.columns:
            world = world[world["continent"] != "Antarctica"]
        LAND = unary_union(world.geometry.values)

        # ensure next48 track cols
        if "lat_next48" not in df or "lon_next48" not in df:
            df["lat_next48"] = df.groupby("storm_id")["lat"].shift(-H48)
            df["lon_next48"] = df.groupby("storm_id")["lon"].shift(-H48)

        def contains(geom, lat, lon):
            m = lat.notna() & lon.notna()
            out = np.zeros(len(lat), dtype=bool)
            if m.any():
                pts = shapely.points(lon[m].to_numpy(), lat[m].to_numpy())
                out[m] = shapely.contains(geom, pts)
            return out

        on_land_now = contains(LAND, df["lat"], df["lon"])
        on6  = contains(LAND, df.get("lat_next6"),  df.get("lon_next6"))   if "lat_next6"  in df else np.zeros(len(df), bool)
        on12 = contains(LAND, df.get("lat_next12"), df.get("lon_next12"))  if "lat_next12" in df else np.zeros(len(df), bool)
        on24 = contains(LAND, df.get("lat_next24"), df.get("lon_next24"))  if "lat_next24" in df else np.zeros(len(df), bool)
        on48 = contains(LAND, df["lat_next48"], df["lon_next48"])

        df["landfall_within24"] = ((~on_land_now) & (on6 | on12 | on24)).astype("int8")
        df["landfall_within48"] = ((~on_land_now) & (on6 | on12 | on24 | on48)).astype("int8")
        print("✔ landfall_within24/48 added")
    except Exception as e:
        print(f"⚠ landfall skipped ({e.__class__.__name__}: {e}) — install geopandas+shapely+geodatasets to enable.")
    return df

def refresh_dictionary(df: pd.DataFrame):
    dd = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes], "hint": ""})
    hints = {
        "rw_next24":"Rapid Weakening (≤ −30 kt in 24h)",
        "max_wmo_wind_next48":"Max wind (kt) within the next 48h (6..48h ahead)",
        "major_within48":"1 if max_wmo_wind_next48 ≥ 96 kt (Cat 3+) within 48h",
        "landfall_within24":"Touches land within 24h (ocean→land)",
        "landfall_within48":"Touches land within 48h (ocean→land)",
        "lat_next48":"Latitude 48h ahead",
        "lon_next48":"Longitude 48h ahead",
    }
    dd["hint"] = dd["column"].map(hints).fillna("")
    dd.to_csv(DICT_CSV, index=False)
    # append nicely to MD
    with open(DICT_MD, "a", encoding="utf-8") as f:
        f.write("### Added extras\n\n")
        sub = dd[dd["column"].isin(list(hints.keys()))]
        if not sub.empty:
            f.write(sub.to_markdown(index=False) + "\n\n")

def main():
    assert MASTER.exists(), f"Missing master v2: {MASTER}"
    df = pd.read_parquet(MASTER)
    df = add_rw_major(df)
    df = add_landfall(df)
    df.to_parquet(MASTER, index=False)
    refresh_dictionary(df)

    print("✔ updated master with extras")
    print("rw_next24:", df["rw_next24"].value_counts(dropna=False).to_dict())
    if "major_within48" in df:
        print("major_within48:", df["major_within48"].value_counts(dropna=False).to_dict())
    if "landfall_within24" in df:
        print("landfall_within24:", df["landfall_within24"].value_counts(dropna=False).to_dict())

if __name__ == "__main__":
    main()
