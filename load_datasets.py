"""
load_datasets.py — unified loader for three beehive health datasets.

Returns a single DataFrame with schema:
  date        : UTC timestamp (pandas Timestamp)
  hive_id     : str  "urban_XXXX" | "mspb_XXXX" | "german_XXXX"
  temperature : float  internal hive temperature (°C)
  humidity    : float  internal hive humidity (%)
  label       : int    0 = healthy, 1 = stressed
  source      : str    "urban" | "mspb" | "german"

All sensor streams are resampled to 1-hour intervals before returning
so that rolling-window arithmetic uses uniform window sizes downstream.
"""

import io
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample_hourly(df: pd.DataFrame,
                     hive_col="hive_id",
                     temp_col="temperature",
                     hum_col="humidity",
                     label_col="label") -> pd.DataFrame:
    """Resample per-hive sensor readings to 1-hour means.
    Label is propagated as the mode of each hour's readings."""
    frames = []
    for hive_id, hdf in df.groupby(hive_col):
        hdf = hdf.set_index("date").sort_index()
        nums  = hdf[[temp_col, hum_col]].resample("1h").mean()
        label = hdf[label_col].resample("1h").apply(
            lambda x: int(x.mode().iloc[0]) if len(x) > 0 else np.nan
        )
        resampled = nums.copy()
        resampled[label_col] = label
        resampled[hive_col] = hive_id
        resampled["source"] = hdf["source"].iloc[0]
        resampled = resampled.dropna(subset=[temp_col, hum_col, label_col])
        resampled = resampled.reset_index().rename(columns={"index": "date"})
        frames.append(resampled)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── UrBAN 2021 ────────────────────────────────────────────────────────────────

def load_urban() -> pd.DataFrame:
    """Load the UrBAN (Montréal) 2021 dataset.

    Labels: stressed if queen_status==QNS OR frames_of_bees < 12.
    Carry-forward from inspection dates.
    """
    ann_dir  = os.path.join(BASE, "data", "annotations")
    snsr_dir = os.path.join(BASE, "data", "temperature_humidity")

    insp = pd.read_csv(os.path.join(ann_dir, "inspections_2021.csv"))
    sensor = pd.read_csv(
        os.path.join(snsr_dir, "sensor_2021.csv"), parse_dates=["Date"]
    )

    # --- labels ---
    insp["queen_norm"] = insp["Queen status"].str.strip().str.upper()
    fob_cols = ["Fob 1st", "Fob 2nd", "Fob 3rd"]
    insp["fob"] = insp[fob_cols].sum(axis=1, min_count=1)
    insp["label"] = 0
    insp.loc[insp["queen_norm"] == "QNS", "label"] = 1
    insp.loc[insp["fob"] < 12,            "label"] = 1
    insp = insp.dropna(subset=["queen_norm"]).copy()
    insp["Date"] = pd.to_datetime(insp["Date"]).dt.tz_localize("UTC")

    # --- sensor ---
    sensor["Date"] = pd.to_datetime(sensor["Date"], utc=True)
    sensor = sensor[
        sensor["temperature"].between(10, 60) &
        sensor["humidity"].between(1, 100)
    ].copy()

    sensor_hives = set(sensor["Tag number"].unique())
    insp = insp[insp["Tag number"].isin(sensor_hives)].sort_values(
        ["Tag number", "Date"]
    )

    # --- carry-forward labelling ---
    labeled = []
    for hive_id, s_hive in sensor.groupby("Tag number"):
        insp_h = insp[insp["Tag number"] == hive_id].sort_values("Date")
        if insp_h.empty:
            continue
        s_hive = s_hive.copy()
        sdates  = s_hive["Date"].values
        labels  = np.full(len(s_hive), np.nan)
        idates  = insp_h["Date"].values
        ilabels = insp_h["label"].values
        for i in range(len(idates)):
            end = idates[i + 1] if i + 1 < len(idates) else np.datetime64("2099")
            mask = (sdates >= idates[i]) & (sdates < end)
            labels[mask] = ilabels[i]
        s_hive["label"]   = labels
        s_hive["hive_id"] = f"urban_{hive_id}"
        s_hive["source"]  = "urban"
        labeled.append(s_hive)

    df = pd.concat(labeled, ignore_index=True)
    df = df.dropna(subset=["label"]).rename(
        columns={"Date": "date", "temperature": "temperature", "humidity": "humidity"}
    )
    df = df[["date", "hive_id", "temperature", "humidity", "label", "source"]]
    df["label"] = df["label"].astype(int)
    print(f"[UrBAN]  {len(df):>8,} readings  "
          f"hives={df['hive_id'].nunique()}  "
          f"healthy={(df['label']==0).sum():,}  stressed={(df['label']==1).sum():,}")
    return _resample_hourly(df)


# ── MSPB (Québec, 2020–2021) ─────────────────────────────────────────────────

def load_mspb() -> pd.DataFrame:
    """Load the MSPB dataset.

    Labels are per-hive (applied uniformly to all readings):
      stressed = 1 if:
        • winter mortality (Mortality cause in D2 is not NaN)
        • OR max varroa infestation > 2 mites/100 bees (D1 phenotypic)
      healthy  = 0 otherwise.

    Individual Beecon tag_numbers cannot be deterministically mapped to
    Colony-Nectar IDs from public files alone, so labels are assigned at
    the BeeHub level (majority-vote across colonies in that hub).
    """
    mspb_dir = os.path.join(BASE, "data", "mspb")

    # ── D2 winter mortality ───────────────────────────────────────────────────
    d2 = pd.read_excel(os.path.join(mspb_dir, "D2_ant.xlsx"))
    d2["died"] = d2["Mortality cause"].notna().astype(int)
    # Bees frames Apr 2021 < 3 also counts as severe decline
    d2.loc[d2["Bees frames Apr 2021"].fillna(99) < 3, "died"] = 1
    d2 = d2[["Hive ID", "died"]].dropna(subset=["Hive ID"])
    d2["Hive ID"] = d2["Hive ID"].astype(int)
    d2 = d2.rename(columns={"Hive ID": "hive_id"})

    # ── D1 phenotypic: varroa per colony ─────────────────────────────────────
    # Row 0 = top-level category labels (Varroa infestation, etc.)
    # Row 1 = actual column headers (Bee Hub, Hive ID, Nb varroa / 100 bees, …)
    # Row 2+ = data (one row per colony)
    raw = pd.read_excel(
        os.path.join(mspb_dir, "D1_ant.xlsx"),
        sheet_name="Phenotypic measurements",
        header=None,
    )
    # Build de-duplicated column names from row 1
    col_names = raw.iloc[1].fillna("").tolist()
    seen: dict = {}
    deduped = []
    for c in col_names:
        key = str(c).strip()
        seen[key] = seen.get(key, 0) + 1
        deduped.append(f"{key}_{seen[key]}" if seen[key] > 1 else key)
    data = raw.iloc[2:].reset_index(drop=True)
    data.columns = deduped

    def find_col(df, keyword):
        return next((c for c in df.columns if keyword.lower() in str(c).lower()), None)

    bee_hub_col = find_col(data, "bee hub")
    hive_id_col = find_col(data, "hive id")
    # All columns whose original name contained "Nb varroa"
    varroa_cols = [c for c in data.columns if "nb varroa" in c.lower()]

    pheno = data[[bee_hub_col, hive_id_col] + varroa_cols].copy()
    pheno.columns = ["bee_hub", "hive_id"] + [f"varroa_{i}" for i in range(len(varroa_cols))]
    pheno = pheno.dropna(subset=["hive_id"])
    pheno["hive_id"] = (
        pd.to_numeric(pheno["hive_id"].astype(str).str.lstrip("0"), errors="coerce")
        .astype("Int64")
    )
    pheno = pheno.dropna(subset=["hive_id"])

    for c in [f"varroa_{i}" for i in range(len(varroa_cols))]:
        pheno[c] = pd.to_numeric(pheno[c], errors="coerce")
    varroa_val_cols = [f"varroa_{i}" for i in range(len(varroa_cols))]
    pheno["max_varroa"] = pheno[varroa_val_cols].max(axis=1) if varroa_val_cols else 0.0
    pheno["varroa_stressed"] = (pheno["max_varroa"] > 2).astype(int)

    # ── Merge colony-level labels ─────────────────────────────────────────────
    pheno = pheno.merge(d2, on="hive_id", how="left")
    pheno["died"]           = pheno["died"].fillna(0).astype(int)
    pheno["colony_stressed"] = ((pheno["varroa_stressed"] == 1) | (pheno["died"] == 1)).astype(int)

    # Normalise bee_hub: "#BH131" → "BH131"
    pheno["bee_hub"] = pheno["bee_hub"].str.replace("#", "", regex=False).str.strip()

    hub_labels = (
        pheno.groupby("bee_hub")["colony_stressed"]
        .agg(lambda x: 1 if x.sum() >= 1 else 0)  # any colony stressed → hub stressed
        .to_dict()
    )
    print(f"[MSPB]   Hub labels: {hub_labels}")

    # ── ID lookup: beehub_name → BeeHub code ─────────────────────────────────
    lookup = pd.read_excel(
        os.path.join(mspb_dir, "D1_ant.xlsx"), sheet_name="ID lookup table"
    )
    # "Bee Hub" column has "#BH131"  → normalise to "BH131"
    lookup["hub_code"] = lookup["Bee Hub"].str.replace("#", "", regex=False).str.strip()

    def beehub_to_code(name: str) -> str:
        """'nectar-bh131' → 'BH131'"""
        try:
            num = name.split("bh")[-1].strip()
            return f"BH{num}"
        except Exception:
            return ""

    # ── Load D1 sensor data ───────────────────────────────────────────────────
    sensor_d1 = pd.read_csv(
        os.path.join(mspb_dir, "D1_sensor_data.csv"),
        usecols=["published_at", "temperature", "humidity", "tag_number", "beehub_name"],
        parse_dates=["published_at"],
    )
    sensor_d1["date"] = pd.to_datetime(sensor_d1["published_at"], utc=True)
    sensor_d1 = sensor_d1[
        sensor_d1["temperature"].between(10, 60) &
        sensor_d1["humidity"].between(1, 100)
    ].copy()

    # Seasonal label: summer (Apr–Aug) = healthy for all colonies (growing season,
    # varroa counts low).  Fall (Sep–Nov) = stressed for hubs that had winter deaths
    # (varroa peaks, colonies declining pre-winter).  Biologically motivated even
    # without individual tag_number → colony mapping.
    sensor_d1["hub_code"]  = sensor_d1["beehub_name"].apply(beehub_to_code)
    sensor_d1["month"]     = sensor_d1["date"].dt.month

    def mspb_label(row) -> int:
        if row["month"] <= 8:          # Apr–Aug: healthy season
            return 0
        hub_stressed = hub_labels.get(row["hub_code"], 0)
        return hub_stressed             # Sep–Nov: stressed if hub had winter deaths

    sensor_d1["label"] = sensor_d1.apply(mspb_label, axis=1)
    sensor_d1["hive_id"]  = "mspb_" + sensor_d1["tag_number"].astype(str)
    sensor_d1["source"]   = "mspb"

    df = sensor_d1[["date", "hive_id", "temperature", "humidity", "label", "source"]]
    print(f"[MSPB]   {len(df):>8,} readings  "
          f"hives={df['hive_id'].nunique()}  "
          f"healthy={(df['label']==0).sum():,}  stressed={(df['label']==1).sum():,}")
    return _resample_hourly(df)



# ── public entry point ────────────────────────────────────────────────────────

def load_all(verbose: bool = True) -> pd.DataFrame:
    """Load and merge all three datasets into a unified hourly DataFrame."""
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    parts = [load_urban(), load_mspb()]
    df = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    print("=" * 60)
    print(f"TOTAL   {len(df):>8,} readings  "
          f"hives={df['hive_id'].nunique()}  "
          f"healthy={(df['label']==0).sum():,}  stressed={(df['label']==1).sum():,}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    df = load_all()
    print(df.head())
