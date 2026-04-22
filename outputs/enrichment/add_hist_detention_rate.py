"""Patch enriched_ltl_training.csv with hist_detention_rate.

Computes empirical-Bayes shrunk lane-level detention rate (same k=20
shrinkage used for all other hist_* features in build_enriched_ltl.py)
and adds it as a new column. Safe to re-run: drops existing column first.
"""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent / "enriched_ltl_training.csv"
K = 20

df = pd.read_csv(CSV, dtype={"Shipper Zip": str, "Consignee Zip": str})
df["Shipper Zip"] = df["Shipper Zip"].str.zfill(5)
df["Consignee Zip"] = df["Consignee Zip"].str.zfill(5)

if "hist_detention_rate" in df.columns:
    df = df.drop(columns=["hist_detention_rate"])

fired = (pd.to_numeric(df["C/Detention"], errors="coerce").fillna(0) != 0).astype(int)
global_rate = fired.mean()
print(f"Global detention rate: {global_rate:.4f} ({fired.sum()} / {len(fired)} shipments)")

df["_det_fired"] = fired
lane = df.groupby(["Shipper Zip", "Consignee Zip"])["_det_fired"].agg(["sum", "count"])
lane["hist_detention_rate"] = (lane["sum"] + K * global_rate) / (lane["count"] + K)
lane = lane.reset_index()[["Shipper Zip", "Consignee Zip", "hist_detention_rate"]]

df = df.drop(columns=["_det_fired"])
df = df.merge(lane, on=["Shipper Zip", "Consignee Zip"], how="left")

non_null = df["hist_detention_rate"].notna().sum()
print(f"hist_detention_rate: {non_null}/{len(df)} rows populated ({non_null/len(df)*100:.1f}%)")
print(f"  min={df['hist_detention_rate'].min():.4f}  "
      f"median={df['hist_detention_rate'].median():.4f}  "
      f"max={df['hist_detention_rate'].max():.4f}")

df.to_csv(CSV, index=False)
print(f"Wrote {CSV}")
