"""Inspect a downloaded delivery dataset and print a compact data-quality report."""
import argparse
import json
import pandas as pd
from eta_model import TARGET, canonicalize

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="Path to the downloaded Kaggle CSV")
args = parser.parse_args()
df = canonicalize(pd.read_csv(args.csv))
target = pd.to_numeric(df[TARGET], errors="coerce") if TARGET in df else None
report = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "column_names": list(df.columns),
    "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
    "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
    "duplicate_rows": int(df.duplicated().sum()),
    "target": None if target is None else {"column": TARGET, "missing": int(target.isna().sum()), "min": float(target.min()), "median": float(target.median()), "max": float(target.max())},
}
print(json.dumps(report, indent=2, default=str))
