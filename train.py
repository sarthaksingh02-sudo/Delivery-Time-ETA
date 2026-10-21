"""Train on the Kaggle CSV: python train.py data/amazon_delivery.csv"""
import argparse
import json
import pandas as pd
from eta_model import train
from model_package import build_package

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="Path to the downloaded Kaggle CSV")
parser.add_argument("--output", default="models", help="Artifact output directory")
parser.add_argument("--package-output", default="deliverables/eta_model_package.zip", help="Deployment package output path")
args = parser.parse_args()
result = train(pd.read_csv(args.csv), args.output)
build_package(f"{args.output}/eta_model.joblib", f"{args.output}/metrics.json", args.package_output)
result["package"] = args.package_output
print(json.dumps(result, indent=2))
