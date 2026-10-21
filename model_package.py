"""Build a versioned deployment package around a fitted ETA model pipeline."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent


def build_package(model_path: Path | str, metrics_path: Path | str, output_path: Path | str | None = None) -> bytes:
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    bundle = joblib.load(model_path)
    if bundle.get("mode") != "real_dataset":
        raise ValueError("Only a model trained on the uploaded delivery dataset can be packaged.")
    model_bytes = model_path.read_bytes()
    metrics_bytes = metrics_path.read_bytes() if metrics_path.is_file() else b"{}"
    manifest = {
        "package": "last-mile-eta-model",
        "format_version": 1,
        "artifact": "eta_model.joblib",
        "model_type": bundle.get("winner"),
        "target": "Delivery_Time (minutes)",
        "training_rows": int(bundle.get("training_rows", 0)),
        "features": bundle.get("features", []),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "load_note": "Load with joblib in a compatible Python/scikit-learn environment; use the included requirements.txt.",
    }
    requirements = (
        f"Flask>=3.0,<4\n"
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"joblib=={joblib.__version__}\n"
        "gunicorn>=22,<24\n"
    ).encode("utf-8")
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("eta_model.joblib", model_bytes)
        archive.writestr("metrics.json", metrics_bytes)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        archive.writestr("requirements.txt", requirements)
        archive.writestr("README.txt", b"Last-mile ETA trained model package\n\nThe joblib artifact includes fitted preprocessing and estimator steps. Install the included requirements, then load eta_model.joblib with joblib.load(). Verify compatible Python/scikit-learn versions from manifest.json. The original training CSV is not included.\n")
    data = memory.getvalue()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "eta_model.joblib"))
    parser.add_argument("--metrics", default=str(ROOT / "models" / "metrics.json"))
    parser.add_argument("--output", default=str(ROOT / "deliverables" / "eta_model_package.zip"))
    args = parser.parse_args()
    build_package(args.model, args.metrics, args.output)
    print(f"Model deployment package created: {args.output}")
