# Routewise — Last-Mile Delivery ETA

A responsive web app for estimating delivery duration at dispatch. The frontend is plain HTML/CSS/JavaScript; Flask exposes prediction, training and status APIs; scikit-learn holds the preprocessing and regression pipeline.

> **Demo data notice:** Without a Kaggle CSV or trained artifact, the app creates an illustrative model from synthetic data. The interface labels this state. Do not present synthetic predictions as real delivery performance.

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Upload the Kaggle CSV in the Model section to train three candidates, run three-fold cross-validation on the training partition, and compare held-out results. Training saves the fitted preprocessing-plus-estimator pipeline as `models/eta_model.joblib` and creates `deliverables/eta_model_package.zip` with the model, metrics, feature schema, runtime versions, checksum, and pinned dependencies. The original CSV is not included.

The package becomes downloadable in the interface after real-data training. You can also rebuild it with `python model_package.py`. The checked-in `deliverables/eta_model_package.zip` contains the fitted real-data pipeline, metrics, manifest, and pinned model dependencies. The server loads `models/eta_model.joblib` when present, or the package ZIP otherwise. The Python and scikit-learn versions are pinned to match the artifact.

## Deploy the web app on Render

GitHub stores the source; it does not run this Flask app. After pushing this repository, connect it to Render as a **Web Service** (or deploy the included `render.yaml` as a Blueprint). Use Python 3, build command `pip install -r requirements.txt`, start command `gunicorn app:app --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT`, and health-check path `/api/status`. The service loads the trained model from `deliverables/eta_model_package.zip`; the source CSV is intentionally excluded.

The frontend and Flask API are served from the same origin. Do not put Kaggle credentials in the repository. Verify the exact Kaggle listing's license before uploading or redistributing its data. The service accepts CSV training uploads up to 120 MB; for a public deployment, restrict access to the training endpoint before using sensitive data.

## Project files

- `frontend/`: responsive interface, styles and browser-side API calls.
- `app.py`, `server.py`: WSGI entry point and Flask routes.
- `eta_model.py`: schema normalization, Haversine feature, fitted preprocessing, training and inference.
- `model_package.py`: builds the standalone trained-model deployment package.
- `train.py`, `inspect_data.py`: command-line training and dataset inspection.
- `project_definition.md`, `dataset_report.md`, `experiments_7_8_plan.md`: lab documentation.
- `model_evaluation.md`: comparison-split results for the CSV currently in `data/`.
- `deliverables/eta_model_package.zip`: generated real-data model bundle for deployment.
- `deliverables/Last_Mile_ETA_Model_Results.pptx`: presentation with the trained model results and package details.
- `assessment_answers.md`: answers to the classification-metrics and time-series-splitting questions.

## Model method

- Regression target: `Delivery_Time` in minutes (`time_taken(min)` in common dataset variants).
- Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted inside each pipeline.
- Compare Ridge, Random Forest and Extra Trees. Use three-fold shuffled CV on up to 12,000 training rows; candidate selection and reported scores use the same random 20% comparison split, so results may be optimistic. Use an untouched temporal or geographic holdout for an unbiased performance estimate.
- Report CV MAE/RMSE and comparison-split MAE, RMSE, MAPE and R².
- Use temporal/geographic validation and monitoring before operational use. A random holdout does not establish production readiness.
