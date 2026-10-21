# Training data

Place the downloaded Kaggle file here as:

```text
data/amazon_delivery.csv
```

The CSV is ignored by Git. You can either upload it from this folder in the web app, or train from PowerShell:

```powershell
python train.py data/amazon_delivery.csv
```

Training creates `models/eta_model.joblib`, evaluation metrics under `models/`, and `deliverables/eta_model_package.zip`. Do not commit or publicly redistribute the source CSV unless its license and your course rules allow it.
