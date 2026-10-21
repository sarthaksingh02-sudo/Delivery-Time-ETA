# 6-slide presentation — Last-Mile Delivery ETA Prediction

## 1. The dispatch decision

**Customer question:** “When will my order arrive?” At dispatch, route, traffic, weather, vehicle and agent context can inform a duration estimate. Goal: improve ETA quality and operational visibility.

**Say:** “We estimate duration at dispatch, not the full end-to-end order promise. The prototype demonstrates the workflow; measured accuracy comes only after we train and test on the source CSV.”

## 2. Define the ML task

- Supervised regression; target is actual delivery duration in minutes (`time_taken(min)` / `Delivery_Time`).
- Primary KPI: held-out MAE in minutes; also RMSE, MAPE and R².
- Stakeholders: customers, dispatchers, agents and product/data teams.
- Proposed pilot target: 15% lower MAE than a simple baseline (to be measured).

## 3. Data and preparation

- Source: community Kaggle Amazon Delivery dataset; listing variants report roughly 43k records. Confirm the actual file's rows, schema and license.
- Inputs: agent age/rating, pickup/drop coordinates, weather, traffic, vehicle, area, festival, multiple deliveries, order time.
- Haversine distance; impute missing fields and one-hot encode categories inside a fitted pipeline.
- Exclude IDs and pickup time to limit memorization and post-dispatch leakage.

## 4. Explore and model

- EDA to show duration distribution and relationships with route distance, traffic, weather and agent context.
- Compare Ridge baseline, Random Forest and Extra Trees.
- Use a fixed 80/20 holdout and choose lowest MAE; use time/grouped validation before operational use.
- The app displays predictions, model status and real test metrics after CSV upload.

## 5. Live prototype

Open the app, set route distance, dispatch hour, traffic, weather, vehicle, area and agent context, then predict. Demonstrate scenario changes (e.g. heavier traffic or longer route) and discuss why the estimate moves.

**Important:** with no uploaded real CSV, the displayed model uses synthetic data and carries a demo warning. Do not present synthetic predictions as measured dataset performance.

## 6. Results, limitations and next steps

Results from the uploaded CSV: Random Forest selected; held-out MAE 16.85 min, RMSE 21.83 min, MAPE 15.57%, R² 0.821; held-out set 8,748 of 43,739 rows. Target range is 10–270 minutes, median 125 minutes. Discuss largest-error deliveries and verify the unusually high target values/units. Next: validate temporal/geographic generalization, calibrate ETA ranges, monitor drift and late-arrival rate, then pilot with human oversight.

## Demo setup checklist

1. Download the CSV from the exact Kaggle listing used by the course.
2. Install requirements; run `python app.py`.
3. Upload the CSV in the Model section; train and review the held-out metrics above.
4. Download `eta_model_package.zip` from the app and include it with the Flask project when deploying on Render. Check license and artifact size before public hosting.
