# Project Definition — Last-Mile Delivery ETA Prediction

## Problem

At dispatch, customers and operations teams need an estimate of the remaining time until a delivery is completed. The project predicts the actual delivery duration (minutes) from information available at dispatch, helping set customer expectations and prioritize operational attention.

## ML formulation and target

Supervised regression. Target: `Delivery_Time` in minutes (often named `time_taken(min)` in the Kaggle CSV). Prediction point: dispatch. Exclude information only available after dispatch, particularly pickup time if dispatch precedes pickup.

## Success measures

Primary offline metrics: MAE and RMSE on held-out deliveries, with MAE as the easiest measure to communicate (“typical absolute error in minutes”). Supporting metrics: MAPE and R². Use three-fold shuffled cross-validation on the training partition for stability checks, then report the candidate comparison on the same held-out records. Proposed pilot KPI: reduce MAE by at least 15% versus a simple distance/traffic baseline, while monitoring the 90th percentile absolute error and late-arrival rate. The 15% is a proposed target, not a measured result.

## Users and stakeholders

- Customers: need a clear, timely ETA.
- Dispatch/operations: need to allocate agent attention and manage exceptions.
- Delivery agents: ETA should account for route and operating conditions without unfair individual ranking.
- Product and data teams: need reliable event timestamps, monitoring and feedback loops.

## Constraints and risks

Predictions depend on reliable route coordinates and dispatch-time weather/traffic snapshots. Historical data may reflect one geography and period, so performance can shift across cities, seasons and festivals. A Kaggle community dataset is not necessarily representative of a live service; validate its provenance/license. Shuffled cross-validation and a random held-out split are classroom evaluation, not production evidence. Do not use the ETA alone for punitive agent decisions. Return a calibrated range and refresh estimates as conditions change before customer launch.

## Deliverable

An interactive Streamlit prototype, reproducible preprocessing/training pipeline, held-out model comparison and documented data limitations. No accuracy claim is made until the actual dataset has been loaded and evaluated.
