# Remaining Lab Experiments — Proposed Scope

The brief supplied details for Experiments 1–6 only. The final two are proposed here so the eight-experiment schedule has a usable completion plan; replace them if the course brief specifies different titles or activities.

## Experiment 7 — Deployment and User Demo (2 hours)

**Input:** selected pipeline and evaluation report.  
**Output:** running Streamlit prototype and deployment instructions.  
**Activities:** run locally with `streamlit run app.py`; demonstrate input conditions and a duration estimate; upload the course CSV to train the selected model; confirm deployment target, repository and license terms; deploy through Streamlit Community Cloud after pushing to a connected GitHub repository. The app labels synthetic demo mode and never presents its output as test accuracy.

## Experiment 8 — Error Review and Operational Readiness (2 hours)

**Input:** real held-out predictions and user feedback.  
**Output:** final presentation, error analysis and follow-up plan.  
**Activities:** review worst absolute errors and segments by traffic/weather/distance; assess temporal and geographic generalization; propose calibrated ETA ranges, drift checks and late-arrival monitoring; present MAE/RMSE with units and limitations. Do not claim a measured improvement until it is calculated from the actual CSV and baseline.
