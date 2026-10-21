# Assessment Answers — Delivery ETA Project

## Question 1: F1 score, precision, accuracy and recall for two models

These four metrics are classification metrics. The ETA model in this project is a **regression** model: it predicts a continuous delivery duration in minutes. Therefore, accuracy, precision, recall and F1 do not directly apply to its predictions. Report MAE, RMSE, MAPE and R² for the current regression task.

To calculate those classification metrics, define a classification target first. A reasonable delivery example is `Late` versus `On time`, but it requires a promised delivery time or an explicitly agreed SLA cutoff. The current CSV contains actual delivery duration and does not include the promised ETA/SLA, so it does not by itself provide a defensible late/on-time label. Do not invent numeric scores or a threshold after seeing the predictions.

If an SLA and two classifiers are available, evaluate both on the **same untouched test rows** and use the same positive class (for example, `Late`). For a confusion matrix:

- **TP:** late deliveries correctly predicted late.
- **FP:** on-time deliveries incorrectly predicted late.
- **FN:** late deliveries incorrectly predicted on time.
- **TN:** on-time deliveries correctly predicted on time.

Then calculate:

```text
Accuracy  = (TP + TN) / (TP + FP + FN + TN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Precision answers “When the model flags a delivery as late, how often is it truly late?” Recall answers “Of all late deliveries, how many did it catch?” F1 balances precision and recall. Accuracy can be misleading when one class is much more common than the other; a model that predicts the majority class every time may still have high accuracy.

Example scikit-learn calculation once true and predicted class labels exist:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

scores = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_late": precision_score(y_true, y_pred, pos_label="Late", zero_division=0),
    "recall_late": recall_score(y_true, y_pred, pos_label="Late", zero_division=0),
    "f1_late": f1_score(y_true, y_pred, pos_label="Late", zero_division=0),
}
```

**Short answer for this project:** “Those four scores do not apply to the ETA regression model. I would use MAE and RMSE. To calculate classification metrics, I would first define an SLA-based late/on-time target, then compare two classifiers on the same test set.”

## Question 2: Why can randomized time-series data score better than timestamp-preserving data?

Randomization often makes train and test rows look more alike. Each split receives a mixture of early and late periods, so the model can see examples of the patterns represented in its test rows. A chronological split instead trains on the past and evaluates on a later period. If demand, traffic, weather, agents or operating policies changed over time, this exposes distribution shift and the score can be worse.

There is another important possibility: randomizing individual time-series rows can leak information. Nearby timestamps are often highly correlated, and repeated orders, agents or routes may appear in both train and test. The model then sees near-duplicates or future-period patterns during training. Random-split performance can look much better without representing future forecasting ability.

Random splitting is appropriate when rows are independent and identically distributed for the intended use. For a model that will predict future deliveries, preserve time order: train on older dates, validate on later dates, and test once on the newest untouched period. Rolling-origin or expanding-window validation gives several past-to-future checks. Group by delivery, agent or route where repeated entities could leak across splits.

**Short answer:** “Randomization can improve the measured score because it makes the train and test distributions more similar, but it may also leak correlated future information. For a time-dependent deployment, chronological validation is the more honest estimate of future performance, even if its score is lower.”
