# Model Evaluation — Uploaded Amazon Delivery CSV

## Dataset and split

- 43,739 records; all target values usable; target `Delivery_Time` ranges from 10 to 270 minutes (median 125).
- 80/20 random split: 34,991 training rows and 8,748 comparison-split rows.
- Three-fold shuffled cross-validation on a reproducible sample of 12,000 training rows.
- Missing ratings and weather are imputed inside each fitted pipeline. The model uses agent age/rating, Haversine distance, weather, traffic, vehicle, area, order hour, and product category.
- `Multiple_Deliveries`, `Festival`, and `City` are not present in this CSV and were excluded. Pickup time and identifiers are excluded.

## Candidate metrics

| Model | CV MAE (min) | CV RMSE (min) | Split MAE (min) | Split RMSE (min) | Split MAPE | Split R² |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 17.92 | 23.15 | **16.85** | **21.83** | 15.57% | **0.821** |
| Extra Trees | 19.02 | 24.66 | 17.87 | 23.17 | 16.84% | 0.799 |
| Ridge baseline | 26.28 | 33.31 | 25.82 | 32.79 | 26.26% | 0.596 |

## Selection and interpretation

Random Forest had the lowest MAE on the 20% comparison split. The same split was used to select among candidates and report these metrics, so the score may be optimistic. Its MAE was 16.85 minutes on this uploaded file; this is not a guarantee of live ETA accuracy.

The target median of 125 minutes and maximum of 270 minutes are unusually high for many last-mile ETA expectations. Verify the target units and that the file matches the course's intended dataset before presenting these results. The evaluation is a shuffled comparison split, not an untouched final test or a temporal/geographic holdout. Review worst errors and evaluate on a separate future-time or location-based set before making an operational claim.
