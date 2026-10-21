# Dataset Description Report

## Source and access

- Candidate source: [Amazon Delivery Dataset on Kaggle](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset)
- Dataset is community-contributed. The exact CSV version, terms and license must be checked on the Kaggle listing used for this submission before redistribution or public hosting. License is **not yet confirmed**.
- Expected scale in the project brief: approximately 43,000 records. **Inspected file: 43,739 rows × 16 columns.**
- License: **not confirmed**. Check the Kaggle listing's license and your course rules before public use or redistribution.

## Feature dictionary

| Dataset field (common variants) | Meaning | Model treatment |
|---|---|---|
| `Delivery_person_Age` / `Agent_Age` | Delivery agent age | Numeric; median imputation and scaling |
| `Delivery_person_Ratings` / `Agent_Rating` | Agent rating | Numeric; median imputation and scaling |
| `Restaurant_latitude`, `Restaurant_longitude` / `Store_Latitude`, `Store_Longitude` | Pickup coordinates | Used with drop coordinates for Haversine route distance |
| `Delivery_location_latitude`, `Delivery_location_longitude` / `Drop_Latitude`, `Drop_Longitude` | Delivery coordinates | Used for Haversine route distance |
| `Weatherconditions` / `Weather` | Weather condition | Categorical; impute and one-hot encode |
| `Road_traffic_density` / `Traffic` | Traffic level | Categorical; impute and one-hot encode |
| `Type_of_vehicle` / `Vehicle` | Agent vehicle type | Categorical; impute and one-hot encode |
| `Area` | Delivery area category | Categorical; impute and one-hot encode |
| `Category` | Product category | Categorical; impute and one-hot encode |
| `multiple_deliveries` | Number of concurrent/combined deliveries | Numeric |
| `Festival` | Festival indicator | Categorical |
| `City` | City category | Categorical |
| `Order_Date`, `Time_Orderd` | Order date and time | Order hour and weekday; confirm dispatch-time availability |
| `Time_Order_picked` | Pickup time | Excluded due possible post-dispatch leakage |
| `Delivery_person_ID`, `ID`, `Order_ID` | Record/agent identifiers | Excluded from model features |
| `time_taken(min)` / `Time_taken (min)` | Actual delivery duration in minutes | Regression target |

Field availability and exact naming differ by dataset variant. Verify against the downloaded file.

## Target profile and quality checks

- Rows × columns: **43,739 × 16**; numeric and categorical/object columns are mixed.
- `Delivery_Time`: **0 missing**, min **10**, median **125**, max **270 minutes**.
- Missing values: `Agent_Rating` **54**, `Weather` **91**; other listed fields **0**.
- Exact duplicate rows: **0**.
- All pickup/drop coordinates are present; check sentinel/out-of-city coordinates during EDA.
- `Multiple_Deliveries`, `Festival`, and `City` are absent from this file. The model excludes them instead of imputing an entirely empty feature.
- Model inputs used: `Agent_Age`, `Agent_Rating`, Haversine `Distance_km`, `Weather`, `Traffic`, `Vehicle`, `Area`, derived order `Hour`, and `Category`.
- **Target sanity check:** the median of 125 minutes and maximum of 270 minutes are high for a delivery-time dataset. Confirm that this exact Kaggle file and target units match the assignment before presenting the metrics as applicable to the intended service.

## Known limitations

- Kaggle uploads may be mirrors or modified versions; data provenance and license need confirmation.
- Coordinates can be missing, zero, or otherwise invalid; Haversine distance may be misleading for such rows.
- Historical city/season conditions may not transfer to a different geography or future period.
- Random split metrics may overstate performance when adjacent records share city, agent, time or operating conditions.
- The case brief's approximate row count is not a substitute for inspecting the exact CSV.
