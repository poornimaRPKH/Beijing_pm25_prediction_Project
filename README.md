# Beijing PM2.5 Prediction API

Predicts the PM2.5 of the **next hour** from the **previous 24 hours**.

## Model
- Selected model: Tuned GRU (32 units, dropout 0.2, learning rate 0.001).
- Chosen on the validation year 2013 (RMSE 23.49, R2 0.942).
- Test year 2014 (selected model only): MAE 12.25, RMSE 21.17, R2 0.949.
- Data split: training 2010 to 2012, validation 2013, test 2014.

## Input
- 24 rows (hours) x 15 columns, **already scaled** with the saved scalers.
- The column order must be exactly the same as in training:

1. `pm25_scaled`
2. `DEWP`
3. `TEMP`
4. `PRES`
5. `Iws`
6. `Is`
7. `Ir`
8. `wind_NE`
9. `wind_NW`
10. `wind_SE`
11. `wind_cv`
12. `hour_sin`
13. `hour_cos`
14. `month_sin`
15. `month_cos`

## Output
- JSON with `predicted_pm25` and `unit` (ug/m3) for the next hour.
- No "confidence" value is given, because this is a regression model (it predicts a number, not a class).

## Run with Docker
```
docker build -t beijing-pm25-api .
docker run -p 8000:8000 beijing-pm25-api
```
Open http://localhost:8000/docs to test the API.

## Test with the sample request
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```
- The notebook gives 58.14 for this sample. The API should return the same value.
- The real PM2.5 of that hour was 35.00.

## Limitations
- Errors are larger at high pollution (average error 28.05 above 300, against 11.36 on normal hours).
- Sudden pollution spikes are not predicted well.
- About 10% of the rows use interpolated values.
- The input must use the same scaling and column order as in training.

## Files
- `app/app.py`: the FastAPI app.
- `app/model.keras`: the trained Tuned GRU model.
- `app/target_scaler.pkl`: converts the model output back to ug/m3.
- `Dockerfile`, `requirements.txt`, `.dockerignore`: Docker setup.
- `sample_request.json`: a real example input.
