from fastapi import FastAPI
from fastapi.responses import JSONResponse

from model.predict import MODEL_VERSION, model, predict_output
from schema.response_model import PredictionResponse
from schema.user_input import UserInput

app = FastAPI()


### human readable
@app.get("/")
def root():
    return {"message": "Welcome to the Insurance Premium Prediction API!"}


### machine readable
@app.get("/health")
def health_check():
    status = "OK" if model else "Error"
    return {
        "status": status,
        "message": "Model loaded" if model else "Model not available",
        "version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):
    user_input = {
        "bmi": data.bmi,
        "lifestyle_risk": data.lifestyle_risk,
        "age_group": data.age_group,
        "city_tier": data.city_tier,
        "occupation": data.occupation,
        "income_lpa": data.income_lpa,
    }
    try:
        prediction = predict_output(user_input)
        return JSONResponse(status_code=200, content={"predicted_category": prediction})

    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            status_code=500, content={"error": "Prediction failed", "details": str(e)}
        )
