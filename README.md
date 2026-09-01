# RiskLens

Predict insurance premium risk category from a person's basic profile, served as a REST API.

## What it does

RiskLens takes in age, weight, height, income, occupation, smoking status, and city, and returns a predicted premium category: **Low**, **Medium**, or **High**.

It's built for two audiences:

- **Individuals** can get a quick sense of where they might fall before speaking with an insurance agent.
- **Insurers** can use it as a fast, consistent first-pass classification layer instead of relying purely on manual review.

## How it works

1. **Feature engineering** — Raw inputs are transformed into signals that actually drive risk:
   - `bmi` calculated from height and weight
   - `age_group` bucketed into young, adult, middle_aged, senior
   - `lifestyle_risk` combining smoking status and BMI together

2. **Model** — A classification model trained on these engineered features to predict premium category.

3. **Serving** — The same feature logic is reimplemented inside the Pydantic input model using `computed_field`, so incoming API requests are transformed the same way the training data was.

4. **API** — FastAPI wraps the trained model and exposes a `/predict` endpoint with a structured response (predicted category, confidence score, and full class probability breakdown), plus a `/health` endpoint for checking the service and model status.

> Currently API-only. A redesigned frontend and Docker support are planned — see [Roadmap](#roadmap) below.

## Tech stack

- **FastAPI** — API layer and request validation
- **Pydantic** — input schema, field constraints, and computed features
- **scikit-learn** *(update if different)* — model training

## Project structure

```
RiskLens/
├── README.md
├── app.py                    # FastAPI backend, route definitions
├── requirements.txt
├── config/
│   └── city_tier.py          # city tier mapping / config values
├── model/
│   ├── model.pkl              # trained model
│   └── predict.py             # loads model, runs prediction
├── schema/
│   ├── user_input.py          # Pydantic input schema + computed_field logic
│   └── response_model.py      # Pydantic response schema
├── model.ipynb                # training and feature engineering notebook
├── insurance.csv              # training dataset
└── .gitignore
```

## Running locally

```bash
# clone the repo
git clone https://github.com/<your-username>/RiskLens.git
cd RiskLens

# install dependencies
pip install -r requirements.txt

# start the API
uvicorn app:app --reload
```

Once running, visit `http://127.0.0.1:8000/docs` for interactive API documentation, or hit `/health` to confirm the model loaded correctly.

## Why this approach

The biggest risk in most ML systems isn't the model itself — it's a mismatch between how features were built during training and how they're built at inference time. RiskLens keeps that logic in sync by defining the feature transformations once and reusing them in both the training notebook and the live API's input model. A lens is only useful if it shows you the same picture every time you look through it.

## Roadmap

- [ ] Redesigned frontend
- [ ] Dockerize the service for consistent local and deployed environments
- [ ] CI checks on push (lint, basic smoke test against `/health`)

## Status

This is a learning project built to understand end-to-end ML serving with FastAPI and Pydantic. Feedback and suggestions are welcome.

## License

MIT
