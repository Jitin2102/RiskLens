import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "insurance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def make_age_group(age: int) -> str:
    if age < 25:
        return "young"
    if age < 45:
        return "adult"
    if age < 65:
        return "middle_aged"
    return "senior"


def make_city_tier(city: str) -> str:
    tier_1_cities = {
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Kolkata",
        "Pune",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
    }
    tier_2_cities = {
        "Nagpur",
        "Pune",
        "Surat",
        "Kolkata",
        "Ahmedabad",
        "Bhopal",
        "Coimbatore",
        "Patna",
        "Vadodara",
        "Allahabad",
        "Visakhapatnam",
        "Vijayawada",
        "Madurai",
        "Rajkot",
        "Nashik",
        "Varanasi",
        "Srinagar",
        "Jodhpur",
        "Amritsar",
        "Ranchi",
        "Guwahati",
        "Dehradun",
    }

    if city in tier_1_cities:
        return "tier_1"
    if city in tier_2_cities:
        return "tier_2"
    return "tier_3"


def make_lifestyle_risk(row: pd.Series) -> str:
    bmi = row["weight"] / (row["height"] ** 2)
    if row["smoker"] and bmi > 30:
        return "high"
    if row["smoker"] or bmi > 27:
        return "medium"
    return "low"


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["bmi"] = df["weight"] / (df["height"] ** 2)
    df["age_group"] = df["age"].apply(make_age_group)
    df["city_tier"] = df["city"].apply(make_city_tier)
    df["lifestyle_risk"] = df.apply(make_lifestyle_risk, axis=1)
    return df


def train_and_save_model() -> float:
    df = build_dataset()

    X = df[
        ["income_lpa", "occupation", "bmi", "age_group", "lifestyle_risk", "city_tier"]
    ]
    y = df["insurance_premium_category"]

    categorical_features = ["occupation", "age_group", "lifestyle_risk", "city_tier"]
    numerical_features = ["income_lpa", "bmi"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(pipeline, file)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    return accuracy


if __name__ == "__main__":
    train_and_save_model()
