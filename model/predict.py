import pickle

import pandas as pd

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

## MlFlow model version
MODEL_VERSION = "2.0.0"

class_labels = model.classes_.tolist()


def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    predicted_class = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    class_probs = dict(zip(class_labels, (round(p, 4) for p in probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs,
    }
