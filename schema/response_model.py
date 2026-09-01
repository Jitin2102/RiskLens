from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ..., description="The predicted insurance premium category", example="medium"
    )
    confidence: float = Field(
        ...,
        description="Model's Confidence score for the predicted class(range: 0 to 1)",
        example=0.85,
    )
    class_probabilities: dict = Field(
        ...,
        description="Probability distribution across all possible classes",
        example={"low": 0.1, "medium": 0.85, "high": 0.05},
    )
