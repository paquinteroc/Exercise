import logging
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from model_loader import load_model
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


app = FastAPI()
model_path = os.getenv("MODEL_PATH")

# Configure logging with custom format including date and time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInput(BaseModel):
    x1: float
    x2: float
    x4: float
    x5: float
    x3: str
    x6: str
    x7: str


# Wrapper model to accept a list of ModelInput
class PredictionInput(BaseModel):
    inputs: List[ModelInput]


class PredictionResult(BaseModel):
    predictions: List[int]


@app.post("/score/", response_model=PredictionResult)
def make_prediction(request: PredictionInput):
    """
    Make predictions using the input data and the model.
    Parameters:
    - request (PredictionInput): Input data to make predictions.
    Returns:
    - PredictionResult: Predicted values.
    """
    model = load_model(model_path)
    try:
        # Convert the list of ModelInput instances to a DataFrame
        input_data = [item.dict() for item in request.inputs]
        df = pd.DataFrame(input_data)
        # Make predictions
        predictions = model.predict(df)

        logger.info(f"predictions: {predictions} made for request {request}")
        return PredictionResult(predictions=predictions.tolist())
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while making predictions."
        )
