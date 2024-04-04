import logging
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
from model_loader import load_model

app = FastAPI()
model_path = os.getenv("MODEL_PATH")

# Configure logging with custom format including date and time
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    data: list


@app.post("/score/")
async def score(request: PredictionRequest):
    try:
        model = load_model(model_path)
        predictions = model.predict(np.array(request.data))
        logger.info(
            "Predictions made: %s", predictions.tolist()
        )  # Log successful predictions
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.exception("An error occurred while processing the request.")
        return {"error": "An error occurred while processing the request."}
