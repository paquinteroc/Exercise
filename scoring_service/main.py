from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
from model_loader import load_model

app = FastAPI()
model_path = os.getenv("MODEL_PATH")


class PredictionRequest(BaseModel):
    data: list


@app.post("/score/")
async def score(request: PredictionRequest):
    model = load_model(model_path)
    predictions = model.predict(np.array(request.data))
    return {"predictions": predictions.tolist()}
