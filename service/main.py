from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model_loader import load_model

app = FastAPI()


class PredictionRequest(BaseModel):
    data: list


@app.post("/score/")
async def score(request: PredictionRequest):
    model = load_model("model/model.pkl")
    predictions = model.predict(np.array(request.data))
    return {"predictions": predictions.tolist()}
