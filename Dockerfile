FROM python:3.9-alpine
RUN apk add --no-cache build-base
WORKDIR /app

COPY requirements-service.txt .

RUN pip install --no-cache-dir -r requirements-service.txt

COPY scoring_service/* ./
COPY models/* ./models/

EXPOSE 8080

ARG MODEL_PATH
ENV MODEL_PATH=$MODEL_PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]

