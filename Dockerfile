# Use a lightweight Python base image
FROM python:3.8-slim

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the application code
COPY . /app

# Set the working directory to the app directory
WORKDIR /app/app/scoring

# Expose the port FastAPI will run on
EXPOSE 80

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
