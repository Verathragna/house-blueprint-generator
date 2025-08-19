FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY . .

# Ensure model weights exist to avoid runtime surprises
RUN test -f checkpoints/model_latest.pth || echo 'Warning: model weights not found'

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
