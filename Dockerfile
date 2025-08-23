# Build front-end assets
FROM node:20 AS frontend
WORKDIR /app/interface/web
COPY interface/web/package*.json ./
RUN npm ci
COPY interface/web .
# Build with API base URL baked in
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# Final image with API and model
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY . .
# Copy built front-end assets from builder stage
COPY --from=frontend /app/interface/web/dist interface/web/dist

# Ensure model weights exist to avoid runtime surprises
RUN test -f checkpoints/model_latest.pth || echo 'Warning: model weights not found'

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
