FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

ENV PORT=6002

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:application", "--host", "0.0.0.0", "--port", "6002"]
