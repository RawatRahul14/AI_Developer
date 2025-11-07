# === Base Image ===
FROM python:3.11-slim

# === Set Working Directory ===
WORKDIR /app

# === Copy project files ===
COPY . /app

# === Copy environment file ===
COPY .env /app/.env

# === Install dependencies from requirements.txt ===
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# === Expose Port ===
EXPOSE 8080

# === Run the FastAPI app ===
CMD ["uvicorn", "Task3.main:app", "--host", "0.0.0.0", "--port", "8080"]