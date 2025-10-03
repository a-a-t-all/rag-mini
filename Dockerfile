FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8000 7860
# Default to FastAPI server
CMD ["python","-m","src.api","--host","0.0.0.0","--port","8000"]
