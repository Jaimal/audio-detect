FROM python:3.11-slim

# ffmpeg + libsndfile are needed by librosa/pydub for m4a/mp3 decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000 \
    HOST=0.0.0.0 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

EXPOSE 10000

# Single worker: TF model is loaded at import time, multiple workers would
# multiply the memory footprint. Threads handle concurrent uploads.
CMD ["sh", "-c", "exec gunicorn -w 1 -k gthread --threads 4 --timeout 180 -b 0.0.0.0:${PORT} web_app:app"]
