FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --system --no-create-home botuser && \
    mkdir -p /data && chown botuser:botuser /data

USER botuser

# DB storage
ENV WEATHER_BOT_DB=/data/trades.db

ENTRYPOINT ["weather-bot"]
CMD ["run", "--verbose"]
