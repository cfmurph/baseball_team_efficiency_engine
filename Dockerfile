# Public BenchOrStart read API. Serves committed fixture current/ until a
# real ARTIFACTS_URI is supplied at runtime. Does not invent 2026 rows.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    ARTIFACTS_URI=file:///app/tests/fixtures/api/lake_current \
    API_CORS_ORIGINS=* \
    API_CORS_ORIGIN_REGEX=https://.*[.]vercel[.]app

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

COPY services /app/services
COPY src /app/src
COPY fantasy /app/fantasy
COPY config /app/config
COPY tests/fixtures/api/lake_current /app/tests/fixtures/api/lake_current

EXPOSE 8000

# Honors $PORT from Railway / Render / Fly. Do not pin API_PORT here.
CMD ["python3", "-m", "services.api"]
