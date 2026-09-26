# One image for both roles: `coordinator` (default) and `agent`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY flotilla ./flotilla
COPY config ./config

RUN useradd --uid 10001 --home-dir /data --create-home --shell /usr/sbin/nologin flotilla
USER flotilla

ENV FLOTILLA_CONFIG=/app/config/flotilla.yaml \
    FLOTILLA_DATA_DIR=/data

EXPOSE 8800 8801
# Checks FLOTILLA_PORT (8800 by default); agent services pass --port 8801.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-m", "flotilla", "healthcheck"]

ENTRYPOINT ["python", "-m", "flotilla"]
CMD ["coordinator"]
