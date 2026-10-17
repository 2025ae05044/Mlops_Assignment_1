# Monitoring and Logging

## Request logging
The FastAPI middleware records the HTTP method, endpoint path,
response status code, and response duration in milliseconds.
Patient input values are not logged.

## Health checks
The API exposes GET /health.
Docker Compose checks this endpoint every 30 seconds.

## Viewing logs
View recent logs:

    docker compose logs --tail=100 heart-disease-api

Follow live logs:

    docker compose logs -f heart-disease-api

## Current limitations
Monitoring currently consists of application logs, Uvicorn access
logs, and Docker health checks. A dedicated metrics dashboard
and alerting system have not been implemented.
