FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

# PORT is read by `rocmhub serve`; Railway injects it at runtime.
CMD ["rocmhub", "serve", "--host", "0.0.0.0"]
