# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt -c constraints.txt

COPY src/ src/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]