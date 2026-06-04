FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY federated_server.py .
ENV DATA_DIR=/data PORT=8080
VOLUME /data
EXPOSE 8080
CMD ["gunicorn", "federated_server:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-"]
