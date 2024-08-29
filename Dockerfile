# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=360 -r requirements.txt


COPY train_stage/train.py  /app/train_stage/train.py
COPY data_stage/data.py /app/data_stage/data.py


COPY train_stage/libt /app/train_stage/libt
COPY data_stage/libd /app/data_stage/libd

COPY data_stage/data_config.yaml /app/data_stage/data_config.yaml
COPY train_stage/train_config.yaml /app/train_stage/train_config.yaml

COPY web_app/server.py /app/web_app/server.py
COPY web_app/utils /app/web_app/utils
COPY web_app/server_config.yaml /app/web_app/server_config.yaml
COPY start.sh .








# Make the shell script executable
RUN chmod +x start.sh

# Expose port 5000 to the outside world
EXPOSE 5000

# Set the entry point to the shell script
ENTRYPOINT ["./start.sh"]

# Run server.py when the container launches
CMD ["python3", "web_app/server.py"]
