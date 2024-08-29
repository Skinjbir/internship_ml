# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt

# Copy the application files in a single command to reduce layers
COPY /train_stage/train.py /data_stage/data.py /app/
COPY /train_stage/libt /app/libt
COPY /data_stage/libd /app/libd
COPY /train_stage/train_config.yaml /data_stage/data_config.yaml /app/
COPY start.sh .

# Make the shell script executable
RUN chmod +x start.sh

# Set the entry point to the shell script
ENTRYPOINT ["./start.sh"]
