FROM python:3.13-slim

WORKDIR /app

# Copy only the essential files first
COPY requirements.txt .
COPY application.py .

# If your app uses other folders (like "src", "templates"), copy them as well:
COPY src ./src
COPY templates ./templates
COPY artifacts ./artifacts
COPY logs ./logs

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "application.py"]
