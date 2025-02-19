FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install any dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["sh", "-c", "curl -X POST http://ollama:11434/api/pull -d '{\"model\": \"llama3.2\"}' && python generate_inference_load.py"]
