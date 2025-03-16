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

# Default environment variables
ENV GPU_MODEL="h100" \
    AI_MODEL="llama3.2" \
    TEST_TIME="240" \
    LIMITING_MODE="none" \
    PRINT_RESPONSES="false" \
    DEBUG="false" \
    OUTPUT_DIR="benchmark_output" \
    IN_DOCKER="true" \
    NO_FIXED_OUTPUT="false" \
    DEMO_MODE="false" \
    LOG_PROMPTS="false" \
    LOG_FILE="prompts_and_responses.log" \
    WARMUP="false"

# Command to run the application with environment variables
CMD python generate_inference_load.py \
    --gpu-model $GPU_MODEL \
    --ai-model $AI_MODEL \
    --test-time $TEST_TIME \
    --limiting-mode $LIMITING_MODE \
    $([ "$PRINT_RESPONSES" = "true" ] && echo "--print-responses") \
    $([ "$DEBUG" = "true" ] && echo "--debug") \
    --output-dir $OUTPUT_DIR \
    $([ "$IN_DOCKER" = "true" ] && echo "--in-docker") \
    $([ "$NO_FIXED_OUTPUT" = "true" ] && echo "--no-fixed-output") \
    $([ "$DEMO_MODE" = "true" ] && echo "--demo-mode 3") \
    $([ "$LOG_PROMPTS" = "true" ] && echo "--log-file ${OUTPUT_DIR}/${LOG_FILE}") \
    $([ "$WARMUP" = "true" ] && echo "--warmup")
