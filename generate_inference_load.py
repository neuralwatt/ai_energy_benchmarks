"""
generate_inference_load.py

This script generates inference load on a specified AI model using GPU resources.
It meausres the gpu utilization and power consumption during the inference and is dependent on ollama an dnvidia-smi.
It supports different limiting modes such as power and frequency to control the GPU behavior.
The script reads prompts from a CSV file and sends them to an endpoint for inference.
The results are logged and saved in CSV format.

Usage:
    python generate_inference_load.py [options]

Command-line options:
    --gpu-model: The model of the GPU being used.
    --ai-model: The AI model to be used for inference.
    --test-time: Duration of each test in seconds.
    --limiting-mode: Mode to limit GPU resources (none, frequency, power).
    --print-responses: Flag to print LLM responses to the console.
    --debug: Flag to enable debug mode with shorter test times and limited variations.
    --model-list: Comma-separated list of AI models to cycle through for inference.
    --output-dir: Directory where output files will be stored.
    --in-docker: Flag to indicate running in Docker container.
    --no-fixed-output: Flag to disable fixed temperature and seed settings.
    --demo-mode: Number of prompts to run or path to custom prompt file.

Copyright (c) 2025 NeuralWatt Corp. All rights reserved.
"""

import subprocess
import time
import json
import csv
import os
import argparse
from datetime import datetime
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate inference load on an AI model')
parser.add_argument('--gpu-model', default='h100', help='The model of the GPU being used')
parser.add_argument('--ai-model', default='llama3.2', help='The AI model to be used for inference')
parser.add_argument('--test-time', type=int, default=240, help='Duration of each test in seconds')
parser.add_argument('--limiting-mode', default='none', choices=['none', 'frequency', 'power'], 
                    help='Mode to limit GPU resources (none, frequency, power)')
parser.add_argument('--print-responses', action='store_true', help='Print LLM responses to the console')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with shorter test times and limited variations')
parser.add_argument('--output-dir', default='benchmark_output', help='Directory where output files will be stored')
parser.add_argument('--in-docker', action='store_true', help='Indicate running in Docker container')
parser.add_argument('--no-fixed-output', action='store_true', help='Disable fixed temperature and seed settings')
parser.add_argument('--demo-mode', default=None, help='Number of prompts to run or path to custom prompt file')

args = parser.parse_args()

# Set parameters from command-line arguments
gpu_model = args.gpu_model
ai_model = args.ai_model
test_time = args.test_time
test_count = 0
limiting_mode = args.limiting_mode
print_responses = args.print_responses
debug = args.debug
in_docker = args.in_docker
output_dir = args.output_dir
no_fixed_output = args.no_fixed_output
demo_mode = args.demo_mode

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory listing: {os.listdir('.')}")

print(f"Output directory path: {os.path.abspath(output_dir)}")
print(f"Output directory exists: {os.path.exists(output_dir)}")
print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")

# Read prompts from file - modified to handle demo mode
prompts_file = 'prompts.csv'

print(f"**Demo mode: {demo_mode}")
# Handle demo-mode parameter
if demo_mode:
    if os.path.isfile(demo_mode):
        # If demo_mode is a valid file path, use it as the prompts file
        prompts_file = demo_mode
        print(f"Using custom prompt file: {prompts_file}")
    elif demo_mode.isdigit():
        # If demo_mode is a number, limit the number of prompts used
        with open(prompts_file, 'r') as file:
            all_prompts = file.read().strip().split('",\n"')
            all_prompts = [prompt.strip('"') for prompt in all_prompts]
        
        num_prompts = min(int(demo_mode), len(all_prompts))
        prompts = all_prompts[:num_prompts]
        print(f"Running in demo mode with {num_prompts} prompts")
    elif demo_mode.lower() in ['true', 'yes']:
        # If demo_mode is true or yes, use a default limited number of prompts (3)
        DEFAULT_DEMO_PROMPTS = 3
        with open(prompts_file, 'r') as file:
            all_prompts = file.read().strip().split('",\n"')
            all_prompts = [prompt.strip('"') for prompt in all_prompts]
        
        prompts = all_prompts[:DEFAULT_DEMO_PROMPTS]
        print(f"Running in demo mode with {DEFAULT_DEMO_PROMPTS} prompts")
    else:
        print(f"Invalid demo-mode value: {demo_mode}. Using default prompt file.")
        with open(prompts_file, 'r') as file:
            prompts = file.read().strip().split('",\n"')
            prompts = [prompt.strip('"') for prompt in prompts]
else:
    # Default behavior - read all prompts from the default file
    with open(prompts_file, 'r') as file:
        prompts = file.read().strip().split('",\n"')
        prompts = [prompt.strip('"') for prompt in prompts]

file_id = f"{limiting_mode}_{gpu_model}_{ai_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Setup
if limiting_mode == "power":
    min_power_limit = int(subprocess.check_output("nvidia-smi -q -d POWER | grep 'Min Power Limit' | awk '{print $5}'", shell=True).strip()) / 100
    max_power_limit = int(subprocess.check_output("nvidia-smi -q -d POWER | grep 'Max Power Limit' | awk '{print $5}'", shell=True).strip()) / 100

    power_limits = list(range(min_power_limit, max_power_limit, int((max_power_limit - min_power_limit) * 0.1)))
    power_limits[-1] = max_power_limit

    print(f"Power limits: {power_limits}")

elif limiting_mode == "frequency":
    gpu_frequencies_output = subprocess.check_output("nvidia-smi --query-supported-clocks=graphics,memory --format=csv", shell=True).decode()
    valid_gpu_frequencies = sorted(set(int(line.split(",")[0].strip().replace(' MHz', '')) for line in gpu_frequencies_output.splitlines() if " MHz" in line))

    gpu_frequencies = [valid_gpu_frequencies[i] for i in range(0, len(valid_gpu_frequencies), int(len(valid_gpu_frequencies) * 0.1))]
    gpu_frequencies[-1] = valid_gpu_frequencies[-1]

    print(f"GPU Frequencies: {gpu_frequencies}")

else:
    print(f"Power limiting mode set: {limiting_mode}; no initialization action for this mode")

nvidia_smi_log_path = os.path.join(output_dir, f"inference.nvidia_smi_log.{file_id}.csv")
process = subprocess.Popen(["python", "monitor_nvidia.py", nvidia_smi_log_path, output_dir])
monitor_id = process.pid

if debug:
    if limiting_mode == "power":
        power_limits = power_limits[-2:]
    elif limiting_mode == "frequency":
        gpu_frequencies = gpu_frequencies[-2:]
    test_time = 20

variation_count = 0
test_count = 0


# Set the endpoint based on the environment
if in_docker:
    print("Running in Docker")
    endpoint = "http://ollama:11434/api/generate"
else:
    print("Running locally")
    endpoint = "http://localhost:11434/api/generate"

while True:
    if limiting_mode == "power":
        if variation_count >= len(power_limits):
            break
        power_limit = power_limits[variation_count]
        print(f"Setting power limit to {power_limit}")
        subprocess.run(["nvidia-smi", "-pl", str(power_limit)])

    elif limiting_mode == "frequency":
        if variation_count >= len(gpu_frequencies):
            print(f"Test count: {variation_count}, breaking")
            break
        gpu_frequency = gpu_frequencies[variation_count]
        print(f"Setting GPU frequency to {gpu_frequency}")
        subprocess.run(["nvidia-smi", "-lgc", f"{gpu_frequency},{gpu_frequency}"])

    elif limiting_mode in ["none", "agent"]:
        # In 'none' or 'agent' mode, do not set power limits or GPU frequencies
        if test_count >= len(prompts):
            break

    else:
        print(f"Unknown power limiting mode set: {limiting_mode}")

    start_time = datetime.now()
    end_time = start_time + pd.Timedelta(seconds=test_time)
    time.sleep(1)

    for i, prompt in enumerate(prompts):
        body = {
            "model": ai_model, 
            "prompt": prompt
        }
        
        # Add temperature and seed for reproducible output unless no-fixed-output is specified
        if not no_fixed_output:
            body["temperature"] = 0
            body["seed"] = 42
            
        print(f"{i} of {len(prompts)} Prompt: {prompt}")

        query_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        response = subprocess.check_output(["curl", "-s", "-X", "POST", endpoint, "-H", "Content-Type: application/json", "-d", json.dumps(body)])
        query_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        response_array = [json.loads(line) for line in response.decode().split("\n") if line]

        if print_responses:
            for res in response_array[:-1]:
                print(res["response"], end="")

        try:
            final_response = next(res for res in response_array if res.get("done"))
        except StopIteration:
            print("No final response found with 'done' key.")
            continue

        total_tokens = len(final_response["context"])
        total_duration_seconds = final_response["total_duration"] / 1e9
        prompt_tokens = final_response["prompt_eval_count"]
        prompt_eval_duration_seconds = final_response["prompt_eval_duration"] / 1e9
        response_tokens = final_response["eval_count"]
        response_eval_duration_seconds = final_response["load_duration"] / 1e9

        if total_duration_seconds > 0:
            tokens_per_second = total_tokens / total_duration_seconds
            prompt_tokens_per_second = prompt_tokens / prompt_eval_duration_seconds
            response_tokens_per_second = response_tokens / response_eval_duration_seconds
            print(f"Total Tokens: {total_tokens}")
            print(f"Total Duration (seconds): {total_duration_seconds}")
            print(f"Tokens per Second: {tokens_per_second}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Prompt Tokens per Second: {prompt_tokens_per_second}")
            print(f"Response Tokens: {response_tokens}")
            print(f"Response Tokens per Second: {response_tokens_per_second}")

            csv_output = {
                "StartTime": query_start_time,
                "EndTime": query_end_time,
                "TotalTokens": float(total_tokens),
                "TotalDuration(seconds)": float(total_duration_seconds),
                "TokensPerSecond": float(tokens_per_second),
                "PromptTokens": float(prompt_tokens),
                "PromptEvalDuration(seconds)": float(prompt_eval_duration_seconds),
                "PromptTokensPerSecond": float(prompt_tokens_per_second),
                "ResponseTokens": float(response_tokens),
                "ResponseEvalDuration(seconds)": float(response_eval_duration_seconds),
                "ResponseTokensPerSecond": float(response_tokens_per_second),
                "LongOrShortPrompt": "Long" if i % 2 == 0 else "Short",
                "Model": body["model"],
            }
            inference_log_path = os.path.join(output_dir, f"inference.load.{file_id}.csv")
            try:
                with open(inference_log_path, "a", newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_output.keys())
                    if test_count == 0:
                        writer.writeheader()
                    writer.writerow(csv_output)
                print(f"Successfully wrote to {inference_log_path}")
            except Exception as e:
                print(f"Error writing to {inference_log_path}: {e}")
        else:
            print("No valid duration found.")

        test_count += 1

    variation_count += 1

process.terminate()
