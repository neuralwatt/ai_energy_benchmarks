## Instructions benchmarking setup, creating a loadline or testing a loadline.

### Disk Space
Ensure your VM has > 60G free space in / to download the model weights. You can configure ollama to put the model weights somewhere else but I haven't covered that in these instructions. If allocating a single OS disk on datacrunch.io one with 100G total capacity is sufficient for this test.

## Quickstart
1. Create & activate the conda environment
```bash
conda create -n ai_energy_benchmark python=3.12
conda activate ai_energy_benchmark
```

2. Run the benchmark
```
 AI_MODEL=llama3.2 WARMUP=true docker compose up
```



## Running Everything with Docker

To start the services defined in the `docker-compose.yml` file, you have two options: `docker-compose up` or `docker-compose build`.

### Using `docker-compose up`

The `docker-compose up` command will start and run all the services defined in the `docker-compose.yml` file. It will download the image from the github container registry for th benchmarks. 

```sh
docker-compose up
```

This command is useful when you want to start the entire application stack from source with a single command. It will also rebuild the images if there are any changes in the Dockerfiles or the context.

### Configure Docker Environment Variables

You can configure the benchmark parameters by modifying the environment variables in the `docker-compose.yml` file:

```yaml
environment:
  - GPU_MODEL=h100
  - AI_MODEL=llama3.2
  - TEST_TIME=240
  - LIMITING_MODE=none
  - PRINT_RESPONSES=False
  - DEBUG=False
  - OUTPUT_DIR=benchmark_output
  - IN_DOCKER=True
  - DEMO_MODE=   # Number of prompts to run, path to custom prompt file, or "true" for default demo (3 prompts)
```

Or you can override them on the command line:

```sh
GPU_MODEL=h100 AI_MODEL=llama3.2 DEMO_MODE=5 docker-compose up
```

For running in demo mode with the default number of prompts:

```sh
DEMO_MODE=true docker-compose up
```

> **Note**: When using environment variables with Docker Compose, the values are passed directly to the Python script as command-line arguments. So `DEMO_MODE=3` will result in running with only 3 prompts from the default prompt file.

### Using `docker-compose build`

The `docker-compose build` command will build the images defined in the `docker-compose.yml` file without starting the containers.

```sh
docker-compose build
```

This command is useful when you want to build the images separately before starting the containers. You can then start the containers using:

```sh
docker-compose up
```

or

```sh
docker-compose up --no-build
```

The `--no-build` flag ensures that the images are not rebuilt if they already exist.

### Summary

- Use `docker-compose up` to build and start the services in one step.
- Use `docker-compose build` to build the images separately, and then use `docker-compose up` to start the services.

## Running each part seperately

### Command Line Options

When running the benchmark script directly, you can use the following command-line options:

```
usage: generate_inference_load.py [-h] [--gpu-model GPU_MODEL] [--ai-model AI_MODEL] [--test-time TEST_TIME]
                                [--limiting-mode {none,frequency,power,agent}] [--print-responses] [--debug]
                                [--model-list MODEL_LIST] [--output-dir OUTPUT_DIR] [--in-docker] [--demo-mode DEMO_MODE]

Generate inference load on an AI model

options:
  -h, --help            show this help message and exit
  --gpu-model GPU_MODEL
                        The model of the GPU being used (default: 1080ti)
  --ai-model AI_MODEL   The AI model to be used for inference (default: llama3.2)
  --test-time TEST_TIME
                        Duration of each test in seconds (default: 240)
  --limiting-mode {none,frequency,power}
                        Mode to limit GPU resources (default: none)
  --print-responses     Print LLM responses to the console
  --debug               Enable debug mode with shorter test times and limited variations
  --output-dir OUTPUT_DIR
                        Directory where output files will be stored (default: benchmark_output)
  --in-docker           Indicate running in Docker container
  --demo-mode DEMO_MODE Number of prompts to run or path to custom prompt file
```

### Install on Linux using Docker
1. Ensure Docker is running
1. Install nvidia container toolkit if not already installed--if you created a VM on datacrunch.io or another ML focused gpu provider it's likely already installed and running.
    https://hub.docker.com/r/ollama/ollama
1. Run ollama 
    ``` 
    docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```

### Running Ollama
1. Pull the model you want
    ```
    curl -s http://localhost:11434/api/pull -d '{
        "model": "llama3.2"
    }'
    ```

1. Send the prompt
    ```
    curl -s http://localhost:11434/api/generate -d '{
        "model": "llama3.2",
        "prompt":"Why is the sky blue?"
    }'
    ```
1. Returns the tokens in json as well as time stats for each which can be used as measure of work

### Verify correct gpu utilization
For now just check task manager or nvidia-smi when sending the prompt. Will come up with programmatic way to do this in the future.

### Models required
We currently use several sizes of llama models for the benchmarking. Install these models to ollama using the above curl commands:
* llama3.2
--also tested with these
* llama3.1:8b
* llama3.3

### Create the environment
```
#install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

#create the conda environment
conda env create -n <env_name> -f requirements.txt
conda activate <env_name> 
```

### Set the correct mode for the test
The file `generate_inference_load.py` is the main inference test driver. There are several modes it can run in:
* none -- for testing the default behavior, only does the power monitoring and runs the test but no power limiting or agent execution
* frequency -- loops through the range of available gpu frequencies to generate a frequency load line. This is a long test.
* power -- loops through the range of available gpu power caps to generate a power load line. This is a long test.

Open `generate_inference_load.py` and ensure the `limiting_mode` parameter is set to "none" for this initial test. This will run a single pass through the test prompts varied across the 3 models and generate the output for that. 

### Run the test
```
python generate_inference_load.py
```

### Troubleshooting
Common mistakes I make
* Ollama docker not running or all required models not available
* If each prompt looks like it's running slow check the cpu utilization and it shouldn't be really high; ollama is really good at spilling to ram if it runs out of GPU memory and that can happen on 48GB or smaller GPUs with this test.
* Sometimes python doesn't get correctly installed from the yaml file. If so just conda install python=3.12 with the gpu_load_line env activated
* Curl command for deterministic output
```bash
curl http://localhost:11434/api/generate -d '{
    "model": "llama3.2",
    "prompt": "Why is the sky blue?",
    "options": {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 2048
    }
}' 
```

### Analyzing the results
A full pass through the prompts takes around 3 minutes on an h100 if using just one model.
There are two files which are output: nvidia_smi_log.csv which are the gpu performance metrics and inference_load.csv which are metrics about the prompt. If you have jupyter installed you can use the notebook analyze_agent.ipynb to check the performance of your current run against a default run which has previously been generated. For example if you just ran this test on an h100 the outputs will be inferece.nvidia_smi_log.none_h100_llama3.2_<timestamp>.csv and inference.load.none_h100_llama3.2_<timestamp>.csv.  You need to set a few parameters at the top of the notebook to indicate the files you are anayzing.

### Demo Mode

The demo mode provides a way to run the benchmarks with a reduced set of prompts or with custom prompts:

1. **Limited Number of Prompts**: To run with a limited number of prompts from the default file, use:
   ```
   python generate_inference_load.py --demo-mode 5
   ```
   This will run only the first 5 prompts from the default prompt file.

2. **Custom Prompt File**: To use a custom prompt file, specify the path:
   ```
   python generate_inference_load.py --demo-mode custom_prompts.csv
   ```
   The custom prompt file should follow the same format as the default prompt file.

3. **Default Demo Mode**: To use the default demo mode with 3 prompts, use:
   ```
   python generate_inference_load.py --demo-mode true
   ```
   or
   ```
   python generate_inference_load.py --demo-mode yes
   ```

## Windows instructions

### Install on Windows using command line (since nvidia container runtime not supported on windows)
1. Install ollama via download exe https://github.com/ollama/ollama/tree/main?tab=readme-ov-file
1. Ensure ollama is running and exposing local api endpoint http://localhost:11434/api/tags should list models installed.

### Alternate Powershell Format for the Curl commands
```
$body = @{
    model = "llama3.2"
    prompt = "Why is the sky blue?"
}

Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body ($body | ConvertTo-Json) -ContentType "application/json"

