# Instructions benchmarking setup, creating a loadline or testing a loadline.

## Branch Management Guide.  This is a private clone of a public repo where we keep neuralwatt specific code.

## Overview
This guide provides instructions on managing branches between a public repository and a private repository. Follow these steps to keep your repositories organized and in sync.

## Setting Up Remotes
1. **Add Remotes**
   Add the public and private repositories as remotes:
   ```bash
   # Add the public repository
   git remote add public https://github.com/neuralwatt/ai_energy_benchmarks.git
   # Add the private repository
   git remote add private https://github.com/neuralwatt/neuralwatt_benchmark.git
1. **Verify Remotes**
    ```bash
    git remote -v
1. **Creating a new Branch and Committing**
    ```bash
    git checkout -b new_branch
    git add .
    git commit -m "message"
1. **Pushing Changes to Private Repo**
    ```bash
    git push private new_branch
1. **Pushing Changes to Public Repo**
    ```bash
    git push public new_branch
1. **Keeping Repositories in Sync**
    ```bash
    git fetch public #Fetch updates from public
    git merge public/main #Merge updates from public in to private
    git push private new_branch #push merged changes to private repo
1. ** Switching branches**
    ```bash
    #If in private repo and want to checkout the public
    git checkout -b public-main public/main



## Quickstart to testing the agent in inference mode
### Disk Space
Ensure your VM has > 60G free space in / to download the model weights. You can configure ollama to put the model weights somewhere else but I haven't covered that in these instructions. If allocating a single OS disk on datacrunch.io one with 100G total capacity is sufficient for this test.

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
    curl http://localhost:11434/api/pull -d '{
        "model": "llama3.2"
    }'
    ```

1. Send the prompt
    ```
    curl http://localhost:11434/api/generate -d '{
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
Change directory back to CarbonAwareLinux/benchmarks/gpu_load_line
```
python generate_inference_load.py
```
You should see both the agent spew as well as the prompts as they get cycled through.

### Troubleshooting
Common mistakes I make
* Ollama docker not running or all required models not available
* If each prompt looks like it's running slow check the cpu utilization and it shouldn't be really high; ollama is really good at spilling to ram if it runs out of GPU memory and that can happen on 48GB or smaller GPUs with this test.
* Sometimes python doesn't get correctly installed from the yaml file. If so just conda install python=3.12 with the gpu_load_line env activated

### Analyzing the results
A full pass through the prompts takes around 3 minutes on an h100 if using just one model.
There are two files which are output: nvidia_smi_log.csv which are the gpu performance metrics and inference_load.csv which are metrics about the prompt. If you have jupyter installed you can use the notebook analyze_agent.ipynb to check the performance of your current run against a default run which has previously been generated. For example if you just ran this test on an h100 the outputs will be inferece.nvidia_smi_log.none_h100_llama3.2_<timestamp>.csv and inference.load.none_h100_llama3.2_<timestamp>.csv.  

## Generating the load line
Create the conda environment
```
conda env create -f gpu_load_line_env.yml
conda activate gpu_load_line
```

Generate monitor the data and run the test

This has a dependency on monitor_nvidia.py (which itself has a dependency on nvidia-smi so make sure nvidia-smi is working as it's not currently installed in the conda env).
```
python generate_inference_load.py
```

Create the analysis images by running this notebook
```
jupyter notebook generate_load_line.ipynb
```

TODO
* Still need to export the optimization data for the agent

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