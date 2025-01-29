## Instructions benchmarking setup, creating a loadline or testing a loadline.

## Quickstart to testing the agent in inference mode
### Disk Space
Enusre your VM has > 60G free space in / to download the model weights.  You can configure ollama to put the model weights somewhere else but I haven't covered that in these instructions. If allocating a single OS disk on datacrunch.io one with 100G total capacity is sufficient for this test.

### Install on Linux using Docker
1. Ensure Docker is running
1. Install nvidia container toolkit if not already installed--if you created a VM on datacrunch.io or another ML focused gpu provider its likely already installed and running.
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
For now just check task manager or nvidia-smi when sending the prompt.  Will come up with programmatic way to do this in the future.

### Models required
We currently use several sizes of llama models for the benchmarking. Install these modesl to ollama using the above curl commands:
* llama3.1:8b
* llama3.2
* llama3.3

### Create the environment
```
#install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

#create the conda environment
cd benchmarks/gpu_load_line/
conda env create -f gpu_load_line_env.yml 
conda activate gpu_load_line

#install powershell
snap install powershell --classic
```

### Build the governor agnet
```
#install rust; defaults are fine
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"  
sudo apt install pkg-config
sudo apt install libssl-dev

#build the agent--executable name is currently co2cpufrequency
cd CarbonAwareLinux/os/carbon_aware_governor
cargo build
```

### Ensure correct environment variables are set
Get the environment keys I posted to slack and append them to /etc/environment 
Then 
```
source \etc\environment
```

### check the agent is running correctly
The agent needs to run with sudo and -E passes in the current environment
```
sudo -E ./target/debug/co2cpufrequency --gpu-energy-mode
```

If running correctly the output should look like this spewing once every defined interval (set as a command line parameter but we're using the default of 1 second for now)
```
Current gpu load is: 0
Load and power quantile found for load: 1 and frequency_quantile: 0.1
getting quantile: 0.1
Quantile: 0.1, Frequencies: [(0, 210)]
Setting Target frequency. Load 1, Frequency Quantile 0.1, Frequency_GPU0 210
{"timestamp":"2025-01-05T16:36:43.561046Z","level":"INFO","fields":{"message":"Setting Target frequency. Simulated Time 2025-01-05 16:36:43 UTC, Load 1, Frequency Quantile 0.1, Frequency_GPU0 210","simulated_time":"2025-01-05 16:36:43 UTC","load":"1","frequency_quantile":"0.1","frequency_gpu0":"210"},"target":"co2cpufrequency::gpu_energy_manager"}
```

Use ctrl-c to kill the agent

### Set the correct mode for the test
The file generate_inference_load.ps1 is the main inference test driver.  There are several modes it can run in
* none -- for testing the default behavior, only does the power monitoring and runs the test but no power limiting or agent execution
* frequency -- loops through the range of available gpu frequencies to generate a frequency load line.  This is a long test.
* power -- loops through the range of available gpu power caps to generate a power load line.  This is a long test.
* agent -- runs the agent in --gpu-energy-mode while running default test.

Open generate_inference_load.ps1 and ensure the $limiting_mode parameter is set to "agent" for this initial test.  This will run a single pass through the test prompts varied across the 3 models and generate the output for that.  The agent will be using the load line installed at ./CarbonAwareLinux/os/carbon_aware_governor/gpu_configs/power_percentages_h100_virtualAmdEpyc.csv


### Run the test
Change directory back to CarbonAwareLinux/benchmarks/gpu_load_line
```
sudo -E pwsh generate_inference_load.ps1
```
You should see both the agent spew as well as the prompts as they get cycled through.

### Troubleshooting
Common mistakes I make
* Ollama docker not running or all required models not available
* Not passing in -E when running agent
* Not having all required environment variables set in current environment; sometimes this looks like this thread 'tokio-runtime-worker' panicked at src/main.rs:281:10:
Failed to read LOCAL_LAT environment variable: EnvVar(NotPresent)
* If each prompt looks like its running slow check the cpu utilization and it should't be really high; ollama is really good at spilling to ram if it runs out of GPU memory and that can happen on 48GB or smaller GPUs with this test.
* sometime python doesn't get correctly installed from the yaml file.  If so just conda install python=3.12 with the gpu_load_line env activated

### Analyzing the results
A full pass through the promts takes around 10 minutes on an h100
There are two files which are output: nvidia_smi_log.csv which are the gpu performance metrics and inference_load.csv which are metrics about the prompt.  If you have jupyter installed you can use the notebook analyze_agent.ipynb to check the performance of your current run against a default run which has previously been generated.  All previous runs are stored in the same directory with lables around the gpu and type of test run.  For example if you just ran this test on an h100 you should rename the files to nvidia_smi_log.agent_h100_mixed.csv and inference_load.agent_h100_mixed.csv.  

## Generating the load line
Install Powershell (I might migrate this dependency to bash in the future)
```
sudo apt-get install -y wget apt-transport-https software-properties-common
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-ubuntu-focal-prod focal main" > /etc/apt/sources.list.d/microsoft.list'
sudo apt-get update
sudo apt-get install -y powershell
```

Create the conda environment
```
conda env create -f gpu_load_line_env.yml
conda activate gpu_load_line
```

Generate monitor the data and run the test

This has a dependency on monitor_nvidia.py (which itself has a dependency on nvidia-smi so make sure nvidia-smi is working as its not currently installed in the conda env).
```
./generate_inference_load.ps1
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
```