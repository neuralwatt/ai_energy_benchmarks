#!/usr/bin/env python3
import argparse
import asyncio
import aiohttp
import csv
import json
import logging
import os
import random
import time
import multiprocessing
import subprocess
import warnings
import numpy as np
import threading
from functools import partial
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PowerMonitor class for GPU power monitoring
@dataclass
class PowerMetrics:
    total_power_joules: float
    peak_power_watts: float
    avg_power_watts: float
    mean_tokens_per_watt: float
    median_tokens_per_watt: float
    p99_tokens_per_watt: float

class PowerMonitor:
    """Monitors GPU power consumption using nvidia-smi."""
    
    def __init__(self, interval=0.1):
        """Initialize the power monitor.
        
        Args:
            interval: Sampling interval in seconds.
        """
        self.interval = interval
        self.power_samples = []
        self.timestamps = []
        self.running = False
        self.thread = None
        
        # Test nvidia-smi availability and output at init time
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            logger.info(f"PowerMonitor initialized. Test reading: {output}")
            self.gpu_available = True
        except Exception as e:
            logger.warning(f"PowerMonitor initialization failed: {e}")
            self.gpu_available = False
    
    def _collect_samples(self):
        """Collect power samples from nvidia-smi."""
        logger.info("Power monitoring thread started")
        sample_count = 0
        last_sample_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                # Use Popen for better control
                process = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=0.5)  # Set a timeout to avoid hanging
                
                if process.returncode != 0:
                    logger.warning(f"nvidia-smi failed: {stderr.strip()}")
                    # Add a zero sample to maintain timing consistency
                    self.power_samples.append(0.0)
                    self.timestamps.append(current_time)
                else:
                    # Parse power values from all GPUs and sum them
                    power_values = [float(val.strip()) for val in stdout.strip().split('\n') if val.strip()]
                    total_power = sum(power_values)
                    
                    # Store sample and timestamp
                    self.power_samples.append(total_power)
                    self.timestamps.append(current_time)
                    
                    # Only log occasionally to avoid flooding logs
                    sample_count += 1
                    if sample_count % 10 == 0:  # Log every 10 samples
                        logger.info(f"Power monitoring: collected {sample_count} samples, latest: {total_power}W")
                
                # Make sure we don't get trapped in a tight loop if something's wrong
                time_since_last = current_time - last_sample_time
                if time_since_last < self.interval:
                    time.sleep(self.interval - time_since_last)
                
                last_sample_time = current_time
                
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("nvidia-smi command timed out")
                # Add a zero sample to maintain timing consistency
                self.power_samples.append(0.0)
                self.timestamps.append(time.time())
            except Exception as e:
                logger.warning(f"Failed to collect power sample: {e}")
                # Add a zero sample to maintain timing consistency
                self.power_samples.append(0.0)
                self.timestamps.append(time.time())
                # Don't spam if we're getting errors
                time.sleep(self.interval)
        
        logger.info(f"Power monitoring thread stopped. Collected {len(self.power_samples)} samples.")
        if len(self.power_samples) > 0:
            non_zero_samples = [s for s in self.power_samples if s > 0]
            if non_zero_samples:
                logger.info(f"Power stats: min={min(non_zero_samples):.2f}W, max={max(non_zero_samples):.2f}W, avg={sum(non_zero_samples)/len(non_zero_samples):.2f}W")
    
    def start(self):
        """Start power monitoring in a background thread."""
        if self.running:
            return self
        
        if not self.gpu_available:
            logger.warning("Starting power monitor with no GPU availability. Power metrics will be reported as 0.")
            # Add some fake samples to ensure the thread doesn't fail
            self.power_samples = [0.0]
            self.timestamps = [time.time()]
            return self
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_samples, daemon=True)
        self.thread.start()
        logger.info("Power monitoring thread created and started")
        return self
    
    def stop(self):
        """Stop power monitoring and return collected samples."""
        if not self.running:
            return self.power_samples, self.timestamps
            
        logger.info("Stopping power monitoring thread")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Power monitoring thread did not terminate properly")
        
        if not self.power_samples:
            logger.warning("No power samples collected during monitoring")
            return [0.0], [time.time()]
            
        logger.info(f"Power monitoring completed. Collected {len(self.power_samples)} samples.")
        logger.info(f"Average power: {sum(self.power_samples)/len(self.power_samples):.2f}W, Peak: {max(self.power_samples):.2f}W")
        
        return self.power_samples, self.timestamps

def calculate_power_metrics(power_samples, timestamps, total_tokens, benchmark_duration):
    """Calculate power-related metrics from collected samples.
    
    Args:
        power_samples: List of power measurements in watts
        timestamps: List of timestamp values when measurements were taken
        total_tokens: Total number of tokens processed
        benchmark_duration: Duration of the benchmark in seconds
        
    Returns:
        PowerMetrics object containing calculated metrics
    """
    if not power_samples:
        logger.warning("No power samples collected, cannot calculate power metrics")
        return PowerMetrics(
            total_power_joules=0.0,
            peak_power_watts=0.0,
            avg_power_watts=0.0,
            mean_tokens_per_watt=0.0,
            median_tokens_per_watt=0.0,
            p99_tokens_per_watt=0.0
        )
        
    # If we have only one sample, use it to estimate power for the whole duration
    if len(power_samples) == 1:
        logger.info(f"Only one power sample available ({power_samples[0]}W), using it for the entire duration")
        avg_power = power_samples[0]
        total_energy = avg_power * benchmark_duration
        tokens_per_watt = total_tokens / total_energy if total_energy > 0 else 0
        
        return PowerMetrics(
            total_power_joules=total_energy,
            peak_power_watts=avg_power,
            avg_power_watts=avg_power,
            mean_tokens_per_watt=tokens_per_watt,
            median_tokens_per_watt=tokens_per_watt,
            p99_tokens_per_watt=tokens_per_watt
        )
    
    # With multiple samples, calculate using intervals
    intervals = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    
    # Calculate energy (joules) for each interval (power × time)
    # Use average power between consecutive samples
    energy_per_interval = [
        ((p1 + p2) / 2) * interval 
        for p1, p2, interval in zip(power_samples[:-1], power_samples[1:], intervals)
    ]
    
    # Calculate total energy consumption in joules
    total_energy = sum(energy_per_interval)
    
    # If the intervals don't cover the full benchmark duration, extrapolate
    covered_duration = sum(intervals)
    if covered_duration < benchmark_duration * 0.9:  # If we're missing more than 10%
        logger.warning(f"Power samples cover only {covered_duration:.2f}s of {benchmark_duration:.2f}s benchmark duration")
        # Extrapolate energy for the remaining time using the average power
        avg_measured_power = total_energy / covered_duration if covered_duration > 0 else 0
        remaining_duration = benchmark_duration - covered_duration
        remaining_energy = avg_measured_power * remaining_duration
        total_energy += remaining_energy
        logger.info(f"Added estimated {remaining_energy:.2f}J for uncovered {remaining_duration:.2f}s")
    
    # Other metrics
    peak_power = max(power_samples)
    avg_power = total_energy / benchmark_duration if benchmark_duration > 0 else 0
    
    # Tokens per watt (energy efficiency)
    tokens_per_watt = total_tokens / total_energy if total_energy > 0 else 0
    
    return PowerMetrics(
        total_power_joules=total_energy,
        peak_power_watts=peak_power,
        avg_power_watts=avg_power,
        mean_tokens_per_watt=tokens_per_watt,
        median_tokens_per_watt=tokens_per_watt,  # Using same value as mean for now
        p99_tokens_per_watt=tokens_per_watt      # Using same value as mean for now
    )

class BackgroundPowerMonitor:
    """A simplified power monitor that runs in a separate process."""
    
    def __init__(self, interval=0.5, output_file=None):
        self.interval = interval
        self.output_file = output_file
        self.process = None
        
    def start(self):
        """Start continuous power monitoring in a separate process."""
        if self.process and self.process.is_alive():
            return self
            
        def monitor_power():
            """Function to run in a separate process to monitor power"""
            try:
                # Verify nvidia-smi is working first
                test_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                test_output = test_result.stdout.strip()
                print(f"[Power Monitor] Test reading: {test_output}")
                
                with open(self.output_file, 'w') as f:
                    f.write("timestamp,power_watts\n")
                    f.flush()
                    
                    start_time = time.time()
                    sample_count = 0
                    
                    while True:
                        try:
                            # Get power reading from nvidia-smi
                            result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            power_values = [float(val.strip()) for val in result.stdout.strip().split('\n') if val.strip()]
                            total_power = sum(power_values)
                            current_time = time.time()
                            
                            # Write to file
                            f.write(f"{current_time},{total_power}\n")
                            f.flush()
                            
                            sample_count += 1
                            if sample_count % 10 == 0:
                                elapsed = current_time - start_time
                                rate = sample_count / elapsed if elapsed > 0 else 0
                                print(f"[Power Monitor] Collected {sample_count} samples ({rate:.1f}/s). Latest: {total_power:.2f}W")
                            
                            # Sleep precisely
                            next_sample_time = current_time + self.interval
                            sleep_time = max(0, next_sample_time - time.time())
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                                
                        except Exception as e:
                            print(f"[Power Monitor] Error: {e}")
                            time.sleep(self.interval)  # Still sleep on error
                            
            except Exception as e:
                print(f"[Power Monitor] Fatal error: {e}")
                        
        if self.output_file:
            # Use separate process instead of thread to avoid GIL issues
            self.process = multiprocessing.Process(target=monitor_power, daemon=True)
            self.process.start()
            logger.info(f"Started background power monitoring process (PID: {self.process.pid})")
            
        return self
        
    def stop(self):
        """Stop power monitoring and return collected samples."""
        if self.process:
            logger.info(f"Stopping background power monitoring process (PID: {self.process.pid})")
            self.process.terminate()
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                logger.warning("Background power monitoring process did not terminate, killing it")
                self.process.kill()
                self.process.join(timeout=0.5)
            
        # Try to read the file with power samples
        power_samples = []
        timestamps = []
        
        if self.output_file and os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    # Skip header
                    next(f)
                    for line in f:
                        if line.strip():
                            parts = line.strip().split(',')
                            if len(parts) == 2:
                                timestamps.append(float(parts[0]))
                                power_samples.append(float(parts[1]))
                logger.info(f"Read {len(power_samples)} power samples from {self.output_file}")
                
                if power_samples:
                    logger.info(f"Background power stats: min={min(power_samples):.2f}W, max={max(power_samples):.2f}W, avg={sum(power_samples)/len(power_samples):.2f}W")
            except Exception as e:
                logger.warning(f"Error reading power samples from file: {e}")
                
        return power_samples, timestamps

class InferenceLoadGenerator:
    def __init__(
        self,
        model_name: str,
        prompts_file: str,
        concurrency: int,
        host: str = "localhost",
        port: int = 8000,
        max_tokens: int = 2000,
        stream: bool = False,
        num_requests: int = 0,  # 0 means unlimited/all available prompts
        output_dir: str = "benchmark_output",
        num_processes: int = 1,  # Number of CPU processes to use
        random: bool = True     # Whether to select prompts randomly or sequentially
    ):
        self.model_name = model_name
        self.prompts_file = prompts_file
        self.concurrency = concurrency
        self.host = host
        self.port = port
        self.max_tokens = max_tokens
        self.stream = stream
        self.num_requests = num_requests
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.random = random
        self.prompts = []
        self.responses = []
        self.completion_times = []
        self.start_time = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log file for prompts and responses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.output_dir, 
            f"prompts_and_responses.{self.get_gpu_type()}_{model_name.replace('/', '-')}_{timestamp}.log"
        )
        
        # CSV file for metrics
        self.metrics_file = os.path.join(
            self.output_dir, 
            f"inference.load.none_{self.get_gpu_type()}_{model_name.replace('/', '-')}_{timestamp}.csv"
        )
        
        # CSV file for power samples
        self.power_samples_file = os.path.join(
            self.output_dir, 
            f"power_samples_{self.get_gpu_type()}_{model_name.replace('/', '-')}_{timestamp}.csv"
        )
        
        # Create a manager for sharing data between processes
        self.manager = multiprocessing.Manager()
        self.shared_responses = self.manager.list()
        self.shared_completion_times = self.manager.list()
        
        # Initialize power monitor separately to avoid serialization issues
        self.power_monitor = None

    def get_gpu_type(self) -> str:
        """Get the GPU type for file naming. Returns h100 as default for now."""
        return "h100"  # This could be made more dynamic in the future

    def load_prompts(self) -> None:
        """Load prompts from the CSV file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                self.prompts = [row[0] for row in reader]
                
            logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
            
            if self.num_requests > 0 and self.num_requests < len(self.prompts):
                if self.random:
                    # Randomly select prompts if random mode is enabled
                    self.prompts = random.sample(self.prompts, self.num_requests)
                    logger.info(f"Randomly selected {len(self.prompts)} prompts for inference")
                else:
                    # Sequentially select prompts if random mode is disabled
                    self.prompts = self.prompts[:self.num_requests]
                    logger.info(f"Sequentially selected the first {len(self.prompts)} prompts for inference")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

    async def make_inference_request(self, session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
        """Make a single inference request and return the result."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": self.stream,
            "max_tokens": self.max_tokens
        }
        
        start_time = time.time()
        ttft = None  # Time to first token
        token_times = []  # List of timestamps when tokens are received
        
        try:
            if self.stream:
                # For streaming responses, we need to track token timing
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {error_text}")
                    
                    # Process the streaming response to get token timing
                    generated_text = ""
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        if not line_text or line_text == "data: [DONE]":
                            continue
                        if line_text.startswith('data: '):
                            token_time = time.time()
                            if ttft is None:
                                ttft = token_time - start_time
                            token_times.append(token_time)
                            
                            # Parse the chunk to get the token text
                            try:
                                chunk = json.loads(line_text[5:])  # Remove "data: " prefix
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        generated_text += delta['content']
                            except json.JSONDecodeError:
                                pass
                    
                    result = {
                        "choices": [{"message": {"content": generated_text}}]
                    }
            else:
                # For non-streaming responses
                async with session.post(url, json=payload) as response:
                    result = await response.json()
                    token_received_time = time.time()
                    ttft = token_received_time - start_time
                    # For non-streaming, we can't track individual tokens, so estimate
                    # Add a single timestamp for the entire response
                    token_times = [token_received_time]
            
            elapsed_time = time.time() - start_time
            
            # Calculate ITL (Inter-Token Latency)
            itl = []
            if len(token_times) > 1:
                itl = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
            
            # Log the prompt and response
            with open(self.log_file, 'a', encoding='utf-8') as log:
                log.write(f"PROMPT: {prompt}\n")
                log.write(f"RESPONSE: {json.dumps(result)}\n")
                log.write("-" * 80 + "\n")
            
            # Extract the generated text from the response
            generated_text = ""
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0]:
                    generated_text = result["choices"][0]["message"].get("content", "")
                elif "text" in result["choices"][0]:
                    generated_text = result["choices"][0]["text"]
            
            return {
                "prompt": prompt,
                "response": result,
                "elapsed_time": elapsed_time,
                "ttft": ttft if ttft is not None else elapsed_time,  # Use full time if no TTFT
                "itl": itl,
                "token_times": token_times,
                "generated_text": generated_text,
                "output_tokens": len(token_times) if self.stream else None  # Only available for streaming
            }
        except Exception as e:
            logger.error(f"Error making inference request: {e}")
            return {
                "prompt": prompt,
                "response": {"error": str(e)},
                "elapsed_time": time.time() - start_time,
                "ttft": None,
                "itl": [],
                "token_times": [],
                "generated_text": "",
                "output_tokens": 0
            }

    async def run_inference_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Run inference on a batch of prompts concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.make_inference_request(session, prompt) for prompt in batch]
            return await asyncio.gather(*tasks)

    async def process_batch_async(self, batch_idx: int, prompts_batch: List[str]):
        """Process a batch of prompts asynchronously."""
        logger.info(f"Process {batch_idx}: Starting batch with {len(prompts_batch)} prompts")
        
        results = await self.run_inference_batch(prompts_batch)
        
        # Add results to shared list for later aggregation
        for result in results:
            self.shared_responses.append(result)
            self.shared_completion_times.append(result['elapsed_time'])
        
        batch_times = [r['elapsed_time'] for r in results]
        avg_time = sum(batch_times) / len(batch_times)
        logger.info(f"Process {batch_idx}: Batch average completion time: {avg_time:.2f}s")
        
        return results

    def process_batch(self, batch_idx: int, prompts_batch: List[str]):
        """Process a batch of prompts in a separate process."""
        # Run asyncio event loop in this process
        asyncio.run(self.process_batch_async(batch_idx, prompts_batch))

    def write_metrics(self, power_metrics=None) -> None:
        """Write performance metrics to a CSV file."""
        with open(self.metrics_file, 'w', newline='') as csvfile:
            fieldnames = [
                'request_id', 'prompt_length', 'completion_tokens', 'elapsed_time',
            ]
            
            # Add power metrics fields if available
            if power_metrics:
                fieldnames.extend([
                    'total_power_joules', 'peak_power_watts', 'avg_power_watts',
                    'mean_tokens_per_watt', 'median_tokens_per_watt', 'p99_tokens_per_watt'
                ])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, result in enumerate(self.responses):
                row = {
                    'request_id': i,
                    'prompt_length': len(result['prompt']),
                    'completion_tokens': self.max_tokens,  # Assuming all requests use max_tokens
                    'elapsed_time': result['elapsed_time']
                }
                
                # Add power metrics if available (same for all rows)
                if power_metrics:
                    row.update({
                        'total_power_joules': power_metrics.total_power_joules,
                        'peak_power_watts': power_metrics.peak_power_watts,
                        'avg_power_watts': power_metrics.avg_power_watts,
                        'mean_tokens_per_watt': power_metrics.mean_tokens_per_watt,
                        'median_tokens_per_watt': power_metrics.median_tokens_per_watt,
                        'p99_tokens_per_watt': power_metrics.p99_tokens_per_watt
                    })
                
                writer.writerow(row)

    def calculate_timing_metrics(self):
        """Calculate detailed timing metrics from the responses."""
        # Extract timing data from all responses
        ttfts = [r.get('ttft', 0) for r in self.responses if r.get('ttft') is not None]
        
        # Calculate TPOT (Time per Output Token, excluding the first token)
        tpots = []
        for r in self.responses:
            ttft = r.get('ttft', 0)
            elapsed = r.get('elapsed_time', 0)
            output_tokens = r.get('output_tokens', self.max_tokens)  # Fallback to max_tokens if not available
            
            # Only calculate TPOT if we have more than one token
            if output_tokens and output_tokens > 1 and ttft is not None:
                tpot = (elapsed - ttft) / (output_tokens - 1)
                tpots.append(tpot)
        
        # Extract all inter-token latencies
        itls = []
        for r in self.responses:
            if 'itl' in r and r['itl']:
                itls.extend(r['itl'])
        
        # Calculate metrics for each timing category
        metrics = {}
        
        percentiles = [50, 90, 95, 99]  # Standard percentiles to report
        
        # Calculate TTFT metrics
        if ttfts:
            metrics['ttft'] = {
                'mean': np.mean(ttfts) * 1000,  # Convert to ms
                'median': np.median(ttfts) * 1000,
                'std': np.std(ttfts) * 1000,
                'percentiles': [(p, np.percentile(ttfts, p) * 1000) for p in percentiles]
            }
        
        # Calculate TPOT metrics
        if tpots:
            metrics['tpot'] = {
                'mean': np.mean(tpots) * 1000,
                'median': np.median(tpots) * 1000,
                'std': np.std(tpots) * 1000,
                'percentiles': [(p, np.percentile(tpots, p) * 1000) for p in percentiles]
            }
        
        # Calculate ITL metrics
        if itls:
            metrics['itl'] = {
                'mean': np.mean(itls) * 1000,
                'median': np.median(itls) * 1000,
                'std': np.std(itls) * 1000,
                'percentiles': [(p, np.percentile(itls, p) * 1000) for p in percentiles]
            }
            
        return metrics

    def run(self) -> None:
        """Run the inference load generation process using multiple processes."""
        self.load_prompts()
        
        if not self.prompts:
            logger.error("No prompts loaded. Exiting.")
            return
        
        # Create a power monitoring file in the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        power_log_file = os.path.join(
            self.output_dir,
            f"power_log_{self.model_name.replace('/', '-')}_{timestamp}.csv"
        )
        
        # Start background power monitoring in a separate process
        bg_power_monitor = BackgroundPowerMonitor(interval=0.2, output_file=power_log_file).start()
        logger.info(f"Started background GPU power monitoring, logging to {power_log_file}")
        
        # Also start regular power monitor as backup
        self.power_monitor = PowerMonitor(interval=0.1).start()
        logger.info("Started GPU power monitoring")
        
        self.start_time = time.time()
        logger.info(f"Starting inference load generation with concurrency {self.concurrency} across {self.num_processes} processes")
        
        # Divide prompts into chunks for each process
        chunks_per_process = len(self.prompts) // self.num_processes
        if chunks_per_process == 0:
            chunks_per_process = 1
            
        prompt_chunks = []
        for i in range(0, len(self.prompts), chunks_per_process):
            if len(prompt_chunks) < self.num_processes:
                chunk = self.prompts[i:i + chunks_per_process]
                prompt_chunks.append(chunk)
            else:
                # Add any remaining prompts to the last chunk
                prompt_chunks[-1].extend(self.prompts[i:])
                break
        
        # If we have fewer prompts than processes, adjust num_processes
        actual_processes = min(self.num_processes, len(prompt_chunks))
        
        # Create and start processes
        processes = []
        for i in range(actual_processes):
            process = multiprocessing.Process(
                target=self.process_batch,
                args=(i, prompt_chunks[i])
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        # Record benchmark duration immediately after processes complete
        benchmark_duration = time.time() - self.start_time
        
        # Take 3 power readings with nvidia-smi to get a better fallback estimate
        fallback_powers = []
        for _ in range(3):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                power_values = [float(val.strip()) for val in result.stdout.strip().split('\n') if val.strip()]
                if power_values:
                    fallback_powers.append(sum(power_values))
                time.sleep(0.5)  # Short pause between readings
            except Exception as e:
                logger.warning(f"Failed to get fallback power reading: {e}")
        
        # Calculate average fallback power if we have readings
        fallback_power = 0.0
        if fallback_powers:
            fallback_power = sum(fallback_powers) / len(fallback_powers)
            logger.info(f"Got average fallback power reading: {fallback_power:.2f}W from {len(fallback_powers)} samples")
        
        # Stop power monitors
        power_samples, timestamps = self.power_monitor.stop()
        logger.info("Stopped in-process GPU power monitoring")
        
        bg_power_samples, bg_timestamps = bg_power_monitor.stop()
        logger.info(f"Stopped background GPU power monitoring, collected {len(bg_power_samples)} samples")
        
        # Choose the best power data source
        # Prefer background monitor data if it has samples
        final_power_samples = bg_power_samples if bg_power_samples else power_samples
        final_timestamps = bg_timestamps if bg_timestamps else timestamps
        
        # If we still don't have good samples but have a fallback, use the fallback
        if not final_power_samples or all(p == 0 for p in final_power_samples):
            if fallback_power > 0:
                logger.info(f"Using fallback power reading: {fallback_power:.2f}W")
                final_power_samples = [fallback_power]
                final_timestamps = [time.time()]
        
        # Save power samples to CSV with timestamps relative to start time
        if final_power_samples and final_timestamps:
            try:
                with open(self.power_samples_file, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp_seconds', 'power_watts']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for ts, pwr in zip(final_timestamps, final_power_samples):
                        # Convert timestamp to seconds relative to start_time
                        relative_time = ts - self.start_time
                        writer.writerow({
                            'timestamp_seconds': f"{relative_time:.3f}",
                            'power_watts': f"{pwr:.2f}"
                        })
                
                logger.info(f"Saved {len(final_power_samples)} power samples to {self.power_samples_file}")
            except Exception as e:
                logger.error(f"Failed to save power samples to CSV: {e}")
        
        # Collect results from shared lists
        self.responses = list(self.shared_responses)
        self.completion_times = list(self.shared_completion_times)
        
        avg_completion_time = sum(self.completion_times) / len(self.completion_times) if self.completion_times else 0
        throughput = len(self.prompts) / benchmark_duration if benchmark_duration > 0 else 0
        
        # Calculate total input and output tokens
        total_input_tokens = sum(len(result['prompt']) for result in self.responses)
        
        # Calculate actual output tokens instead of using max_tokens approximation
        total_output_tokens = 0
        for result in self.responses:
            # First check if the API response includes a completion_tokens field
            if 'response' in result and isinstance(result['response'], dict):
                # Check for completion_tokens in the top level response
                if 'completion_tokens' in result['response']:
                    total_output_tokens += result['response']['completion_tokens']
                    continue
                # Check for completion_tokens in usage field (OpenAI format)
                elif 'usage' in result['response'] and isinstance(result['response']['usage'], dict):
                    if 'completion_tokens' in result['response']['usage']:
                        total_output_tokens += result['response']['usage']['completion_tokens']
                        continue
                
                # Extract and parse the message content directly from the response
                try:
                    if 'choices' in result['response'] and len(result['response']['choices']) > 0:
                        choice = result['response']['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            # If content appears to be truncated, try to get as much as possible
                            if isinstance(content, str):
                                # A simple token estimation (approximation)
                                estimated_tokens = len(content.split()) * 1.3  # Average 1.3 tokens per word
                                total_output_tokens += int(estimated_tokens)
                                continue
                except Exception as e:
                    logger.warning(f"Error parsing response content: {e}")
            
            # If no completion_tokens found, fall back to our existing methods
            if 'output_tokens' in result and result['output_tokens'] is not None:
                total_output_tokens += result['output_tokens']
            elif 'token_times' in result and result['token_times']:
                # Use the length of token_times if available (for streaming responses)
                total_output_tokens += len(result['token_times'])
            elif 'generated_text' in result and result['generated_text']:
                # Improved token estimation for the text
                text = result['generated_text']
                # Count words and estimate tokens (GPT models average ~1.3 tokens per word)
                word_count = len(text.split())
                estimated_tokens = int(word_count * 1.3)
                total_output_tokens += estimated_tokens
            else:
                # Last resort, check the logged response
                logger.warning("Attempting to parse truncated response from log")
                try:
                    if 'response' in result and isinstance(result['response'], dict) and 'choices' in result['response']:
                        # If we have a truncated response, try to count visible words
                        raw_response = str(result['response'])
                        visible_content = raw_response.split('content": "')[1].split('"')[0] if 'content": "' in raw_response else ""
                        if visible_content:
                            word_count = len(visible_content.split())
                            estimated_tokens = int(word_count * 1.3)
                            total_output_tokens += estimated_tokens
                            continue
                except Exception as e:
                    logger.warning(f"Failed to parse truncated response: {e}")
                
                # Absolute last resort, use max_tokens
                logger.warning("Could not determine actual token count for a response, using max_tokens")
                total_output_tokens += self.max_tokens
        
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate power metrics
        power_metrics = calculate_power_metrics(
            power_samples=final_power_samples,
            timestamps=final_timestamps,
            total_tokens=total_tokens,
            benchmark_duration=benchmark_duration
        )
        
        # Calculate detailed timing metrics
        timing_metrics = self.calculate_timing_metrics()
        
        # Print performance summary
        print("{s:{c}^{n}}".format(s=' Inference Load Generator Results ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", len(self.responses)))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", total_input_tokens))
        print("{:<40} {:<10}".format("Total generated tokens:", total_output_tokens))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", throughput))
        print("{:<40} {:<10.2f}".format("Average completion time (s):", avg_completion_time))
        
        # Add token throughput metrics
        output_throughput = total_output_tokens / benchmark_duration if benchmark_duration > 0 else 0
        total_throughput = total_tokens / benchmark_duration if benchmark_duration > 0 else 0
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", output_throughput))
        print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", total_throughput))
        
        # Print detailed timing metrics
        if 'ttft' in timing_metrics:
            print("{s:{c}^{n}}".format(s=' Time to First Token ', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", timing_metrics['ttft']['mean']))
            print("{:<40} {:<10.2f}".format("Median TTFT (ms):", timing_metrics['ttft']['median']))
            for p, value in timing_metrics['ttft']['percentiles']:
                print("{:<40} {:<10.2f}".format(f"P{int(p) if p.is_integer() else p} TTFT (ms):", value))
        
        if 'tpot' in timing_metrics:
            print("{s:{c}^{n}}".format(s=' Time per Output Token (excl. 1st token) ', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", timing_metrics['tpot']['mean']))
            print("{:<40} {:<10.2f}".format("Median TPOT (ms):", timing_metrics['tpot']['median']))
            for p, value in timing_metrics['tpot']['percentiles']:
                print("{:<40} {:<10.2f}".format(f"P{int(p) if p.is_integer() else p} TPOT (ms):", value))
        
        if 'itl' in timing_metrics:
            print("{s:{c}^{n}}".format(s=' Inter-token Latency ', n=50, c='-'))
            print("{:<40} {:<10.2f}".format("Mean ITL (ms):", timing_metrics['itl']['mean']))
            print("{:<40} {:<10.2f}".format("Median ITL (ms):", timing_metrics['itl']['median']))
            for p, value in timing_metrics['itl']['percentiles']:
                print("{:<40} {:<10.2f}".format(f"P{int(p) if p.is_integer() else p} ITL (ms):", value))
        
        # Print power metrics
        print("{s:{c}^{n}}".format(s=' Power Consumption Metrics ', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Total Power (J):", power_metrics.total_power_joules))
        print("{:<40} {:<10.2f}".format("Peak Power (W):", power_metrics.peak_power_watts))
        print("{:<40} {:<10.2f}".format("Average Power (W):", power_metrics.avg_power_watts))
        print("{:<40} {:<10.2f}".format("Mean Tokens/Watt:", power_metrics.mean_tokens_per_watt))
        print("{:<40} {:<10.2f}".format("Median Tokens/Watt:", power_metrics.median_tokens_per_watt))
        print("{:<40} {:<10.2f}".format("P99 Tokens/Watt:", power_metrics.p99_tokens_per_watt))
        print("=" * 50)
        
        # Write metrics to CSV file
        self.write_metrics(power_metrics)
        
        logger.info(f"Metrics written to {self.metrics_file}")
        logger.info(f"Prompts and responses logged to {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate inference load with adjustable concurrency')
    parser.add_argument('--model', type=str, required=True, help='Model name to use for inference')
    parser.add_argument('--prompts-file', type=str, default='prompts.csv', help='CSV file containing prompts')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrent requests per process')
    parser.add_argument('--host', type=str, default='localhost', help='Inference server host')
    parser.add_argument('--port', type=int, default=8000, help='Inference server port')
    parser.add_argument('--max-tokens', type=int, default=30, help='Maximum tokens to generate')
    parser.add_argument('--stream', action='store_true', help='Use streaming mode')
    parser.add_argument('--num-requests', type=int, default=0, help='Number of requests to make (0 = all prompts)')
    parser.add_argument('--output-dir', type=str, default='benchmark_output', help='Output directory for logs and metrics')
    parser.add_argument('--processes', type=int, default=0, help='Number of CPU processes to use (0 = auto-detect)')
    parser.add_argument('--no-random', action='store_true', help='Select prompts sequentially instead of randomly')
    
    args = parser.parse_args()
    
    # Auto-detect number of CPU cores if not specified
    if args.processes <= 0:
        args.processes = multiprocessing.cpu_count()
        logger.info(f"Auto-detected {args.processes} CPU cores")
    
    load_generator = InferenceLoadGenerator(
        model_name=args.model,
        prompts_file=args.prompts_file,
        concurrency=args.concurrency,
        host=args.host,
        port=args.port,
        max_tokens=args.max_tokens,
        stream=args.stream,
        num_requests=args.num_requests,
        output_dir=args.output_dir,
        num_processes=args.processes,
        random=not args.no_random  # Use sequential selection if --no-random is specified
    )
    
    # Use multiprocessing for main run
    if __name__ == "__main__":
        load_generator.run()
    else:
        # When running in a spawned process
        asyncio.run(load_generator.process_batch_async(0, load_generator.prompts))

if __name__ == "__main__":
    # Fix for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
