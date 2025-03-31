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
from functools import partial
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        num_processes: int = 1  # Number of CPU processes to use
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
        
        # Create a manager for sharing data between processes
        self.manager = multiprocessing.Manager()
        self.shared_responses = self.manager.list()
        self.shared_completion_times = self.manager.list()

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
                # If num_requests is specified, select that many prompts randomly
                self.prompts = random.sample(self.prompts, self.num_requests)
                logger.info(f"Randomly selected {len(self.prompts)} prompts for inference")
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
        
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                elapsed_time = time.time() - start_time
                
                # Log the prompt and response
                with open(self.log_file, 'a', encoding='utf-8') as log:
                    log.write(f"PROMPT: {prompt}\n")
                    log.write(f"RESPONSE: {json.dumps(result)}\n")
                    log.write("-" * 80 + "\n")
                
                return {
                    "prompt": prompt,
                    "response": result,
                    "elapsed_time": elapsed_time
                }
        except Exception as e:
            logger.error(f"Error making inference request: {e}")
            return {
                "prompt": prompt,
                "response": {"error": str(e)},
                "elapsed_time": time.time() - start_time
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

    def write_metrics(self) -> None:
        """Write performance metrics to a CSV file."""
        with open(self.metrics_file, 'w', newline='') as csvfile:
            fieldnames = ['request_id', 'prompt_length', 'completion_tokens', 'elapsed_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, result in enumerate(self.responses):
                writer.writerow({
                    'request_id': i,
                    'prompt_length': len(result['prompt']),
                    'completion_tokens': self.max_tokens,  # Assuming all requests use max_tokens
                    'elapsed_time': result['elapsed_time']
                })

    def run(self) -> None:
        """Run the inference load generation process using multiple processes."""
        self.load_prompts()
        
        if not self.prompts:
            logger.error("No prompts loaded. Exiting.")
            return
        
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
        
        # Collect results from shared lists
        self.responses = list(self.shared_responses)
        self.completion_times = list(self.shared_completion_times)
        
        total_time = time.time() - self.start_time
        avg_completion_time = sum(self.completion_times) / len(self.completion_times) if self.completion_times else 0
        throughput = len(self.prompts) / total_time if total_time > 0 else 0
        
        logger.info(f"Inference load generation complete")
        logger.info(f"Total requests: {len(self.prompts)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average completion time: {avg_completion_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} requests/second")
        
        # Write metrics to CSV file
        self.write_metrics()
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
        num_processes=args.processes
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
