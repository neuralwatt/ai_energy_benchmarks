import subprocess
import pandas as pd
import time
import sys
import os

def get_nvidia_smi_output():
    result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,memory.total,memory.free,memory.used,temperature.gpu,clocks.gr,clocks.mem,power.limit,power.draw', 
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def log_nvidia_smi_to_csv(logfile, interval=.1):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(logfile) or '.', exist_ok=True)
    
    while True:
        output = get_nvidia_smi_output()
        data = [line.split(', ') for line in output.strip().split('\n')]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'index', 'name', 'utilization.gpu(%)', 'utilization.memory(%)',
            'clocks_throttle_reasons.sw_power_cap', 'clocks_throttle_reasons.sw_thermal_slowdown',
            'memory.total(MiB)', 'memory.free(MiB)', 'memory.used(MiB)', 'temperature.gpu(C)', 
            'gpu.frequency(Ghz)', 'vram.frequency(Ghz)', 'power.limit(W)', 'power.draw(W)'
        ])
        df.to_csv(logfile, mode='a', header=not pd.io.common.file_exists(logfile), index=False)
        time.sleep(interval)

if __name__ == '__main__':
    output_dir = "benchmark_output"  # Default output directory
    
    if len(sys.argv) > 1:
        logfile = sys.argv[1]
    else:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        logfile = os.path.join(output_dir, 'nvidia_smi_log.csv')
    
    # If an output directory is specified as second argument, use it
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        # If logfile doesn't already have a path, put it in the output directory
        if not os.path.dirname(logfile):
            logfile = os.path.join(output_dir, os.path.basename(logfile))
    
    log_nvidia_smi_to_csv(logfile)