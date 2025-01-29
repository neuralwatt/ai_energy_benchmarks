import subprocess
import pandas as pd
import time

def get_nvidia_smi_output():
    result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu,clocks.gr,clocks.mem,power.limit,power.draw', 
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def log_nvidia_smi_to_csv(logfile, interval=.1):
    while True:
        output = get_nvidia_smi_output()
        data = [line.split(', ') for line in output.strip().split('\n')]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'index', 'name', 'utilization.gpu(%)', 'utilization.memory(%)', 
            'memory.total(MiB)', 'memory.free(MiB)', 'memory.used(MiB)', 'temperature.gpu(C)', 
            'gpu.frequency(Ghz)', 'vram.frequency(Ghz)', 'power.limit(W)', 'power.draw(W)'
        ])
        df.to_csv(logfile, mode='a', header=not pd.io.common.file_exists(logfile), index=False)
        time.sleep(interval)

if __name__ == '__main__':
    logfile = 'nvidia_smi_log.csv'
    log_nvidia_smi_to_csv(logfile)