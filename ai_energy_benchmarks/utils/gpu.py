"""GPU monitoring utilities."""

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GPUStats:
    """GPU statistics snapshot."""

    gpu_id: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: Optional[float]
    power_draw_w: Optional[float]
    timestamp: float


class GPUMonitor:
    """Monitor GPU utilization and statistics."""

    @staticmethod
    def get_gpu_stats(gpu_id: int = 0) -> Optional[GPUStats]:
        """Get current GPU statistics.

        Args:
            gpu_id: GPU device ID

        Returns:
            GPUStats object or None if failed
        """
        try:
            # Try nvidia-smi first
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    memory_used = float(parts[2])
                    memory_total = float(parts[3])

                    return GPUStats(
                        gpu_id=int(parts[0]),
                        utilization_percent=float(parts[1]),
                        memory_used_mb=memory_used,
                        memory_total_mb=memory_total,
                        memory_percent=(
                            (memory_used / memory_total * 100) if memory_total > 0 else 0
                        ),
                        temperature_c=(
                            float(parts[4]) if len(parts) > 4 and parts[4].strip() else None
                        ),
                        power_draw_w=(
                            float(parts[5]) if len(parts) > 5 and parts[5].strip() else None
                        ),
                        timestamp=time.time(),
                    )
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
            FileNotFoundError,
        ):
            pass

        # Try PyTorch CUDA if nvidia-smi fails
        try:
            import torch

            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(gpu_id)
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**2)  # MB
                memory_total = props.total_memory / (1024**2)  # MB

                return GPUStats(
                    gpu_id=gpu_id,
                    utilization_percent=0.0,  # PyTorch doesn't provide utilization
                    memory_used_mb=memory_reserved,
                    memory_total_mb=memory_total,
                    memory_percent=(
                        (memory_reserved / memory_total * 100) if memory_total > 0 else 0
                    ),
                    temperature_c=None,
                    power_draw_w=None,
                    timestamp=time.time(),
                )
        except ImportError:
            pass

        return None

    @staticmethod
    def monitor_during_operation(
        operation_func, gpu_id: int = 0, interval: float = 0.5, *args, **kwargs
    ) -> Dict[str, Any]:
        """Monitor GPU during an operation.

        Args:
            operation_func: Function to execute
            gpu_id: GPU device ID
            interval: Monitoring interval in seconds
            *args, **kwargs: Arguments for operation_func

        Returns:
            Dict with operation result and GPU stats
        """
        import threading

        stats_list: List[GPUStats] = []
        monitoring = True

        def monitor_loop():
            while monitoring:
                stats = GPUMonitor.get_gpu_stats(gpu_id)
                if stats:
                    stats_list.append(stats)
                time.sleep(interval)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        # Execute operation
        start_time = time.time()
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            monitoring = False
            monitor_thread.join(timeout=2)

        end_time = time.time()

        # Calculate statistics
        if stats_list:
            avg_utilization = sum(s.utilization_percent for s in stats_list) / len(stats_list)
            max_utilization = max(s.utilization_percent for s in stats_list)
            avg_memory = sum(s.memory_used_mb for s in stats_list) / len(stats_list)
            max_memory = max(s.memory_used_mb for s in stats_list)
            avg_power = None
            max_power = None

            power_stats = [s.power_draw_w for s in stats_list if s.power_draw_w is not None]
            if power_stats:
                avg_power = sum(power_stats) / len(power_stats)
                max_power = max(power_stats)

            gpu_active = max_utilization > 5.0  # Consider GPU active if utilization > 5%
        else:
            avg_utilization = 0
            max_utilization = 0
            avg_memory = 0
            max_memory = 0
            avg_power = None
            max_power = None
            gpu_active = False

        return {
            "result": result,
            "success": success,
            "error": error,
            "duration": end_time - start_time,
            "gpu_stats": {
                "gpu_id": gpu_id,
                "samples": len(stats_list),
                "avg_utilization_percent": avg_utilization,
                "max_utilization_percent": max_utilization,
                "avg_memory_mb": avg_memory,
                "max_memory_mb": max_memory,
                "avg_power_w": avg_power,
                "max_power_w": max_power,
                "gpu_active": gpu_active,
            },
        }

    @staticmethod
    def check_gpu_available(gpu_id: int = 0) -> bool:
        """Check if GPU is available.

        Args:
            gpu_id: GPU device ID

        Returns:
            bool: True if GPU is available
        """
        stats = GPUMonitor.get_gpu_stats(gpu_id)
        return stats is not None

    @staticmethod
    def print_gpu_info(gpu_id: int = 0):
        """Print GPU information.

        Args:
            gpu_id: GPU device ID
        """
        stats = GPUMonitor.get_gpu_stats(gpu_id)
        if stats:
            print(f"GPU {stats.gpu_id}:")
            print(f"  Utilization: {stats.utilization_percent:.1f}%")
            print(
                f"  Memory: {stats.memory_used_mb:.0f}/{stats.memory_total_mb:.0f} MB ({stats.memory_percent:.1f}%)"
            )
            if stats.temperature_c is not None:
                print(f"  Temperature: {stats.temperature_c:.1f}°C")
            if stats.power_draw_w is not None:
                print(f"  Power: {stats.power_draw_w:.1f}W")
        else:
            print(f"GPU {gpu_id}: Not available or unable to query")
