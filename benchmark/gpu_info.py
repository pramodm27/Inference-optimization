import json
import subprocess

def gpu_info():
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
           "--format=csv,noheader"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

if __name__ == "__main__":
    print(json.dumps(gpu_info(), indent=2))
