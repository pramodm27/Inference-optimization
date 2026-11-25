import subprocess
from datetime import datetime, timezone

def snapshot():
    fields = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    rows = []
    for line in result.stdout.splitlines():
        values = [x.strip() for x in line.split(",")]
        if len(values) == 8:
            rows.append(dict(zip(
                ["index","name","gpu_util_pct","memory_util_pct","memory_used_mib",
                 "memory_total_mib","temperature_c","power_w"], values)))
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "gpus": rows}
