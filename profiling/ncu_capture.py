import argparse, shutil, subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--set", choices=["basic", "full"], default="basic")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not shutil.which("ncu"):
        raise SystemExit("ncu not found; install Nsight Compute from NVIDIA.")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("pass a command after --")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ncu", "--set", args.set, "--target-processes", "all",
           "--force-overwrite", "-o", args.output, *command]
    subprocess.run(cmd, check=True, timeout=args.timeout)

if __name__ == "__main__":
    main()
