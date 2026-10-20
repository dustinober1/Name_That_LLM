import subprocess
import time
import sys
import re
from pathlib import Path
from datetime import datetime

KERNEL_SLUG = "dustinober/name-that-llm-solution-pipeline"
COMPETITION = "name-that-llm"
VERSION = "10"
SUBMISSION_MESSAGE = "Ensemble v10: Master Quad-Model + StyleGrams (Shape/Function-Word) + Sinkhorn Full Matrix Balancing + Hungarian"
LOG_FILE = Path("v10_submission_monitor.log")

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def run_cmd(cmd: list) -> tuple[int, str, str]:
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()

def main():
    log("=== Starting Automated Kernel Monitor & Submission for Version 10 ===")
    
    # Phase 1: Wait for Kernel Execution to Finish
    prev_status = None
    start_time = time.time()
    
    while True:
        code, out, err = run_cmd(["kaggle", "kernels", "status", KERNEL_SLUG])
        combined = (out + " " + err).lower()
        
        if combined != prev_status:
            log(f"Kernel status update: {out}")
            prev_status = combined
            
        if "complete" in combined:
            log("✓ Kernel execution completed successfully!")
            break
        elif "error" in combined or "failed" in combined or "cancel" in combined:
            log(f"✗ Kernel execution failed or encountered error:\n{out}\n{err}")
            sys.exit(1)
            
        time.sleep(20)
        
    # Phase 2: Submit to Kaggle Competition
    log(f"Submitting submission.parquet from kernel {KERNEL_SLUG} (Version {VERSION}) to {COMPETITION}...")
    submit_cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-f", "submission.parquet",
        "-k", KERNEL_SLUG,
        "-v", VERSION,
        "-m", SUBMISSION_MESSAGE
    ]
    code, out, err = run_cmd(submit_cmd)
    log(f"Submit command output:\n{out}\n{err}")
    if code != 0:
        log("✗ Submission command returned non-zero exit code!")
        sys.exit(code)
        
    log("✓ Successfully submitted to Kaggle! Monitoring leaderboard evaluation...")
    
    # Phase 3: Wait for Competition Submission Evaluation
    eval_start = time.time()
    while time.time() - eval_start < 3600:  # Max 1 hour evaluation wait
        time.sleep(25)
        code, out, err = run_cmd(["kaggle", "competitions", "submissions", "-c", COMPETITION])
        lines = out.splitlines()
        if len(lines) >= 3:
            header = lines[0]
            latest = lines[2]
            log(f"Latest submission row:\n{latest}")
            
            if "complete" in latest.lower():
                log(f"🎉 SUBMISSION COMPLETE! Final result:\n{header}\n{latest}")
                break
            elif "error" in latest.lower() or "failed" in latest.lower():
                log(f"✗ Evaluation failed:\n{latest}")
                break
        else:
            log(f"Unexpected submissions output:\n{out}")

    log("=== Monitor & Submission Process Finished ===")

if __name__ == "__main__":
    main()
