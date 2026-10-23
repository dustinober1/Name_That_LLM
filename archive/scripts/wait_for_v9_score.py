import subprocess
import time
import sys
import re
from datetime import datetime

SUBMISSION_REF = "57406098"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def main():
    log(f"Monitoring submission {SUBMISSION_REF} until scored...")
    while True:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if SUBMISSION_REF in line:
                log(f"Current status line: {line.strip()}")
                if "complete" in line.lower():
                    log(f"🎉 SUBMISSION COMPLETE! Final row:\n{line}")
                    sys.exit(0)
                elif "error" in line.lower() or "failed" in line.lower():
                    log(f"✗ Submission failed:\n{line}")
                    sys.exit(1)
        time.sleep(25)

if __name__ == "__main__":
    main()
