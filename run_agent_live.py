import time
import subprocess
from datetime import datetime

INPUT_CSV = "sample_telemetry.csv"
OUTPUT_CSV = "agent_results_live.csv"
AGENT_FILE = "ai_agent_telemetry_updated.py"

REFRESH_SECONDS = 3

while True:
    print(f"\nRunning AI agent at {datetime.now().strftime('%H:%M:%S')}...")

    try:
        subprocess.run(
            [
                "python",
                AGENT_FILE,
                "--input",
                INPUT_CSV,
                "--output",
                OUTPUT_CSV
            ],
            check=True
        )

        print(f"Updated {OUTPUT_CSV}")

    except Exception as e:
        print(f"Error running agent: {e}")

    time.sleep(REFRESH_SECONDS)
