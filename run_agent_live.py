import time
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = "sample_telemetry.csv"

# Latest/current output
LIVE_OUTPUT_CSV = "agent_results_live.csv"

# Full running history log
HISTORY_OUTPUT_CSV = "agent_results_history.csv"

AGENT_FILE = "ai_agent_telemetry_updated.py"

REFRESH_SECONDS = 3

# =========================================================
# LIVE LOOP
# =========================================================

print("=" * 70)
print("Starting LIVE AI Agent Monitoring")
print(f"Reading telemetry from: {INPUT_CSV}")
print(f"Live output file:       {LIVE_OUTPUT_CSV}")
print(f"History output file:    {HISTORY_OUTPUT_CSV}")
print(f"Refresh interval:       {REFRESH_SECONDS} seconds")
print("Press CTRL + C to stop.")
print("=" * 70)

try:

    while True:

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[{current_time}] Running AI agent...")

        try:

            # -------------------------------------------------
            # RUN AI AGENT
            # -------------------------------------------------

            subprocess.run(
                [
                    "python",
                    AGENT_FILE,
                    "--input",
                    INPUT_CSV,
                    "--output",
                    LIVE_OUTPUT_CSV
                ],
                check=True
            )

            print(f"Updated latest results: {LIVE_OUTPUT_CSV}")

            # -------------------------------------------------
            # APPEND TO HISTORY CSV
            # -------------------------------------------------

            if Path(LIVE_OUTPUT_CSV).exists():

                live_df = pd.read_csv(LIVE_OUTPUT_CSV)

                # Add timestamp for historical tracking
                live_df["log_timestamp"] = current_time

                # Append instead of overwrite
                if Path(HISTORY_OUTPUT_CSV).exists():

                    history_df = pd.read_csv(HISTORY_OUTPUT_CSV)

                    combined_df = pd.concat(
                        [history_df, live_df],
                        ignore_index=True
                    )

                    combined_df.to_csv(
                        HISTORY_OUTPUT_CSV,
                        index=False
                    )

                else:
                    # Create history file first time
                    live_df.to_csv(
                        HISTORY_OUTPUT_CSV,
                        index=False
                    )

                print(f"Appended results to: {HISTORY_OUTPUT_CSV}")

            else:
                print("Live output CSV not found.")

        except Exception as e:
            print(f"Error running AI agent: {e}")

        # -----------------------------------------------------
        # WAIT
        # -----------------------------------------------------

        time.sleep(REFRESH_SECONDS)

# =========================================================
# STOP CLEANLY
# =========================================================

except KeyboardInterrupt:
    print("\n")
    print("=" * 70)
    print("LIVE AI AGENT STOPPED")
    print("=" * 70)
