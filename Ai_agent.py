"""
ai_agent_telemetry_updated.py

Updated AI Agent Layer for EE 460 Edge-Based Anomaly Detection Project

This version is designed for your UPDATED notebook/pipeline where the ML models
already output telemetry probabilities such as:

    {
        "Vibration_Anomaly_Confidence": 0.9622,
        "Thermal_Anomaly_Confidence": 0.0025
    }

The models detect anomalies.
The AI agent interprets model telemetry, combines vibration + thermal risk,
triggers alerts, and creates technician-friendly maintenance recommendations.

Best project architecture:
    raw vibration/temp data
        -> feature extraction
        -> scaling/normalization
        -> Logistic Regression / MLP / LSTM / Autoencoder
        -> telemetry probabilities
        -> AI agent risk assessment
        -> optional Gemini technician report

Quick tests:
    python ai_agent_telemetry_updated.py --demo
    python ai_agent_telemetry_updated.py --make-sample-csv sample_telemetry.csv
    python ai_agent_telemetry_updated.py --input sample_telemetry.csv --output agent_report.csv

Optional Gemini:
    pip install google-generativeai
    export GEMINI_API_KEY="your_key_here"        # Mac/Linux
    set GEMINI_API_KEY=your_key_here             # Windows CMD
    $env:GEMINI_API_KEY="your_key_here"          # PowerShell
    python ai_agent_telemetry_updated.py --demo --use-gemini
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class AgentTelemetry:
    """One telemetry reading passed from the ML models to the AI agent."""

    timestamp: str = ""

    # Main updated notebook outputs
    vibration_anomaly_confidence: Optional[float] = None
    thermal_anomaly_confidence: Optional[float] = None

    # Optional PyTorch/LSTM-style outputs from your notebook
    pytorch_vibration_anomaly_confidence: Optional[float] = None
    pytorch_thermal_anomaly_confidence: Optional[float] = None

    # Optional classifier outputs if teammates still pass 0/1 predictions
    logistic_prediction: Optional[int] = None
    mlp_prediction: Optional[int] = None
    lstm_prediction: Optional[int] = None

    # Optional reconstruction error if using an autoencoder output
    lstm_reconstruction_error: Optional[float] = None
    lstm_threshold: Optional[float] = None

    # Optional raw features, useful for evidence/context only
    rms: Optional[float] = None
    kurtosis: Optional[float] = None
    delta_t: Optional[float] = None
    temperature_c: Optional[float] = None

    model_family: str = "multi-model telemetry"
    edge_device: str = "Raspberry Pi"


@dataclass
class AgentDecision:
    """Final AI agent decision."""

    timestamp: str
    risk_level: str
    combined_risk_score: float
    vibration_risk: Optional[float]
    thermal_risk: Optional[float]
    trigger_alert: bool
    explanation: str
    likely_cause: str
    recommended_action: str
    urgency: str
    edge_note: str
    evidence: str
    llm_used: bool



# Utility helpers


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp_probability(value: Optional[float]) -> Optional[float]:
    """Keep confidence values in the valid 0 to 1 range."""
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def get_first_existing(row: pd.Series, possible_columns: list[str], default: Any = None) -> Any:
    for col in possible_columns:
        if col in row.index:
            return row[col]
    return default


def probability_from_binary_prediction(pred: Optional[int]) -> Optional[float]:
    if pred is None:
        return None
    return 1.0 if pred == 1 else 0.0


# Core updated telemetry logic


def extract_best_vibration_risk(t: AgentTelemetry) -> Optional[float]:
    """
    Choose the best available vibration risk signal.

    Priority:
    1. Vibration_Anomaly_Confidence from scikit-learn telemetry
    2. PyTorch_Vibration_Anomaly_Confidence from LSTM/PyTorch telemetry
    3. LSTM reconstruction error compared to threshold
    4. binary predictions such as logistic/mlp/lstm
    """

    if t.vibration_anomaly_confidence is not None:
        return clamp_probability(t.vibration_anomaly_confidence)

    if t.pytorch_vibration_anomaly_confidence is not None:
        return clamp_probability(t.pytorch_vibration_anomaly_confidence)

    if t.lstm_reconstruction_error is not None and t.lstm_threshold is not None and t.lstm_threshold > 0:
        # Converts reconstruction error into a rough confidence score.
        # error == threshold -> 0.50, error much higher than threshold -> closer to 1.0
        ratio = t.lstm_reconstruction_error / t.lstm_threshold
        return clamp_probability(0.5 * ratio)

    votes = [
        probability_from_binary_prediction(t.logistic_prediction),
        probability_from_binary_prediction(t.mlp_prediction),
        probability_from_binary_prediction(t.lstm_prediction),
    ]
    votes = [v for v in votes if v is not None]
    if votes:
        return sum(votes) / len(votes)

    return None


def extract_best_thermal_risk(t: AgentTelemetry) -> Optional[float]:
    """
    Choose the best available thermal risk signal.

    Priority:
    1. Thermal_Anomaly_Confidence from scikit-learn telemetry
    2. PyTorch_Thermal_Anomaly_Confidence from LSTM/PyTorch telemetry
    3. temperature/delta_T fallback rules if probabilities are unavailable
    """

    if t.thermal_anomaly_confidence is not None:
        return clamp_probability(t.thermal_anomaly_confidence)

    if t.pytorch_thermal_anomaly_confidence is not None:
        return clamp_probability(t.pytorch_thermal_anomaly_confidence)

    # Fallback if only raw temperature is available.
    if t.temperature_c is not None:
        if t.temperature_c >= 95:
            return 1.0
        if t.temperature_c >= 85:
            return 0.75
        if t.temperature_c >= 75:
            return 0.45
        return 0.10

    # Fallback if only Delta_T is available. Adjust these after seeing your dataset range.
    if t.delta_t is not None:
        if t.delta_t >= 25:
            return 0.90
        if t.delta_t >= 15:
            return 0.65
        if t.delta_t >= 8:
            return 0.35
        return 0.10

    return None


def assess_risk_from_telemetry(
    t: AgentTelemetry,
    vibration_weight: float = 0.70,
    thermal_weight: float = 0.30,
) -> tuple[str, float, Optional[float], Optional[float], list[str]]:
    """
    Combine vibration and thermal model telemetry into a final risk level.

    The weights reflect that bearing failure is usually more directly visible in
    vibration, while temperature is an important supporting signal.
    """

    vib_risk = extract_best_vibration_risk(t)
    temp_risk = extract_best_thermal_risk(t)
    evidence: list[str] = []

    # If only one risk stream exists, use it directly.
    if vib_risk is not None and temp_risk is not None:
        combined = (vibration_weight * vib_risk) + (thermal_weight * temp_risk)
        evidence.append(f"Vibration anomaly confidence = {vib_risk:.4f}.")
        evidence.append(f"Thermal anomaly confidence = {temp_risk:.4f}.")
        evidence.append(
            f"Combined score used {vibration_weight:.0%} vibration weight and {thermal_weight:.0%} thermal weight."
        )
    elif vib_risk is not None:
        combined = vib_risk
        evidence.append(f"Vibration anomaly confidence = {vib_risk:.4f}.")
        evidence.append("Thermal confidence was unavailable, so vibration risk was used directly.")
    elif temp_risk is not None:
        combined = temp_risk
        evidence.append(f"Thermal anomaly confidence = {temp_risk:.4f}.")
        evidence.append("Vibration confidence was unavailable, so thermal risk was used directly.")
    else:
        combined = 0.0
        evidence.append("No anomaly confidence values were provided; defaulted to low risk.")

    # Extra context from raw features if available.
    if t.rms is not None:
        evidence.append(f"RMS context = {t.rms:.4f}.")
    if t.kurtosis is not None:
        evidence.append(f"Kurtosis context = {t.kurtosis:.4f}.")
    if t.delta_t is not None:
        evidence.append(f"Delta_T context = {t.delta_t:.4f}.")
    if t.temperature_c is not None:
        evidence.append(f"Temperature context = {t.temperature_c:.2f} °C.")
    if t.lstm_reconstruction_error is not None and t.lstm_threshold is not None:
        evidence.append(
            f"LSTM reconstruction error = {t.lstm_reconstruction_error:.4f}; threshold = {t.lstm_threshold:.4f}."
        )

    if combined >= 0.85:
        risk = "HIGH"
    elif combined >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return risk, round(combined, 4), vib_risk, temp_risk, evidence


def local_report(
    t: AgentTelemetry,
    risk: str,
    combined_score: float,
    vib_risk: Optional[float],
    temp_risk: Optional[float],
    evidence: list[str],
) -> AgentDecision:
    """Generate a technician report locally without Gemini."""

    trigger_alert = risk in {"MEDIUM", "HIGH"}

    if risk == "HIGH":
        explanation = "The telemetry indicates a high likelihood of abnormal machine behavior."
        likely_cause = "Possible bearing degradation, lubrication issue, overheating, or mechanical imbalance."
        recommended_action = (
            "Trigger an alert, reduce load if safe, inspect the bearing, check lubrication, "
            "and schedule urgent maintenance."
        )
        urgency = "Immediate inspection or within 24 hours."
    elif risk == "MEDIUM":
        explanation = "The telemetry shows early warning signs that may indicate developing degradation."
        likely_cause = "Possible early-stage bearing wear, mild overheating, or unstable vibration pattern."
        recommended_action = (
            "Continue close monitoring, run an additional diagnostic check, and inspect during the next maintenance window."
        )
        urgency = "Monitor closely and inspect soon."
    else:
        explanation = "The telemetry does not show strong evidence of a current anomaly."
        likely_cause = "No likely failure cause detected from current model outputs."
        recommended_action = "No immediate action required. Continue normal monitoring."
        urgency = "Routine monitoring."

    edge_note = (
        f"Edge note: the device only needs to pass compressed model telemetry to the agent, "
        f"not raw vibration windows. This is more realistic for deployment on {t.edge_device}."
    )

    timestamp = t.timestamp or datetime.now().isoformat(timespec="seconds")

    return AgentDecision(
        timestamp=timestamp,
        risk_level=risk,
        combined_risk_score=combined_score,
        vibration_risk=None if vib_risk is None else round(vib_risk, 4),
        thermal_risk=None if temp_risk is None else round(temp_risk, 4),
        trigger_alert=trigger_alert,
        explanation=explanation,
        likely_cause=likely_cause,
        recommended_action=recommended_action,
        urgency=urgency,
        edge_note=edge_note,
        evidence=" ".join(evidence),
        llm_used=False,
    )


def gemini_report(
    t: AgentTelemetry,
    risk: str,
    combined_score: float,
    vib_risk: Optional[float],
    temp_risk: Optional[float],
    evidence: list[str],
    model_name: str = "gemini-1.5-flash",
) -> Optional[AgentDecision]:
    """Use Gemini as an optional explanation layer."""

    api_key = os.getenv("GEMINI_API_KEY")
    if genai is None or not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        prompt = f"""
You are an industrial maintenance AI agent for an edge-based anomaly detection system.

The ML models already generated anomaly telemetry. Do not redo the ML prediction.
Explain the result clearly for a technician.

Telemetry:
{json.dumps(asdict(t), indent=2)}

Agent risk level: {risk}
Combined risk score: {combined_score}
Vibration risk: {vib_risk}
Thermal risk: {temp_risk}
Evidence:
{json.dumps(evidence, indent=2)}

Return only valid JSON with these exact keys:
- explanation
- likely_cause
- recommended_action
- urgency
- trigger_alert

Keep it concise and practical.
"""

        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        timestamp = t.timestamp or datetime.now().isoformat(timespec="seconds")

        return AgentDecision(
            timestamp=timestamp,
            risk_level=risk,
            combined_risk_score=combined_score,
            vibration_risk=None if vib_risk is None else round(vib_risk, 4),
            thermal_risk=None if temp_risk is None else round(temp_risk, 4),
            trigger_alert=bool(parsed.get("trigger_alert", risk in {"MEDIUM", "HIGH"})),
            explanation=str(parsed.get("explanation", "No explanation returned.")),
            likely_cause=str(parsed.get("likely_cause", "No likely cause returned.")),
            recommended_action=str(parsed.get("recommended_action", "No recommendation returned.")),
            urgency=str(parsed.get("urgency", "No urgency returned.")),
            edge_note=(
                f"Edge note: ML inference runs locally on {t.edge_device}; Gemini is only used after "
                "the edge device has created compressed telemetry."
            ),
            evidence=" ".join(evidence),
            llm_used=True,
        )

    except Exception as exc:
        print(f"Gemini failed, using local report. Error: {exc}")
        return None


def run_agent(t: AgentTelemetry, use_gemini: bool = False) -> AgentDecision:
    risk, combined, vib_risk, temp_risk, evidence = assess_risk_from_telemetry(t)

    if use_gemini:
        llm_decision = gemini_report(t, risk, combined, vib_risk, temp_risk, evidence)
        if llm_decision is not None:
            return llm_decision

    return local_report(t, risk, combined, vib_risk, temp_risk, evidence)


def run_ai_agent(data: dict[str, Any] | AgentTelemetry, use_gemini: bool = False) -> dict[str, Any]:
    """
    Notebook-friendly wrapper.

    Example using your updated notebook output:

        agent_data = get_agent_telemetry(sample_vib, sample_temp)
        report = run_ai_agent(agent_data)
        print(report)

    Example using PyTorch telemetry:

        agent_data = get_pytorch_agent_telemetry(sample_vib_seq, sample_temp_seq)
        report = run_ai_agent(agent_data)
    """

    if isinstance(data, AgentTelemetry):
        telemetry = data
    else:
        telemetry = telemetry_from_dict(data)

    return asdict(run_agent(telemetry, use_gemini=use_gemini))


# -----------------------------------------------------------------------------
# Conversion from dicts/CSV rows
# -----------------------------------------------------------------------------

def telemetry_from_dict(data: dict[str, Any]) -> AgentTelemetry:
    return AgentTelemetry(
        timestamp=str(data.get("timestamp", data.get("time", datetime.now().isoformat(timespec="seconds")))),
        vibration_anomaly_confidence=safe_float(
            data.get("Vibration_Anomaly_Confidence", data.get("vibration_anomaly_confidence")), None
        ),
        thermal_anomaly_confidence=safe_float(
            data.get("Thermal_Anomaly_Confidence", data.get("thermal_anomaly_confidence")), None
        ),
        pytorch_vibration_anomaly_confidence=safe_float(
            data.get("PyTorch_Vibration_Anomaly_Confidence", data.get("pytorch_vibration_anomaly_confidence")), None
        ),
        pytorch_thermal_anomaly_confidence=safe_float(
            data.get("PyTorch_Thermal_Anomaly_Confidence", data.get("pytorch_thermal_anomaly_confidence")), None
        ),
        logistic_prediction=safe_int(data.get("logistic_prediction", data.get("log_reg_pred")), None),
        mlp_prediction=safe_int(data.get("mlp_prediction", data.get("mlp_pred")), None),
        lstm_prediction=safe_int(data.get("lstm_prediction", data.get("lstm_pred")), None),
        lstm_reconstruction_error=safe_float(
            data.get("lstm_reconstruction_error", data.get("reconstruction_error")), None
        ),
        lstm_threshold=safe_float(data.get("lstm_threshold", data.get("threshold")), None),
        rms=safe_float(data.get("rms", data.get("RMS")), None),
        kurtosis=safe_float(data.get("kurtosis", data.get("Vib_Kurtosis")), None),
        delta_t=safe_float(data.get("Delta_T", data.get("delta_t")), None),
        temperature_c=safe_float(data.get("temperature_c", data.get("temperature")), None),
        model_family=str(data.get("model_family", "multi-model telemetry")),
        edge_device=str(data.get("edge_device", "Raspberry Pi")),
    )


def telemetry_from_row(row: pd.Series, edge_device: str = "Raspberry Pi") -> AgentTelemetry:
    data = {col: row[col] for col in row.index}
    data["edge_device"] = edge_device
    return telemetry_from_dict(data)


def process_csv(input_csv: str, output_csv: str, use_gemini: bool = False, edge_device: str = "Raspberry Pi") -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    decisions: list[Dict[str, Any]] = []

    for _, row in df.iterrows():
        telemetry = telemetry_from_row(row, edge_device=edge_device)
        decisions.append(asdict(run_agent(telemetry, use_gemini=use_gemini)))

    result_df = pd.DataFrame(decisions)
    result_df.to_csv(output_csv, index=False)
    return result_df



# Testing/demo utilities


def build_sample_cases() -> list[AgentTelemetry]:
    return [
        AgentTelemetry(
            timestamp="demo-healthy",
            vibration_anomaly_confidence=0.08,
            thermal_anomaly_confidence=0.04,
            rms=0.20,
            kurtosis=2.6,
            delta_t=2.0,
        ),
        AgentTelemetry(
            timestamp="demo-vibration-warning",
            vibration_anomaly_confidence=0.72,
            thermal_anomaly_confidence=0.22,
            rms=0.58,
            kurtosis=5.1,
            delta_t=7.5,
        ),
        AgentTelemetry(
            timestamp="demo-high-failure",
            vibration_anomaly_confidence=0.96,
            thermal_anomaly_confidence=0.74,
            rms=0.90,
            kurtosis=8.2,
            delta_t=18.0,
        ),
        AgentTelemetry(
            timestamp="demo-pytorch-telemetry",
            pytorch_vibration_anomaly_confidence=0.91,
            pytorch_thermal_anomaly_confidence=0.61,
        ),
    ]


def make_sample_csv(path: str) -> None:
    rows = [asdict(case) for case in build_sample_cases()]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Sample telemetry CSV created: {path}")
    print(f"Next run: python ai_agent_telemetry_updated.py --input {path} --output agent_report.csv")


def print_decision_card(decision: AgentDecision) -> None:
    print("\n" + "=" * 80)
    print(f"Timestamp:       {decision.timestamp}")
    print(f"Risk Level:      {decision.risk_level}")
    print(f"Combined Score:  {decision.combined_risk_score}")
    print(f"Vibration Risk:  {decision.vibration_risk}")
    print(f"Thermal Risk:    {decision.thermal_risk}")
    print(f"Trigger Alert:   {decision.trigger_alert}")
    print(f"LLM Used:        {decision.llm_used}")
    print("-" * 80)
    print(f"Explanation:     {decision.explanation}")
    print(f"Likely Cause:    {decision.likely_cause}")
    print(f"Action:          {decision.recommended_action}")
    print(f"Urgency:         {decision.urgency}")
    print(f"Evidence:        {decision.evidence}")
    print(f"Edge Note:       {decision.edge_note}")


def demo(use_gemini: bool = False, output_csv: Optional[str] = None) -> pd.DataFrame:
    decisions = []

    for case in build_sample_cases():
        decision = run_agent(case, use_gemini=use_gemini)
        decisions.append(asdict(decision))
        print_decision_card(decision)

    result_df = pd.DataFrame(decisions)

    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\nDemo report saved to: {output_csv}")

    print("\nExpected behavior:")
    print("  demo-healthy -> LOW risk")
    print("  demo-vibration-warning -> MEDIUM risk")
    print("  demo-high-failure -> HIGH risk")
    print("  demo-pytorch-telemetry -> usually HIGH or MEDIUM depending confidence")

    return result_df

# Command line interface


def main() -> None:
    parser = argparse.ArgumentParser(description="Telemetry-based AI Agent for EE 460 anomaly detection")
    parser.add_argument("--input", type=str, help="Input CSV with telemetry confidence values")
    parser.add_argument("--output", type=str, default="agent_report.csv", help="Output CSV for agent reports")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo cases")
    parser.add_argument("--make-sample-csv", type=str, help="Create a fake telemetry CSV for testing")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini for natural-language report generation")
    parser.add_argument("--edge-device", type=str, default="Raspberry Pi", help="Target edge device name")

    args = parser.parse_args()

    if args.make_sample_csv:
        make_sample_csv(args.make_sample_csv)
        return

    if args.demo:
        demo(use_gemini=args.use_gemini, output_csv=args.output)
        return

    if args.input:
        result = process_csv(
            input_csv=args.input,
            output_csv=args.output,
            use_gemini=args.use_gemini,
            edge_device=args.edge_device,
        )
        print(f"Agent report saved to: {args.output}")
        print(result.head())
        return

    print("No input provided, running demo. Use --input your_file.csv for real telemetry.")
    demo(use_gemini=args.use_gemini, output_csv=args.output)


if __name__ == "__main__":
    main()
