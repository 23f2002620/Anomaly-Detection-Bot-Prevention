import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

MODEL_PATH = "isolation_forest_model.pkl"
INPUT_JSON_PATH = "input_events.json"
OUTPUT_JSON_PATH = "output_results.json"

FEATURE_COLUMNS = [
    "login_hour",
    "logins_from_ip_24h",
    "accounts_created_from_ip_24h",
    "messages_sent_last_hour",
    "reports_last_24h",
    "is_duplicate_device"
]


def load_input_events(path: str):
    """
    Load JSON events.
    Supports:
    - A JSON array: [ {...}, {...}, ... ]
    - Or newline-delimited JSON: {...}\n{...}\n
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()

    if not data:
        return []

    try:
        # Try as a normal JSON array
        events = json.loads(data)
        if isinstance(events, dict):
            # Single object → wrap into list
            events = [events]
    except json.JSONDecodeError:
        # Fallback: newline-delimited JSON
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))

    return events


def compute_rule_flags(event):
    """Return list of rule-based flags for the given event dict."""
    flags = []
    if event.get("accounts_created_from_ip_24h", 0) > 5:
        flags.append("rapid_account_creation")
    if event.get("messages_sent_last_hour", 0) > 50:
        flags.append("mass_messaging")
    if event.get("reports_last_24h", 0) > 3:
        flags.append("multiple_reports")
    if event.get("is_duplicate_device", 0) == 1:
        flags.append("duplicate_device")
    if event.get("login_hour") in [1, 2, 3, 4]:
        flags.append("odd_login_time")
    return flags


def compute_severity_and_action(score, flags):
    """
    Map anomaly score + rule flags to (severity, recommended_action).
    Lower score = more anomalous.
    """
    if score < -0.25 or len(flags) >= 3:
        severity = "critical"
        action = "permanent_ban_recommended"
    elif score < -0.15 or len(flags) == 2:
        severity = "high"
        action = "temporary_ban_24h_recommended"
    elif score < -0.05 or len(flags) == 1:
        severity = "medium"
        action = "captcha_and_rate_limit"
    else:
        severity = "low"
        action = "monitor_only"
    return severity, action


def detect_anomaly_for_event(event, model):
    """
    Takes one input event (dict) and returns one output (dict) with only
    native Python types (no numpy.bool_, numpy.float64, etc.).
    """

    # Extract features
    X = np.array([[event[col] for col in FEATURE_COLUMNS]])

    # Model prediction
    score = model.decision_function(X)[0]
    pred = model.predict(X)[0]

    # Convert Numpy values → Python values
    score = float(score)
    anomaly_detected = bool(pred == -1)

    # Rule-based flags
    flags = compute_rule_flags(event)

    # Severity & action
    severity, action = compute_severity_and_action(score, flags)

    return {
        "user_id": str(event.get("user_id")),
        "anomaly_detected": anomaly_detected,
        "anomaly_score": score,
        "severity": str(severity),
        "recommended_action": str(action),
        "rule_flags": list(flags),
        "timestamp_utc": datetime.utcnow().isoformat()
    }

def main():
    # Check files
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            f"Train and save IsolationForest before running this script."
        )
    if not Path(INPUT_JSON_PATH).exists():
        raise FileNotFoundError(
            f"Input JSON file not found: {INPUT_JSON_PATH}"
        )

    # Load model
    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Load events
    print(f"[INFO] Loading input events from {INPUT_JSON_PATH}")
    events = load_input_events(INPUT_JSON_PATH)
    print(f"[INFO] Loaded {len(events)} events")

    results = []
    for i, event in enumerate(events, start=1):
        try:
            result = detect_anomaly_for_event(event, model)
            results.append(result)
        except KeyError as e:
            print(f"[WARN] Event {i} missing field {e}; skipping this event.")
        except Exception as e:
            print(f"[WARN] Error processing event {i}: {e}")

    # Save to output JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Wrote {len(results)} results to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
