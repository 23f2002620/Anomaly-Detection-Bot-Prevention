"""
python anomaly_detection.py --mode generate
python anomaly_detection.py --mode train
python anomaly_detection.py --mode evaluate
python anomaly_detection.py --mode monitor --delay 0.5
"""

import argparse
import random
import string
import time
from datetime import datetime
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "isolation_forest_model.pkl"
DATA_PATH = "activity_logs.csv"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -----------------------------
# MOCK DATA GENERATION
# -----------------------------

def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


def random_device_fingerprint():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


def generate_mock_activity_logs(
    n_normal: int = 80,
    n_suspicious: int = 20
) -> pd.DataFrame:
    """
    Generate mock activity logs with labels.
    Normal and suspicious patterns are encoded according to your rules.
    """
    logs: List[Dict] = []

    # --- Normal users ---
    for i in range(n_normal):
        user_id = f"user_{i+1}"
        ip = random_ip()
        device = random_device_fingerprint()

        login_hour = np.random.choice(range(7, 23))  # more typical human hours
        accounts_created_from_ip_24h = np.random.randint(0, 4)  # usually low
        logins_from_ip_24h = np.random.randint(1, 10)
        messages_sent_last_hour = np.random.randint(0, 40)
        reports_last_24h = np.random.randint(0, 2)
        is_duplicate_device = 0  # mostly unique devices

        logs.append({
            "user_id": user_id,
            "ip_address": ip,
            "device_fingerprint": device,
            "login_hour": login_hour,
            "logins_from_ip_24h": logins_from_ip_24h,
            "accounts_created_from_ip_24h": accounts_created_from_ip_24h,
            "messages_sent_last_hour": messages_sent_last_hour,
            "reports_last_24h": reports_last_24h,
            "is_duplicate_device": is_duplicate_device,
            "label": 0  # normal
        })

    # --- Suspicious users ---
    for i in range(n_suspicious):
        user_id = f"suspicious_{i+1}"
        ip = random_ip()
        device = random_device_fingerprint()

        # Make them more extreme
        # Randomly pick which suspicious pattern to emphasize
        pattern_type = random.choice(["rapid_accounts", "mass_messages",
                                      "multi_reports", "duplicate_device",
                                      "odd_login_time"])

        login_hour = np.random.choice(range(0, 24))
        accounts_created_from_ip_24h = np.random.randint(0, 4)
        messages_sent_last_hour = np.random.randint(0, 60)
        reports_last_24h = np.random.randint(0, 4)
        is_duplicate_device = 0

        if pattern_type == "rapid_accounts":
            accounts_created_from_ip_24h = random.randint(6, 15)
        elif pattern_type == "mass_messages":
            messages_sent_last_hour = random.randint(51, 200)
        elif pattern_type == "multi_reports":
            reports_last_24h = random.randint(4, 10)
        elif pattern_type == "duplicate_device":
            is_duplicate_device = 1
        elif pattern_type == "odd_login_time":
            login_hour = random.choice([1, 2, 3, 4])  # unusual hours

        logs.append({
            "user_id": user_id,
            "ip_address": ip,
            "device_fingerprint": device,
            "login_hour": login_hour,
            "logins_from_ip_24h": np.random.randint(1, 30),
            "accounts_created_from_ip_24h": accounts_created_from_ip_24h,
            "messages_sent_last_hour": messages_sent_last_hour,
            "reports_last_24h": reports_last_24h,
            "is_duplicate_device": is_duplicate_device,
            "label": 1  # suspicious
        })

    df = pd.DataFrame(logs)
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Generated dataset with shape {df.shape} and saved to {DATA_PATH}")
    print(df["label"].value_counts())
    return df


# -----------------------------
# MODEL TRAINING & SAVING
# -----------------------------

FEATURE_COLUMNS = [
    "login_hour",
    "logins_from_ip_24h",
    "accounts_created_from_ip_24h",
    "messages_sent_last_hour",
    "reports_last_24h",
    "is_duplicate_device",
]


def train_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    """
    Train IsolationForest on NORMAL data only (label == 0).
    """
    normal_df = df[df["label"] == 0]
    X_normal = normal_df[FEATURE_COLUMNS].values

    model = IsolationForest(
        n_estimators=150,
        contamination=0.2,  # expected proportion of anomalies in production
        max_samples="auto",
        random_state=RANDOM_SEED
    )

    model.fit(X_normal)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Isolation Forest model trained and saved to {MODEL_PATH}")
    return model


# -----------------------------
# EVALUATION METRICS
# -----------------------------

def evaluate_model(model: IsolationForest, df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate model on full dataset (normal + suspicious).
    """
    X = df[FEATURE_COLUMNS].values
    y_true = df["label"].values  # 0 normal, 1 suspicious

    # IsolationForest: 1 = inlier, -1 = outlier
    y_pred_if = model.predict(X)
    y_pred = np.where(y_pred_if == -1, 1, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n[METRICS]")
    print(f"Precision:          {precision:.3f}")
    print(f"Recall:             {recall:.3f}")
    print(f"F1-score:           {f1:.3f}")
    print(f"False Positive Rate:{false_positive_rate:.3f}")
    print("\nConfusion Matrix (tn, fp, fn, tp):", (tn, fp, fn, tp))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# -----------------------------
# SEVERITY & BAN RECOMMENDATION
# -----------------------------

def compute_severity(
    anomaly_score: float,
    row: pd.Series
) -> Tuple[str, str]:
    """
    Map anomaly score and rule-based flags to severity & recommended action.
    Lower score = more anomalous in IsolationForest.
    """

    rule_flags = []

    if row["accounts_created_from_ip_24h"] > 5:
        rule_flags.append("rapid_account_creation")
    if row["messages_sent_last_hour"] > 50:
        rule_flags.append("mass_messaging")
    if row["reports_last_24h"] > 3:
        rule_flags.append("multiple_reports")
    if row["is_duplicate_device"] == 1:
        rule_flags.append("duplicate_device")
    if row["login_hour"] in [1, 2, 3, 4]:
        rule_flags.append("odd_login_time")

    # severity logic (simple but clear)
    # NOTE: anomaly_score is typically negative for anomalies; we invert thresholds accordingly.
    if anomaly_score < -0.25 or len(rule_flags) >= 3:
        severity = "critical"
        action = "permanent_ban_recommended"
    elif anomaly_score < -0.15 or len(rule_flags) == 2:
        severity = "high"
        action = "temporary_ban_24h_recommended"
    elif anomaly_score < -0.05 or len(rule_flags) == 1:
        severity = "medium"
        action = "captcha_and_rate_limit"
    else:
        severity = "low"
        action = "monitor_only"

    return severity, action


def admin_alert_template(
    user_id: str,
    severity: str,
    action: str,
    anomaly_score: float,
    row: pd.Series
) -> str:
    """
    Returns a formatted alert message string for admins.
    """
    now_str = datetime.utcnow().isoformat()

    return f"""
[ALERT - {severity.upper()}]
Timestamp (UTC): {now_str}
User ID: {user_id}
IP Address: {row['ip_address']}
Device Fingerprint: {row['device_fingerprint']}

Anomaly Score: {anomaly_score:.4f}
Accounts from IP (24h): {row['accounts_created_from_ip_24h']}
Logins from IP (24h):  {row['logins_from_ip_24h']}
Messages last hour:    {row['messages_sent_last_hour']}
Reports last 24h:      {row['reports_last_24h']}
Duplicate device:      {bool(row['is_duplicate_device'])}
Login hour (0-23):     {row['login_hour']}

Recommended Action: {action}

Notes:
- Severity levels: low, medium, high, critical
- Please review this user in the admin panel and confirm the action.
"""


# -----------------------------
# REAL-TIME MONITORING DASHBOARD (CLI)
# -----------------------------

def realtime_monitor_dashboard(
    df: pd.DataFrame,
    model: IsolationForest,
    delay_seconds: float = 0.5
):
    """
    Simulated real-time monitoring:
    - Streams rows from the dataset one by one.
    - Prints severity, action, and optionally full alert for high/critical.
    """
    X = df[FEATURE_COLUMNS].values
    anomaly_scores = model.decision_function(X)  # higher = more normal
    preds = model.predict(X)  # 1 = normal, -1 = anomaly

    print("\n=== REAL-TIME SAFETY MONITORING (SIMULATED) ===\n")
    print("Press Ctrl+C to stop.\n")

    for idx, row in df.iterrows():
        score = anomaly_scores[idx]
        pred = preds[idx]  # 1 or -1
        is_anomaly = (pred == -1)

        severity, action = compute_severity(score, row)

        indicator = "OK"
        if is_anomaly:
            indicator = "ANOMALY"

        print(
            f"[{indicator}] user={row['user_id']:<15} "
            f"severity={severity:<8} action={action:<30} "
            f"score={score:.4f}"
        )

        # For medium+ severity, show an alert body
        if severity in ["medium", "high", "critical"]:
            print(admin_alert_template(
                user_id=row["user_id"],
                severity=severity,
                action=action,
                anomaly_score=score,
                row=row
            ))

        time.sleep(delay_seconds)


# -----------------------------
# MAIN CLI ENTRYPOINT
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Safety Monitoring System (Anomaly Detection & Bot Prevention)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate", "train", "evaluate", "monitor", "all"],
        help="Operation mode"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (seconds) between events in monitor mode"
    )

    args = parser.parse_args()

    if args.mode == "generate":
        generate_mock_activity_logs()
    else:
        # load or generate data if missing
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"[INFO] Loaded dataset from {DATA_PATH}, shape={df.shape}")
        except FileNotFoundError:
            print("[WARN] Dataset not found. Generating new dataset.")
            df = generate_mock_activity_logs()

    if args.mode in ["train", "all"]:
        train_isolation_forest(df)

    if args.mode in ["evaluate", "all"]:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            print("[WARN] Model not found. Training a new one.")
            model = train_isolation_forest(df)
        evaluate_model(model, df)

    if args.mode in ["monitor", "all"]:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            print("[WARN] Model not found. Training a new one.")
            model = train_isolation_forest(df)
        realtime_monitor_dashboard(df, model, delay_seconds=args.delay)


if __name__ == "__main__":
    main()
