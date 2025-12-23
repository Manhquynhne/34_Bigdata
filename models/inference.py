import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
from datetime import datetime, timedelta
import joblib
import numpy as np

# =========================
# 1. Telegram config
# =========================
TELEGRAM_TOKEN = "8313428656:AAHbp8fcQmtd0Oi5nTfEEdCj4zxxEIXqnUQ"
TELEGRAM_CHAT_ID = "7209657864"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"❌ Lỗi gửi Telegram: {e}")

# =========================
# 2. Load best model
# =========================
def load_best_model():
    experiment = mlflow.get_experiment_by_name("IoT Predictive Maintenance")
    if experiment is None:
        raise ValueError("❌ Experiment 'IoT Predictive Maintenance' chưa tồn tại. Chạy mlflow_experiment.py trước.")
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"❌ Chưa có run nào trong experiment {experiment.name}.")
    
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Loading model from: {model_uri}")
    
    return mlflow.pyfunc.load_model(model_uri), best_run.info.run_id

# =========================
# 3. Predict & alert
# =========================
FEATURES = ["temperature", "vibration", "pressure", "rpm", "current", "load"]

def predict_and_alert():
    try:
        model, run_id = load_best_model()
    except Exception as e:
        print(e)
        return

    # -------------------------
    # Load dữ liệu
    # -------------------------
    # Dữ liệu đã scale cho model
    data_scaled = pd.read_csv("data/processed/processed_data.csv", parse_dates=["timestamp"])
    # Dữ liệu thực (raw) để gửi Telegram
    data_real = pd.read_csv("data/processed/raw_window_input.csv", parse_dates=["timestamp"])


    if data_scaled.empty or data_real.empty:
        print("⚠ Không có dữ liệu để dự đoán")
        return

    # -------------------------
    # Lọc dữ liệu 24h qua
    # -------------------------
    cutoff_24h = datetime.now() - timedelta(hours=24)
    data_scaled_24h = data_scaled[data_scaled["timestamp"] >= cutoff_24h]
    data_real_24h = data_real[data_real["timestamp"] >= cutoff_24h]

    # -------------------------
    # Load scaler
    # -------------------------
    scaler = joblib.load("data/processed/scaler.pkl")

    # ===== 1. Thông số hiện tại =====
    latest_scaled = data_scaled[FEATURES].iloc[-1:]
    X_current = scaler.transform(latest_scaled)
    preds_current = model.predict(X_current)
    current_failure_rate = preds_current.mean()

    latest_real = data_real[FEATURES].iloc[-1:]

    # ===== 2. Trung bình 24h =====
    mean_scaled_24h = data_scaled_24h[FEATURES].mean()
    X_mean_24h = scaler.transform(mean_scaled_24h.to_frame().T)
    preds_mean_24h = model.predict(X_mean_24h)
    mean_24h_failure_rate = preds_mean_24h.mean()

    mean_real_24h = data_real_24h[FEATURES].mean().to_frame().T

    # ===== 3. Dự đoán 1h tiếp theo =====
    future_real = {}
    for col in FEATURES:
        recent_values = data_real[col].tail(12).values
        if len(recent_values) < 2:
            future_real[col] = recent_values[-1]
        else:
            slope = (recent_values[-1] - recent_values[0]) / (len(recent_values)-1)
            future_real[col] = recent_values[-1] + slope

    # Chuyển sang DataFrame để model dự đoán
    X_future = scaler.transform(pd.DataFrame([future_real], columns=FEATURES))
    future_failure_rate = model.predict(X_future).mean()

    # ===== 4. Xác định trạng thái =====
    if future_failure_rate > 0.5:
        status = "🔴 <b>CẢNH BÁO NGUY HIỂM</b>"
    elif future_failure_rate > 0.2:
        status = "🟠 <b>CẢNH BÁO SỚM</b>"
    else:
        status = "🟢 <b>HỆ THỐNG ỔN ĐỊNH</b>"

    # ===== 5. Chuẩn bị message Telegram =====
    message = f"{status}\n🤖 Model Run ID: <code>{run_id}</code>\n\n"

    message += "<b>🔥 Thông số hiện tại (giá trị thực):</b>\n"
    for col in FEATURES:
        message += f"{col}: {latest_real[col].values[0]:.2f}\n"
    message += f"Failure rate hiện tại: {current_failure_rate:.2%}\n\n"

    message += "<b>📊 Trung bình 24h qua (giá trị thực):</b>\n"
    for col in FEATURES:
        message += f"{col}: {mean_real_24h[col].values[0]:.2f}\n"
    message += f"Failure rate trung bình 24h: {mean_24h_failure_rate:.2%}\n\n"

    message += "<b>⏭ Dự đoán 1h tiếp theo (giá trị thực):</b>\n"
    for col in FEATURES:
        message += f"{col}: {future_real[col]:.2f}\n"
    message += f"Failure rate 1h tới: {future_failure_rate:.2%}"

    print(message.replace("<b>", "").replace("</b>", ""))
    send_telegram_alert(message)

# =========================
# 6. Main
# =========================
if __name__ == "__main__":
    predict_and_alert()
