import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import joblib

# =========================
# Config
# =========================
RAW_FILE = "data/raw_data.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_data.csv")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.pkl")
RAW_WINDOW_FILE = os.path.join(OUTPUT_DIR, "raw_window_input.csv")  # 🔥 file lưu dữ liệu window

WINDOW_DAYS = 2   # 🔥 lấy dữ liệu 2 NGÀY để train

# =========================
# Load dữ liệu theo window 2 ngày
# =========================
def load_last_window_days(file_path, days):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    now = datetime.now()
    window_start = now - timedelta(days=days)

    df_window = df[df["timestamp"] >= window_start]

    # 🔥 Lưu một bản copy dữ liệu gốc window
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_window.to_csv(RAW_WINDOW_FILE, index=False)
    print(f"📄 Đã lưu bản sao dữ liệu window {days} ngày tại: {RAW_WINDOW_FILE}")

    return df_window

# =========================
# Preprocess
# =========================
def preprocess(df):
    if df.empty:
        print("⚠ Không có dữ liệu trong window 2 ngày.")
        return df

    # Drop missing values
    df = df.dropna().reset_index(drop=True)

    # Các feature dùng để train
    feature_cols = [
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "current",
        "load"
    ]

    # Scale feature (KHÔNG scale label)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Lưu scaler để dùng cho predict
    joblib.dump(scaler, SCALER_FILE)

    return df

# =========================
# Main
# =========================
if __name__ == "__main__":
    if not os.path.exists(RAW_FILE):
        print("❌ raw_data.csv chưa tồn tại")
        exit()

    print("🔄 Loading raw data...")
    data_window = load_last_window_days(RAW_FILE, WINDOW_DAYS)

    print(f"📊 Records in last {WINDOW_DAYS} days: {len(data_window)}")

    processed_data = preprocess(data_window)

    if processed_data.empty:
        print("⚠ Không có dữ liệu sau preprocess.")
        exit()

    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lưu data đã xử lý
    processed_data.to_csv(OUTPUT_FILE, index=False)

    print("✅ Preprocessing complete for TRAINING")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print(f"📦 Records used for training: {len(processed_data)}")
