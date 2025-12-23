import schedule
import time
import subprocess
import threading
from datetime import datetime

# =========================
# 1. Pipeline chính: tiền xử lý + dự báo
# =========================
def run_full_pipeline():
    print(f"\n🔔 [{datetime.now().strftime('%H:%M:%S')}] Bắt đầu chu trình tự động pipeline...")

    try:
        # 1. Tiền xử lý dữ liệu
        print("  - Đang tiền xử lý dữ liệu...")
        subprocess.run(["python", "data_pipeline/preprocess_data.py"], check=True)

        # 2. Dự báo & gửi thông báo
        print("  - Đang thực hiện dự báo & gửi thông báo...")
        subprocess.run(["python", "models/inference.py"], check=True)

        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Chu trình pipeline hoàn tất.")
    except Exception as e:
        print(f"❌ Lỗi khi chạy pipeline: {e}")

# =========================
# 2. Training model 2 ngày 1 lần (chạy background)
# =========================
def run_training():
    def train_background():
        print(f"\n🧠 [{datetime.now().strftime('%H:%M:%S')}] Bắt đầu training model (2 ngày)...")
        subprocess.run(["python", "models/mlflow_experiment.py"])
        print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Training model hoàn tất.")

    threading.Thread(target=train_background).start()

# =========================
# 3. Lập lịch
# =========================
# 5 phút chạy pipeline
schedule.every(60).minutes.do(run_full_pipeline)

# 2 ngày chạy training
schedule.every(1).day.do(run_training)

print("🚀 Hệ thống tự động Training & Dự báo đã khởi động!")
print("🕐 Pipeline sẽ chạy mỗi 5 phút. Training model sẽ chạy mỗi 2 ngày.\nVui lòng không tắt cửa sổ này.")

# =========================
# 4. Vòng lặp chạy schedule
# =========================
while True:
    schedule.run_pending()
    time.sleep(1)
