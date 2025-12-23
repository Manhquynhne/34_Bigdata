# data_pipeline/kafka_consumer.py
from kafka import KafkaConsumer
import json
import csv
import os
from datetime import datetime

# =========================
# Kafka consumer
# =========================
consumer = KafkaConsumer(
    'sensor_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True
)

# =========================
# File lưu dữ liệu
# =========================
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "raw_data.csv")

# Tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)

# Kiểm tra file đã tồn tại chưa
file_exists = os.path.isfile(RAW_FILE)

# =========================
# Ghi dữ liệu
# =========================
with open(RAW_FILE, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Nếu file chưa tồn tại → ghi header
    if not file_exists:
        writer.writerow([
            "timestamp",
            "temperature",
            "vibration",
            "pressure",
            "rpm",
            "current",
            "load",
            "label",
            "failure_type"
        ])

    print("✅ Kafka Consumer started. Saving data to raw_data.csv...")

    for message in consumer:
        data = message.value

        row = [
            datetime.fromtimestamp(data["timestamp"]),
            data["temperature"],
            data["vibration"],
            data["pressure"],
            data["rpm"],
            data["current"],
            data["load"],
            data["label"],
            data["failure_type"]
        ]

        writer.writerow(row)
        f.flush()  # ghi ngay xuống disk

        print(f"Saved data: {row}")
