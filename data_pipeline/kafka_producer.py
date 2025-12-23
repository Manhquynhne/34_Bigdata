# kafka_producer.py
import json
import yaml
import time
import random
from kafka import KafkaProducer

# =========================
# Load Kafka configuration
# =========================
with open('configs/kafka_config.yaml', 'r') as f:
    kafka_config = yaml.safe_load(f)['kafka']

producer = KafkaProducer(
    bootstrap_servers=kafka_config['bootstrap_servers'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# =========================
# Label & failure definition
# =========================
"""
label:
0 = Normal
1 = Warning
2 = Failure

failure_type:
0 = None
1 = Bearing Wear
2 = Overheating
3 = Imbalance
"""

def generate_label_and_failure(temperature, vibration, pressure):
    # Failure
    if temperature > 80 or vibration > 3.0 or pressure < 1.3:
        if temperature > 90:
            return 2, 2  # Overheating
        elif vibration > 3.5:
            return 2, 3  # Imbalance
        else:
            return 2, 1  # Bearing wear

    # Warning
    if temperature >= 60 or vibration >= 1.5:
        return 1, 0

    # Normal
    return 0, 0


# =========================
# Generate & send data
# =========================
def generate_and_send_data():
    while True:
        temperature = round(random.uniform(40, 95), 2)      # °C
        vibration = round(random.uniform(0.2, 4.5), 2)     # mm/s
        pressure = round(random.uniform(1.2, 2.0), 2)      # bar
        rpm = random.randint(1400, 1500)                    # vòng/phút
        current = round(random.uniform(10, 18), 2)         # A
        load = random.randint(50, 100)                      # %

        label, failure_type = generate_label_and_failure(
            temperature, vibration, pressure
        )

        data = {
            "timestamp": int(time.time()),
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "current": current,
            "load": load,
            "label": label,
            "failure_type": failure_type
        }

        producer.send(kafka_config['topic'], data)
        print(f"Sent data: {data}")

        time.sleep(5)


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Kafka Producer started...")
    generate_and_send_data()
