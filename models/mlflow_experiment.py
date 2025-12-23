import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 1. Load config
# =========================
with open('configs/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

EXPERIMENT_NAME = "IoT Predictive Maintenance"

# =========================
# 2. Load processed data
# =========================
data = pd.read_csv("data/processed/processed_data.csv")

if len(data) < 10:
    raise ValueError("❌ Chưa đủ dữ liệu để train")

# =========================
# 3. Rule-based labeling
# =========================
data["failure"] = (
    (data["temperature"] > 1.5) |
    (data["vibration"] > 1.2) |
    (data["current"] > 1.3) |
    (data["load"] > 1.2)
).astype(int)

# =========================
# 4. Feature selection
# =========================
FEATURES = ["temperature", "vibration", "pressure", "rpm", "current", "load"]

X = data[FEATURES]
y = data["failure"]

# =========================
# 5. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. Train model
# =========================
model = RandomForestClassifier(
    **model_config["model"]["parameters"]
)
model.fit(X_train, y_train)

# =========================
# 7. Evaluate
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# 8. MLflow logging
# =========================
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_params(model_config["model"]["parameters"])
    mlflow.log_param("features_used", FEATURES)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ MLflow experiment logged successfully")
    print(f"📊 Accuracy: {accuracy:.4f}")
