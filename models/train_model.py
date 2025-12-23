# train model.py
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Config
# =========================
PROCESSED_FILE = "data/processed/processed_data.csv"
MODEL_DIR = "models/saved"
EXPERIMENT_NAME = "Streaming Predictive Maintenance"

os.makedirs(MODEL_DIR, exist_ok=True)

# 🔥 DÙNG NHIỀU FEATURE HƠN
FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "current",
    "load"
]

TARGET = "label"

# =========================
# Main
# =========================
if __name__ == "__main__":

    if not os.path.exists(PROCESSED_FILE):
        print("❌ processed_data.csv chưa tồn tại")
        exit()

    # Load data đã preprocess
    data = pd.read_csv(PROCESSED_FILE)

    if len(data) < 100:
        print("⚠ Chưa đủ dữ liệu để train")
        exit()

    # =========================
    # Split data
    # =========================
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # Model (GIỮ NGUYÊN BẢN CHẤT)
    # =========================
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    # =========================
    # MLflow
    # =========================
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features_used", FEATURES)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Save model
        model_path = os.path.join(
            MODEL_DIR,
            f"pm_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "model")

        print("✅ Training completed")
        print(f"📊 Accuracy: {acc:.4f}")
        print(f"💾 Model saved at: {model_path}")
