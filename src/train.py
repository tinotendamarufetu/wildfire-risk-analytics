import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.sklearn


class WildfireModelTrainer:
    """Trains, evaluates, and tracks ML models for wildfire risk prediction using MLflow."""

    def __init__(
        self, 
        data_path: str = "data/processed/engineered_wildfire_features.parquet",
        models_dir: str = "models",
        experiment_name: str = "Wildfire_Risk_Underwriting"
    ):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize MLflow experiment
        mlflow.set_experiment(experiment_name)

    def load_and_prepare_data(self):
        """Loads dataset and splits into feature matrix X and target y."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")
            
        df = pd.read_parquet(self.data_path)
        
        feature_cols = [
            "elevation_m", "slope_pct", "dist_nearest_fire_m", 
            "fires_within_5km", "fires_within_10km", "years_since_last_fire",
            "max_vpd", "max_wind_speed", "total_precip_annual"
        ]
        
        X = df[feature_cols]
        y = df["is_high_risk"]

        class_counts = y.value_counts()
        singleton_mask = y.map(class_counts).eq(1)
        if singleton_mask.any():
            singleton_labels = class_counts[class_counts == 1].index.tolist()
            singleton_positions = np.flatnonzero(singleton_mask.to_numpy())
            remaining_positions = np.flatnonzero(~singleton_mask.to_numpy())
            if len(remaining_positions) < 2:
                raise ValueError(
                    "Not enough samples remain to create a train/test split after "
                    "keeping single-sample classes in the training set."
                )

            remaining_y = y.iloc[remaining_positions]
            remaining_counts = remaining_y.value_counts()
            test_count = int(np.ceil(len(remaining_positions) * 0.2))
            stratify = (
                remaining_y
                if remaining_counts.min() >= 2
                and test_count >= len(remaining_counts)
                and len(remaining_positions) - test_count >= len(remaining_counts)
                else None
            )
            remaining_train, test_positions = train_test_split(
                remaining_positions,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
            train_positions = np.concatenate([remaining_train, singleton_positions])

            warnings.warn(
                f"Class(es) {singleton_labels} have only one sample; keeping them "
                "in the training set. Their performance cannot be evaluated on "
                "the held-out test set.",
                UserWarning,
                stacklevel=2,
            )
            return (
                X.iloc[train_positions],
                X.iloc[test_positions],
                y.iloc[train_positions],
                y.iloc[test_positions],
            )

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    def _plot_and_save_confusion_matrix(self, y_true, y_pred, model_name: str) -> str:
        """Generates and saves confusion matrix heatmap artifact."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Low Risk", "High Risk"],
                    yticklabels=["Low Risk", "High Risk"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        
        artifact_path = os.path.join(self.models_dir, f"cm_{model_name.lower()}.png")
        plt.savefig(artifact_path, bbox_inches="tight")
        plt.close()
        return artifact_path

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Trains an XGBoost classifier and logs metrics & artifacts to MLflow."""
        print("\n[+] Training XGBoost Classifier...")
        
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

        with mlflow.start_run(run_name="XGBoost_Classifier"):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions & Probabilities
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else y_pred
            
            # Evaluation Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            # Log Parameters & Metrics to MLflow
            mlflow.log_params(params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
                
            # Log Confusion Matrix Plot Artifact
            cm_path = self._plot_and_save_confusion_matrix(y_test, y_pred, "XGBoost")
            mlflow.log_artifact(cm_path)
            
            # Log Model Binary
            mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
            
            # Save Local Model Binary
            local_model_path = os.path.join(self.models_dir, "xgboost_wildfire_model.joblib")
            joblib.dump(model, local_model_path)
            print(f"[OK] XGBoost Model logged to MLflow & saved locally: {local_model_path}")
            
            return model, metrics

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Trains a Random Forest classifier as a baseline model."""
        print("\n[+] Training Random Forest Classifier...")
        
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }

        with mlflow.start_run(run_name="Random_Forest_Classifier"):
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else y_pred
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            mlflow.log_params(params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
                
            cm_path = self._plot_and_save_confusion_matrix(y_test, y_pred, "RandomForest")
            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(
                model,
                artifact_path="rf_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            
            print("[OK] Random Forest Model logged to MLflow.")
            return model, metrics


if __name__ == "__main__":
    trainer = WildfireModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data()
    
    xgb_model, xgb_metrics = trainer.train_xgboost(X_train, X_test, y_train, y_test)
    rf_model, rf_metrics = trainer.train_random_forest(X_train, X_test, y_train, y_test)
    
    print("\n--- Model Training Summary ---")
    print("XGBoost Metrics:      ", xgb_metrics)
    print("Random Forest Metrics:", rf_metrics)