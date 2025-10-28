import os
from typing import Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(csv_path)
	# Target column is "disease"; features are all other columns (binary symptom indicators)
	X = df.drop(columns=["disease"])
	y = df["disease"]
	return X, y


def train_and_save_model(csv_path: str, model_path: str) -> None:
	X, y = load_dataset(csv_path)
	n_samples = len(y)
	n_classes = y.nunique()
	# Compute a safe test size: at least n_classes, but not more than 40% of data
	suggested = max(int(round(0.2 * n_samples)), n_classes)
	test_size = min(max(suggested, 1), max(int(0.4 * n_samples), 1))

	# Use stratify only if each class has at least 2 samples and test_size >= n_classes
	class_counts = y.value_counts()
	can_stratify = (class_counts.min() >= 2) and (test_size >= n_classes)
	try:
		X_train, X_test, y_train, y_test = train_test_split(
			X,
			y,
			test_size=test_size,
			random_state=42,
			stratify=y if can_stratify else None,
		)
	except ValueError:
		# Fallback: non-stratified small split
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=max(int(round(0.2 * n_samples)), 1), random_state=42
		)

	model = RandomForestClassifier(
		n_estimators=300,
		max_depth=None,
		random_state=42,
		class_weight="balanced_subsample",
	)
	model.fit(X_train, y_train)

	# Evaluate and print to console for visibility
	pred = model.predict(X_test)
	acc = accuracy_score(y_test, pred)
	print(f"Validation Accuracy: {acc:.3f}")
	print(classification_report(y_test, pred))

	# Ensure directory exists
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	model_bundle = {
		"model": model,
		"feature_columns": list(X.columns),
	}
	joblib.dump(model_bundle, model_path)
	print(f"Model saved to {model_path}")


if __name__ == "__main__":
	DATASET_PATH = os.path.join("data", "symptom_disease_dataset.csv")
	MODEL_PATH = os.path.join("models", "model.joblib")
	train_and_save_model(DATASET_PATH, MODEL_PATH)


