import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import pandas as pd


MODEL_PATH = os.path.join("models", "model.joblib")
DATASET_PATH = os.path.join("data", "symptom_disease_dataset.csv")


def ensure_directories():
	for folder in ["models", "data", os.path.join("static", "css"), "templates"]:
		os.makedirs(folder, exist_ok=True)


def load_or_train_model():
	"""Load the model if available; otherwise train a new one from the dataset."""
	from train import train_and_save_model
	if not os.path.exists(MODEL_PATH):
		if not os.path.exists(DATASET_PATH):
			raise FileNotFoundError(
				"Dataset not found. Please place the CSV at data/symptom_disease_dataset.csv or use README instructions."
			)
		train_and_save_model(DATASET_PATH, MODEL_PATH)
	return joblib.load(MODEL_PATH)


def build_app():
	ensure_directories()
	app = Flask(__name__)
	app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
	app.wsgi_app = ProxyFix(app.wsgi_app)

	# Lazy-load the model at first request to speed up startup in some environments
	_model = {"bundle": None}

	# Keep the list of symptoms centralized to match training columns
	SYMPTOMS = [
		"fever",
		"cough",
		"fatigue",
		"headache",
		"sore_throat",
		"runny_nose",
		"shortness_of_breath",
		"chest_pain",
		"nausea",
		"diarrhea",
		"rash",
		"joint_pain",
	]

	# Simple non-medical recommendations for demo purposes only
	RECOMMENDATIONS = {
		"Common Cold": {
			"food": ["Warm soups", "Herbal tea with honey", "Vitamin C-rich fruits"],
			"tablets": ["Paracetamol 500mg", "Antihistamine (e.g., Cetirizine)"]
		},
		"Flu": {
			"food": ["Hydration (water, ORS)", "Broth-based soups", "Bananas, rice"],
			"tablets": ["Paracetamol 500mg", "Oseltamivir (if prescribed)"]
		},
		"COVID-19": {
			"food": ["Hydration", "High-protein diet", "Fruits and vegetables"],
			"tablets": ["Paracetamol 500mg", "Consult physician for antivirals"]
		},
		"Migraine": {
			"food": ["Magnesium-rich foods", "Ginger tea", "Avoid triggers (aged cheese, caffeine)"]
			,
			"tablets": ["NSAIDs (e.g., Ibuprofen)", "Triptans (if prescribed)"]
		},
		"Food Poisoning": {
			"food": ["ORS/fluids", "BRAT diet (Banana, Rice, Applesauce, Toast)", "Avoid dairy/oily foods"],
			"tablets": ["ORS packets", "Zinc (as advised)"]
		},
		"Dengue": {
			"food": ["Papaya leaf extract (folk remedy)", "Hydration", "High-protein diet"],
			"tablets": ["Paracetamol 500mg", "Avoid NSAIDs (aspirin/ibuprofen)"]
		},
		"Pneumonia": {
			"food": ["Warm fluids", "Nutritious high-calorie foods", "Fruits"],
			"tablets": ["Antibiotics only if prescribed", "Paracetamol 500mg"]
		},
		"Allergy": {
			"food": ["Warm water", "Local honey (anecdotal)", "Avoid known allergens"],
			"tablets": ["Antihistamine (Cetirizine)"]
		},
		"Bronchitis": {
			"food": ["Warm soups/tea", "Hydration", "Ginger/turmeric preparations"],
			"tablets": ["Cough suppressant (as advised)", "Bronchodilator (if prescribed)"]
		},
		"Gastritis": {
			"food": ["Small frequent meals", "Low-spice diet", "Plain yogurt"],
			"tablets": ["Antacid", "PPI (e.g., Omeprazole) as prescribed"]
		},
		"Chickenpox": {
			"food": ["Soft foods", "Hydration", "Avoid scratching, keep nails short"],
			"tablets": ["Paracetamol 500mg", "Antivirals only if prescribed"]
		},
	}

	def get_model_bundle():
		if _model["bundle"] is None:
			_model["bundle"] = load_or_train_model()
		return _model["bundle"]

	@app.route("/")
	def index():
		return render_template("index.html", symptoms=SYMPTOMS)

	@app.route("/predict", methods=["POST"]) 
	def predict():
		selected = request.form.getlist("symptoms")
		age_raw = request.form.get("age", "").strip()
		blood_group = request.form.get("blood_group", "").strip()
		if not selected:
			flash("Please select at least one symptom.")
			return redirect(url_for("index"))

		# Build a single-row dataframe with binary indicators for each symptom
		row = {symptom: (1 if symptom in selected else 0) for symptom in SYMPTOMS}
		X = pd.DataFrame([row])
		bundle = get_model_bundle()
		clf = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
		feature_columns = bundle.get("feature_columns") if isinstance(bundle, dict) else None
		if feature_columns is not None:
			# Ensure exact ordering and presence of columns expected by the model
			for col in feature_columns:
				if col not in X.columns:
					X[col] = 0
			X = X[feature_columns]

		if hasattr(clf, "predict_proba"):
			proba = clf.predict_proba(X)[0]
			classes = list(clf.classes_)
			ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
			top3 = [(c, float(p)) for c, p in ranked[:3]]
			prediction, confidence = top3[0][0], float(top3[0][1])
		else:
			prediction = clf.predict(X)[0]
			# Fallback confidence not available
			confidence = None
			top3 = [(prediction, None)]

		# Prepare view-friendly items to avoid Python calls in Jinja
		top3_view = []
		for cls_name, prob in top3:
			if prob is None:
				top3_view.append({"cls": cls_name, "percent": None, "bar": None})
			else:
				percent = int(round(prob * 100))
				bar = percent if percent >= 5 else 5
				top3_view.append({"cls": cls_name, "percent": percent, "bar": bar})

		# Validate auxiliary fields (not used by the model in this demo)
		age = None
		try:
			if age_raw:
				age_val = int(age_raw)
				if 0 <= age_val <= 120:
					age = age_val
				else:
					flash("Age must be between 0 and 120.")
		except ValueError:
			flash("Age must be a whole number.")

		# Pick recommendations based on the top prediction
		rec = RECOMMENDATIONS.get(prediction, {"food": [], "tablets": []})

		return render_template(
			"result.html",
			prediction=prediction,
			top3=top3,
			top3_view=top3_view,
			selected_symptoms=selected,
			age=age,
			blood_group=blood_group or None,
			recommendations=rec,
		)

	@app.route("/train", methods=["POST"]) 
	def retrain():
		from train import train_and_save_model
		try:
			train_and_save_model(DATASET_PATH, MODEL_PATH)
			_model["bundle"] = joblib.load(MODEL_PATH)
			flash("Model retrained successfully.")
		except Exception as exc:
			flash(f"Training failed: {exc}")
		return redirect(url_for("index"))

	return app


app = build_app()


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)


