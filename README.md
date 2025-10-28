# Disease Prediction System (Flask + ML)

A minimal end-to-end example of a disease prediction web app using Flask and a scikit-learn model trained on a small demo symptom dataset. Not medical advice.

## Features
- Simple symptom checklist UI (HTML/CSS)
- Flask backend with `/predict` and `/train`
- RandomForestClassifier trained on `data/symptom_disease_dataset.csv`
- Model auto-trains on first run if not present
- Shows top-3 predicted diseases with probabilities

## Project Structure
```
.
├── app.py                    # Flask app
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── data/
│   └── symptom_disease_dataset.csv
├── models/
│   └── model.joblib          # Saved model (generated)
├── templates/
│   ├── index.html            # Form
│   └── result.html           # Results
└── static/
    └── css/
        └── styles.css
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Usage
1. Check symptoms and click Predict.
2. See most likely condition and top-3 probabilities.
3. To retrain on the bundled dataset, use the Retrain Model button on the homepage or run:
```powershell
python train.py
```

## Customize
- Add/remove symptoms by editing the columns in `data/symptom_disease_dataset.csv` and keeping `disease` as the target label. Ensure the symptom list in `app.py` matches the dataset columns.
- Replace the dataset with a larger one of the same schema for better performance.

## Notes
- Educational demo; not for clinical use.
- The small dataset is synthetic and only for demonstration.
