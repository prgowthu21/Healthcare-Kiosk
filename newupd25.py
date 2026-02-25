"""
============================================================
HEALTHCARE KIOSK — IEEE-GRADE (NO CHOL / NO OLDPEAK)
============================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
import lightgbm as lgb


# ============================================================
# MESSAGE ENGINE
# ============================================================

def medical_message(prob, label="Normal"):
    if prob < 0.4:
        return f"✅ {label} – No immediate concern"
    elif prob < 0.65:
        return f"⚠ {label} – Borderline, monitor regularly"
    elif prob < 0.85:
        return f"🚨 {label} – Elevated risk, consult a doctor"
    else:
        return f"🏥 {label} – High risk, seek medical attention immediately"


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# BMI LOGIC
# ============================================================

def bmi_calc(w, h):
    return round(w / ((h / 100) ** 2), 2)

def obesity_level(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity Class I"
    elif bmi < 40:
        return "Obesity Class II"
    else:
        return "Obesity Class III (Morbid Obesity)"

def nutrition_plan(bmi):
    if bmi < 18.5:
        return "🥗 High-Calorie Balanced Plan"
    elif bmi < 25:
        return "🥗 Balanced Maintenance Diet"
    elif bmi < 30:
        return "🥗 Weight Management Plan"
    elif bmi < 35:
        return "🥗 High-Protein Structured Diet"
    else:
        return "🥗 Clinical Nutrition Plan"


# ============================================================
# 1️⃣ HYPERTENSION MODEL
# ============================================================

header("TRAINING HYPERTENSION MODEL")

bp_df = pd.read_csv("data/hypertension_dataset.csv")

features_bp = [
    "Systolic_BP", "Diastolic_BP", "Smoking_Status",
    "Family_History", "Sleep_Duration", "Heart_Rate"
]
target_bp = "Hypertension"

bp_df = bp_df[features_bp + [target_bp]].dropna()

encoders_bp = {}
for col in ["Smoking_Status", "Family_History"]:
    le = LabelEncoder()
    bp_df[col] = le.fit_transform(bp_df[col])
    encoders_bp[col] = le

X_bp = bp_df[features_bp].values
y_bp = bp_df[target_bp].values

X_train, X_test, y_train, y_test = train_test_split(
    X_bp, y_bp, test_size=0.2, random_state=42, stratify=y_bp
)

scaler_bp = StandardScaler()
X_train_s = scaler_bp.fit_transform(X_train)

rf_bp = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=42))
rf_bp.fit(X_train_s, y_train)

lr_bp = CalibratedClassifierCV(LogisticRegression(max_iter=500))
lr_bp.fit(X_train_s, y_train)


# ============================================================
# 2️⃣ TEMPERATURE REGRESSOR
# ============================================================

header("TRAINING TEMPERATURE REGRESSOR")

spo2_df = pd.read_csv("data/spo2_dataset.csv")
spo2_df = spo2_df[["pulse", "SpO2", "body temperature"]].dropna()
spo2_df.columns = ["pulse", "spo2", "temp"]

X_s = spo2_df[["pulse", "spo2"]].values
y_s = spo2_df["temp"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42
)

scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train)

spo2_model = RandomForestRegressor(n_estimators=300, random_state=42)
spo2_model.fit(X_train_s, y_train)


# ============================================================
# 3️⃣ CARDIAC MODEL (IEEE-GRADE)
# ============================================================

header("CARDIAC MODEL")

heart_df = pd.read_csv("data/heart_disease_complete.csv")

features_card = ["age", "trestbps", "thalch"]
heart_df = heart_df[features_card + ["num"]].dropna()

X_card = heart_df[features_card].copy()
y_card = (heart_df["num"] > 0).astype(int)

# Feature Engineering
X_card["pulse_pressure"] = X_card["trestbps"] - X_card["thalch"]

# SMOTE-Tomek
smote = SMOTETomek(random_state=42)
X_res, y_res = smote.fit_resample(X_card, y_card)

# Power Transform
pt_card = PowerTransformer()
X_res_transformed = pt_card.fit_transform(X_res)

# LightGBM Feature Selection
lgb_selector = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgb_selector.fit(X_res_transformed, y_res)

importances = lgb_selector.feature_importances_
selected_idx = np.argsort(importances)[-4:]  # keep top 4 features
X_selected = X_res_transformed[:, selected_idx]

# Train/Test
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_selected, y_res, test_size=0.2,
    stratify=y_res, random_state=42
)

rf_base = RandomForestClassifier(n_estimators=300, random_state=42)
gb_base = GradientBoostingClassifier(random_state=42)
lr_base = LogisticRegression(max_iter=1000)

stacking_card = StackingClassifier(
    estimators=[("rf", rf_base), ("gb", gb_base), ("lr", lr_base)],
    final_estimator=RandomForestClassifier(n_estimators=200, random_state=42)
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking_card, X_train_c, y_train_c, cv=cv, scoring="roc_auc")
print(f"10-Fold CV ROC-AUC: {np.mean(cv_scores):.4f}")

stacking_card.fit(X_train_c, y_train_c)


def predict_cardiac(age, trestbps, thalch):
    row = np.array([[age, trestbps, thalch]])
    pulse_pressure = trestbps - thalch
    row = np.hstack([row, [[pulse_pressure]]])
    row_transformed = pt_card.transform(row)
    row_selected = row_transformed[:, selected_idx]
    return stacking_card.predict_proba(row_selected)[0][1]


# ============================================================
# 4️⃣ META MODEL
# ============================================================

header("TRAINING META HEALTH MODEL")

meta_X = []
meta_y = []

for i in range(500):
    bp_prob = np.random.rand()
    cardiac_prob = np.random.rand()
    bmi = np.random.uniform(18, 35)
    temp = np.random.uniform(36, 39)

    meta_X.append([bp_prob, cardiac_prob, bmi, temp])
    meta_y.append(1 if (bp_prob + cardiac_prob) / 2 > 0.6 else 0)

meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
meta_model.fit(np.array(meta_X), np.array(meta_y))


# ============================================================
# USER INTERACTION
# ============================================================

header("ADVANCED HEALTHCARE KIOSK")

while True:

    print("\n1. Full Health Assessment")
    print("2. Exit")

    choice = input("Enter choice: ").strip()

    try:
        if choice == "1":

            s = float(input("Systolic BP: "))
            d = float(input("Diastolic BP: "))
            smoking = input("Smoking Status: ").title()
            family = input("Family History: ").title()
            sleep = float(input("Sleep Duration (hrs): "))
            hr = float(input("Heart Rate: "))

            age = float(input("Age: "))
            max_hr = float(input("Max Heart Rate (thalch): "))

            weight = float(input("Weight (kg): "))
            height = float(input("Height (cm): "))
            pulse = float(input("Pulse: "))
            spo2 = float(input("SpO2 (%): "))

            smoking_enc = encoders_bp["Smoking_Status"].transform([smoking])[0]
            family_enc = encoders_bp["Family_History"].transform([family])[0]

            bp_input = scaler_bp.transform([[s, d, smoking_enc, family_enc, sleep, hr]])
            bp_prob = (
                rf_bp.predict_proba(bp_input)[0][1] +
                lr_bp.predict_proba(bp_input)[0][1]
            ) / 2

            cardiac_prob = predict_cardiac(age, s, max_hr)

            temp_input = scaler_s.transform([[pulse, spo2]])
            temp_pred = spo2_model.predict(temp_input)[0]

            bmi = bmi_calc(weight, height)

            meta_prob = meta_model.predict_proba(
                [[bp_prob, cardiac_prob, bmi, temp_pred]]
            )[0][1]

            print("\n=== HEALTH REPORT ===")
            print("Hypertension :", medical_message(bp_prob))
            print("Cardiac       :", medical_message(cardiac_prob, "Cardiac Risk"))
            print(f"BMI           : {bmi}")
            print("Obesity Level :", obesity_level(bmi))
            print("Nutrition Plan:", nutrition_plan(bmi))
            print(f"Predicted Temp: {temp_pred:.2f} °C")

            print("\n🧠 OVERALL AI HEALTH SCORE:")
            print(medical_message(meta_prob, "Overall Health"))

        elif choice == "2":
            break

    except Exception as e:
        print("Error:", e)