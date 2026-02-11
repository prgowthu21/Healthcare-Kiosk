"""
============================================================
HEALTHCARE KIOSK
============================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score   


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
# SAFE LABEL ENCODER
# ============================================================

def safe_encode(encoder, value, field_name):
    value = value.strip().title()
    if value not in encoder.classes_:
        raise ValueError(
            f"Invalid {field_name}. Allowed values: {list(encoder.classes_)}"
        )
    return encoder.transform([value])[0]


# ============================================================
# ACTIVITY LEVEL MAP
# ============================================================

ACTIVITY_MAP = {
    "Low": "Sedentary",
    "Moderate": "Moderate",
    "High": "Active"
}

# ============================================================
# BMI & NUTRITION LOGIC
# ============================================================

def bmi_calc(w, h):
    return round(w / ((h / 100) ** 2), 2)

def obesity_level(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity Type I"
    elif bmi < 40:
        return "Obesity Type II"
    else:
        return "Obesity Type III"

def nutrient_focus(bmi):
    if bmi < 18.5:
        return "Increase calorie intake with balanced protein & healthy fats"
    elif bmi < 25:
        return "Maintain balanced diet with regular activity"
    elif bmi < 30:
        return "Reduce refined carbs & increase fiber"
    elif bmi < 35:
        return "High-protein structured diet plan"
    else:
        return "Strict medical nutrition therapy recommended"


# ============================================================
# 1. HYPERTENSION MODEL 
# ============================================================

header("TRAINING MODEL 1: HYPERTENSION RISK")

bp_df = pd.read_csv("data/hypertension_dataset.csv")

# FEATURE LIST
features = [
    "Systolic_BP",
    "Diastolic_BP",
    "Smoking_Status",
    "Family_History",
    "Sleep_Duration",
    "Heart_Rate"
]

target = "Hypertension"

bp_df = bp_df[features + [target]].dropna()

encoders_bp = {}
for col in ["Smoking_Status", "Family_History"]:
    le = LabelEncoder()
    bp_df[col] = le.fit_transform(bp_df[col])
    encoders_bp[col] = le

X = bp_df[features].values
y = bp_df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_bp = StandardScaler()
X_train_s = scaler_bp.fit_transform(X_train)
X_test_s  = scaler_bp.transform(X_test)

bp_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

bp_model.fit(X_train_s, y_train)


# ============================================================
# 2. BODY TEMPERATURE PREDICTOR
# ============================================================

header("TRAINING MODEL 2: BODY TEMPERATURE PREDICTOR")

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
X_test_s  = scaler_s.transform(X_test)

spo2_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

spo2_model.fit(X_train_s, y_train)


# ============================================================
# 3. MULTI-VITAL DISEASE MODEL - CARDIAC
# ============================================================

header("TRAINING: MULTI-VITAL DISEASE MODEL")

heart_df = pd.read_csv("data/heart_disease_complete.csv")

features = ["age", "trestbps", "thalch"]

heart_df = heart_df[features + ["num"]].dropna()

X_m = heart_df[features].values
y_m = (heart_df["num"] > 0).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X_m, y_m,
    test_size=0.2,
    random_state=42,
    stratify=y_m
)

scaler_m = StandardScaler()
X_train_s = scaler_m.fit_transform(X_train)
X_test_s  = scaler_m.transform(X_test)

multi_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

multi_model.fit(X_train_s, y_train)


# ============================================================
# 4. DIET MODEL
# ============================================================

header("TRAINING: OBESITY & NUTRITION MODEL")

diet_df = pd.read_csv("data/diet_recommendations_dataset.csv")

diet_features = [
    "Age","Gender","Weight_kg","Height_cm",
    "Physical_Activity_Level","Weekly_Exercise_Hours","Blood_Pressure_mmHg"
]

target = "Diet_Recommendation"

diet_df = diet_df[diet_features + [target]].dropna()

encoders = {}

for col in ["Gender", "Physical_Activity_Level"]:
    le = LabelEncoder()
    diet_df[col] = le.fit_transform(diet_df[col])
    encoders[col] = le

target_enc = LabelEncoder()
diet_df[target] = target_enc.fit_transform(diet_df[target])

X_o = diet_df[diet_features].values
y_o = diet_df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X_o, y_o,
    test_size=0.2,
    random_state=42,
    stratify=y_o
)

scaler_o = StandardScaler()
X_train_s = scaler_o.fit_transform(X_train)
X_test_s  = scaler_o.transform(X_test)

diet_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

diet_model.fit(X_train_s, y_train)


# ============================================================
# USER INTERACTION
# ============================================================

header("HEALTHCARE KIOSK - USER MODE")

while True:

    print("\nChoose option:")
    print("1. BP & Heart Rate")
    print("2. SpO2 & Temperature")
    print("3. Multi-Vital Disease Risk")
    print("4. Obesity & Nutrition")
    print("5. Exit")

    choice = input("Enter choice (1-5): ").strip()

    try:
        if choice == "1":

            print("\nEnter BP & Lifestyle Details:")

            s = float(input("Systolic BP: "))
            d = float(input("Diastolic BP: "))
            smoking = input("Smoking Status (Former/Current/Never): ").strip().title()
            family = input("Family History (Yes/No): ").strip().title()
            sleep = float(input("Sleep Duration (hours): "))
            hr = float(input("Heart Rate: "))

            smoking_enc = encoders_bp["Smoking_Status"].transform([smoking])[0]
            family_enc = encoders_bp["Family_History"].transform([family])[0]

            user_X = [[s,d,smoking_enc,family_enc,sleep,hr]]

            user_X_s = scaler_bp.transform(user_X)
            prob = bp_model.predict_proba(user_X_s)[0][1]

            print("\nHypertension Risk Assessment")
            print("Guidance:", medical_message(prob))

        elif choice == "2":

            pulse = float(input("Pulse (bpm): "))
            spo2  = float(input("SpO2 (%): "))

            user_X = [[pulse, spo2]]
            user_X_s = scaler_s.transform(user_X)

            predicted_temp = spo2_model.predict(user_X_s)[0]

            print("\nPredicted Body Temperature")
            print(f"Estimated Temperature: {predicted_temp:.2f} °C")
            if predicted_temp < 37.5: 
                print("Guidance: ✅ Normal temperature range") 
            elif predicted_temp < 38.5: 
                print("Guidance: ⚠ Mild fever possible") 
            else: 
                print("Guidance: 🚨 High fever risk – consult a doctor")
        
        elif choice == "3": 
            print("\nEnter Cardiac Vital Details:") 
            age = float(input("Age: ")) 
            bp = float(input("Resting BP: ")) 
            hr = float(input("Maximum Heart Rate: ")) 
            X = scaler_m.transform([[age, bp, hr]]) 
            prob = multi_model.predict_proba(X)[0][1] 
            print("\nAssessment: Cardiac Risk Screening")
            print("Guidance:", medical_message(prob, "Cardiac Risk")) 
        
        elif choice == "4": 
            print("\nEnter Obesity & Nutrition Details:") 
            age = float(input("Age: ")) 
            gender = input("Gender (Male/Female): ") 
            weight = float(input("Weight (kg): ")) 
            height = float(input("Height (cm): ")) 
            activity_input = input("Activity Level (Low/Moderate/High): ").strip().title() 
            if activity_input not in ACTIVITY_MAP: 
                raise ValueError("Activity Level must be Low, Moderate, or High")
            activity = ACTIVITY_MAP[activity_input]
            bp = float(input("Blood Pressure: ")) 
            exercise = float(input("Weekly Exercise Hours: ")) 
            bmi = bmi_calc(weight, height) 
            obesity = obesity_level(bmi) 
            user = [[ age, safe_encode(encoders["Gender"], gender, "Gender"), weight, height, safe_encode(encoders["Physical_Activity_Level"], activity, "Activity Level"), exercise, bp ]] 
            user_s = scaler_o.transform(user) 
            diet_pred = target_enc.inverse_transform(diet_model.predict(user_s))[0] 
            print("\n--- RESULT ---") 
            print(f"BMI: {bmi}") 
            print(f"Obesity Level: {obesity}") 
            print(f"Diet Recommendation: {diet_pred}") 
            print(f"Nutrient Focus: {nutrient_focus(bmi)}")

        elif choice == "5":
            print("\nThank you for using the Healthcare Kiosk. Stay healthy!")
            break

    except Exception as e:
        print(f"\n❌ Error: {e}")
