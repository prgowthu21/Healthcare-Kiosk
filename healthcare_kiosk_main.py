"""
============================================================
HEALTHCARE KIOSK
============================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score   


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
# ACTIVITY LEVEL MAPPING FOR DIET MODEL
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


def nutrient_focus(score):
    if score <= 3:
        return "Balanced diet"
    elif score <= 6:
        return "Increase fiber & protein"
    else:
        return "Strict diet control & medical nutrition therapy"

# ============================================================
# 1. HYPERTENSION RISK (BP-FOCUSED MODEL)
# ============================================================

header("TRAINING MODEL 1: HYPERTENSION RISK")

bp_df = pd.read_csv("data/hypertension_dataset.csv")

features = [           # FEATURES & TARGET
    "Systolic_BP",
    "Diastolic_BP",
    "Smoking_Status",
    "Family_History",
    "Stress_Level",
    "Salt_Intake",
    "Sleep_Duration",
    "Heart_Rate"
]

target = "Hypertension"

bp_df = bp_df[features + [target]].dropna()

categorical_cols = ["Smoking_Status", "Family_History"] # ENCODE CATEGORICAL FEATURES
encoders_bp = {}

for col in categorical_cols:
    le = LabelEncoder()
    bp_df[col] = le.fit_transform(bp_df[col])
    encoders_bp[col] = le

X = bp_df[features].values # TRAIN–TEST SPLIT
y = bp_df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_bp = StandardScaler() # SCALING
X_train_s = scaler_bp.fit_transform(X_train) 
X_test_s  = scaler_bp.transform(X_test)

bp_model = RandomForestClassifier( # MODEL TRAINING
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

bp_model.fit(X_train_s, y_train)

bp_acc = accuracy_score(y_test, bp_model.predict(X_test_s))
bp_auc = roc_auc_score(y_test, bp_model.predict_proba(X_test_s)[:, 1])

# ============================================================
# MODEL 2 – SpO2 & TEMPERATURE
# ============================================================

header("TRAINING MODEL 2: SpO2 & TEMPERATURE")

spo2_df = pd.read_csv("data/spo2_dataset.csv")
spo2_df = spo2_df[["SpO2", "body temperature"]].dropna()
spo2_df.columns = ["spo2", "temp"]

X_s = spo2_df.values
y_s = np.array([1 if (s < 95 or t > 38.5) else 0 for s, t in X_s])

X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
)

scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train)
X_test_s = scaler_s.transform(X_test)

spo2_model = SVC(kernel="rbf", probability=True, random_state=42)
spo2_model.fit(X_train_s, y_train)

spo2_acc = accuracy_score(y_test, spo2_model.predict(X_test_s))
spo2_auc = roc_auc_score(y_test, spo2_model.predict_proba(X_test_s)[:, 1])


# ============================================================
# 3. MULTI-VITAL DISEASE RISK MODEL
# ============================================================

header("TRAINING: MULTI-VITAL DISEASE MODEL")

heart_df = pd.read_csv("data/heart_disease_complete.csv")
features = ["age", "trestbps", "chol", "thalch", "oldpeak"]

heart_df = heart_df[features + ["num"]].dropna()

X_m = heart_df[features].values
y_m = (heart_df["num"] > 0).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42, stratify=y_m
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

multi_accuracy = accuracy_score(y_test, multi_model.predict(X_test_s))


# ============================================================
# 4. OBESITY + NUTRITION MODEL
# ============================================================

header("TRAINING: OBESITY & NUTRITION MODEL")

diet_df = pd.read_csv("data/diet_recommendations_dataset.csv")

diet_features = [
    "Age", "Gender", "Weight_kg", "Height_cm", "BMI",
    "Physical_Activity_Level", "Daily_Caloric_Intake",
    "Cholesterol_mg/dL", "Blood_Pressure_mmHg",
    "Glucose_mg/dL", "Weekly_Exercise_Hours",
    "Dietary_Nutrient_Imbalance_Score"
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
    X_o, y_o, test_size=0.2, random_state=42, stratify=y_o
)

scaler_o = StandardScaler()
X_train_s = scaler_o.fit_transform(X_train)
X_test_s = scaler_o.transform(X_test)

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
            stress = float(input("Stress Level (1-10): "))
            salt = float(input("Salt Intake (g/day): "))
            sleep = float(input("Sleep Duration (hours): "))
            hr = float(input("Heart Rate: "))

            # Encode categorical values safely
            smoking_enc = encoders_bp["Smoking_Status"].transform([smoking])[0]
            family_enc = encoders_bp["Family_History"].transform([family])[0]

            user_X = [[
                s,
                d,
                smoking_enc,
                family_enc,
                stress,
                salt,
                sleep,
                hr
            ]]

            user_X_s = scaler_bp.transform(user_X)
            prob = bp_model.predict_proba(user_X_s)[0][1]

            print("\nHypertension Risk Assessment")
            print("Guidance:", medical_message(prob))
            print(f"Model Reliability: {bp_acc*100:.1f}%")

        elif choice == "2":
            print("\nEnter SpO2 & Vital Details:")

            pulse = float(input("Pulse (bpm): "))
            temp = float(input("Body Temperature (°C): "))
            spo2 = float(input("SpO2 (%): "))

            user_X = [[pulse, temp, spo2]]
            user_X_s = scaler_s.transform(user_X)

            prob = spo2_model.predict_proba(user_X_s)[0][1]

            print("\nSpO2 Health Assessment")
            print("Guidance:", medical_message(prob))
            print(f"Model Reliability: {spo2_acc*100:.1f}%")


        elif choice == "3":
            age = float(input("Age: "))
            bp = float(input("Resting BP: "))
            chol = float(input("Cholesterol: "))
            hr = float(input("Max Heart Rate: "))
            st = float(input("ST Depression: "))

            X = scaler_m.transform([[age, bp, chol, hr, st]])
            prob = multi_model.predict_proba(X)[0][1]

            print("\nAssessment: Cardiac Risk Screening")
            print("Guidance:", medical_message(prob, "Low Risk"))

        elif choice == "4":
            age = float(input("Age: "))
            gender = input("Gender (Male/Female): ")

            weight = float(input("Weight (kg): "))
            height = float(input("Height (cm): "))

            activity_input = input("Activity Level (Low/Moderate/High): ").strip().title()
            if activity_input not in ACTIVITY_MAP:
                raise ValueError("Activity Level must be Low, Moderate, or High")

            activity = ACTIVITY_MAP[activity_input]

            calories = float(input("Daily Calories: "))
            chol = float(input("Cholesterol: "))
            bp = float(input("Blood Pressure: "))
            glucose = float(input("Glucose: "))
            exercise = float(input("Weekly Exercise Hours: "))
            imbalance = float(input("Nutrient Imbalance Score (0–10): "))

            bmi = bmi_calc(weight, height)
            obesity = obesity_level(bmi)

            user = [[
                age,
                safe_encode(encoders["Gender"], gender, "Gender"),
                weight,
                height,
                bmi,
                safe_encode(encoders["Physical_Activity_Level"], activity, "Activity Level"),
                calories,
                chol,
                bp,
                glucose,
                exercise,
                imbalance
            ]]

            user_s = scaler_o.transform(user)
            diet_pred = target_enc.inverse_transform(diet_model.predict(user_s))[0]

            print("\n--- RESULT ---")
            print(f"BMI: {bmi}")
            print(f"Obesity Level: {obesity}")
            print(f"Diet Recommendation: {diet_pred}")
            print(f"Nutrient Focus: {nutrient_focus(imbalance)}")

        elif choice == "5":
            print("\nThank you for using the Healthcare Kiosk. Stay healthy!")
            break

        else:
            print("Invalid choice. Try again.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
