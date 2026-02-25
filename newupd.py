"""
============================================================
HEALTHCARE KIOSK — SIMPLIFIED RF + LR VERSION
============================================================
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

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
# 1️⃣ HYPERTENSION MODEL — RF + LR ONLY
# ============================================================

header("TRAINING HYPERTENSION MODEL")

bp_df = pd.read_csv("data/hypertension_dataset.csv")

features = [
    "Systolic_BP","Diastolic_BP","Smoking_Status",
    "Family_History","Sleep_Duration","Heart_Rate"
]

target = "Hypertension"
bp_df = bp_df[features + [target]].dropna()

encoders_bp = {}
for col in ["Smoking_Status","Family_History"]:
    le = LabelEncoder()
    bp_df[col] = le.fit_transform(bp_df[col])
    encoders_bp[col] = le

X = bp_df[features].values
y = bp_df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

scaler_bp = StandardScaler()
X_train_s = scaler_bp.fit_transform(X_train)

# ✅ RandomForest
rf_bp = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200,random_state=42))
rf_bp.fit(X_train_s,y_train)

# ✅ LogisticRegression
lr_bp = CalibratedClassifierCV(LogisticRegression(max_iter=500))
lr_bp.fit(X_train_s,y_train)


# ============================================================
# 2️⃣ TEMPERATURE REGRESSOR
# ============================================================

header("TRAINING TEMPERATURE REGRESSOR")

spo2_df = pd.read_csv("data/spo2_dataset.csv")
spo2_df = spo2_df[["pulse","SpO2","body temperature"]].dropna()
spo2_df.columns = ["pulse","spo2","temp"]

X_s = spo2_df[["pulse","spo2"]].values
y_s = spo2_df["temp"].values

X_train,X_test,y_train,y_test = train_test_split(X_s,y_s,test_size=0.2,random_state=42)

scaler_s = StandardScaler()
X_train_s = scaler_s.fit_transform(X_train)

spo2_model = RandomForestRegressor(n_estimators=300,random_state=42)
spo2_model.fit(X_train_s,y_train)


# ============================================================
# 3️⃣ CARDIAC MODEL — RF + LR ONLY
# ============================================================

header("TRAINING CARDIAC MODEL")

heart_df = pd.read_csv("data/heart_disease_complete.csv")

features = ["age","trestbps","thalch"]
heart_df = heart_df[features + ["num"]].dropna()

X_m = heart_df[features].values
y_m = (heart_df["num"] > 0).astype(int).values

X_train,X_test,y_train,y_test = train_test_split(
    X_m,y_m,test_size=0.2,random_state=42,stratify=y_m
)

scaler_m = StandardScaler()
X_train_s = scaler_m.fit_transform(X_train)

rf_card = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200,random_state=42))
rf_card.fit(X_train_s,y_train)

lr_card = CalibratedClassifierCV(LogisticRegression(max_iter=500))
lr_card.fit(X_train_s,y_train)


# ============================================================
# 4️⃣ META MODEL
# ============================================================

header("TRAINING META HEALTH MODEL")

meta_X = []
meta_y = []

for i in range(len(X_train)):
    bp_prob = np.random.rand()
    cardiac_prob = np.random.rand()
    bmi = np.random.uniform(18,35)
    temp = np.random.uniform(36,39)

    meta_X.append([bp_prob,cardiac_prob,bmi,temp])
    meta_y.append(1 if (bp_prob+cardiac_prob)/2 > 0.6 else 0)

meta_model = RandomForestClassifier(n_estimators=200,random_state=42)
meta_model.fit(np.array(meta_X),np.array(meta_y))


# ============================================================
# USER INTERACTION
# ============================================================

header("ADVANCED HEALTHCARE KIOSK")

while True:

    print("\n1. Full Health Assessment")
    print("2. Exit")

    choice = input("Enter choice: ").strip()

    try:
        if choice=="1":

            s=float(input("Systolic BP: "))
            d=float(input("Diastolic BP: "))
            smoking=input("Smoking Status: ").title()
            family=input("Family History: ").title()
            sleep=float(input("Sleep Duration: "))
            hr=float(input("Heart Rate: "))

            age=float(input("Age: "))
            max_hr=float(input("Max Heart Rate: "))

            weight=float(input("Weight: "))
            height=float(input("Height: "))
            pulse=float(input("Pulse: "))
            spo2=float(input("SpO2: "))

            smoking_enc = encoders_bp["Smoking_Status"].transform([smoking])[0]
            family_enc = encoders_bp["Family_History"].transform([family])[0]

            bp_input = scaler_bp.transform([[s,d,smoking_enc,family_enc,sleep,hr]])

            # ⭐ Average RF + LR probability
            bp_prob = (rf_bp.predict_proba(bp_input)[0][1] +
                       lr_bp.predict_proba(bp_input)[0][1]) / 2

            cardiac_input = scaler_m.transform([[age,s,max_hr]])

            cardiac_prob = (rf_card.predict_proba(cardiac_input)[0][1] +
                            lr_card.predict_proba(cardiac_input)[0][1]) / 2

            temp_input = scaler_s.transform([[pulse,spo2]])
            temp_pred = spo2_model.predict(temp_input)[0]

            bmi = bmi_calc(weight,height)

            meta_prob = meta_model.predict_proba([[bp_prob,cardiac_prob,bmi,temp_pred]])[0][1]

            print("\n=== HEALTH REPORT ===")
            print("Hypertension:",medical_message(bp_prob))
            print("Cardiac:",medical_message(cardiac_prob,"Cardiac Risk"))
            print("BMI:",bmi)
            print("Obesity Level:",obesity_level(bmi))
            print("Nutrition Plan:",nutrition_plan(bmi))
            print(f"Predicted Temp: {temp_pred:.2f}")

            print("\n🧠 OVERALL AI HEALTH SCORE:")
            print(medical_message(meta_prob,"Overall Health"))

        elif choice=="2":
            break

    except Exception as e:
        print("Error:",e)
