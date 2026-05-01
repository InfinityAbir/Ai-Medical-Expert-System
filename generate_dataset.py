import pandas as pd
import random

# Base patterns (knowledge-driven)
diseases = {
    "Flu": {
        "fever": 1, "cough": 1, "fatigue": 1, "headache": 1,
        "sore_throat": 1, "runny_nose": 0, "nausea": 1,
        "body_pain": 1, "skin_rash": 0, "chills": 1
    },
    "Cold": {
        "fever": 0, "cough": 1, "fatigue": 0, "headache": 0,
        "sore_throat": 1, "runny_nose": 1, "nausea": 0,
        "body_pain": 0, "skin_rash": 0, "chills": 0
    },
    "Allergy": {
        "fever": 0, "cough": 0, "fatigue": 0, "headache": 0,
        "sore_throat": 1, "runny_nose": 1, "nausea": 0,
        "body_pain": 0, "skin_rash": 0, "chills": 0
    },
    "Dengue": {
        "fever": 1, "cough": 0, "fatigue": 1, "headache": 1,
        "sore_throat": 0, "runny_nose": 0, "nausea": 1,
        "body_pain": 1, "skin_rash": 0, "chills": 1
    },
    "Chickenpox": {
        "fever": 1, "cough": 0, "fatigue": 1, "headache": 0,
        "sore_throat": 0, "runny_nose": 0, "nausea": 1,
        "body_pain": 1, "skin_rash": 1, "chills": 1
    },
    "Healthy": {
        "fever": 0, "cough": 0, "fatigue": 0, "headache": 0,
        "sore_throat": 0, "runny_nose": 0, "nausea": 0,
        "body_pain": 0, "skin_rash": 0, "chills": 0
    }
}

def generate_data(samples_per_disease=150):
    data = []

    for disease, base in diseases.items():
        for _ in range(samples_per_disease):
            row = base.copy()

            # Add randomness (realistic variation)
            for key in row:
                if random.random() < 0.1:  # 10% noise
                    row[key] = 1 - row[key]

            row["disease"] = disease
            data.append(row)

    return pd.DataFrame(data)


df = generate_data(150)  # 150 × 6 diseases = 900 rows
df.to_csv("dataset/dataset.csv", index=False)

print("✅ Dataset generated with", len(df), "rows")