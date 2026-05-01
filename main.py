import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from knowledge_base import match_rules
from bayes import train_probabilities, bayes_predict
from search import best_first_search
from model import train_and_evaluate, predict_model, train_second_model

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset/dataset.csv")

# -----------------------------
# Train components
# -----------------------------
probabilities = train_probabilities(df)
model, acc, prec, rec, f1, cm, y_test, y_pred = train_and_evaluate(df)
nb_acc = train_second_model(df)

# -----------------------------
# User input
# -----------------------------
symptoms = {
    "fever": 1,
    "cough": 1,
    "fatigue": 1,
    "headache": 0,
    "sore_throat": 1,
    "runny_nose": 0,
    "nausea": 1,
    "body_pain": 1,
    "skin_rash": 0,
    "chills": 1
}

# -----------------------------
# Rule-based reasoning
# -----------------------------
rule_scores = match_rules(symptoms)

# -----------------------------
# Bayesian reasoning
# -----------------------------
bayes_scores = bayes_predict(symptoms, probabilities)

# -----------------------------
# Combine scores
# -----------------------------
combined_scores = {}
for disease in rule_scores:
    combined_scores[disease] = rule_scores[disease] + bayes_scores.get(disease, 0)

# -----------------------------
# Search (Best First)
# -----------------------------
results = best_first_search(combined_scores)

# -----------------------------
# ML Prediction
# -----------------------------
ml_result = predict_model(model, symptoms)

# -----------------------------
# Output (Improved)
# -----------------------------
print("\n🔍 Top Diagnoses (with confidence):")

total_score = sum(combined_scores.values())

for disease, score in results[:3]:
    confidence = score / total_score if total_score != 0 else 0
    print(f"{disease}: {confidence:.2%}")

print(f"\n🤖 ML Prediction: {ml_result}")

# -----------------------------
# Evaluation Metrics
# -----------------------------
print("\n📊 Model Evaluation:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")

# -----------------------------
# Model Comparison
# -----------------------------
print("\n📊 Model Comparison:")
print(f"Decision Tree Accuracy: {acc:.2f}")
print(f"Naive Bayes Accuracy: {nb_acc:.2f}")

# -----------------------------
# Feature Importance (Advanced)
# -----------------------------
features = df.drop("disease", axis=1).columns
importance = model.feature_importances_

print("\n📊 Feature Importance:")
for f, imp in zip(features, importance):
    print(f"{f}: {imp:.3f}")

# -----------------------------
# Save results to file
# -----------------------------
with open("results.txt", "w") as f:
    f.write("Top Diagnoses:\n")
    for disease, score in results[:3]:
        f.write(f"{disease}: {score:.4f}\n")

    f.write(f"\nML Prediction: {ml_result}\n")
    f.write(f"\nAccuracy: {acc:.2f}\n")

# -----------------------------
# Confusion Matrix
# -----------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# Performance Graph
# -----------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [acc, prec, rec, f1]

plt.figure()
plt.bar(metrics, values)
plt.title("Model Performance")
plt.savefig("performance.png")
plt.show()

# -----------------------------
# Model Comparison Graph
# -----------------------------
models = ["Decision Tree", "Naive Bayes"]
scores = [acc, nb_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison")
plt.savefig("model_comparison.png")
plt.show()

# -----------------------------
# Ethical Disclaimer
# -----------------------------
print("\n⚠️  Disclaimer: This system is for educational purposes only and should not replace professional medical advice.")