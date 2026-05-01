import pandas as pd

def train_probabilities(df):
    diseases = df['disease'].unique()
    probabilities = {}

    for disease in diseases:
        subset = df[df['disease'] == disease]
        probabilities[disease] = subset.mean(numeric_only=True)

    return probabilities


def bayes_predict(symptoms, probabilities):
    scores = {}

    for disease, probs in probabilities.items():
        score = 1
        for symptom, value in symptoms.items():
            if symptom in probs:
                prob = probs[symptom]
                score *= prob if value == 1 else (1 - prob)
        scores[disease] = score

    return scores