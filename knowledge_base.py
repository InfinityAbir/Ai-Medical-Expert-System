rules = {
    "Flu": ["fever", "cough", "fatigue", "chills"],
    "Cold": ["cough", "sore_throat", "runny_nose"],
    "Allergy": ["runny_nose", "sore_throat"],
    "Dengue": ["fever", "fatigue", "body_pain", "nausea"],
    "Chickenpox": ["skin_rash", "fever", "fatigue"],
    "Healthy": []
}


def match_rules(symptoms):
    scores = {}

    for disease, rule_symptoms in rules.items():
        match_count = sum([1 for s in rule_symptoms if symptoms.get(s, 0) == 1])
        scores[disease] = match_count / (len(rule_symptoms) + 1)

    return scores