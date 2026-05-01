def best_first_search(scores):
    # Sort diseases by probability
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_diseases