from scipy import stats

def majority_voting(texts):
    final_results = []
    for t in zip(*texts):
        most_common = stats.mode(t)[0][0]
        final_results.append(most_common)
    return final_results
