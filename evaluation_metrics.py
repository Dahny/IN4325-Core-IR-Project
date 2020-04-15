import math
import numpy as np


def compute_NDCG(rankings, predictions, n):
    """
    Computing the Normalized Discounted Cumulative Gain (NDCG)
    :param rankings:
    :param predictions:
    :param n:
    :return:
    """
    if len(predictions) < n:
        predictions.extend([0] * (n - len(predictions)))
        rankings.extend([0] * (n - len(rankings)))
    indices = np.argsort(predictions)[::-1]
    relevant_values = np.array(rankings)[indices]
    DCG = 0
    IDCG = 0
    sorted_rankings = np.sort(relevant_values[:n])[::-1]
    for i in range(n):
        DCG += (pow(2, relevant_values[i]) - 1) / math.log(i + 2, 2)
        IDCG += (pow(2, sorted_rankings[i]) - 1) / math.log(i + 2, 2)
    return DCG / IDCG


def compute_ERR(rankings, predictions, n):
    """
    Computing the Expected Reciprocal Rank (ERR)
    :param rankings:
    :param predictions:
    :param n:
    :return:
    """
    if len(predictions) < n:
        predictions.extend([0] * (n - len(predictions)))
        rankings.extend([0] * (n - len(rankings)))
    indices = np.argsort(predictions)[::-1]
    relevant_values = np.array(rankings)[indices]
    p = 1
    ERR = 0
    for r in range(n):
        R_r = (pow(2, relevant_values[r]) - 1) / (pow(2, np.max(relevant_values)))
        ERR += p * (R_r / (r+1))
        p = p * (1 - R_r)
    return ERR


if __name__ == '__main__':
    ranks = [1, 2, 3, 0, 3, 2]
    predictions = [0.2, 0.1, 0.5, 0.4, 0.7, 0.6]
    performance = compute_ERR(ranks, predictions, 10)
    print("performance ", performance)