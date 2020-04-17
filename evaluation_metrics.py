import math
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from extraction import read_qrels, INPUT_FILE_QRELS


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
    if IDCG == 0:
        return
    else:
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

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def sort_qrels(qrels):
    qrels_sorted = {}
    for _, field in qrels.iterrows():
        if qrels_sorted.get(field['query_id']) is None:
            qrels_sorted[field['query_id']] = {field['table_id']: field['rel']}
        else:
            qrels_sorted[field['query_id']][field['table_id']] = field['rel']
    return qrels_sorted

# ignoring query 12 as it has no valuable ground truth
def get_avg_NDCG_Score(sorted_rels, sorted_scores, n=20):
    output = 0

    for i, query_ranks in enumerate(sorted_rels):
        # Using the method of lisa (query 12 gives NaN so I just add 1)
        ndcg = compute_NDCG(query_ranks, sorted_scores[i], n)
        if math.isnan(ndcg):
            output += 0
        else:
            output += ndcg

    output = output / 59
    return output


def get_avg_ERR_Score(sorted_rels, sorted_scores, n=20):
    output = 0

    for i, query_ranks in enumerate(sorted_rels):
        # Using the method of lisa (query 12 gives NaN so I just add 1)
        ndcg = compute_ERR(query_ranks, sorted_scores[i], n)
        if math.isnan(ndcg):
            output += 0
        else:
            output += ndcg

    output = output / 59
    return output


def compute_NDCG_multi_field_example():
    multi_field = pd.read_csv("C:/Users/Dplen/Documents/IN4325-Core-IR-Project/data/multi_field.txt", sep="\t",
                              header=None)
    qrels = read_qrels(INPUT_FILE_QRELS)

    # Get rankings and predictions
    scores = multi_field[3]
    multi_field = multi_field[2]

    qrels_sorted = {}
    for _, field in qrels.iterrows():
        if qrels_sorted.get(field['query_id']) is None:
            qrels_sorted[field['query_id']] = {field['table_id']: field['rel']}
        else:
            qrels_sorted[field['query_id']][field['table_id']] = field['rel']

    sorted_ranks = []
    sorted_scores = []
    i = 0
    query = 1
    for j in range(20, len(multi_field) + 20, 20):
        query_ranks = []
        query_scores = scores[i:j].to_list()
        for table_id in multi_field[i:j]:
            query_ranks.append(qrels_sorted[query][table_id])
        sorted_ranks.append(query_ranks)
        sorted_scores.append(query_scores)
        query += 1
        i = j

    output = 0

    for i, query_ranks in enumerate(sorted_ranks):
        # Using method I copied from kaggle
        #output += ndcg_at_k(query_ranks, 20, method=0)

        # Using sklearn method
        #output += ndcg_score(np.asarray([query_ranks]), np.asarray([sorted_scores[i]]), 10)

        # Using the method of lisa (query 12 gives NaN so I just add 1)
        ndcg = compute_NDCG(query_ranks, sorted_scores[i], 20)
        if math.isnan(ndcg):
            output += 0
        else:
            output += ndcg

    output = output / 59
    return output



if __name__ == '__main__':
    ranks = [1, 2, 3, 0, 3, 2]
    predictions = [0.2, 0.1, 0.5, 0.4, 0.7, 0.6]
    performance = compute_ERR(ranks, predictions, 10)
    print("performance ", performance)