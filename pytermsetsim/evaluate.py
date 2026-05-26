from typing import List, Tuple

from scipy.optimize import linear_sum_assignment
import numpy as np
from sentence_transformers import SentenceTransformer


def term_set_sim_by_sim_matrix(
    sim_matrix: np.ndarray
) -> Tuple[float, list, list]:
    """
    sim_matrix: row indicates gold terms, column indicates predicted terms. Each entry is the similarity between the corresponding gold and predicted term.
    """
    cost_matrix = 1 - sim_matrix  # Convert similarity to cost
    # Hungarian algorithm to find the optimal assignment; ECCV's DETR paper also uses similar idea
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sim_score_sum = sim_matrix[row_ind, col_ind].sum()
    sim_score_avg = sim_score_sum / len(row_ind)
    return sim_score_avg, row_ind, col_ind


def term_set_sim_by_sentence_transformer(
    gold: List[str],
    pred: List[str],
    model: SentenceTransformer
) -> Tuple[float, list, list, np.ndarray]:
    gold_embs = model.encode(gold)
    pred_embs = model.encode(pred)
    sim_matrix = model.similarity(gold_embs, pred_embs)
    sim_score,row_ind, col_ind = term_set_sim_by_sim_matrix(sim_matrix)
    return sim_score, row_ind, col_ind, sim_matrix


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # case 1
    gold = [
        "high blood pressure",
        "cough",
        "pyrexia"
    ]
    pred = [
        "dry cough",
        "fever",
        "hypertension",
    ]
    gold_embs = model.encode(gold)
    pred_embs = model.encode(pred)
    sim_matrix = model.similarity(gold_embs, pred_embs)
    print(f"Similarity matrix:", sim_matrix)
    sim_score, row_ind, col_ind = term_set_sim_by_sim_matrix(sim_matrix)
    print("Optimal assignment:", list(zip(row_ind, col_ind)))
    print(f'Similarity score: {sim_score}')

    # case 2
    gold = [
        "high blood pressure",
        "cough",
        "pyrexia"
    ]
    pred = [
        "fever",
        "hypertension",
    ]
    sim_score, row_ind, col_ind, sim_matrix = term_set_sim_by_sentence_transformer(gold, pred, model)
    print(f"Similarity matrix:", sim_matrix)
    print("Optimal assignment:", list(zip(row_ind, col_ind)))
    print(f'Similarity score: {sim_score}')