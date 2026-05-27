from typing import List, Tuple

from scipy.optimize import linear_sum_assignment
import numpy as np
from sentence_transformers import SentenceTransformer


def _helper_SBMS(
    sim_matrix: np.ndarray
) -> Tuple[float, list, list]:
    """
    sim_matrix: row indicates gold terms, column indicates predicted terms. Each entry is the similarity between the corresponding gold and predicted term.
    """
    cost_matrix = 1 - sim_matrix  # Convert similarity to cost
    # Hungarian algorithm to find the optimal assignment; ECCV's DETR paper also uses similar idea
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sim_score_sum = sim_matrix[row_ind, col_ind].sum()
    return sim_score_sum, row_ind, col_ind


def SBMS_by_sentence_transformer(
    gold: List[str],
    pred: List[str],
    model: SentenceTransformer
) -> Tuple[float, list, list, np.ndarray]:
    """
    SBMS stands for Soft Bipartite Matching Score.
    """
    gold_embs = model.encode(gold)
    pred_embs = model.encode(pred)
    sim_matrix = model.similarity(gold_embs, pred_embs)
    sim_score_sum, row_ind, col_ind = _helper_SBMS(sim_matrix)
    sim_score_prec = sim_score_sum / len(pred) if len(pred) > 0 else 0.0
    sim_score_rec = sim_score_sum / len(gold) if len(gold) > 0 else 0.0
    sim_score_f1 = 2 * sim_score_prec * sim_score_rec / (sim_score_prec + sim_score_rec) if (sim_score_prec + sim_score_rec) > 0 else 0.0
    return sim_score_f1, sim_score_prec, sim_score_rec, row_ind, col_ind, sim_matrix


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
    SBMS_f1, SBMS_prec, SBMS_rec, row_ind, col_ind, sim_matrix = SBMS_by_sentence_transformer(gold, pred, model)
    print(f"Similarity matrix:", sim_matrix)
    print("Optimal assignment:", list(zip(row_ind, col_ind)))
    print(f'Similarity f1: {SBMS_f1}, Precision: {SBMS_prec}, Recall: {SBMS_rec}')

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
    SBMS_f1, SBMS_prec, SBMS_rec, row_ind, col_ind, sim_matrix = SBMS_by_sentence_transformer(gold, pred, model)
    print(f"Similarity matrix:", sim_matrix)
    print("Optimal assignment:", list(zip(row_ind, col_ind)))
    print(f'Similarity f1: {SBMS_f1}, Precision: {SBMS_prec}, Recall: {SBMS_rec}')