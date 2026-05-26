from typing import List

from scipy.optimize import linear_sum_assignment
import numpy as np
from sentence_transformers import SentenceTransformer


def term_set_sim_by_sim_matrix(
    sim_matrix: np.ndarray
):
    cost_matrix = 1 - sim_matrix  # Convert similarity to cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # case 1
    gold = [
        "high blood pressure",
        "cough",
        "pyrexia"
    ]
    pred = [
        "hypertension",
        "dry cough",
        "fever"
    ]
    gold_embs = model.encode(gold)
    pred_embs = model.encode(pred)
    sim_matrix = model.similarity(gold_embs, pred_embs)
    print(f"Similarity matrix:", sim_matrix)
    row_ind, col_ind = term_set_sim_by_sim_matrix(sim_matrix)
    print("Optimal assignment:", list(zip(row_ind, col_ind)))

    # case 2
    gold = [
        "high blood pressure",
        "cough",
        "pyrexia"
    ]
    pred = [
        "hypertension",
        "fever"
    ]