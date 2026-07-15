import os
import json

import pronto
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.preprocessing import normalize


def load_cl_terms():
    cl = pronto.Ontology("text_baselines/cl.owl")
    term_names = []
    term_texts = []
    term_ids = []

    for term in cl.terms():
        if term.obsolete: 
            continue
        label = term.name if term.name else ""
        # 收集所有 exact synonym
        synonyms = [syn.description for syn in term.synonyms if syn.scope == 'EXACT']
        # 可以加入 broad synonym，但注意可能引入噪声
        # 构造文本：label + 同义词
        combined = label
        if synonyms:
            combined += " ; " + " ; ".join(synonyms)
        # 可选：添加定义的第一句
        if term.definition:
            combined += " . " + term.definition[:200]
        term_texts.append(combined)
        term_ids.append(term.id)
        term_names.append(term.name)
    print(f"Total terms: {len(term_texts)}")
    # 检查一个示例
    print(f"Sample terms: {term_texts[:3]}")
    return term_texts, term_ids, term_names


def prepare_cl_term_embs(
    term_texts: list,
    model: SentenceTransformer,
    emb_out_path: str = "text_baselines/cl_term_embeddings.pt"
):
    # 使用 SentenceTransformer 生成嵌入
    term_embeddings = model.encode(term_texts, show_progress_bar=True, convert_to_tensor=True)
    torch.save(term_embeddings, emb_out_path)


def search_numpy(
    query_text: str, 
    model: SentenceTransformer, 
    term_texts: list, 
    term_embeddings_norm: np.ndarray
):
    # 编码查询并归一化
    query_emb = model.encode([query_text], convert_to_tensor=False)  # 返回 numpy
    query_emb_norm = normalize(query_emb, norm='l2')
    
    # 计算余弦相似度（内积）
    scores = np.dot(query_emb_norm, term_embeddings_norm.T).flatten()
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    
    return term_texts[best_idx], best_score.item()


def gen_result_stage2(
    data_dir: str,
    out_dir: str,
    model: SentenceTransformer,
    term_names: list,
    term_embeddings: np.ndarray,
):
    """
    Use pipeline stage 2 data as input
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # read json files from data_dir
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            # ad-hoc skip one json file
            if file_name == "paper_label_data_annotation_mapping.summary.json":
                continue
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
            input_names = [item['original_label'] for item in data['items']]
            overall_results = []
            overall_parsed_results = []
            for cell_name in input_names:
                best_term, best_score = search_numpy(cell_name, model, term_names, term_embeddings)
                overall_results.append((cell_name, best_term, best_score))
                overall_parsed_results.append((cell_name, best_term))
                #print(f"Processed cell name: {cell_name}, mapped name: {best_term}, score: {best_score}...")
            out_items = [{'original_label': original, 'mapped_label': mapped} for original, mapped in overall_parsed_results]
            output = {
                'title': data['title'],
                'rule_version': data['rule_version'],
                'items': out_items,
                'plm_raw_response': overall_results,
            }
            out_file_path = os.path.join(out_dir, file_name)
            with open(out_file_path, "w") as f:
                json.dump(output, f, indent=2)


if __name__ == '__main__':
    # model from PubMedBert
    #model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    term_texts, term_ids, term_names = load_cl_terms()
    #prepare_cl_term_embs(term_texts, model)
    #term_embeddings = torch.load("text_baselines/cl_term_embeddings_PubMedBERT.pt")

    # model from SapBERT
    model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    term_texts, term_ids, term_names = load_cl_terms()
    #prepare_cl_term_embs(term_texts, model, emb_out_path="text_baselines/cl_term_embeddings_sapbert.pt")
    term_embeddings = torch.load("text_baselines/cl_term_embeddings_sapbert.pt")
    term_embeddings = term_embeddings.cpu().numpy().astype('float32')
    term_embeddings = normalize(term_embeddings, axis=1, norm='l2')

    # PubmedBERT results
    #gen_result_stage2(data_dir="./text_baselines/all_journals_gpt55_thinking_20260711/stage2/", out_dir="./text_baselines/all_journals_gpt55_thinking_20260711_stage2_PubMedBert_results/", model=model, term_names=term_names, term_embeddings=term_embeddings)
    #gen_result_stage2(data_dir="./text_baselines/all_journals_deepseek_v4_pro_thinking_20260711/stage2/", out_dir="./text_baselines/all_journals_deepseek_v4_pro_thinking_20260711_stage2_PubMedBert_results/", model=model, term_names=term_names, term_embeddings=term_embeddings)

    # SapBERT results
    gen_result_stage2(data_dir="./text_baselines/all_journals_gpt55_thinking_20260711/stage2/", out_dir="./text_baselines/all_journals_gpt55_thinking_20260711_stage2_SapBERT_results/", model=model, term_names=term_names, term_embeddings=term_embeddings)
    gen_result_stage2(data_dir="./text_baselines/all_journals_deepseek_v4_pro_thinking_20260711/stage2/", out_dir="./text_baselines/all_journals_deepseek_v4_pro_thinking_20260711_stage2_SapBERT_results/", model=model, term_names=term_names, term_embeddings=term_embeddings)
