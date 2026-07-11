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
    
    return term_texts[best_idx], best_score


if __name__ == '__main__':
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    term_texts, term_ids, term_names = load_cl_terms()
    #prepare_cl_term_embs(term_texts, model)
    
    # load embeddings
    term_embeddings = torch.load("text_baselines/cl_term_embeddings.pt")
    term_embeddings = term_embeddings.cpu().numpy().astype('float32')
    term_embeddings = normalize(term_embeddings, axis=1, norm='l2')
    best_term, best_score = search_numpy("follicular helper T (TFH) cells (CD4+ T cells:CXCL13/c3)", model, term_names, term_embeddings)
    print(f"Best term: {best_term}, Score: {best_score}")
    # TODO: finish all