# src/fintune_rl/utils.py
import os
import numpy as np
import torch

def load_embeddings(embedding_dir="data/embedding"):
    """
    加载 baseline 预先算好的 image_embeddings 和 text_items（文本）
    我们在 RL 训练中：
    - image_embeddings 作为固定环境（不训练）
    - 文本从 text_items 重新走 CLIP text encoder（有梯度）
    """
    image_embeddings = np.load(os.path.join(embedding_dir, "clip_image_features.npy"))
    image_paths = np.load(os.path.join(embedding_dir, "clip_image_items.npy"), allow_pickle=True)
    text_items = np.load(os.path.join(embedding_dir, "clip_text_items.npy"), allow_pickle=True)

    # 统一成 float32
    image_embeddings = image_embeddings.astype(np.float32)

    return image_embeddings, image_paths, text_items


def recall_at_k(all_text_embs: torch.Tensor,
                all_image_embs: torch.Tensor,
                k_list=(1, 5, 10)):
    """
    使用 *向量* 直接计算 Recall@K（不再走 encode_text）
    all_text_embs: [N, D]
    all_image_embs: [N, D]  （注意这里我们假设和 baseline 一样：第 i 个 text 对应第 i 个 image）
    """
    device = all_text_embs.device
    N = all_text_embs.size(0)
    # 归一化，保证是余弦相似度
    text_norm = all_text_embs / all_text_embs.norm(dim=-1, keepdim=True)
    img_norm = all_image_embs / all_image_embs.norm(dim=-1, keepdim=True)

    # [N, N] 相似度矩阵
    sims = text_norm @ img_norm.T

    recalls = {f"Recall@{k}": [] for k in k_list}

    for i in range(N):
        sim_row = sims[i]  # [N]
        # 排序得到从大到小的索引
        ranked_indices = torch.argsort(sim_row, descending=True)

        for k in k_list:
            topk = ranked_indices[:k]
            if i in topk:
                recalls[f"Recall@{k}"].append(1)
            else:
                recalls[f"Recall@{k}"].append(0)

    avg_recalls = {}
    for k in k_list:
        avg_recalls[f"Recall@{k}"] = float(np.mean(recalls[f"Recall@{k}"]))

    return avg_recalls
