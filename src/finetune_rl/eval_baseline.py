# src/finetune_rl/eval_baseline.py
import os
import numpy as np
import torch

from .utils import load_embeddings, recall_at_k

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    embedding_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/baseline/data/embedding")
    )

    # 1) 加载 image_embeddings + 文本列表
    image_embeddings_np, image_paths, text_items = load_embeddings(embedding_dir)
    print(f"载入 image_embeddings: {image_embeddings_np.shape}, 文本数量: {len(text_items)}")

    # 2) 加载“原始 CLIP 文本向量”（clip_process.py 生成的那份）
    text_embeddings_np = np.load(os.path.join(embedding_dir, "clip_text_features.npy"))
    print(f"载入 text_embeddings: {text_embeddings_np.shape}")

    # 转成 torch
    image_embs = torch.from_numpy(image_embeddings_np).float().to(device)
    text_embs = torch.from_numpy(text_embeddings_np).float().to(device)

    # 3) 统一归一化（跟 RL 那边一样）
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # 4) 评估 Recall@K
    print("\n===== Baseline (Zero-shot CLIP) Recall@K =====")
    avg_recalls = recall_at_k(text_embs, image_embs, k_list=(1, 5, 10))
    for k, v in avg_recalls.items():
        print(f"Recall@{k}: {v:.4f} ({v * 100:.2f}%)")

if __name__ == "__main__":
    main()
