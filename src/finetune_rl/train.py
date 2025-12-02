# # ==================== train_rl.py (RL-inspired Rank-Aware Adapter) =====================

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import clip

from .reward import compute_rank_and_reward
from .utils import load_embeddings, recall_at_k


# ---------------- dataset ----------------
class TextImageRLDataset(Dataset):
    def __init__(self, text_items):
        self.text_items = list(text_items)

    def __len__(self):
        return len(self.text_items)

    def __getitem__(self, idx):
        # idx 同时作为 “正确图片” 在 image_embs 中的索引
        return idx, self.text_items[idx]


# ---------------- small adapter (只训练它) ----------------
class RankAdapter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        # 初始化为接近恒等映射，避免一开始就把空间搞坏
        nn.init.eye_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


# ---------------- seed ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- main train ----------------
def main():
    set_seed(42)
    
    # ===== 超参区域 =====
    NUM_EPOCHS = 10          
    LR = 5e-5               
    ALPHA = 0.5             
    L2_LAMBDA = 1e-4        

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) 加载预先算好的 image_embeddings & 文本列表
    embedding_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../src/baseline/data/embedding")
    )
    image_embeddings_np, image_paths, text_items = load_embeddings(embedding_dir)
    print(f"载入 image_embeddings: {image_embeddings_np.shape}, 文本数量: {len(text_items)}")

    # [N, D]
    image_embs = torch.from_numpy(image_embeddings_np).float().to(device)
    image_embs.requires_grad_(False)
    feat_dim = image_embs.shape[1]

    # 2) Dataset & DataLoader
    dataset = TextImageRLDataset(text_items)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) 冻结 CLIP，只用它做 feature extractor
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # 4) 只训练一个小 adapter（不会动 CLIP 原始空间）
    adapter = RankAdapter(dim=feat_dim).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)
    temperature = 0.1
    num_epochs = NUM_EPOCHS
    alpha = ALPHA

    identity = torch.eye(feat_dim, device=device)

    print("\n===== Start RL-inspired Rank-Aware Training (Adapter Only) =====")

    for epoch in range(num_epochs):
        adapter.train()
        epoch_losses = []
        epoch_weights = []

        pbar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{num_epochs}]")

        for indices, texts in pbar:
            indices = indices.to(device)  # [B]

            # --- 1) 用 CLIP 提取文本特征（不求导，保证稳定） ---
            with torch.no_grad():
                tokens = clip.tokenize(list(texts)).to(device)    # [B, L]
                text_features = clip_model.encode_text(tokens)    # [B, D]
                text_features = text_features.float() 

            # --- 2) 通过 adapter 做轻微变换（只训这个） ---
            text_features = adapter(text_features)                # [B, D]

            # 归一化，避免数值爆炸，加入 eps 防止除 0
            norms = text_features.norm(dim=-1, keepdim=True) + 1e-8
            text_features = text_features / norms

            # --- 3) 与所有图片算相似度 & log_softmax ---
            sims = text_features @ image_embs.T                  # [B, N]
            logits = sims / temperature
            log_probs = F.log_softmax(logits, dim=-1)            # [B, N]

            # --- 4) 逐样本计算 rank-aware 加权 CE ---
            loss_sum = 0.0
            batch_ce = []
            batch_w = []

            B = indices.size(0)
            for i in range(B):
                true_idx = indices[i].item()          # 正确图片在 image_embs 中的索引

                sim_row = sims[i]                     # [N]
                logprob_row = log_probs[i]            # [N]
                log_prob_true = logprob_row[true_idx] # scalar

                # rank & inv_rank (reward=1/rank)
                rank, inv_rank = compute_rank_and_reward(sim_row, true_idx)

                # 基本 CE：-log P(true)
                ce_i = -log_prob_true                 # scalar tensor

                # rank-aware 权重：rank 越小(越前)，inv_rank 越大，weight 越大
                weight = 1.0 + alpha * inv_rank       # scalar tensor, 介于 [1, 1+alpha]

                loss_sum += weight * ce_i

                batch_ce.append(ce_i.item())
                batch_w.append(weight.item())

            loss = loss_sum / B

            # L2 正则：让 adapter.weight 接近恒等矩阵
            reg = ((adapter.linear.weight - identity) ** 2).mean()
            total_loss = loss + L2_LAMBDA * reg

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_weights.extend(batch_w)

            pbar.set_postfix(
                loss=f"{np.mean(epoch_losses):.4f}",
                avg_w=f"{np.mean(epoch_weights):.3f}"
            )

        print(f">>> Epoch {epoch+1} | Avg Loss={np.mean(epoch_losses):.4f}, Avg Weight={np.mean(epoch_weights):.3f}\n")

    # 5) 训练结束后，用 adapter 重新编码所有文本并保存
    print("\n===== Encoding text features with RL-inspired Adapter =====")
    adapter.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(text_items), batch_size):
            batch = text_items[i:i + batch_size]
            tokens = clip.tokenize(batch).to(device)
            text_features = clip_model.encode_text(tokens)
            text_features = text_features.float()
            text_features = adapter(text_features)
            norms = text_features.norm(dim=-1, keepdim=True) + 1e-8
            text_features = text_features / norms
            all_embs.append(text_features)

    all_text_embs = torch.cat(all_embs, dim=0).float()  # [N, D]
    save_path = os.path.join(embedding_dir, "clip_text_features_rl.npy")
    np.save(save_path, all_text_embs.cpu().numpy())
    print("保存 RL-inspired 文本向量到:", save_path)

    # 6) 用 RL-inspired 文本向量做一次 Recall@K 评估
    print("\n===== Evaluating Recall@K (Adapter Text vs. Original Image Embeddings) =====")
    avg_recalls = recall_at_k(all_text_embs, image_embs, k_list=(1, 5, 10))
    for k, v in avg_recalls.items():
        print(f"Recall@{k}: {v:.4f} ({v * 100:.2f}%)")

    print("\n🎉 RL-inspired 训练完成！模型和向量已保存，可以进行检索对比测试。")


if __name__ == "__main__":
    main()
