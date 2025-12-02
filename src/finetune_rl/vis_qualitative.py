# src/finetune_rl/vis_qualitative.py

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# --------- 1. 载入所有 embedding 和路径 ---------
def load_all():
    # baseline 这边的 embedding 目录
    embedding_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../baseline/data/embedding")
    )

    img_embs = np.load(os.path.join(embedding_dir, "clip_image_features.npy"))
    txt_base = np.load(os.path.join(embedding_dir, "clip_text_features.npy"))
    txt_rl   = np.load(os.path.join(embedding_dir, "clip_text_features_rl.npy"))

    # 原来保存的图片路径（里面是老项目的绝对路径）
    raw_img_paths = np.load(
        os.path.join(embedding_dir, "clip_image_items.npy"),
        allow_pickle=True
    )
    texts = np.load(
        os.path.join(embedding_dir, "clip_text_items.npy"),
        allow_pickle=True
    )

    # ==== 关键：把旧路径修成当前项目的路径 ====
    # 当前文件:   <project>/src/finetune_rl/vis_qualitative.py
    # 项目根目录: <project>/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 你现在的数据目录结构：data/Flickr8k_Dataset/Flicker8k_Dataset/*.jpg
    correct_img_dir = os.path.join(
        project_root, "data", "Flickr8k_Dataset", "Flicker8k_Dataset"
    )

    fixed_paths = []
    for p in raw_img_paths:
        filename = os.path.basename(str(p))          # 只要文件名
        new_path = os.path.join(correct_img_dir, filename)
        fixed_paths.append(new_path)

    img_paths = np.array(fixed_paths)

    return img_embs, txt_base, txt_rl, img_paths, texts, embedding_dir


# --------- 2. 给定 caption index，用 text embedding 做 Top-K 检索 ---------
def retrieve_topk_for_index(idx, text_embs, img_embs, k=10):
    """
    idx: 文本在数据集中的索引（0 ~ N-1）
    text_embs: [N, D] 里的某一行作为 query
    img_embs:  [N, D]
    """
    q = text_embs[idx:idx + 1]              # [1, D]
    sims = (q @ img_embs.T).reshape(-1)     # [N]
    order = sims.argsort()[::-1][:k]        # 从大到小取前 K
    return order, sims[order]


# --------- 3. 画一张 “Baseline vs Adapter” 的对比图 ---------
def plot_case(idx, caption, img_paths,
              base_idx, base_sims,
              rl_idx, rl_sims,
              out_dir="results_vis_adapter",
              k=10):
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Caption #{idx}: {caption}", fontsize=11)

    # 第一行：Baseline
    for j, (img_i, sim) in enumerate(zip(base_idx, base_sims)):
        ax = plt.subplot(2, k, j + 1)
        img = Image.open(img_paths[img_i]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"B#{j+1}\n{sim:.3f}", fontsize=7)

    # 第二行：RL-inspired Adapter
    for j, (img_i, sim) in enumerate(zip(rl_idx, rl_sims)):
        ax = plt.subplot(2, k, k + j + 1)
        img = Image.open(img_paths[img_i]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"A#{j+1}\n{sim:.3f}", fontsize=7)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"case_{idx}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# --------- 4. 主函数：选几个 caption 做可视化 ---------
def main():
    img_embs, txt_base, txt_rl, img_paths, texts, _ = load_all()

    # 这里随便举几个 index，你可以自己改成你想看的样本
    # 可以后面改成：找包含 "car" / "children" 等关键词的 index
    case_indices = [10, 123, 500]

    for idx in case_indices:
        caption = str(texts[idx])

        base_idx, base_sims = retrieve_topk_for_index(idx, txt_base, img_embs, k=10)
        rl_idx, rl_sims     = retrieve_topk_for_index(idx, txt_rl,   img_embs, k=10)

        plot_case(
            idx, caption, img_paths,
            base_idx, base_sims,
            rl_idx, rl_sims,
            out_dir="results_vis_adapter",
            k=10
        )


if __name__ == "__main__":
    main()
