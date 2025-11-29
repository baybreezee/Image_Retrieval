import os
import sys
import torch
import numpy as np
import math  # 新增数学库用于计算行数
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- 配置部分 ---
EMBEDDING_DIR = "data/embedding_finetuned"
MODEL_PATH = "./finetuned_model"
SAMPLE_SIZE = 1000


# ----------------

class FinetunedRetriever:
    def __init__(self, text_embeddings, image_embeddings, texts, image_paths, model_path, device="cuda"):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.texts = texts
        self.image_paths = image_paths
        self.device = device

        print(f"正在加载微调后的模型用于查询: {model_path} ...")
        try:
            self.model = CLIPModel.from_pretrained(model_path, use_safetensors=True).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        except:
            print("Safetensors加载失败，尝试默认加载...")
            self.model = CLIPModel.from_pretrained(model_path).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def retrieve_images(self, query_text, top_k=5):
        query_embedding = self.encode_text(query_text)
        similarities = cosine_similarity(query_embedding, self.image_embeddings).flatten()

        # 取更多候选以防去重后不够
        candidate_indices = similarities.argsort()[::-1][:top_k * 10]

        results = []
        seen_paths = set()

        for idx in candidate_indices:
            path = self.image_paths[idx]
            if path not in seen_paths:
                results.append({
                    "image_path": path,
                    "token": self.texts[idx],
                    "similarity": similarities[idx]
                })
                seen_paths.add(path)

            if len(results) >= top_k:
                break

        return results

    def evaluate_retrieval(self, k_values=[1, 5, 10]):
        print(f"\n开始评估 Recall@{k_values} (基于图片路径匹配)...")
        sim_matrix = cosine_similarity(self.text_embeddings, self.image_embeddings)
        num_samples = len(self.text_embeddings)
        recalls = {k: 0 for k in k_values}
        max_k = max(k_values)

        for i in range(num_samples):
            correct_path = self.image_paths[i]
            scores = sim_matrix[i]
            sorted_indices = scores.argsort()[::-1]
            top_k_indices = sorted_indices[:max_k]
            top_k_paths = [self.image_paths[idx] for idx in top_k_indices]

            for k in k_values:
                if correct_path in top_k_paths[:k]:
                    recalls[k] += 1

        for k in k_values:
            recalls[k] = recalls[k] / num_samples
            print(f"Recall@{k}: {recalls[k]:.4f} ({recalls[k] * 100:.2f}%)")
        return recalls


def visualize_search_results(query, results, save_name=None):
    """画出 Query 和 Top-K 图片 (支持多行显示)"""
    top_k = len(results)

    # 动态计算布局：每行最多5张
    cols = 5
    rows = math.ceil(top_k / cols)

    # 画布大小随行数变化
    plt.figure(figsize=(15, 3.5 * rows))
    plt.suptitle(f"Finetuned Query: '{query}'", fontsize=16, fontweight='bold', y=0.98)

    for i, res in enumerate(results):
        img_path = res['image_path']
        score = res['similarity']

        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Rank {i + 1}\nSim: {score:.4f}", color='darkgreen', fontsize=10)
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
        print(f"  [图片已保存]: {save_name}")
    try:
        plt.show()
    except:
        pass


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("正在加载向量数据...")
    try:
        img_embeds = np.load(os.path.join(EMBEDDING_DIR, "clip_image_features.npy"))
        txt_embeds = np.load(os.path.join(EMBEDDING_DIR, "clip_text_features.npy"))
        img_paths = np.load(os.path.join(EMBEDDING_DIR, "image_paths.npy"))
        texts = np.load(os.path.join(EMBEDDING_DIR, "texts.npy"))
    except FileNotFoundError:
        print("❌ 错误：找不到 Finetuned 向量文件。请先运行 generate_vectors.py")
        return

    test_img_embeds = img_embeds[-SAMPLE_SIZE:]
    test_txt_embeds = txt_embeds[-SAMPLE_SIZE:]
    test_img_paths = img_paths[-SAMPLE_SIZE:]
    test_texts = texts[-SAMPLE_SIZE:]

    retriever = FinetunedRetriever(
        test_txt_embeds, test_img_embeds, test_texts, test_img_paths, MODEL_PATH, device
    )

    queries = ["a dog running", "a car on the road", "children playing"]
    print("\n=== 案例检索演示 (Top 10) ===")

    os.makedirs("results_vis", exist_ok=True)

    for i, q in enumerate(queries):
        print(f"\n查询: '{q}'")
        # --- 这里改成 top_k=10 ---
        res = retriever.retrieve_images(q, top_k=10)

        for r_idx, r in enumerate(res):
            print(f"Rank {r_idx + 1}: {r['similarity']:.4f} | {os.path.basename(r['image_path'])}")

        visualize_search_results(q, res, save_name=f"results_vis/query_{i}_top10.png")

    print(f"\n=== 使用 {SAMPLE_SIZE} 个样本进行量化评估 ===")
    retriever.evaluate_retrieval()


if __name__ == "__main__":
    main()