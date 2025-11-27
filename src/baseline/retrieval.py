import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
import os


class ImageRetriever:
    def __init__(self, text_embeddings, image_embeddings, texts, image_paths, device="cuda"):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.texts = texts
        self.image_paths = image_paths
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP模型加载完成!")

    def encode_text(self, text):
        """编码文本为向量"""
        if isinstance(text, str):
            text = [text]
        text_tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def compute_similarity(self, query_embedding, target_embeddings):
        """计算余弦相似度"""
        similarities = cosine_similarity(query_embedding.reshape(1, -1), target_embeddings)
        return similarities[0]

    def retrieve_images(self, query_text, top_k=5):
        """根据文本查询检索图像"""
        # 使用CLIP模型编码查询文本
        query_embedding = self.encode_text(query_text)

        # 计算相似度
        similarities = self.compute_similarity(query_embedding, self.image_embeddings)

        # 去重图片：保留每张图片最大相似度
        img_to_score = {}
        img_to_token = {}
        for idx, path in enumerate(self.image_paths):
            score = float(similarities[idx])
            if path not in img_to_score or score > img_to_score[path]:
                img_to_score[path] = score
                img_to_token[path] = self.texts[idx]

        # 按相似度排序
        sorted_items = sorted(img_to_score.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for path, score in sorted_items:
            results.append({
                'image_path': path,
                'similarity': score,  # 转换为Python float
                'token': img_to_token[path]
            })
        return results

    def evaluate_retrieval(self, k_values=[1, 5, 10]):
        """评估检索性能"""
        recalls = {f"Recall@{k}": [] for k in k_values}

        # 对于每个查询，检查正确图像是否在top-k中
        for i, query_text in enumerate(self.texts):
            results = self.retrieve_images(query_text, top_k=max(k_values))

            # 找到正确图像的排名
            correct_image_path = self.image_paths[i]
            ranked_paths = [result['image_path'] for result in results]

            for k in k_values:
                if correct_image_path in ranked_paths[:k]:
                    recalls[f"Recall@{k}"].append(1)
                else:
                    recalls[f"Recall@{k}"].append(0)

        # 计算平均召回率
        avg_recalls = {}
        for k in k_values:
            avg_recalls[f"Recall@{k}"] = np.mean(recalls[f"Recall@{k}"])

        return avg_recalls, recalls

    def print_evaluation_results(self, avg_recalls):
        """打印评估结果"""
        print("\n=== 检索性能评估结果 ===")
        for metric, score in avg_recalls.items():
            print(f"{metric}: {score:.4f} ({score * 100:.2f}%)")


# ======================== 测试代码 ========================
if __name__ == "__main__":
 # 加载已生成的向量文件
    save_path = "data/embedding"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # 加载图像和文本特征和路径
        image_embeddings = np.load(os.path.join(save_path, "clip_image_features.npy"))
        image_paths = np.load(os.path.join(save_path, "clip_image_items.npy"), allow_pickle=True)

        text_embeddings = np.load(os.path.join(save_path, "clip_text_features.npy"))
        texts = np.load(os.path.join(save_path, "clip_text_items.npy"), allow_pickle=True)

        print(f"加载的图片向量形状: {image_embeddings.shape}")
        print(f"加载的文本向量形状: {text_embeddings.shape}")
        print(f"图片数量: {len(image_paths)}")
        print(f"文本数量: {len(texts)}")

        # 确保数据类型一致
        image_embeddings = image_embeddings.astype(np.float32)
        text_embeddings = text_embeddings.astype(np.float32)

        # 初始化检索器
        retriever = ImageRetriever(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            texts=texts,
            image_paths=image_paths,
            device=device
        )

        print("\n=== 测试检索功能 ===")

        # 测试几个查询
        test_queries = [
            "a dog running",
            "a person walking",
            "a car on the road",
            "a group of people",
            "a child playing"
        ]

        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = retriever.retrieve_images(query, top_k=3)

            for i, result in enumerate(results):
                print(f"Rank {i + 1}: 相似度 {result['similarity']:.4f}")
                print(f"  描述: {result['token'][:50]}...")  # 只显示前50个字符
                print(f"  图片: {os.path.basename(result['image_path'])}")

        print("\n=== 测试评估功能 ===")
        # 使用小样本测试评估功能（避免时间太长）
        sample_size = min(50, len(texts))
        print(f"使用 {sample_size} 个样本进行快速评估...")

        # 创建小样本的检索器
        sample_retriever = ImageRetriever(
            text_embeddings=text_embeddings[:sample_size],
            image_embeddings=image_embeddings[:sample_size],
            texts=texts[:sample_size],
            image_paths=image_paths[:sample_size],
            device=device
        )

        # 运行评估
        avg_recalls, detailed_recalls = sample_retriever.evaluate_retrieval(k_values=[1, 5, 10])
        sample_retriever.print_evaluation_results(avg_recalls)

        print("\n=== 检索系统测试完成 ===")

    except FileNotFoundError as e:
        print(f"错误: 找不到向量文件 {e}")
        print("请先运行CLIP模型生成向量文件")
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()