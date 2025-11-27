import os
import torch
import numpy as np
from PIL import Image
import clip
from data import Flickr8kDataset


class CLIPModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP模型加载完成!")

    def encode_text(self, text):
        """编码文本为向量"""
        if isinstance(text, str):
            text = [text]

        text_token = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_token)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return text_feature.cpu().numpy().astype(np.float32)  # 确保返回float32

    def encode_image(self, image_path):
        """编码单张图片为向量"""
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def batch_encode(self, item, batch_size=32, mode="image", save_path=None):
        """
        统一批量编码函数
        item: 图片路径列表 或 文本列表
        mode: "image" 或 "text"
        batch_size: 批量大小
        save_path: 保存向量文件的目录
        """
        all_features = []
        valid_items = []

        for i in range(0, len(item), batch_size):
            batch = item[i:i + batch_size]
            batch_features = []

            if mode == "image":
                for path in batch:
                    features = self.encode_image(path)
                    if features is not None:
                        batch_features.append(features)
                        valid_items.append(path)
            elif mode == "text":
                features = self.encode_text(batch)
                batch_features.append(features)
                valid_items.extend(batch)
            else:
                raise ValueError("mode 必须是 'image' 或 'text'")

            if batch_features:
                all_features.append(np.vstack(batch_features))

            print(f"已处理 {min(i + batch_size, len(item))}/{len(item)} {mode}")

        if all_features:
            features_array = np.vstack(all_features)
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, f"clip_{mode}_features.npy"), features_array)
                np.save(os.path.join(save_path, f"clip_{mode}_items.npy"), np.array(valid_items))
                print(f"{mode}向量已保存到 {save_path}")
            return features_array, valid_items
        else:
            return np.array([]), []

    @staticmethod
    def load_features(save_path, mode="image"):
        """加载已经编码好的向量"""
        features = np.load(os.path.join(save_path, f"clip_{mode}_features.npy"))
        items = np.load(os.path.join(save_path, f"clip_{mode}_items.npy"), allow_pickle=True)
        return features, items


if __name__ == "__main__":
    # 图片路径
    image_path, token_path = Flickr8kDataset.get_path()
    dataset = Flickr8kDataset(image_path, token_path)
    df = dataset.load_dataset()
    df = dataset.validate_dataset(df)
    image = df['image_path'].tolist()

    # 文本
    token = df['token'].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel(device=device)

    save_path = "data/embedding"

    # 批量编码图片
    image_features, valid_image_paths = clip_model.batch_encode(image, batch_size=100, mode="image",
                                                                save_path=save_path)

    # 批量编码文本
    text_features, valid_texts = clip_model.batch_encode(token, batch_size=100, mode="text", save_path=save_path)

    print("图片编码完成:", len(valid_image_paths))
    print("文本编码完成:", len(valid_texts))
