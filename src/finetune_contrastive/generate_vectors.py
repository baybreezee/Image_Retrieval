import os
import sys
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baseline.data import Flickr8kDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载Finetune好的模型
    model_path = "./finetuned_model"

    print(f"正在加载微调后的模型: {model_path} ...")
    try:
        model = CLIPModel.from_pretrained(model_path, use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"加载失败")
        model = CLIPModel.from_pretrained(model_path).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)

    # 准备数据
    img_dir, token_path = Flickr8kDataset.get_path()
    dataset_handler = Flickr8kDataset(img_dir, token_path)
    df = dataset_handler.load_dataset()
    df = dataset_handler.validate_dataset(df)

    # 确保顺序
    image_paths = df['image_path'].tolist()
    texts = df['token'].tolist()

    print(f"准备处理 {len(df)} 条数据...")

    # 提取特征
    batch_size = 64
    image_embeddings = []
    text_embeddings = []

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="生成向量"):
            batch_img_paths = image_paths[i: i + batch_size]
            batch_texts = texts[i: i + batch_size]

            # 读取图片
            images = []
            valid_indices = []
            for idx, p in enumerate(batch_img_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except:
                    print(f"跳过损坏图片: {p}")

            if not images: continue

            # 处理输入
            inputs = processor(
                text=batch_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)

            # 获取特征
            outputs = model(**inputs)

            # 归一化特征
            img_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

            image_embeddings.append(img_embeds.cpu().numpy())
            text_embeddings.append(txt_embeds.cpu().numpy())

    # 合并并保存
    final_img_embeds = np.concatenate(image_embeddings, axis=0)
    final_txt_embeds = np.concatenate(text_embeddings, axis=0)

    # 保存
    output_dir = "data/embedding_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "clip_image_features.npy"), final_img_embeds)
    np.save(os.path.join(output_dir, "clip_text_features.npy"), final_txt_embeds)

    np.save(os.path.join(output_dir, "image_paths.npy"), np.array(image_paths))
    np.save(os.path.join(output_dir, "texts.npy"), np.array(texts))

    print(f"新向量已保存到: {output_dir}")
    print(f"图片向量形状: {final_img_embeds.shape}")


if __name__ == "__main__":

    main()
