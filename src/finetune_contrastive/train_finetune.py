import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from baseline.data import Flickr8kDataset
except ImportError:
    print("错误")
    sys.exit(1)


# Dataset 定义
class FinetuneDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        text = row['token']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # 如果图片读不出来，随机读下一张
            return self.__getitem__((idx + 1) % len(self))

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
        }


# Loss 函数
def contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size).to(logits_per_image.device)
    loss_img = nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2


# 画图
def plot_loss_curve(steps, losses, save_path="finetune_loss.png"):
    print(f"正在生成 Loss 曲线: {save_path} ...")
    plt.figure(figsize=(10, 6), dpi=100)

    plt.scatter(steps, losses, alpha=0.2, color='gray', s=5, label='Raw Step Loss')

    window_size = 20
    if len(losses) > window_size:
        smooth_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        smooth_steps = steps[window_size - 1:]
        plt.plot(smooth_steps, smooth_losses, color='#1f77b4', linewidth=2, label='Smoothed Trend')
    else:
        plt.plot(steps, losses, color='#1f77b4', linewidth=2, label='Loss')

    plt.title('CLIP Finetuning Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('InfoNCE Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss 曲线已保存！")

def main():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    batch_size = 32
    epochs = 3
    lr = 1e-5
    save_path = "./finetuned_model"

    # 测试集
    TEST_SIZE = 1000  # 保留最后 1000 条不参与训练

    print(f"正在使用设备: {device}")

    # 加载全量数据
    print("加载数据")
    img_dir, token_path = Flickr8kDataset.get_path()
    dataset_handler = Flickr8kDataset(img_dir, token_path)
    df = dataset_handler.load_dataset()
    df = dataset_handler.validate_dataset(df)

    print(f"原始数据总量: {len(df)}")

    # 剔除测试集数据
    # df.iloc[:-1000] 取从头开始直到倒数第1000个
    train_df = df.iloc[:-TEST_SIZE]

    print(f"-" * 30)
    print(f"训练集数量: {len(train_df)} ")
    print(f"测试集数量: {TEST_SIZE} (用于后续 retrieval 评估)")
    print(f"-" * 30)

    # 初始化模型
    processor = CLIPProcessor.from_pretrained(model_name)
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    except:
        print("Safetensors 加载失败")
        model = CLIPModel.from_pretrained(model_name).to(device)

    # 冻结参数 (只训练 Projection 层)
    for name, param in model.named_parameters():
        if "projection" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 准备 DataLoader (只使用 train_df)
    train_ds = FinetuneDataset(train_df, processor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 记录绘图数据
    history_steps = []
    history_losses = []
    global_step = 0

    # 训练循环
    model.train()
    print("开始训练")

    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )

            loss = contrastive_loss(outputs.logits_per_image, outputs.logits_per_text)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss

            history_steps.append(global_step)
            history_losses.append(current_loss)
            global_step += 1

            if step % 50 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {current_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} 完成, 平均 Loss: {avg_loss:.4f} ===")

    # 保存模型
    print(f"正在保存模型到 {save_path}")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    # 画图
    plot_loss_curve(history_steps, history_losses, save_path="finetune_loss.png")

if __name__ == "__main__":

    main()
