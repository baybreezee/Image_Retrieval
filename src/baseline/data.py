import os
import pandas as pd
from PIL import Image
import torch



class FlickrDataset:
    def __init__(self, data_root="./data"):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "archive", "Images")

    def load_dataset(self):
        """加载数据集"""
        print("正在加载数据集...")
        """扫描 Images 文件夹，加载所有 jpg 图片"""
        if not os.path.exists(self.images_dir):
            print(f"图像目录不存在: {self.images_dir}")
            return None

        # 获取所有 jpg 文件
        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(".jpg")]

        if not image_files:
            print("Images 文件夹中没有 jpg 图片")
            return None

        # 加载标注文件
        if os.path.exists(self.captions_file):
            try:
                df = pd.read_csv(self.captions_file)
                print(f"成功加载标注文件，共 {len(df)} 条数据")
                return df
            except Exception as e:
                print(f"加载标注文件失败: {e}")
        else:
            print(f"标注文件不存在: {self.captions_file}")

    def get_image_path(self, image_name):
        """获取图像完整路径"""
        image_path = os.path.join(self.images_dir, image_name)

        # 如果图像文件不存在，返回None
        if not os.path.exists(image_path):
            # 尝试其他可能的扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
                alt_path = os.path.join(self.images_dir, os.path.splitext(image_name)[0] + ext)
                if os.path.exists(alt_path):
                    return alt_path
            print(f"图像文件不存在: {image_path}")
            return None

        return image_path

    def validate_dataset(self, df):
        """验证数据集，移除不存在的图像"""
        valid_data = []
        missing_images = []

        for idx, row in df.iterrows():
            image_path = self.get_image_path(row['image'])
            if image_path and os.path.exists(image_path):
                valid_data.append({
                    'image': row['image'],
                    'caption': row['caption'],
                    'image_path': image_path
                })
            else:
                missing_images.append(row['image'])

        if missing_images:
            print(f"发现 {len(missing_images)} 个缺失的图像文件")

        valid_df = pd.DataFrame(valid_data)
        print(f"有效数据: {len(valid_df)} 条")
        return valid_df

if __name__ == "__main__":
    # 初始化数据集对象
    dataset = FlickrDataset(data_root="./data")

    # 1. 加载数据集（会自动创建示例数据，如果真实数据不存在）
    df = dataset.load_dataset()
    print("原始数据：")
    print(df.head())

    # 2. 验证数据集（检查图片是否存在，移除缺失的条目）
    valid_df = dataset.validate_dataset(df)
    print("验证后的数据：")
    print(valid_df.head())

    # 3. 测试获取图片完整路径
    for img_name in valid_df['image'].tolist():
        img_path = dataset.get_image_path(img_name)
        print(f"{img_name} -> {img_path}")

    # 4. 可选：尝试用Pillow加载第一张图片
    first_image_path = valid_df['image_path'].iloc[0]
    if first_image_path:
        image = Image.open(first_image_path)
        print(f"第一张图片大小: {image.size}")
        image.show()  # 会弹出图片窗口
