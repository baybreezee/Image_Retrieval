import os
import glob
import pandas as pd
from PIL import Image

class Flickr8kDataset:
    def __init__(self, image_path, token_path):
        """
        images_dir: 图片文件夹路径
        caption_file: captions 文件路径（Flickr8k.token.txt）
        """
        self.image_path = image_path
        self.token_path = token_path

    def load_dataset(self):
        """加载 captions 文件"""
        if not os.path.exists(self.token_path):
            raise FileNotFoundError(f"标注文件不存在: {self.token_path}")
        # Flickr8k.token.txt 用空格或制表符分隔
        df = pd.read_csv(self.token_path, sep='\t', names=['image', 'caption'], header=None)
        # 去掉 image 后面的 #0, #1 等
        df['image'] = df['image'].apply(lambda x: x.split('#')[0])

        return df

    def get_all_image_paths(self):
        """获取所有本地图片路径"""
        paths = glob.glob(os.path.join(self.image_path, "*.jpg"))
        return {os.path.basename(p): p for p in paths}

    def validate_dataset(self, df):
        """过滤掉不存在的图片，并添加 image_path 列"""
        image_dict = self.get_all_image_paths()
        df = df[df['image'].isin(image_dict.keys())].copy()
        df['image_path'] = df['image'].apply(lambda x: image_dict[x])
        return df


if __name__ == "__main__":
    import os
    print(os.getcwd())

    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    image_path = os.path.join(directory, "Flickr8K_Dataset/Flicker8k_Dataset")
    token_path = os.path.join(directory, "Flickr8k_text/Flickr8k.token.txt")

    dataset = Flickr8kDataset(image_path, token_path)
    df = dataset.load_dataset()
    df = dataset.validate_dataset(df)
    print(df.head())
