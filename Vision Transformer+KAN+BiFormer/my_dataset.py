from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_label: list, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform
        # 过滤非RGB图片
        self.images_path, self.images_label = self._filter_and_convert()

    def _filter_and_convert(self):
        filtered_paths = []
        filtered_labels = []
        for path, label in zip(self.images_path, self.images_label):
            try:
                img = Image.open(path)
                if img.mode in ['RGB', 'RGBA']:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    filtered_paths.append(path)
                    filtered_labels.append(label)
            except Exception as e:
                print(f"Warning: Failed to load image {path}. Error: {e}")
        return filtered_paths, filtered_labels

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} isn't RGB mode.")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images_label[item]

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels