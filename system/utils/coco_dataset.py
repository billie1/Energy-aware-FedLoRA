# utils/coco_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class COCODetectionDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        """
        root/
          ├── train/
          │   ├── *.jpg
          │   └── _annotations.coco.json
          ├── test/
        """
        self.root = root
        self.split = split
        self.transforms = transforms

        ann_path = os.path.join(root, split, "_annotations.coco.json")
        img_dir = os.path.join(root, split)

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.images = coco["images"]
        self.annotations = coco.get("annotations", [])

        # image_id → annotations
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        # category id mapping（你这个数据集本身就是从 0 开始，其实可以直接用）
        self.cat_id_map = {
            cat["id"]: idx
            for idx, cat in enumerate(coco["categories"])
        }

        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        anns = self.img_to_anns.get(img_id, [])

        # ===== 核心修复点 =====
        # 始终返回 Tensor，dtype=torch.long
        if len(anns) == 0:
            label = torch.tensor(0, dtype=torch.long)
        else:
            label_id = anns[0]["category_id"]
            label = torch.tensor(
                self.cat_id_map[label_id],
                dtype=torch.long
            )
        # =====================

        return {
            "image": image,
            "label": label
        }
