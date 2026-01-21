# utils/dataset.py

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pyarrow.parquet as pq
import torchvision.transforms as T
import numpy as np


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


class CityscapesSegmentationDataset(Dataset):
    def __init__(self, parquet_files, transforms=None):
        """
        parquet_files: List[str]
        """
        self.tables = []
        self.cum_sizes = []
        self.target_size = (224, 224)

        total = 0
        for f in parquet_files:
            table = pq.read_table(f)
            self.tables.append(table)
            total += table.num_rows
            self.cum_sizes.append(total)

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])

    def __len__(self):
        return self.cum_sizes[-1]

    def _locate_table(self, idx):
        for i, size in enumerate(self.cum_sizes):
            if idx < size:
                prev = 0 if i == 0 else self.cum_sizes[i - 1]
                return i, idx - prev
        raise IndexError

    def __getitem__(self, idx):
        table_idx, row_idx = self._locate_table(idx)
        # 读取一行数据
        row = self.tables[table_idx].slice(row_idx, 1).to_pydict()

        # ---------------------------------------------------------------------
        # 1. 处理图像 (Key: "image")
        # 数据集格式: list<list<list<float>>> 形状 (3, 128, 256)
        # ---------------------------------------------------------------------
        img_data = np.array(row["image"][0])  # 变成 numpy 数组: [3, 128, 256], float

        # 步骤 A: 转换维度 [3, H, W] -> [H, W, 3] 以适应 PIL
        img_data = img_data.transpose(1, 2, 0) 

        # 步骤 B: 处理数值范围和类型
        # 如果数据是 0.0-1.0 的 float，需要乘 255 转 uint8
        # 如果数据已经是 0-255 的 float，直接转 uint8
        if img_data.max() <= 1.5:
            img_data = (img_data * 255).astype(np.uint8)
        else:
            img_data = img_data.astype(np.uint8)
        
        image = Image.fromarray(img_data)
        
        # 应用 Transforms (Resize 等)
        image = self.transforms(image)

        # ---------------------------------------------------------------------
        # 2. 处理标签 (Key: "segmentation_2")
        # 我们需要 19 类分割任务，所以必须取 "segmentation_2" 列
        # 数据集格式: list<list<float>> 形状 (128, 256)
        # ---------------------------------------------------------------------
        mask_data = np.array(row["segmentation_2"][0]) # [128, 256]
        
        # 转为 PIL 以便进行 Resize
        mask_img = Image.fromarray(mask_data.astype(np.uint8))

        # 关键: Resize 必须用 NEAREST (最近邻)，保证类别ID不变
        mask_img = mask_img.resize(self.target_size, resample=Image.NEAREST)

        # 转回 Tensor (Long类型用于 CrossEntropyLoss)
        mask = torch.from_numpy(np.array(mask_img)).long()

        return {
            "image": image,        # Tensor [3, 224, 224]
            "label": mask          # Tensor [224, 224] (注意这里叫 label 对应你训练代码)
        }



class TrafficSignClassificationDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        """
        root/
          ├── train/
          ├── test/
          ├── TsignRecgTrainAnnotation.txt
          ├── TsignRecgTestAnnotation.txt
        """
        self.root = root
        self.split = split

        if split == "train":
            ann_file = "TsignRecgTrainAnnotation.txt"
            img_dir = "train"
        else:
            ann_file = "TsignRecgTestAnnotation.txt"
            img_dir = "test"

        self.img_dir = os.path.join(root, img_dir)
        self.ann_path = os.path.join(root, ann_file)

        self.samples = self._load_annotations()

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def _load_annotations(self):
        samples = []
        with open(self.ann_path, "r") as f:
            for line in f:
                line = line.strip().strip(";")
                fields = line.split(";")
                # filename;W;H;x1;y1;x2;y2;class
                filename = fields[0]
                x1, y1, x2, y2 = map(int, fields[3:7])
                label = int(fields[7])

                samples.append({
                    "filename": filename,
                    "bbox": (x1, y1, x2, y2),
                    "label": label
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample["filename"])

        image = Image.open(img_path).convert("RGB")

        # ✨ 裁剪 traffic sign 区域（非常关键）
        x1, y1, x2, y2 = sample["bbox"]
        image = image.crop((x1, y1, x2, y2))

        image = self.transforms(image)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": image,
            "label": label
        }