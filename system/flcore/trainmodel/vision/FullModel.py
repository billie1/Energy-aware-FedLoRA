import torch.nn as nn

from .model import DetectionHead, SegmentationHead, ClassificationHead

class FullModel(nn.Module):
    def __init__(self, vit_model, task, num_det_classes=None, num_seg_classes=None, num_cls=None):
        super().__init__()

        self.vit = vit_model
        # -------------------------------------------

        # 获取 hidden_dim
        if hasattr(vit_model, 'config'):
            hidden_dim = vit_model.config.hidden_size
        else:
            hidden_dim = getattr(vit_model, 'embed_dim', 768)

        # 初始化 Heads
        if task == "detection":
            self.det_head = DetectionHead(hidden_dim, num_det_classes)
        elif task == "segmentation":
            self.seg_head = SegmentationHead(hidden_dim, num_seg_classes)
        elif task == "classification":
            self.cls_head = ClassificationHead(hidden_dim, num_cls)
        else:
            raise ValueError(f"Unknown task {task}")

    def forward(self, images, task, H=None, W=None):
        
        # 1. 运行 ViT
        outputs = self.vit(images)
        features = outputs.last_hidden_state

        # 2. 运行任务头
        if task == "detection":
            return self.det_head(features)
        elif task == "segmentation":
            return self.seg_head(features, H, W)
        elif task == "classification":
            return self.cls_head(features)
        else:
            raise ValueError(f"Unknown task {task}")