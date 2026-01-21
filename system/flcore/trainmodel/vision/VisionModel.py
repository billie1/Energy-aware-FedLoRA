import torch
import torch.nn as nn

from .model import LinearWithLoRA, DetectionHead, SegmentationHead, ClassificationHead

class VisionModel(nn.Module):
    def __init__(self, model, task, lora_rank, lora_alpha, lora_dropout,
                 num_det_classes=None, num_seg_classes=None, num_cls=None):
        super().__init__()

        self.base_model = model
        self.lora_rank = lora_rank
        self.hetlora_gamma = 0.99

        for p in self.base_model.parameters():
            p.requires_grad = False

        # --- 核心修改：递归替换 ViT 内部的 Linear 层 ---
        # 我们定义一个列表来引用所有的 LoRA 模块，方便后续 adjust_rank
        self.lora_modules = [] 
        
        self._replace_layers_with_lora(
            self.base_model, lora_rank, lora_alpha, lora_dropout
        )

        self.original_norms = self.calculate_original_norms()
        # -------------------------------------------

        hidden_dim = model.config.hidden_size

        # 初始化 Heads
        if task == "detection":
            self.det_head = DetectionHead(hidden_dim, num_det_classes)
        elif task == "segmentation":
            self.seg_head = SegmentationHead(hidden_dim, num_seg_classes)
        elif task == "classification":
            self.cls_head = ClassificationHead(hidden_dim, num_cls)
        else:
            raise ValueError(f"Unknown task {task}")

    def _replace_layers_with_lora(self, module, rank, alpha, dropout):
        """
        递归遍历模型，将满足条件的 nn.Linear 替换为 LinearWithLoRA
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if('ViT' in str(self.base_model)):
                    if child.in_features == child.out_features:
                        layer_with_lora = LinearWithLoRA(child, rank, alpha, dropout)
                        setattr(module, name, layer_with_lora)
                        self.lora_modules.append(layer_with_lora)
                elif('Swin' in str(self.base_model)):
                    lname = name.lower()
                    inject = any(
                        key in lname
                        for key in [
                            "query", "key", "value", "dense"
                        ]
                    )
                    # 显式排除 head / classifier
                    if "head" in lname or "classifier" in lname:
                        inject = False

                    if inject:
                        layer_with_lora = LinearWithLoRA(
                            child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout
                        )
                        setattr(module, name, layer_with_lora)
                        self.lora_modules.append(layer_with_lora)
                else:
                    print("replace_layers_with_lora wrong!")
                    aaa
            else:
                # 如果不是 Linear，继续递归查找子模块
                self._replace_layers_with_lora(child, rank, alpha, dropout)

    def adjust_lora_rank(self, new_rank):
        """调整所有注入模块的 rank"""
        self.lora_rank = new_rank
        for mod in self.lora_modules:
            mod.adjust_rank(new_rank)

    def calculate_original_norms(self):
        original_norms = []
        for lora_module in self.lora_modules:
            original_norm = torch.norm(
                lora_module.lora_layer.lora_A) * torch.norm(lora_module.lora_layer.lora_B)
            original_norms.append(original_norm)
        return original_norms

    def rank_self_pruning(self):
        for idx, lora_module in enumerate(self.lora_modules):
            original_norm = self.original_norms[idx]
            ranks_to_prune = int(lora_module.lora_layer.r * (1 - self.hetlora_gamma))
            pruned_norm = torch.norm(
                lora_module.lora_layer.lora_A[-ranks_to_prune:, :]) * torch.norm(lora_module.lora_layer.lora_B[:, -ranks_to_prune:])
            if pruned_norm < original_norm:
                lora_module.lora_layer.lora_A = nn.Parameter(
                    lora_module.lora_layer.lora_A[:-ranks_to_prune, :])
                lora_module.lora_layer.lora_B = nn.Parameter(
                    lora_module.lora_layer.lora_B[:, :-ranks_to_prune])
                lora_module.lora_layer.r -= ranks_to_prune

    def calculate_lora_params(self):
        """
        计算模型中 LoRA 权重的总参数字节数。
        """
        total_lora_bytes = 0
        
        for lora_module in self.lora_modules:
            # 获取 lora_layer 中的参数 (lora_A, lora_B)
            # 注意：LinearWithLoRA 内部结构是 self.lora_layer -> self.lora_A, self.lora_B
            params = [
                lora_module.lora_layer.lora_A, 
                lora_module.lora_layer.lora_B
            ]
            
            for p in params:
                # numel() 获取元素个数，element_size() 获取单个元素的字节大小
                # 例如 float32: element_size=4, float16: element_size=2
                total_lora_bytes += p.numel() * p.element_size()
        
        return total_lora_bytes

    def forward(self, images, task, H=None, W=None):
        # --- 这里的 forward 就变得非常干净了 ---
        
        # 1. 运行 ViT
        # 由于内部的 Linear 已经被我们偷梁换柱成了 LinearWithLoRA，
        # 所以调用 self.vit(images) 时，LoRA 已经在各层内部自动生效了。
        outputs = self.base_model(images)
        features = outputs.last_hidden_state

        if len(features.shape) == 4: # [B, H, W, C]
            batch, h, w, c = features.shape
            features = features.view(batch, h * w, c) # 转为 [B, 49, 1024]

        # 2. 运行任务头
        if task == "detection":
            return self.det_head(features)
        elif task == "segmentation":
            return self.seg_head(features, H, W)
        elif task == "classification":
            return self.cls_head(features)
        else:
            raise ValueError(f"Unknown task {task}")