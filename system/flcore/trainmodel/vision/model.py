import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, lora_alpha, lora_dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.lora_A = nn.Parameter(torch.randn(input_dim, r))
        self.lora_B = nn.Parameter(torch.zeros(r, output_dim))
        self.scaling = self.lora_alpha / self.r

    def adjust_lora_rank(self, new_r):
        self.r = new_r
        # 重新初始化参数（注意：这样会丢失之前的训练权重，如果是动态调整通常需要更复杂的策略，这里按你原逻辑实现）
        self.lora_A = nn.Parameter(torch.randn(self.lora_A.shape[0], self.r).to(self.lora_A.device))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.lora_B.shape[1]).to(self.lora_B.device))
        self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        # x: [B, N, D]
        # 注意：这里我们只计算 LoRA 的增量 (BAx * scale)
        # 不需要加原输入 x，因为 Hook 会把这个结果加到原 Linear 的输出上
        lora_out = self.lora_dropout(x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_out
    

class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank, alpha, dropout):
        super().__init__()
        self.original_linear = original_linear
        self.lora_layer = LoRALayer(
            original_linear.in_features, 
            original_linear.out_features, 
            rank, alpha, dropout
        )
        
        # 冻结原始参数
        for p in self.original_linear.parameters():
            p.requires_grad = False

    def adjust_rank(self, new_rank):
        self.lora_layer.adjust_lora_rank(new_rank)

    def forward(self, x):
        # 这里的 x 是网络中间层的输入
        # 同时计算：原路径 + LoRA路径
        return self.original_linear(x) + self.lora_layer(x)
    

class DetectionHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.box = nn.Linear(hidden_dim, 4)

    def forward(self, features):
        # 使用 CLS token
        cls_token = features[:, 0]
        return {
            "pred_logits": self.cls(cls_token),
            "pred_boxes": self.box(cls_token).sigmoid()
        }
    


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, patch_size=16):
        super().__init__()
        self.decoder = nn.Conv2d(hidden_dim, num_classes, 1)
        self.patch_size = patch_size

    def forward(self, features, H, W):
        # features: [B, N+1, D] (假设带 CLS token)
        # 去掉 CLS token
        feat = features[:, 1:, :]  # [B, N, D]
        B, N, D = feat.shape
        
        # 计算特征图的宽高 (例如 14x14)
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size
        
        # 检查 N 是否匹配
        if h_feat * w_feat != N:
            # 简单容错，防止形状对应不上
            h_feat = int(N**0.5)
            w_feat = N // h_feat

        # [B, N, D] -> [B, D, N] -> [B, D, h, w]
        feat = feat.permute(0, 2, 1).view(B, D, h_feat, w_feat)
        
        # 1x1 卷积得到 logits [B, num_classes, 14, 14]
        logits = self.decoder(feat)
        
        # 关键修改：上采样回原图大小 [B, num_classes, H, W]
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits
    

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        """
        features: [B, N, D] from ViT backbone
        """
        # 使用 CLS token
        cls_token = features[:, 0]   # [B, D]

        return {
            "logits": self.cls(cls_token)
        }

