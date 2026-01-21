import numpy as np
import time
import math
import torch
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from flcore.trainmodel.vision.VisionModel import VisionModel
from transformers import ViTModel, SwinModel
from torch.utils.data import DataLoader, Subset, RandomSampler
from utils.rank_adaption import shannon_rate, discount_factor


class clientFedRA(Client):
    def __init__(self, args, id, train_samples, test_samples, task, pos, client_layers, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.local_learning_rate = args.local_learning_rate
        self.task = task
        self.position = pos
        self.client_layers = client_layers
        self.train_samples = train_samples
        self.curr_acc = None
        self.args = args
        if args.base_model == 'vit':
            base_model_path = "/home/zhengbk/vit-finetune/src/model/vit_base_patch16_224"
            backbone = ViTModel.from_pretrained(base_model_path)
            self.total_blocks = 12
        elif args.base_model == 'swin':
            base_model_path = "/home/zhengbk/PFLoRA-lib/swin-base-patch4-window7-224"
            backbone = SwinModel.from_pretrained(base_model_path)
            self.total_blocks = 24
        if 'detection' in self.task:
            args.num_classes = 8
            self.model = VisionModel(
            backbone, self.task, self.lora_rank, self.lora_alpha, args.lora_dropout, num_det_classes=args.num_classes).to(args.device)
            self.rsu_loca = (250, 250)
        elif 'segmentation' in self.task:
            args.num_classes = 19
            self.model = VisionModel(
            backbone, self.task, self.lora_rank, self.lora_alpha, args.lora_dropout, num_seg_classes=args.num_classes).to(args.device)
            self.rsu_loca = (500, 750)
        elif 'classification' in self.task:
            args.num_classes = 58
            self.model = VisionModel(
            backbone, self.task, self.lora_rank, self.lora_alpha, args.lora_dropout, num_cls=args.num_classes).to(args.device)
            self.rsu_loca = (750, 250)
        else:
            raise NotImplementedError

        self.num_labels = args.num_classes
        self.trainloader = self.load_train_data(train_samples)
        # Initialize test loader
        self.test_loader = self.load_test_data(test_samples)
        self.time_cost = 0
        self.energy_cost = 0
        self.total_lora_params = 0
        self.bandwidth = 2e5  # 2 MHz
        self.noise = 1e-9     # 噪声功率
        self.p_rsu = 2.0      # RSU发送功率（单位W）
        self.p_vehicle = 1.0  # 车辆发送功率（单位W）
        self.R_d = 93213
        self.p = 5
        self.f_v = 1e3
        self.cal = 9.996
        self.k = 1e-8

    def get_block_idx_from_name(self, name: str) -> int | None:
        """
        从参数名中提取所属 block 的全局索引（从 0 开始）
        ViT:  layer.0 ~ layer.11  → 0~11
        Swin: layers.0.blocks.0~1 → 0~1
              layers.1.blocks.0~1 → 2~3
              layers.2.blocks.0~17 → 4~21
              layers.3.blocks.0~1 → 22~23
        """
        if 'lora_layer' not in name:
            return None

        if self.args.base_model == 'vit':
            if 'base_model.encoder.layer.' in name:
                try:
                    return int(name.split('base_model.encoder.layer.')[1].split('.')[0])
                except:
                    return None

        elif self.args.base_model == 'swin':
            if 'base_model.encoder.layers.' in name:
                try:
                    parts = name.split('base_model.encoder.layers.')[1].split('.')
                    stage_idx = int(parts[0])
                    if parts[1] == 'blocks':
                        block_idx = int(parts[2])
                        # 计算全局 block 索引
                        if stage_idx == 0:
                            return block_idx  # 0~1
                        elif stage_idx == 1:
                            return 2 + block_idx  # 2~3
                        elif stage_idx == 2:
                            return 4 + block_idx  # 4~21
                        elif stage_idx == 3:
                            return 22 + block_idx  # 22~23
                except:
                    return None
        return None


    def train(self, net, params):
        self.total_lora_params = params
        # 参数下载
        dist = math.sqrt((self.rsu_loca[0] - self.position[0]) ** 2 + (self.rsu_loca[1] - self.position[1]) ** 2)
        # 参数下载修改
        rate_down = shannon_rate(self.bandwidth, self.p_rsu, dist, self.noise)
        time_down = self.total_lora_params / rate_down
        energy_down = self.p_rsu * time_down
        print("download time: ", time_down)
        print("download energy: ", energy_down)
        self.time_cost += time_down
        self.energy_cost += energy_down
        self.model.load_state_dict(net.state_dict())
        self.model.train()
        num_train = min(self.client_layers, self.total_blocks)
        chosen_blocks = np.random.choice(self.total_blocks, num_train, replace=False)
        chosen_set = set(chosen_blocks)
        print(f"Client {self.id} ({self.args.base_model}) trains {num_train}/{self.total_blocks} blocks: {sorted(chosen_blocks)}")
        for name, param in self.model.named_parameters():
            if 'head' in name or 'norm' in name:
                param.requires_grad = True
            # LoRA 参数：只对 attention 的 q,k,v,out 和 pooler 注入的
            elif 'lora_layer' in name:
                block_idx = self.get_block_idx_from_name(name)
                if block_idx is not None and block_idx in chosen_set:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # print(trainable_params)

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.local_learning_rate
        )
        self.loss = torch.nn.CrossEntropyLoss()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        print(f"Client {self.id} starts training:")
        # print(f"  Number of training samples: {self.train_samples}")
        print(f"  Local epochs: {max_local_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.local_learning_rate}")
        
        self.start_time = time.time()
        for epoch in range(max_local_epochs):
            epoch_loss = 0.0
            # 保存初始LoRA模块的参数状态
            initial_state_dict = {name: param.clone() for name, param in self.model.named_parameters()}
            for batch in self.trainloader:
                self.optimizer.zero_grad()
                # 前向传递和损失计算
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images, task=self.task,
                            H=images.shape[2],
                            W=images.shape[3])

                if self.task == "detection":
                    # labels: dict {cls, bbox} or cls only（按你当前实现）
                    loss = self.loss(outputs["pred_logits"], labels)

                elif self.task == "segmentation":
                    # labels: [B, H, W]
                    loss = self.loss(outputs, labels)

                elif self.task == "classification":
                    # labels: regression target
                    loss = self.loss(outputs["logits"], labels)
                else:
                    raise NotImplementedError
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            # self.check_lora_updates(initial_state_dict)
            

            epoch_loss /= len(self.trainloader)
            print(
                f"  Epoch {epoch+1}/{max_local_epochs} | Loss: {epoch_loss:.4f}")
        # print("finetune time", end_time - start_time)
        # comp = (-0.00008) * (self.lora_rank)**2 + 0.011 * (self.lora_rank) + 10
        # self.time_cost += comp
        # self.energy_cost += self.f_v * self.f_v * self.f_v * self.k * comp
        # print("weitiao: ", comp)
        # print("wei eng: ", self.f_v * self.f_v * self.f_v * self.k * comp)
        # time_comp = (time.time() - self.start_time) * math.log(self.model.lora_rank)
        time_comp = (time.time() - self.start_time) * discount_factor(self.model.lora_rank)
        # time_comp = (time.time() - self.start_time)
        # if self.task == 'classification':
        #     time_comp *= 2
        # if self.task == 'detection':
        #     time_comp *= 1.25
            
        energy_comp = self.f_v * self.f_v * self.f_v * self.k * time_comp
        self.time_cost += time_comp
        self.energy_cost += energy_comp
        print("finetune time: ", time_comp)
        print("finetune energy: ", energy_comp)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - self.start_time

        # 参数上传
        # time_up = (((self.total_lora_params)/ self.R_d) / self.model.lora_rank + (self.model.lora_rank -  (self.model.lora_rank)**(0.985))) / 10
        # energy_up = self.p * time_up
        partial_state_dict = self.get_partial_para()
        lora_keys = [k for k in partial_state_dict if 'lora' in k]
        head_norm_keys = [k for k in partial_state_dict if 'head' in k or 'norm' in k or 'Prompt' in k]
        print(f"Uploaded LoRA layers: {len(lora_keys)//4 if lora_keys else 0} blocks")  # 每个LoRA模块有A/B，通常4个param per block if 2 modules
        print(f"Uploaded fixed parts: {len(head_norm_keys)} keys")
        # 计算实际上传的参数字节数（用于通信开销）
        self.uploaded_bytes = sum(p.numel() * p.element_size() for p in partial_state_dict.values())
        rate_up = shannon_rate(self.bandwidth, self.p_vehicle, dist, self.noise)
        time_up = self.uploaded_bytes / rate_up
        energy_up = self.p_vehicle * time_up
        print(f"Client {self.id} uploaded {self.uploaded_bytes / 1e6:.2f} MB parameters "
              f"(time: {time_up:.2f}s, energy: {energy_up:.2f}J)")
        self.time_cost += time_up
        self.energy_cost += energy_up
        print(f"Client {self.id} finished training.")
    
    def get_partial_para(self):
        """
        返回本次实际训练过的参数（包括所有 head/norm/Prompt + 本轮训练的 LoRA 层）
        """
        partial = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # 只上传本次有梯度的参数
                partial[name] = param.detach().clone()

        # 确保 head、norm、Prompt 一定包含（即使某些情况 requires_grad 被误设）
        always_include = ['head.weight', 'head.bias', 'norm.weight', 'norm.bias', 'Prompt_Tokens']
        for key in always_include:
            full_key = f'back.base_model.model.{key}' if not key.startswith('back') else key
            if full_key in self.model.state_dict():
                partial[full_key] = self.model.state_dict()[full_key].clone()

        return partial
    

    def clear_count(self):
        self.time_cost = 0
        self.energy_cost = 0

    def check_lora_updates(self, initial_state_dict):
        """
        检查LoRA模块参数是否有更新
        """
        lora_updated = False
        for name, param in self.model.named_parameters():
            if 'lora' in name and not torch.equal(param, initial_state_dict[name]):
                print(f"LoRA parameter {name} updated.")
                lora_updated = True
                break
        if not lora_updated:
            print("No LoRA parameters were updated.")

    def load_train_data(self, train_samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # 读取完整的训练数据集
        full_train_data = read_client_data(self.task, self.id, is_train=True)

        # 获取数据集的总长度
        full_data_size = len(full_train_data)
        print("FULL TRAIN DATA SIZE: ", full_data_size)
        sample_size = train_samples
        # 确保采样大小不会大于训练数据集的总长度
        sample_size = min(sample_size, full_data_size)
        
        # 随机选择样本索引
        sampled_indices = torch.randperm(full_data_size)[:sample_size]

        # 创建一个Subset来提取采样的数据
        sampled_train_data = Subset(full_train_data, sampled_indices)

        # 使用RandomSampler来打乱采样数据
        sampler = RandomSampler(sampled_train_data)
        print("SAMPLER TRAIN DATA SIZE: ", len(sampled_train_data))
        # 返回包含采样数据的DataLoader
        return DataLoader(sampled_train_data, batch_size=batch_size, drop_last=True, sampler=sampler)

    def load_test_data(self, test_samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.task, self.id, is_train=False)
        # 获取数据集的总长度
        full_data_size = len(test_data)
        print("FULL TEST DATA SIZE: ", full_data_size)
        sample_size = test_samples
        # 确保采样大小不会大于训练数据集的总长度
        sample_size = min(sample_size, full_data_size)
        
        # 随机选择样本索引
        sampled_indices = torch.randperm(full_data_size)[:sample_size]

        # 创建一个Subset来提取采样的数据
        sampled_test_data = Subset(test_data, sampled_indices)

        # 使用RandomSampler来打乱采样数据
        sampler = RandomSampler(sampled_test_data)
        print("SAMPLER TEST DATA SIZE: ", len(sampled_test_data))
        return DataLoader(sampled_test_data, batch_size, drop_last=False, shuffle=True)
