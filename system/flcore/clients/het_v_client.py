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


class clientHetLoRA(Client):
    def __init__(self, args, id, train_samples, test_samples, init_rank, task, position, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.hetlora_min_rank = args.hetlora_min_rank
        self.hetlora_max_rank = args.hetlora_max_rank
        self.hetlora_gamma = args.hetlora_gamma
        self.local_learning_rate = args.local_learning_rate
        self.task = task
        self.position = position
        self.train_samples = train_samples
        self.curr_acc = None
        self.local_learning_rate = 1.5e-5

        if args.base_model == 'vit':
            base_model_path = "/home/zhengbk/vit-finetune/src/model/vit_base_patch16_224"
            backbone = ViTModel.from_pretrained(base_model_path)
        elif args.base_model == 'swin':
            base_model_path = "/home/zhengbk/PFLoRA-lib/swin-base-patch4-window7-224"
            backbone = SwinModel.from_pretrained(base_model_path)
        if 'detection' in self.task:
            args.num_classes = 8
            self.model = VisionModel(
            backbone, self.task, init_rank, args.lora_alpha, args.lora_dropout, num_det_classes=args.num_classes).to(args.device)
            self.rsu_loca = (250, 250)
        elif 'segmentation' in self.task:
            args.num_classes = 19
            self.model = VisionModel(
            backbone, self.task, init_rank, args.lora_alpha, args.lora_dropout, num_seg_classes=args.num_classes).to(args.device)
            self.rsu_loca = (500, 750)
        elif 'classification' in self.task:
            args.num_classes = 58
            self.model = VisionModel(
            backbone, self.task, init_rank, args.lora_alpha, args.lora_dropout, num_cls=args.num_classes).to(args.device)
            self.rsu_loca = (750, 250)
        else:
            raise NotImplementedError
        
        self.num_labels = args.num_classes
        self.trainloader = self.load_train_data(train_samples)
        # Initialize test loader
        self.test_loader = self.load_test_data(test_samples)
        self.bandwidth = 2e5  # 2 MHz
        self.noise = 1e-9     # 噪声功率
        self.p_rsu = 2.0      # RSU发送功率（单位W）
        self.p_vehicle = 1.0  # 车辆发送功率（单位W）
        self.time_cost = 0
        self.energy_cost = 0
        self.total_lora_params = 0
        self.R_d = 93213
        self.p = 5
        self.f_v = 1e3
        self.cal = 9.996
        self.k = 1e-8
        

    def train(self):
        max_local_epochs = self.local_epochs
        print(f"Client {self.id} starts training:")
        print(f"Client {self.id} current rank: {self.model.lora_rank}")
        print(f"  Local epochs: {max_local_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.local_learning_rate}")
        self.total_lora_params = self.model.calculate_lora_params()
        # 参数下载
        # time_down = 0.46
        # energy_down = self.p * time_down
        dist = math.sqrt((self.rsu_loca[0] - self.position[0]) ** 2 + (self.rsu_loca[1] - self.position[1]) ** 2)
        # 参数下载修改
        rate_down = shannon_rate(self.bandwidth, self.p_rsu, dist, self.noise)
        time_down = self.total_lora_params / rate_down
        energy_down = self.p_rsu * time_down
        print("download time: ", time_down)
        print("download energy: ", energy_down)
        self.time_cost += time_down
        self.energy_cost += energy_down
        self.model.train()
        trainable_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad
        ]

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.local_learning_rate
        )
        self.loss = torch.nn.CrossEntropyLoss()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        self.start_time = time.time()
        for epoch in range(max_local_epochs):
            epoch_loss = 0.0
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

            epoch_loss /= len(self.trainloader)
            print(
                f"  Epoch {epoch+1}/{max_local_epochs} | Loss: {epoch_loss:.4f}")

        self.model.rank_self_pruning()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()

        time_comp = (time.time() - self.start_time) * discount_factor(self.model.lora_rank)

        energy_comp = self.f_v * self.f_v * self.f_v * self.k * time_comp
        self.time_cost += time_comp
        self.energy_cost += energy_comp
        print("finetune time: ", time_comp)
        print("finetune energy: ", energy_comp)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - self.start_time
        
        # 参数上传
        rate_up = shannon_rate(self.bandwidth, self.p_vehicle, dist, self.noise)
        time_up = self.total_lora_params / rate_up
        energy_up = self.p_vehicle * time_up
        print("upload time: ", time_up)
        print("upload energy: ", energy_up)
        self.time_cost += time_up
        self.energy_cost += energy_up
        print(f"Client {self.id} finished training.")

    def clear_count(self):
        self.time_cost = 0
        self.energy_cost = 0

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


        
