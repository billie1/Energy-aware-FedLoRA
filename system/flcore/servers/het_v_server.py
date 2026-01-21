# ./system/flcore/servers/serverhetlora.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server

from flcore.clients.het_v_client import clientHetLoRA
from flcore.trainmodel.vision.VisionModel import VisionModel
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import concurrent.futures
import h5py
import csv
from transformers import ViTModel, SwinModel 
from utils.rank_adaption import shannon_rate


class serverHetLoRA(Server):
    def __init__(self, args, times, env, rsu, task):
        super().__init__(args, times)
        self.rs_auc = []
        self.rs_test_acc = []
        self.updates = []
        self.rs_time_cost = []
        self.task = task
        self.env = env
        self.total_rank = args.total_rank / 3
        # # Initialize test loader
        # self.test_loader = self.load_test_data()
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientHetLoRA)
        self.loss = nn.CrossEntropyLoss()

        max_rank = 0
        for client in self.clients:
            rank = client.model.lora_rank
            if rank > max_rank:
                max_rank = rank

        if args.base_model == 'vit':
            base_model_path = "/home/zhengbk/vit-finetune/src/model/vit_base_patch16_224"
            backbone = ViTModel.from_pretrained(base_model_path)
        elif args.base_model == 'swin':
            base_model_path = "/home/zhengbk/PFLoRA-lib/swin-base-patch4-window7-224"
            backbone = SwinModel.from_pretrained(base_model_path)
        if 'detection' in self.task:
            self.rsu_rect = rsu[0]
            args.num_classes = 8
            self.global_model = VisionModel(
            backbone, self.task, max_rank, args.lora_alpha, args.lora_dropout, num_det_classes=args.num_classes).to(args.device)
        elif 'segmentation' in self.task:
            self.rsu_rect = rsu[2]
            args.num_classes = 19
            self.global_model = VisionModel(
            backbone, self.task, max_rank, args.lora_alpha, args.lora_dropout, num_seg_classes=args.num_classes).to(args.device)
        elif 'classification' in self.task:
            self.rsu_rect = rsu[1]
            args.num_classes = 58
            self.global_model = VisionModel(
            backbone, self.task, max_rank, args.lora_alpha, args.lora_dropout, num_cls=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        self.test_losses = []
        self.eps_rewards = []
        self.eps_times = []
        self.eps_energys = []


        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        self.gamma = 2
        self.k = 1e-8
        self.f_k = 1e3
        self.p_vehicle = 1.0
        self.time_cost = 0
        self.energy_cost = 0
        self.reward = 0

        self.total_lora_params = self.calculate_lora_parameters(self.global_model)
        print(f"Total LoRA parameters: {self.total_lora_params*3/1e6} M")

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train_one_eps(self, i):
        start_time = time.time()
        self.eps_reward = 0
        print(f"\n-------------Task: {self.task}-------------")
        if i > 0:
            self.env.step()
            self.send_model()

        if i == 0:
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            accs = 0
            test_los = 0
            for client in self.clients:
                accuracy, test_loss = self.evaluate(
                    client.model, client.test_loader)
                accs += accuracy
                test_los += test_loss
                    
            self.rs_test_acc.append(accs / len(self.clients))
            self.test_losses.append(test_los / len(self.clients))
            self.eps_rewards.append((accs / len(self.clients)) * 2)
            self.eps_times.append(0.0)
            self.eps_energys.append(0.0)


        self.clients, self.clients_leave = self.vehicle_transitions()
        # 都选择直接上传
        for client in self.clients_leave:
            cost = self.gamma * max(0, self.max_acc - client.curr_acc)
            self.eps_reward -= cost

        self.max_acc = 0
        self.eps_time = 0
        self.eps_energy = 0
        self.clear_count()
        temp = 0
        for client in self.clients:
            client.train()
            temp = max(temp, client.time_cost)
            self.eps_energy += client.energy_cost
        self.eps_time += temp
        print(f"\n-------------Round number: {i+1}-------------")
        print("\nEvaluate global model")
        accs = 0
        test_los = 0
        for client in self.clients:
            accuracy, test_loss = self.evaluate(
                    client.model, client.test_loader)
            client.curr_acc = accuracy
            accs += accuracy
            self.max_acc = max(self.max_acc, accuracy)
            test_los += test_loss
        self.acc = accs / len(self.clients)
        self.test_loss = test_los / len(self.clients)
        self.rs_test_acc.append(self.acc)
        self.test_losses.append(self.test_loss)
        self.receive_models()
        agg_s = time.time()
        self.aggregate_parameters()
        agg_e = time.time()
        agg_time = agg_e - agg_s
        agg_energy = (agg_e - agg_s) * self.f_k * self.f_k * self.f_k * self.k
        print("aggregate time: ", agg_time)
        print("aggregate energy", agg_energy)
        self.eps_time += agg_time
        self.eps_energy += agg_energy
        end_time = time.time()
        self.rs_time_cost.append(end_time - start_time)
        self.time_cost += self.eps_time
        self.energy_cost += self.eps_energy
        self.eps_reward = self.calculate_reward(self.eps_time, self.eps_energy, accs / len(self.clients))
        print("eps_reward: ", self.eps_reward)
        print("eps_time: ", self.eps_time)
        print("eps_energy: ", self.eps_energy)
        print("eps_acc: ", self.acc)
        self.eps_rewards.append(self.eps_reward)
        self.eps_times.append(self.eps_time)
        self.eps_energys.append(self.eps_energy)

    def clear_count(self):
        for client in self.clients:
            client.clear_count()

    def train(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(self.global_rounds + 1):
                start_time = time.time()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(self.global_model, self.test_loader)


                self.receive_models()
                self.aggregate_parameters()
                end_time = time.time()
                self.rs_time_cost.append(end_time - start_time)

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.rs_time_cost[1:]) / len(self.rs_time_cost[1:]))

        self.save_results()
        self.save_global_model()
    
    def vehicle_transitions(self):
        """
        返回将从当前 RSU 迁移到其他 RSU 的车辆列表
        :return: Dict[rsu_id -> List[vehicle_id]]
        """
        pred_left = []
        pred_leave = []

        for client in self.clients:
            vid = client.id
            next_pos = self.env.get_vehicle_position(vid)
            if self.rsu_rect['x_min'] <= next_pos[0] <= self.rsu_rect['x_max'] and self.rsu_rect['y_min'] <= next_pos[1] <= self.rsu_rect['y_max']:
                pred_left.append(client)
            else:
                pred_leave.append(client)            

        return pred_left, pred_leave

    def set_clients(self, clientObj):
        rsu_clients = self.env.get_rsu_clients()
        if self.task == 'detection':
            data_lens = np.array([150, 300, 450])
            client_traces = rsu_clients[0]
            self.num_clients = self.env.get_rsu_client_counts()[0]
        elif self.task == 'segmentation':
            data_lens = np.array([70, 140, 210])
            client_traces = rsu_clients[2]
            self.num_clients = self.env.get_rsu_client_counts()[2]
        else:
            data_lens = np.array([200, 400, 600])
            client_traces = rsu_clients[1]
            self.num_clients = self.env.get_rsu_client_counts()[1]

        if self.num_clients <= 3:
            data_lens = data_lens[:self.num_clients]  # 直接取前n个值
        else:
            # 线性插值生成新值（保持原始最小值和最大值）
            min_val, max_val = data_lens.min(), data_lens.max()
            data_lens = np.linspace(min_val, max_val, num=self.num_clients)
            data_lens = np.round(data_lens).astype(int)  # 四舍五入取整
        self.data_distribution = data_lens / data_lens.sum()

        # 使用 np.round 进行四舍五入
        ranks = np.round(self.total_rank * self.data_distribution).astype(int)
        # 调整最后一项以确保总和为 total_rank
        ranks[-1] = self.total_rank - ranks[:-1].sum()
        print(f"init_ranks_array: {ranks}")

        # for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
        for idx, trace in enumerate(client_traces):
            vid = trace['vehicle_id']
            pos = trace['position']
            train_data = read_client_data(self.task, vid, is_train=True)
            test_data = read_client_data(self.task, vid, is_train=False)
            if self.task == 'detection':
                test_sample_rate = data_lens[idx] / max(data_lens)
            elif self.task == 'segmentation':
                test_sample_rate = data_lens[idx] / max(data_lens)
            else:
                test_sample_rate = data_lens[idx] / max(data_lens)
            client = clientObj(self.args,
                               id=vid,
                               train_samples=data_lens[idx],
                               test_samples= int(len(test_data) * test_sample_rate),
                               init_rank = ranks[idx],
                               task = self.task,
                               position = pos
                               )
            self.clients.append(client)

    def calculate_reward(self, tau_v, e_v, q_v):
        """
        计算服务器端的奖励
        """
        self.alpha = 0.5  # 时延权重
        # self.beta = 0.124  # 能耗权重
        self.gamma = 2  # 精度权重
        # return - (self.alpha * tau_v + self.beta * e_v - self.gamma * q_v)
        return - (self.alpha * tau_v - self.gamma * q_v)

    def receive_models(self):
        self.updates = []
        for client in self.clients:
            client_model = client.model
            self.updates.append(client_model.state_dict())

    def send_model(self):
        for client in self.clients:
            client_rank = client.model.lora_rank
            
            client_state_dict = client.model.state_dict()
            
            # 遍历全局参数
            for key, global_param in self.global_model.state_dict().items():
                # 只处理 LoRA 参数
                if "lora_A" in key or "lora_B" in key:
                    
                    # 确定当前参数是 A 还是 B
                    is_matrix_A = "lora_A" in key
                    
                    # 进行切片操作
                    if is_matrix_A:
                        # Global A: [in, global_rank] -> Slice -> [in, client_rank]
                        sliced_param = global_param[:, :client_rank]
                    else:
                        # Global B: [global_rank, out] -> Slice -> [client_rank, out]
                        sliced_param = global_param[:client_rank, :]

                    # 赋值给客户端
                    client_state_dict[key].copy_(sliced_param)

    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        agg_state_dict = self.global_model.state_dict()
        lora_prefixes = list(set(
            key.replace(".lora_A", "") 
            for key in agg_state_dict.keys() 
            if key.endswith(".lora_A")
        ))
        for layer_prefix in lora_prefixes:
            # 全局层的A/B参数
            global_A = agg_state_dict[f"{layer_prefix}.lora_A"]
            global_B = agg_state_dict[f"{layer_prefix}.lora_B"]
            # print(layer_prefix)
            # 收集所有客户端的参数（已零填充对齐）
            padded_client_As, padded_client_Bs = [], []
            for update in self.updates:
                # 零填充对齐客户端参数
                client_A = self.zero_padding(global_A, update[f"{layer_prefix}.lora_A"])
                client_B = self.zero_padding(global_B, update[f"{layer_prefix}.lora_B"])
                padded_client_As.append(client_A)
                padded_client_Bs.append(client_B)
            
            # 计算稀疏权重（基于Frobenius范数）
            sparsity_weights = [
                torch.norm(a @ b, p='fro')  # 计算A@B的Frobenius范数
                for a, b in zip(padded_client_As, padded_client_Bs)
            ]
            total_weight = sum(sparsity_weights) + 1e-8  # 防止除零
            sparsity_weights = [w / total_weight for w in sparsity_weights]

            # 聚合参数
            agg_A = torch.zeros_like(global_A)
            agg_B = torch.zeros_like(global_B)
            for w, a, b in zip(sparsity_weights, padded_client_As, padded_client_Bs):
                agg_A += w * a
                agg_B += w * b

            # 更新到全局参数
            agg_state_dict[f"{layer_prefix}.lora_A"] = agg_A
            agg_state_dict[f"{layer_prefix}.lora_B"] = agg_B
        
        # 加载聚合后的参数
        self.global_model.load_state_dict(agg_state_dict)
        
        # 清空客户端更新
        self.updates = []

    def zero_padding(self, global_lora_layer, client_lora_layer):
        padded = torch.zeros_like(global_lora_layer)
        slices = tuple(slice(0, min(t, s)) for t, s in zip(global_lora_layer.shape, client_lora_layer.shape))
        padded[slices] = client_lora_layer[slices]
        return padded

    def calculate_lora_parameters(self, model):
        """
        计算模型中 LoRA 权重的总参数字节数。
        """
        total_lora_bytes = 0
        
        for lora_module in model.lora_modules:
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


    def load_test_data(self):
        # Assuming the test data is stored with client id 0
        test_data = read_client_data(self.dataset, 0, is_train=False)
        return DataLoader(test_data, batch_size=self.args.batch_size, drop_last=False, shuffle=False)

    def evaluate(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        test_acc = 0
        test_num = 0
        total_loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = model(images, task=self.task,
                            H=images.shape[2],
                            W=images.shape[3])
                if self.task == "detection":
                    loss = self.loss(
                        outputs["pred_logits"], labels
                    )
                    preds = torch.argmax(outputs["pred_logits"], dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                elif self.task == "segmentation":
                    loss = self.loss(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

                elif self.task == "classification":
                    loss = self.loss(
                        outputs["logits"], labels
                    )
                    preds = torch.argmax(outputs["logits"], dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                
                total_loss += loss.item()


        accuracy = 100 * correct / total
        average_loss = total_loss / total

        print(f"Test Accuracy: {accuracy}%")
        print(f"Evaluation - Loss: {average_loss:.4f}")
        return accuracy, average_loss

    def record(self, algorithm):
        # file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.task + "_results.csv"
        # file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_results.csv"
        file_name = "./out/" + str(self.args.num_clients) + "_client/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_results.csv"
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Accuracy", "Test Loss", "Eps_reward", "Eps_time", "Eps_energy"])  # CSV文件的表头
            for acc, loss, r, t, e in zip(self.rs_test_acc, self.test_losses, self.eps_rewards, self.eps_times, self.eps_energys):
                writer.writerow([acc, loss, r, t, e])  # 将每个数据点写入CSV文件

        print("\nBest accuracy.")
        if self.rs_test_acc:
            print(max(self.rs_test_acc))
        else:
            print("No accuracy results found.")
        
        
        print("\nTime and Energy Cost.")
        print("time: ", self.time_cost)
        print("energy: ",self.energy_cost)

        print("\nAverage time cost per round.")
        if len(self.rs_time_cost) > 1:
            print(sum(self.rs_time_cost[1:]) / len(self.rs_time_cost[1:]))
        else:
            print("No time cost data available.")
            
    def save_results(self):
        algo = self.algorithm
        result_path = os.path.join("..", "results", "GLUE")

        # 逐级创建文件夹路径
        os.makedirs(result_path, exist_ok=True)
        if len(self.rs_test_acc):
            algo = self.dataset.split(
                "/")[-1]+"_" + algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_time_cost', data=self.rs_time_cost)
                # Add any other results you need to save
