
import numpy as np
from sklearn import metrics
from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server
from flcore.trainmodel.vision.VisionModel import VisionModel
from flcore.clients.my_v_client import Myclientlora
import concurrent.futures
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import time
import os
import h5py
import json
import csv
import torch.linalg as linalg
import math
from utils.rank_adaption import UCBRankAllocator, FaultToleranceScheduler, shannon_rate


class Myserverlora(Server):
    def __init__(self, args, idx, env, rsu, tr, task):
        super().__init__(args, idx)
        self.rs_auc = []
        self.rs_test_acc = []
        self.rs_time_cost = []
        self.updates = []
        
        self.result_path = "../results/"
        self.task = task
        if 'detection' in self.task:
            args.num_classes = 8
            self.rsu_rect = rsu[0]
        elif 'segmentation' in self.task:
            args.num_classes = 19
            self.rsu_rect = rsu[2]
        elif 'classification' in self.task:
            args.num_classes = 58
            self.rsu_rect = rsu[1]
        self.global_theta_dict = {}

        self.env = env
        self.lambda_ = 0.0
        self.lr = 0.01

        # # Initialize test loader
        # self.test_loader = self.load_test_data()

        # select slow clients
        self.set_slow_clients()
        self.total_rank = tr
        self.current_ranks = tr
        self.set_clients(Myclientlora)

        # rank 调度
        # self.rank_adapter = UCBRankAllocator(V = self.num_clients, total_rank = 300, alpha = 0.2, beta = 0.001, gamma = 1.4)
        self.fault_scheduler = FaultToleranceScheduler(alpha = 0.2, beta = 0.05, gamma = 2)
        self.test_losses = []
        self.eps_rewards = []
        self.eps_times = []
        self.eps_energys = []
        self.eps_paras = []
        self.ranks = []
        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        self.k = 1e-8
        self.f_k = 1e3
        self.p_vehicle = 1.0
        self.time_cost = 0
        self.energy_cost = 0
        self.reward = 0

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train_one_eps(self, i):
        start_time = time.time()
        self.eps_reward = 0
        # self.selected_clients = self.select_clients()
        print(f"\n-------------Task: {self.task}-------------")
        if i > 0:
            self.env.step()
            self.send_models()
        
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

        # TODO: 对即将离开RSU范围的车辆进行容错机制判断
        for client in self.clients_leave:
            # 计算任务迁移损耗
            cli_pos = self.env.get_vehicle_position(client.id)
            cloest_dist = float('inf')
            for useful_c in self.clients:
                usec_pos = self.env.get_vehicle_position(useful_c.id)
                dist = math.sqrt((cli_pos[0] - usec_pos[0]) ** 2 + (cli_pos[1] - usec_pos[1]) ** 2)
                cloest_dist = min(cloest_dist, dist)
            rate_migration = shannon_rate(client.bandwidth, client.p_vehicle, dist, client.noise)
            migration_delay = client.total_lora_params / rate_migration
            migration_energy = client.p_vehicle * migration_delay
            strategy, fault_cost = self.fault_scheduler.select_strategy(self.max_acc, client.tot_energy, migration_delay, migration_energy, client.curr_acc)
            if strategy == 1:
                self.clients.append(client)
            self.eps_reward -= fault_cost

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
        self.selected_ranks = []
        for client in self.clients:
            accuracy, test_loss = self.evaluate(
                    client.model, client.test_loader)
            client.curr_acc = accuracy
            accs += accuracy
            self.max_acc = max(self.max_acc, accuracy)
            test_los += test_loss
            if self.args.ablation == "n":
                client.rank_adapter.update_ucb(client, accuracy, i+1, self.lambda_)
                if (i+1) % 3 == 0:
                    client.selected_rank = client.rank_adapter.select_rank(client)
                    self.selected_ranks.append(client.selected_rank)
        if (i+1) % 3 == 0 and self.args.ablation == "n":
            total_used = sum(self.selected_ranks)
            diff = total_used - self.total_rank
            self.lambda_ = max(0.0, self.lambda_ + self.lr * diff)
            print(f"[Dual-UCB] λ updated to: {self.lambda_:.4f}, total_rank_used: {total_used}")
        self.acc = accs / len(self.clients)
        self.test_loss = test_los / len(self.clients)
        self.rs_test_acc.append(self.acc)
        self.test_losses.append(self.test_loss)
        self.receive_models()
        # print(self.updates)
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
        self.eps_reward += self.calculate_reward(self.eps_time, accs / len(self.clients))
        print("eps_reward: ", self.eps_reward)
        print("eps_time: ", self.eps_time)
        print("eps_energy: ", self.eps_energy)
        print("eps_acc: ", self.acc)
        self.eps_rewards.append(self.eps_reward)
        self.eps_times.append(self.eps_time)
        self.eps_energys.append(self.eps_energy)
        
        self.current_ranks = 0
        self.eps_para = 0
        cur_ranks = []
        for client in self.clients:
            self.current_ranks += client.model.lora_rank
            cur_ranks.append(client.model.lora_rank)
            self.eps_para += client.model.calculate_lora_params()
        self.ranks.append(cur_ranks)
        self.eps_paras.append(self.eps_para/1e6)

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

    def adj_rank(self):
        # self.total_rank = 120
        # 使用 np.round 进行四舍五入
        ranks = np.round(self.total_rank * self.data_distribution).astype(int)

        # 调整最后一项以确保总和为 total_rank
        ranks[-1] = self.total_rank - ranks[:-1].sum()
        for i in range(len(self.clients)):
            self.clients[i].rank_adapter.update_max_rank(ranks[i])
            if self.clients[i].curr_acc < 64:
                self.clients[i].selected_rank = ranks[i]
                

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
        # print(client_traces)
        # 使用 np.round 进行四舍五入
        ranks = np.round(self.total_rank * self.data_distribution).astype(int)

        # 调整最后一项以确保总和为 total_rank
        ranks[-1] = self.total_rank - ranks[:-1].sum()
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
    
    def clear_count(self):
        for client in self.clients:
            client.clear_count()

    def calculate_reward(self, tau_v, q_v):
        """
        计算服务器端的奖励
        """
        self.alpha = 0.5  # 时延权重
        # self.beta = 0.124  # 能耗权重
        self.gamma = 2  # 精度权重
        return - (self.alpha * tau_v - self.gamma * q_v)


    def send_models(self):
        assert (len(self.clients) > 0)
        print("\n-------------Begin UCB Rank Allocator-------------")
        is_change = False
        for client in self.clients:
            # print("client.selected_rank: ", client.selected_rank)
            if client.selected_rank != client.model.lora_rank:
                is_change = True
                client.model.adjust_lora_rank(client.selected_rank)
                print(f"Client {client.id}: current rank is {client.selected_rank}\n")
        if not is_change:
            print("Remain Same\n")
        start_time = time.time()
        total_svd_time = 0.0
        for key, global_param in self.global_theta_dict.items():
            global_param = global_param.float() 
            t0 = time.time()
            U, Sigma, V_T = linalg.svd(global_param, full_matrices=False)  # 进行SVD分解，得到U, Sigma, V^T
            t1 = time.time()
            total_svd_time += (t1 - t0)
             # 遍历每个客户端，按照local_rank截断奇异矩阵
            for client in self.clients:
                rank = client.model.lora_rank
                U_v = U[:, :rank]  # 截取U的前rank列
                Sigma_v = torch.diag(Sigma[:rank])  # 截取Sigma的前rank个奇异值
                # print(Sigma_v)
                V_T_v = V_T[:rank, :]  # 截取V_T的前rank行

                # 得到客户端对应的B_v和A_v
                A_v = U_v @ Sigma_v  
                B_v = V_T_v
                # sqrt_Sigma = torch.sqrt(Sigma_v)
                # A_v = U_v @ sqrt_Sigma
                # B_v = sqrt_Sigma @ V_T_v
                    
                # 将B_v和A_v存入客户端的模型状态字典
                client_model_state_dict = client.model.state_dict()
                client_model_state_dict[f'{key}.lora_B'].copy_(B_v)
                client_model_state_dict[f'{key}.lora_A'].copy_(A_v)
        end_time = time.time()
        print(f"SVD calculation time: {total_svd_time:.4f}s")
        print("SVD total: ", end_time-start_time)


    def receive_models(self):
        # assert (len(self.cl) > 0)
        self.updates = []
        for client in self.clients:
            client_model = client.model
            self.updates.append(client_model.state_dict())

    def calculate_global_parameters(self, model):
        total_params = 0
        for lora_layer in model.lora_layers:
            input_dim = lora_layer.lora_A.shape[0]  # Input dimension of the layer (usually 768)
            output_dim = lora_layer.lora_B.shape[1]  # Output dimension of the layer (usually 768)
            total_params += (input_dim * output_dim)
        return total_params

    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        agg_state_dict = self.global_theta_dict
        client_data_sizes = [client.train_samples for client in self.clients]


        for key in self.clients[0].model.state_dict():
            if 'lora' in key and 'A' in key:  # 只针对包含A矩阵的key
                # 提取出对应的lora层前缀，例如'lora_layers.0'
                layer_key_prefix = key.rsplit('.', 1)[0]  # 获取lora_layers.0这种前缀
                # print(layer_key_prefix)
                # 收集每个客户端的A和B矩阵
                client_updates_A = [update[key] for update in self.updates]
                client_updates_B = [update[layer_key_prefix + '.lora_B'] for update in self.updates]  # 对应的B矩阵

                # 计算每个客户端B和A的乘积
                # sum_updates = sum(
                #     (client_updates_A[i] @ client_updates_B[i]) 
                #     for i in range(len(client_updates_A))
                # )

                # # 直接除以客户端数量 (简单平均)
                # agg_state_dict[layer_key_prefix] = sum_updates / self.num_clients
                 # 计算每个客户端B和A的乘积
                weighted_updates = sum(
                    (client_updates_A[i] @ client_updates_B[i]) * client_data_sizes[i]  # A * B
                    for i in range(len(client_updates_A))
                )

                # 计算数据量总和
                total_data_size = sum(client_data_sizes)

                # 存储加权的矩阵乘积结果
                agg_state_dict[layer_key_prefix] = weighted_updates / total_data_size


                self.global_theta_dict = agg_state_dict

        self.updates = []
    
    def check_lora_updates(self, initial_state_dict):
        """
        检查LoRA模块参数是否有更新
        """
        lora_updated = False
        for name, param in self.global_model.named_parameters():
            if 'lora' in name and not torch.equal(param, initial_state_dict[name]):
                print(f"LoRA parameter {name} updated.")
                lora_updated = True
                break
        if not lora_updated:
            print("No LoRA parameters were updated.")

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

    def load_test_data(self):
        # Assuming the test data is stored with client id 0
        test_data = read_client_data(self.dataset, 0, is_train=False)
        return DataLoader(test_data, batch_size=self.args.batch_size, drop_last=False, shuffle=False)

    def load_server_data(self):
        # 加载服务器端的训练数据
        server_data = read_client_data(
            self.dataset, 0, is_train=True)  # 假设服务器端的数据客户端ID为-1
        server_trainloader = DataLoader(
            server_data, batch_size=self.args.batch_size, shuffle=True)
        return server_trainloader

    def record(self, algorithm):
        file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_results.csv"
        # file_name = "./out/" + str(self.args.num_clients) + "_client/" + algorithm + "_" + self.task + "_results.csv"
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Accuracy", "Test Loss", "Eps_reward", "Eps_time", "Eps_energy", "Eps_para"])  # CSV文件的表头
            for acc, loss, r, t, e, p in zip(self.rs_test_acc, self.test_losses, self.eps_rewards, self.eps_times, self.eps_energys, self.eps_paras):
                writer.writerow([acc, loss, r, t, e, p])  # 将每个数据点写入CSV文件

        rank_file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_ranks.csv"
        with open(rank_file_name, mode="w", newline="", encoding="utf-8") as rank_file:
            rank_writer = csv.writer(rank_file)
            # 直接写入self.ranks的二维列表，每行对应一个rank数组 [17,33,50] / [14,55,55]...
            rank_writer.writerows(self.ranks)

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
        result_path = os.path.join("..", "results", self.task)

        # 逐级创建文件夹路径
        os.makedirs(result_path, exist_ok=True)
        if len(self.rs_test_acc):
            algo = self.dataset.split(
                "/")[-1]+"_" + algo + "_" + self.goal + "_" + str(self.idx)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_time_cost', data=self.rs_time_cost)
                # Add any other results you need to save
