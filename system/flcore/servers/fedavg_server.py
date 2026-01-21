
import numpy as np
from sklearn import metrics
from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server
from flcore.clients.fedavg_client import clientFed
from flcore.trainmodel.vision.FullModel import FullModel
import concurrent.futures
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import time
import os
import h5py
import csv
from transformers import ViTModel
from utils.rank_adaption import shannon_rate


class serverFed(Server):
    def __init__(self, args, times, env, rsu, task):
        super().__init__(args, times)
        self.rs_auc = []
        self.rs_test_acc = []
        self.rs_time_cost = []
        self.updates = []
        self.result_path = "../results/"
        self.task = task
        self.env = env
        self.lora_alpha = args.lora_rank
        # self.global_model = self.clone_model(args,args.model)
        vit_model_path = "/home/zhengbk/vit-finetune/src/model/vit_base_patch16_224"
        if 'detection' in self.task:
            self.rsu_rect = rsu[0]
            args.num_classes = 8
            vit_backbone = ViTModel.from_pretrained(vit_model_path)
            self.global_model = FullModel(
            vit_backbone, self.task, num_det_classes=args.num_classes).to(args.device)
        elif 'segmentation' in self.task:
            self.rsu_rect = rsu[2]
            args.num_classes = 19
            vit_backbone = ViTModel.from_pretrained(vit_model_path)
            self.global_model = FullModel(
            vit_backbone, self.task, num_seg_classes=args.num_classes).to(args.device)
        elif 'classification' in self.task:
            self.rsu_rect = rsu[1]
            args.num_classes = 58
            vit_backbone = ViTModel.from_pretrained(vit_model_path)
            self.global_model = FullModel(
            vit_backbone, self.task, num_cls=args.num_classes).to(args.device)
        else:
            raise NotImplementedError
        

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFed)
        
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
        
        self.total_params = self.calculate_parameters(self.global_model)
        print(f"Total train parameters(Full finetuning): {self.total_params}")

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    
    def train_one_eps(self, i):
        start_time = time.time()
        self.eps_reward = 0
        print(f"\n-------------Task: {self.task}-------------")
        if i > 0:
            self.env.step()

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
            self.eps_rewards.append((accs / len(self.clients)) * self.gamma)
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
            client.train(self.global_model, self.total_params)
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
        self.eps_reward += self.calculate_reward(self.eps_time, self.eps_energy, accs / len(self.clients))
        print("eps_reward: ", self.eps_reward)
        print("eps_time: ", self.eps_time)
        print("eps_energy: ", self.eps_energy)
        print("eps_acc: ", self.acc)
        self.eps_rewards.append(self.eps_reward)
        self.eps_times.append(self.eps_time)
        self.eps_energys.append(self.eps_energy)

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
            # print(test_sample_rate)
            client = clientObj(self.args,
                               id=vid,
                               train_samples=data_lens[idx],
                               test_samples= int(len(test_data) * test_sample_rate),
                               task = self.task,
                               pos = pos
                               )
            self.clients.append(client)

    def clear_count(self):
        for client in self.clients:
            client.clear_count()

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

    def calculate_parameters(self, model):
        """
        计算模型中权重的总参数字节数。
        """
        total_bytes = 0
    
        # model.parameters() 会递归地返回模型中所有注册的参数 (包括权重和偏置)
        for param in model.parameters():
            # param.numel(): 获取该张量的元素总数 (例如 [768, 768] -> 589824)
            # param.element_size(): 获取单个元素占用的字节数 (例如 float32 -> 4 bytes, float16 -> 2 bytes)
            total_bytes += param.numel() * param.element_size()
        
        return total_bytes

    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        
        # 1. 获取全局模型当前的状态字典
        agg_state_dict = self.global_model.state_dict()

        # 2. 遍历所有参数键值
        for key in agg_state_dict:
            # 确保该 key 存在于客户端的更新中
            if key in self.updates[0]:
                
                # 收集所有客户端该参数的更新
                client_updates = [update[key] for update in self.updates]
                
                if client_updates[0].is_floating_point():
                    stacked_updates = torch.stack(client_updates)
                    # 执行简单平均 (FedAvg)
                    agg_state_dict[key] = torch.mean(stacked_updates, dim=0)
                else:
                    # 对于整数类型的参数（例如 BatchNorm 的计数器 num_batches_tracked）
                    # 无法求平均，通常取第一个客户端的值，或者保持全局原值
                    agg_state_dict[key] = client_updates[0]

        # 3. 加载聚合后的参数到全局模型
        self.global_model.load_state_dict(agg_state_dict)
        

        # 4. 清空更新列表
        self.updates = []
    

    def evaluate(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

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


        accuracy = 100.0 * correct / max(total, 1)
        average_loss = total_loss / len(testloader)

        print(f"Test Accuracy: {accuracy}%")
        print(f"Evaluation - Loss: {average_loss:.4f}")
        return accuracy, average_loss

    def record(self, algorithm):
        file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.task + "_results.csv"
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
