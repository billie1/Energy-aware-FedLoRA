
import numpy as np
from sklearn import metrics
from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server
from flcore.clients.fedra_client import clientFedRA
from flcore.trainmodel.vision.VisionModel import VisionModel
import concurrent.futures
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import time
import os
import h5py
import csv
from transformers import ViTModel, SwinModel 
from utils.rank_adaption import shannon_rate


class serverFedRA(Server):
    def __init__(self, args, times, env, rsu, task):
        super().__init__(args, times)
        self.rs_auc = []
        self.rs_test_acc = []
        self.rs_time_cost = []
        self.updates = []
        self.result_path = "../results/"
        self.task = task
        self.env = env
        # self.global_model = self.clone_model(args,args.model)
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
            backbone, self.task, args.lora_rank, args.lora_alpha, args.lora_dropout, num_det_classes=args.num_classes).to(args.device)
        elif 'segmentation' in self.task:
            self.rsu_rect = rsu[2]
            args.num_classes = 19
            self.global_model = VisionModel(
            backbone, self.task, args.lora_rank, args.lora_alpha, args.lora_dropout, num_seg_classes=args.num_classes).to(args.device)
        elif 'classification' in self.task:
            self.rsu_rect = rsu[1]
            args.num_classes = 58
            self.global_model = VisionModel(
            backbone, self.task, args.lora_rank, args.lora_alpha, args.lora_dropout, num_cls=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRA)
        
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
        print(f"Total LoRA parameters: {self.total_lora_params}")

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
            client.train(self.global_model, self.total_lora_params)
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
        eps_para = 0
        for client in self.clients:
            eps_para += client.uploaded_bytes
        print("eps_para: ", eps_para/1e6 + 7)
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
        print(rsu_clients)
        if self.task == 'detection':
            data_lens = np.array([150, 300, 450])
            client_traces = rsu_clients[0]
            self.num_clients = self.env.get_rsu_client_counts()[0]
        elif self.task == 'segmentation':
            data_lens = np.array([70, 140, 210])
            client_traces = rsu_clients[2]
            self.num_clients = self.env.get_rsu_client_counts()[2]
        else:
            # data_lens = np.array([100, 200, 300])
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
            
            if self.args.base_model == 'vit':
                min_layers_limit = 4
                max_layers_limit = 11
            elif self.args.base_model == 'swin':
                min_layers_limit = 6
                max_layers_limit = 23
            

            client_layers = int(min_layers_limit + (max_layers_limit - min_layers_limit) * test_sample_rate)
            # print(test_sample_rate)
            client = clientObj(self.args,
                               id=vid,
                               train_samples=data_lens[idx],
                               test_samples= int(len(test_data) * test_sample_rate),
                               task = self.task,
                               pos = pos,
                               client_layers = client_layers
                               )
            self.clients.append(client)
        print(self.clients)

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


    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        agg_state_dict = self.global_model.state_dict()
        # 保存初始LoRA模块的参数状态
        initial_state_dict = {name: param.clone() for name, param in self.global_model.named_parameters()}
        target_keys = [k for k in agg_state_dict if 'lora' in k or 'head' in k or 'norm' in k or 'Prompt_Tokens' in k]
        global_dict = {k: agg_state_dict[k].clone() * 0 for k in target_keys}
        count_dict = {k: 0 for k in target_keys}
        
        for client_update in self.updates:
            for key, value in client_update.items():
                if key in global_dict:
                    global_dict[key] += value
                    count_dict[key] += 1
        
        for key in target_keys:
            if count_dict[key] > 0:
                global_dict[key] = global_dict[key] / count_dict[key]
            else:
                global_dict[key] = agg_state_dict[key]
        
        self.global_model.load_state_dict(global_dict, strict=False)
        # 在每个epoch后检查LoRA模块的更新情况
        self.check_lora_updates(initial_state_dict)
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
        file_name = "./out/" + str(self.args.num_clients) + "_client/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_results.csv"
        # file_name = "./out/" + str(self.args.num_of_task) + "_task/" + algorithm + "_" + self.args.base_model + "_" + self.task + "_results.csv"
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
