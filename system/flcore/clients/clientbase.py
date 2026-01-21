import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Subset, RandomSampler
# from flcore.trainmodel.models import *


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        # self.model = self.clone_model(args,args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # # check BatchNorm
        # self.has_BatchNorm = False
        # for layer in self.model.children():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         self.has_BatchNorm = True
        #         break

        self.train_slow = kwargs.get('train_slow', False)
        self.send_slow = kwargs.get('send_slow', False)
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )
        self.learning_rate_decay = args.learning_rate_decay

    def clone_model(self, args, model):
        # 将模型的参数保存到一个 OrderedDict 中
        params = model.state_dict()
        if args.algorithm == "HetLoRA":
            cloned_model = type(model)(
                model.base_model,
                model.hetlora_min_rank,
                model.hetlora_max_rank,
                model.hetlora_gamma,
                model.lora_alpha,
                model.lora_dropout
            )
            # 将保存的参数加载到新的模型实例中
            cloned_model.load_state_dict(params)
        elif args.algorithm == "HomoLoRA":
            cloned_model = type(model)(
                model.base_model,
                model.lora_rank,
                model.lora_alpha,
                model.lora_dropout
            )
            # 将保存的参数加载到新的模型实例中
            cloned_model.load_state_dict(params)

        else:
            cloned_model = copy.deepcopy(model)

        return cloned_model

    def load_train_data(self, train_samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # 读取完整的训练数据集
        full_train_data = read_client_data(self.dataset, self.id, is_train=True)

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
        test_data = read_client_data(self.dataset, self.id, is_train=False)
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

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(),
                                   classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" +
                  str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
