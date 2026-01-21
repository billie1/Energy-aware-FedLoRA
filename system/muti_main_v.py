#!/usr/bin/env python
import os

import copy
import torch
import argparse

import time
import warnings
import numpy as np
import torchvision
import logging

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.rank_adaption import TaskRankAllocator


from flcore.trainmodel.vision.VisionModel import VisionModel
from flcore.servers.fedra_server import serverFedRA
from flcore.servers.fedavg_server import serverFed
from flcore.servers.homo_v_server import serverHomoLoRA
from flcore.servers.het_v_server import serverHetLoRA
from flcore.servers.my_v_server import Myserverlora

from flcore.src.env import *

import json
import csv
import psutil
from pynvml import *

nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def set_all_seeds(seed):
    """设置所有常见库的随机种子"""
    import random
    import numpy as np
    import torch
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 额外设置避免非确定性算法
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 使用示例
set_all_seeds(0)

rsu_rects = {
    0: {'x_min': 0, 'x_max': 500, 'y_min': 0, 'y_max': 500},
    1: {'x_min': 500, 'x_max': 1000, 'y_min': 0, 'y_max': 500},
    2: {'x_min': 0, 'x_max': 1000, 'y_min': 500, 'y_max': 1000}
}

# rsu_rects = {
#     0: {'x_min': 0, 'x_max': 1000, 'y_min': 0, 'y_max': 1000},
#     1: {'x_min': 0, 'x_max': 1000, 'y_min': 0, 'y_max': 1000},
#     2: {'x_min': 0, 'x_max': 1000, 'y_min': 0, 'y_max': 1000}
# }


def run(args):
    time_list = []
    reporter = MemReporter()
    task_allocator = TaskRankAllocator(args.total_rank, num_tasks=args.num_of_task, warmup_rounds=6)
    # 示例使用
    folder_path = "/home/zhengbk/PFLoRA-lib/dataset/T-driven"  # 包含1.txt, 2.txt等文件的文件夹路径
    relative_motions = extract_relative_motion(folder_path, vehicle_num=args.num_clients, traj_len=args.global_rounds)
    vehicle_trajs = generate_vehicle_trajs_with_init(relative_motions, rsu_rects, args.num_of_task)
    env = VehicleNetEnv(map_size=(1000, 1000), rsus=rsu_rects, vehicle_trajs=vehicle_trajs)
    print("Creating server and clients ...")
    start = time.time()

    # 任务队列（按 Token -> Seq -> Choice 轮流执行）
    if args.num_of_task == 1:
        # task_list = ["detection"]
        # task_list = ["segmentation"]
        task_list = ["classification"]
    elif args.num_of_task == 2:
        task_list = ["detection", "segmentation"]
    else:
        task_list = ["detection", "segmentation", "classification"]
    server_list = []

    # 初始化   
    for i in range(len(task_list)):
        if args.algorithm == "HomoLoRA":
            server = serverHomoLoRA(args, i, env, rsu_rects, task_list[i])
        elif args.algorithm == "FedAvg":
            server = serverFed(args, i, env, rsu_rects, task_list[i])
        elif args.algorithm == "FedRA":
            server = serverFedRA(args, i, env, rsu_rects, task_list[i])
        elif args.algorithm == "MyLoRA":
            server = Myserverlora(args, i, env, rsu_rects, task_allocator.eta[i], task_list[i])
        elif args.algorithm == "HetLoRA":
            server = serverHetLoRA(args, i, env, rsu_rects, task_list[i])
        else:
            raise NotImplementedError
        server_list.append(server)
    reward_list = []
    
    for i in range(args.global_rounds):
        if args.algorithm == "MyLoRA":
            ser_eta = []
            ser_tot = []
            ser_acc = []
        for server in server_list:
            server.train_one_eps(i)
            if args.algorithm == "MyLoRA":
                ser_acc.append(server.acc)
                ser_eta.append(server.current_ranks)
                ser_tot.append(server.total_rank)
        if args.algorithm == "MyLoRA" and args.ablation == "n":
            print(ser_eta)
            new_etas = task_allocator.update_rank(i+1, ser_acc, ser_eta)
            if new_etas != ser_tot:
                print("\n-------------Begin Task Rank Allocator-------------")
                for i in range(len(server_list)):
                    print(f"{task_list[i]} Task New Rank Budget: {new_etas[i]}\n")
                    server_list[i].total_rank = new_etas[i]
                    server_list[i].adj_rank()
        # for server in server_list:
        #     import json
        #     import csv
        #     save_file = "./out/rank_com/rank_" + str(server.task) + "_" + str(server.args.lora_rank) + "_" + "results.csv"
        #     # save_file = "./out/rank_com/rank_" + str(server.task) + "_full_results.csv"
        #     with open(save_file, mode="w", newline="") as file:
        #         writer = csv.writer(file)
        #         writer.writerow(["Accuracy", "Test Loss"])  # CSV文件的表头
        #         for acc, loss in zip(server.rs_test_acc, server.test_losses):
        #             writer.writerow([acc, loss])  # 将每个数据点写入CSV文件
                        
    if args.ablation == "y":
        args.algorithm += "_wo"
    for server in server_list:
        server.record(args.algorithm)

    all_eps_rewards = [server.eps_rewards for server in server_list]

    lengths = set(len(rewards) for rewards in all_eps_rewards)
    if len(lengths) != 1:
        raise ValueError("所有服务器的 eps_rewards 长度不一致！")

    # 逐项求和
    sum_rewards = [sum(items) for items in zip(*all_eps_rewards)]
    

    file_name = "./out/" + str(args.num_of_task) + "_task/" + args.algorithm + "_" + args.base_model + "_reward.csv"
    # file_name = "./out/" + str(args.num_clients) + "_client/" + args.algorithm + "_" + args.base_model + "_reward.csv"
    # file_name = "./out/" + str(args.num_clients) + "_client/" + args.algorithm + "_reward.csv"
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Eps_reward"])  # CSV文件的表头，放在列表中
        for r in sum_rewards:
            writer.writerow([r])  # 将每个数据点放在列表中写入CSV文件

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-abl', "--ablation", type=str, default="n", choices=["y", "n"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=1e-5,
                        help="Local learning rate")
    parser.add_argument('-nt', "--num_of_task", type=int, default=3)
    parser.add_argument('-ld', "--learning_rate_decay",
                        type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma",
                        type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=40)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="MyLoRA")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=9,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name",
                        type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new",
                        type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    # LoRA
    parser.add_argument('--hetlora_min_rank', type=int,
                       default=8, help="minimum lora rank for HetLoRA")
    parser.add_argument('--hetlora_max_rank', type=int,
                       default=64, help="maximum lora rank for HetLoRA")
    parser.add_argument('--hetlora_gamma', type=float, default=0.99,
                       help="gamma for self rank pruning in HetLoRA")

    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of global epochs')
    parser.add_argument('--lora_alpha', type=int, default=32,
                   help='Alpha parameter for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                   help='Dropout rate for LoRA layers')
    parser.add_argument('-rank', '--lora_rank', type=int, default=33,
                   help='Rank for LoRA layers')

    parser.add_argument('-tr', "--total_rank", type=int, default=300,
                   help='Total Rank for all tasks')
    parser.add_argument('-bm', "--base_model", type=str, default="vit",
                   help='Base model')


    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(
            args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda")
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(
        args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)

