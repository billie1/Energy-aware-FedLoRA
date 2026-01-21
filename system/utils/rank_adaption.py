import numpy as np
import math

class TaskRankAllocator:
    def __init__(self, eta_total, num_tasks, warmup_rounds, lambda_=0.9, gamma=2, epsilon=1e-5):
        self.eta_total = eta_total
        self.T = num_tasks
        self.Q = warmup_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize rank allocation
        base = eta_total // num_tasks
        remainder = eta_total % num_tasks
        self.eta = [base + 1 if t < remainder else base for t in range(num_tasks)]
        
        # State tracking
        self.h = np.zeros(num_tasks)  # 任务难度系数
        self.L_prev = [None] * num_tasks  # 上一轮损失值
        self.sum_delta_L = np.zeros(num_tasks)  # 累计损失变化
        self.count_delta_L = np.zeros(num_tasks)  # 更新次数统计

    def update_rank(self, round, current_acc, effective_rank):
        """
        Update rank allocation for current communication round
        
        :param round:       当前通信轮次 (1-based)
        :param current_acc: 各任务当前损失值列表
        :param effective_rank: 各任务实际使用秩数列表
        :return:            新的秩分配列表
        """

        # Calculate task weights
        weights = []
        for t in range(self.T):
            # 1. Update task difficulty coefficient
            # if self.L_prev[t] is not None:
            #     delta_L = current_acc[t] - self.L_prev[t]
            # else: 
            #     delta_L = 0
            #     self.L_prev[t] = current_acc[t]
            self.h[t] = self.lambda_ * self.h[t] + (1 - self.lambda_) * (self.eta[t] / current_acc[t])
            
            # 2. Calculate resource utilization
            utilization = effective_rank[t] / self.eta[t] if self.eta[t] != 0 else 0
            
            # 3. Compute dynamic weight
            weights.append((self.h[t] ** self.gamma) * utilization)

        # Normalize weights and allocate remaining budget
        total_weight = sum(weights)
        remaining = self.eta_total - sum(effective_rank)

        if round % self.Q != 0 or round >= 18:
            return self.eta.copy()
        
        new_eta = []
        for t in range(self.T):
            if total_weight > 0:
                delta = np.round(weights[t] * remaining / total_weight)
            else:
                delta = 0
                
            # Apply allocation constraints
            allocated = min(effective_rank[t] + delta, 0.7 * self.eta_total)
            allocated = max(allocated, 24)  # Ensure non-negative
            new_eta.append(allocated)
        
        # Update system state
        self.eta = new_eta
        
        return self.eta.copy()


class UCBRankAllocator:
    def __init__(self, total_rank, task, alpha, beta, gamma):
        """
        初始化UCBRankAllocator
        total_rank: 总的rank预算
        alpha: 时延的权重
        beta: 能耗的权重
        gamma: 精度的权重
        """
        self.total_rank = 0  # 总的rank预算
        self.alpha = alpha  # 时延权重
        self.beta = beta  # 能耗权重
        self.gamma = gamma  # 精度权重
        self.task = task
        self.ini_ucb = {'detection': 0, 'segmentation': 155, 'classification': 0}
        # self.ini_ucb = {'detection': 200, 'segmentation': 200, 'classification': 200}
        self.base_rank_options = [8,16,24,32,40,48,56,64,128,168,200]
        self.stats = {}  # 动态维护的统计信息
        
        # 初始化时没有候选，需要后续设置初始rank
        self.rank_options = []

        self.update_max_rank(total_rank, from_base=True)

    def update_max_rank(self, new_max, from_base=False):
        if from_base:
            self.total_rank = new_max
            temp_rank_options = [r for r in self.base_rank_options if r <= self.total_rank]
        else:
            self.total_rank = new_max
            # 1. 保留原 rank_options 中不超过 new_max 的部分
            temp_rank_options = [r for r in self.rank_options if r <= self.total_rank]

            # 2. 添加 base_rank_options 中不超过 new_max 且不在 temp 中的部分
            for r in self.base_rank_options:
                if r <= self.total_rank and r not in temp_rank_options:
                    temp_rank_options.append(r)
        
        if self.total_rank not in temp_rank_options:
            temp_rank_options.append(self.total_rank)
        
        self.rank_options = sorted(set(temp_rank_options))
        
        print(f"Task {self.task} rank list: {self.rank_options}")
        
        # 初始化新候选的统计信息
        for r in self.rank_options:
            if r not in self.stats:
                self.stats[r] = {
                    'total_reward': 0.0,
                    'n': 0,
                    'ucb': self.ini_ucb[self.task]
                }


    def calculate_reward(self, tau_v, e_v, q_v):
        """
        计算客户端的奖励
        """
        return - (self.alpha * tau_v + self.beta * e_v - self.gamma * q_v)

    def update_ucb(self, client, acc, t, lam):
        """
        使用UCB算法更新客户端的选择
        """
        reward = self.calculate_reward(client.time_cost, client.energy_cost, acc)

        # 只更新当前rank的统计
        stats = self.stats[client.model.lora_rank]
        stats['total_reward'] += reward
        stats['n'] += 1

        # 计算UCB值
        if stats['n'] > 0:
            avg_reward = stats['total_reward'] / stats['n'] - lam * client.model.lora_rank
            exploration = np.sqrt(2 * np.log(t) / stats['n'])
            if avg_reward + exploration < stats['ucb'] and stats['ucb'] != self.ini_ucb[self.task]:
                print(f"Task: {client.task}, current rank:{client.model.lora_rank}, ucb value:{stats['ucb']}")
            else:    
                stats['ucb'] = avg_reward + exploration
                print(f"Task: {client.task}, current rank:{client.model.lora_rank}, ucb value:{stats['ucb']}")
            
    def select_rank(self, client):
        """选择相邻rank中UCB值最高的选项"""
        if client.model.lora_rank > self.total_rank:
            return self.total_rank
        print("rank_options: ", self.rank_options)
        candidates = self._get_adjacent_ranks(client)
        
        best_rank = client.model.lora_rank
        best_ucb = self.stats[best_rank]['ucb']
        for r in candidates:
            current_ucb = self.stats[r]['ucb']
            if current_ucb > best_ucb:
                best_ucb = current_ucb
                best_rank = r

        return best_rank

    def _find_max_allowed(self, target):
        """找到不大于target的最大候选值"""
        idx = bisect_right(self.base_rank_options, target) - 1
        return self.base_rank_options[idx] if idx >= 0 else self.base_rank_options[0]

    def _get_adjacent_ranks(self, client):
        """获取当前rank相邻的可选rank"""
        idx = self.rank_options.index(client.model.lora_rank)
        start = max(0, idx-1)
        end = min(len(self.rank_options), idx+2)
        return self.rank_options[start:end]

def shannon_rate(bandwidth, power, distance, noise_power=1e-9, path_loss_exponent=2):
    """根据香农公式计算通信速率（bps）"""
    h = 1 / (distance ** path_loss_exponent + 1e-6)  # 防止除0
    snr = power * h / noise_power
    return bandwidth * math.log2(1 + snr)

class FaultToleranceScheduler:
    def __init__(self, alpha, beta, gamma):
        """
        初始化容错调度模块
        :param alpha: 时延权重
        :param beta: 能耗权重
        :param gamma: 精度权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_cost(self, strategy, target_accuracy, energy_cost, migration_delay, migration_energy, current_accuracy):
        """
        根据选定策略计算代价函数
        :param strategy: 当前选择的策略（0, 1, 2）
        :param target_accuracy: 目标精度阈值
        :param energy_cost: 训练过程中的能耗
        :param migration_delay: 任务迁移时的延迟
        :param migration_energy: 任务迁移时的能耗
        :param current_accuracy: 当前车辆的精度
        :return: 策略对应的代价
        """

        # 策略0：提前上传
        if strategy == 0:
            cost = self.gamma * max(0, target_accuracy - current_accuracy)

        # 策略1：任务迁移
        elif strategy == 1:
            cost = self.alpha * migration_delay + self.beta * migration_energy

        # 策略2：任务放弃
        elif strategy == 2:
            cost = self.beta * energy_cost + self.gamma * target_accuracy

        return cost

    def select_strategy(self, target_accuracy, migration_delay, migration_energy, energy_cost, current_accuracy):
        """
        为每个客户端选择最优策略
        :param target_accuracy: 目标精度
        :param migration_delay: 任务迁移时的延迟
        :param migration_energy: 任务迁移时的能耗
        :param energy_cost: 当前的能耗
        :param current_accuracy: 当前精度
        :return: 最优策略
        """
        # 计算每种策略的代价
        costs = []
        strategy_names = {0: "提前上传", 1: "任务迁移", 2: "任务放弃"}
        for strategy in [0, 1, 2]:
            cost = self.calculate_cost(strategy, target_accuracy, energy_cost, migration_delay, migration_energy, current_accuracy)
            print(f"策略{strategy}的代价：{cost}")
            costs.append(cost)

        # 选择代价最小的策略
        optimal_strategy = np.argmin(costs)

        
        print(f"\n最优策略: {optimal_strategy} ({strategy_names[optimal_strategy]}), 最小 cost = {costs[optimal_strategy]:.4f}")

        return optimal_strategy, costs[optimal_strategy]


def discount_factor(x, k=2.0):
    
    normalized = (51 - x) / (51 - 8)
    return 1 - 0.5 * normalized ** k



