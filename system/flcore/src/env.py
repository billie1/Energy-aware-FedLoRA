# env.py

import os
import numpy as np
from collections import defaultdict
import random
from math import ceil

random.seed(0)

def extract_relative_motion(folder_path, vehicle_num=9, traj_len=100, max_motion_range=10):
    """
    提取相对轨迹位移，并统一缩放，使每辆车轨迹总变化不超过max_motion_range
    """
    motions = []
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    random.shuffle(file_list)

    for fname in file_list:
        with open(os.path.join(folder_path, fname), 'r') as f:
            lines = f.readlines()

        raw_traj = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            _, _, lon, lat = parts
            try:
                lat = float(lat)
                lon = float(lon)
            except:
                continue
            raw_traj.append((lat, lon))
            if len(raw_traj) >= traj_len:
                break

        if len(raw_traj) < 2:
            continue

        base_lat, base_lon = raw_traj[0]
        rel_traj = []
        for lat, lon in raw_traj:
            dx = (lon - base_lon) * 100000
            dy = (lat - base_lat) * 100000
            rel_traj.append((dx, dy))

        # 缩放：限制每个轨迹在x/y方向最大跨度不超过 max_motion_range
        xs = [p[0] for p in rel_traj]
        ys = [p[1] for p in rel_traj]
        dx_range = max(xs) - min(xs)
        dy_range = max(ys) - min(ys)
        max_range = max(dx_range, dy_range)
        if max_range == 0:
            scale = 1.0
        else:
            scale = max_motion_range / max_range

        scaled_traj = [(round(dx * scale), round(dy * scale)) for dx, dy in rel_traj]
        motions.append(scaled_traj)

        if len(motions) >= vehicle_num:
            break

    print(f"✅ 提取并缩放 {len(motions)} 条轨迹，最大位移限制：{max_motion_range}")
    return motions


def generate_vehicle_trajs_with_init(relative_motions, rsu_rects, num_tasks):
    """
    将每条相对轨迹叠加在初始点上生成完整轨迹

    :param relative_motions: List[List[(dx, dy)]]
    :param rsu_rects: Dict[rsu_id -> {'x_min', 'x_max', 'y_min', 'y_max'}]
    :return: Dict[vehicle_id -> List[(x, y)]]
    """
    vehicle_trajs, vid = {}, 0
    total = len(relative_motions)
    
    # ✅ 核心：根据num_tasks确定车辆生成的RSU区域列表（一行搞定分配规则）
    if num_tasks == 1:
        use_rsus = [rsu_rects[1]]  # 仅rsu2
    elif num_tasks == 2:
        use_rsus = [rsu_rects[0], rsu_rects[2]]
    else:  # num_tasks=3 直接取全部
        use_rsus = list(rsu_rects.values())

    # ✅ 通用均匀分配逻辑（所有场景复用，无重复代码）
    per_rsu, remain = total // len(use_rsus), total % len(use_rsus)
    for rect in use_rsus:
        curr_num = per_rsu + (1 if remain > 0 else 0)
        remain -= 1 if remain > 0 else 0
        # 生成当前RSU区域的车辆轨迹
        for _ in range(curr_num):
            if vid >= total: break
            x0 = random.uniform(rect['x_min']+20, rect['x_max']-20)
            y0 = random.uniform(rect['y_min']+20, rect['y_max']-20)
            traj = [(round(min(max(0, x0+dx), 1000)), round(min(max(0, y0+dy), 1000))) for dx, dy in relative_motions[vid]]
            vehicle_trajs[vid] = traj
            vid += 1

    print(f"✅ 生成{len(vehicle_trajs)}条轨迹 | num_tasks={num_tasks}")
    return vehicle_trajs


class VehicleNetEnv:
    def __init__(self, map_size=(1000, 1000), rsus=None, vehicle_trajs=None):
        """
        :param map_size: tuple, 整个区域的大小 (x_max, y_max)
        :param rsus: Dict[rsu_id -> Dict{'x_min', 'x_max', 'y_min', 'y_max'}]
        :param vehicle_trajs: Dict[vehicle_id -> List[(x, y)]]
        """
        self.map_size = map_size
        self.rsus = rsus or {}  # 注意：现在是 Dict[int -> dict]
        self.vehicles = vehicle_trajs or {}  # {vehicle_id: [(x, y), ...]}
        self.round = 0

    def step(self):
        """前进一步，模拟车辆移动"""
        self.round += 1

    def get_rsu_clients(self):
        """
        获取每个 RSU 当前覆盖范围内的客户端信息（包含车辆id和位置）
        :return: Dict[rsu_id -> List[{'vehicle_id': int, 'position': (x, y)}]]
        """
        rsu_clients = defaultdict(list)

        for vid, traj in self.vehicles.items():
            if self.round >= len(traj):
                continue
            vx, vy = traj[self.round]
            for rsu_id, rect in self.rsus.items():
                if rect['x_min'] <= vx <= rect['x_max'] and rect['y_min'] <= vy <= rect['y_max']:
                    rsu_clients[rsu_id].append({
                        'vehicle_id': vid,
                        'position': (vx, vy)
                    })
        # print(rsu_clients)
        return rsu_clients

    def get_rsu_client_counts(self):
        """
        获取每个 RSU 当前轮的覆盖车辆数量
        :return: Dict[rsu_id -> int]
        """
        client_map = self.get_rsu_clients()
        return {rsu_id: len(clients) for rsu_id, clients in client_map.items()}

    def get_vehicle_position(self, vid):
        """
        获取车辆当前位置（根据当前通信轮数）
        """
        if self.round < len(self.vehicles[vid]):
            return self.vehicles[vid][self.round]
        return self.vehicles[vid][-1]

    def get_all_positions(self):
        """
        获取所有车辆当前的位置字典
        :return: Dict[vehicle_id -> (x, y)]
        """
        positions = {}
        for vid in self.vehicles:
            pos = self.get_vehicle_position(vid)
            if pos is not None:
                positions[vid] = pos
        return positions

    def reset(self):
        """重置轨迹回到初始位置"""
        self.round = 0


