import matplotlib.pyplot as plt
from env import VehicleNetEnv, extract_relative_motion, generate_vehicle_trajs_with_init

# 1. 设定RSU矩形区域（范围为左下角和右上角）
rsu_rects = {
    0: {'x_min': 0, 'x_max': 500, 'y_min': 0, 'y_max': 500},
    1: {'x_min': 500, 'x_max': 1000, 'y_min': 0, 'y_max': 500},
    2: {'x_min': 0, 'x_max': 1000, 'y_min': 500, 'y_max': 1000}
}

# 2. 加载轨迹数据（根据你的文件夹路径修改）
folder_path = "/home/zhengbk/PFLoRA-lib/dataset/T-driven"  # 修改为真实路径
relative_motions = extract_relative_motion(folder_path, vehicle_num=9, traj_len=10)
vehicle_trajs = generate_vehicle_trajs_with_init(relative_motions, rsu_rects)

# 3. 初始化模拟环境（RSU为中心点加半径，用不到也没关系）
# dummy_rsus = []  # 不使用圆形范围，这里传空
# env = VehicleNetEnv(map_size=(1000, 1000), rsus=dummy_rsus, vehicle_trajs=vehicle_trajs)

# 4. 可视化
colors = plt.cm.get_cmap('tab10', len(vehicle_trajs))  # 不同轨迹不同颜色

plt.figure(figsize=(8, 8))
plt.title("车辆移动轨迹与RSU矩形覆盖区域")

# 绘制每个车辆的轨迹线
for vid, traj in vehicle_trajs.items():
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.plot(xs, ys, label=f"Vehicle {vid}", color=colors(vid), linewidth=2)
    plt.scatter(xs[0], ys[0], marker='o', color=colors(vid), edgecolors='black', zorder=5)

# 绘制每个RSU的矩形区域
for rsu_id, rect in rsu_rects.items():
    width = rect['x_max'] - rect['x_min']
    height = rect['y_max'] - rect['y_min']
    plt.gca().add_patch(
        plt.Rectangle((rect['x_min'], rect['y_min']), width, height,
                      fill=False, edgecolor='gray', linestyle='--', linewidth=2, label=f"RSU {rsu_id}")
    )
    # 显示编号
    cx = (rect['x_min'] + rect['x_max']) / 2
    cy = (rect['y_min'] + rect['y_max']) / 2
    plt.text(cx, cy, f"RSU {rsu_id}", color='black', fontsize=12, ha='center', va='center')

# 设置边界和图例
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("vehicle_traj.png", dpi=300)