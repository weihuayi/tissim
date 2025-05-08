import jax
#jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
import enum
import jax.numpy as jnp
import numpy as np
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver import SPHSolver,TimeLine
from fealpy.backend.jax import partition 
from fealpy.backend.jax.jax_md.partition import Sparse
from fealpy.cfd.sph.particle_kernel_function import QuinticKernel
#from fealpy.jax.sph.jax_md import space
from jax_md import space
from jax import ops, vmap
from jax import lax, jit
import matplotlib.pyplot as plt
import warnings
import time
from fealpy.backend import backend_manager as bm
bm.set_backend("jax")


EPS = jnp.finfo(float).eps
dx = 0.02
dy = 0.02
h = dx #平滑长度 实际上是3dx
Vmax = 1.0 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1.0 #参考密度
eta0 = 0.01
nu = 1.0 #运动粘度
T = 2 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
dim = 2 #维数
box_size = jnp.array([1.0,1.0]) #模拟区域
path = "./"

#初始化
mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size) #TODO
timeline = TimeLine

#邻近搜索
neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=QuinticKernel(h=h, dim=2).cutoff_radius,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=mesh.nodedata["position"].shape[0],
    num_partitions=mesh.nodedata["position"].shape[0],
    pbc=[True, True, True],
)
forward = solver.forward_wrapper(displacement, kernel)

advance = TimeLine(forward, shift)
advance = jit(advance)
##数据类型？？
warnings.filterwarnings("ignore", category=FutureWarning)
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=mesh.nodedata["position"].shape[0])

def compute_pressure(mesh, rho0, k):
    """计算压力并将其更新到 mesh.nodedata 中"""
    densities = mesh.nodedata["rho"]
    pressure = k * (densities - rho0)  # 使用理想气体状态方程计算压力
    mesh.nodedata["pressure"] = pressure
    return pressure

def check_stability(mesh):
    pressures = mesh.nodedata["pressure"]
    max_pressure = jnp.max(pressures)  # 找到压力的最大值
    return max_pressure

# 在模拟中加入稳定性检查
start = time.time()

# 设定常数 k 和参考密度 rho0
k = 1.0  # 假设常数为 1.0，可以根据需要调整
rho0 = 1.0  # 假设参考密度为 1.0

# 初始化压力
compute_pressure(mesh, rho0, k)

# 用于存储每一步的最大压力
max_pressures = []

for i in range(10000):
    print(i)
    mesh.nodedata, neighbors = advance(dt, mesh.nodedata, neighbors)
    
    if i % 100 == 0:  # 每100步检查一次稳定性
        # 更新压力并检查稳定性
        compute_pressure(mesh, rho0, k)
        max_pressure = check_stability(mesh)
        max_pressures.append(max_pressure)
        
end = time.time()
print(end-start)
print("Max Pressure at each step:", max_pressures)

max_pressures = [float(x) for x in max_pressures]

# 生成步数的列表
steps = list(range(0, len(max_pressures) * 100, 100))

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(steps, max_pressures, marker='o', color='b', label='Max Pressure')

# 添加标题和标签
plt.title("Max Pressure Over Time", fontsize=16)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Max Pressure", fontsize=12)

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.savefig('poly6_kernel_visualization.png', dpi=300)
plt.show()