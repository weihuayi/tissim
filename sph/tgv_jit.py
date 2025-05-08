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

start = time.time()
for i in range(1000):
    print(i)
    mesh.nodedata, neighbors = advance(dt, mesh.nodedata, neighbors) 
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)

end = time.time()
print(end-start)

# 解析解参数
Re = 100
b = -8 * np.pi / Re
U = 1
L = 1

# 获取粒子位置
x = np.array(mesh.nodedata["position"][:, 0])
y = np.array(mesh.nodedata["position"][:, 1])

# 计算当前时间
t = i * dt  # 假设 i 是当前时间步索引

# 计算解析解
u_exact = -U * np.exp(b * t) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
v_exact = U * np.exp(b * t) * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

# 读取数值解
u_numeric = np.array(mesh.nodedata["mv"][:, 0])
v_numeric = np.array(mesh.nodedata["mv"][:, 1])

# 计算误差
error_u = u_numeric - u_exact
error_v = v_numeric - v_exact

# 计算均方误差 (MSE)
mse_u = np.mean(error_u ** 2)
mse_v = np.mean(error_v ** 2)
l2_error = np.sqrt(mse_u + mse_v)

# 输出误差
print(f"Time: {t:.4f}, MSE_u: {mse_u:.6f}, MSE_v: {mse_v:.6f}, L2 Error: {l2_error:.6f}")

# 可视化误差分布
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sc1 = ax[0].scatter(x, y, c=error_u, cmap='coolwarm', s=10)
ax[0].set_title("Error in u-velocity")
plt.colorbar(sc1, ax=ax[0])

sc2 = ax[1].scatter(x, y, c=error_v, cmap='coolwarm', s=10)
ax[1].set_title("Error in v-velocity")
plt.colorbar(sc2, ax=ax[1])

plt.savefig('particle_positions.png', dpi=300)