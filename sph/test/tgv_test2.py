from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver_new import SPHSolver, Space
from fealpy.cfd.sph.particle_kernel_function_new import QuinticKernel

EPS = bm.finfo(float).eps
dx = 0.003
dy = dx
h = dx 
Vmax = 1.0 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1.0 #参考密度
eta0 = 0.01
nu = 1.0 #运动粘度
T = 2 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
dim = 2 #维数
box_size = bm.array([1.0,1.0], dtype=bm.float64) #模拟区域
path = "./"

mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
space = Space()
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

node_self, neighbors = bm.query_point(mesh.nodedata["position"], mesh.nodedata["position"], 3*h, box_size, True, [True, True, True])

for i in range(100):
    print(i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    r = mesh.nodedata["position"]
    node_self, neighbors = bm.query_point(r, r, 3*h, box_size, True, [True, True, True])

    r_i_s, r_j_s = r[neighbors], r[node_self]
    dr_i_j = bm.vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = bm.vmap(kernel.value)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS) # (dr/dx,dr/dy)
    grad_w_dist_norm = bm.vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    mesh.nodedata['rho'] = solver.compute_rho(mesh.nodedata['mass'], neighbors, w_dist)
    p = solver.tait_eos(mesh.nodedata['rho'], c0, rho0)
    background_pressure_tvf = solver.tait_eos(bm.zeros_like(p), c0, rho0)

    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(\
            mesh.nodedata, neighbors, node_self, dr_i_j, dist, grad_w_dist_norm, p)

    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)
    

import numpy as np
import matplotlib.pyplot as plt

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

# 计算逐个粒子的误差
error_u = u_numeric - u_exact
error_v = v_numeric - v_exact

# 计算每个粒子的均方误差 (MSE)
mse_u = error_u ** 2
mse_v = error_v ** 2

# 计算每个粒子的 L2 误差
l2_error = np.sqrt(mse_u + mse_v)

# 输出误差
print(f"Time: {t:.4f}, MSE_u: {np.mean(mse_u):.6f}, MSE_v: {np.mean(mse_v):.6f}, L2 Error: {np.mean(l2_error):.6f}")

# 可视化误差图
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 绘制 u 的 MSE 误差图
sc1 = ax[0].scatter(x, y, c=mse_u, cmap='coolwarm', s=10)
ax[0].set_title("MSE in u-velocity")
plt.colorbar(sc1, ax=ax[0])

# 绘制 v 的 MSE 误差图
sc2 = ax[1].scatter(x, y, c=mse_v, cmap='coolwarm', s=10)
ax[1].set_title("MSE in v-velocity")
plt.colorbar(sc2, ax=ax[1])

# 保存图像
plt.savefig('mse_error.png', dpi=300)
plt.show()
