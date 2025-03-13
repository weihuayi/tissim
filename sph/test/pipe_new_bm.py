from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver_new import SPHSolver
from fealpy.cfd.sph.particle_kernel_function_new import WendlandC2Kernel

import torch # 打印
import numpy as np
torch.set_printoptions(threshold=torch.inf)
torch.set_printoptions(precision=16, threshold=torch.inf)
np.set_printoptions(precision=16, threshold=np.inf)
import matplotlib.pyplot as plt

dx = 1.25e-4
H = 1.5 * dx
dt = bm.array([1e-7], dtype=bm.float64)
u_in = bm.array([5.0, 0.0], dtype=bm.float64)
domain = [0.0, 0.05, 0.0, 0.005]
init_domain = [0.0, 0.005, 0, 0.005]
side = bm.array([0.0508, 0.0058], dtype=bm.float64)

#粘性本构方程参数
n = 0.3083
tau_star = 16834.4
mu_0 = 938.4118
#控制方程参数
eta = 0.5
#Tait 模型状态方程参数
c_1 = 0.0894
B = 5.914e7
rho_0 = 737.54

mesh = NodeMesh.from_pipe_domain(domain, init_domain, H, dx)
solver = SPHSolver(mesh)
kernel = WendlandC2Kernel(h=H, dim=2)

#wall-virtual
v_node_self, w_neighbors = solver.wall_virtual(mesh.nodedata["position"], mesh.nodedata["tag"])

for i in range(15000):
    print(i)
    #阀门粒子更新
    state = mesh.nodedata
    state = solver.gate_change(state, dx, domain, H, u_in, rho_0, dt)

    r = state["position"]
    rho = state["rho"]
    u  =state["u"]
    node_self, neighbors = bm.query_point(r, r, 2*H, side, True, [False, False, False])
    w_dist, grad_w_dist, dr_i_j, dist = solver.kernel_grad(r, node_self, neighbors, kernel) #kernel,grad_kernel,位移差，距离

    # fuild-wall-virtual
    f_node, fwvg_neighbors, dr, dis, w, dw = solver.fuild_fwvg(state, node_self, neighbors, dr_i_j, dist, w_dist, grad_w_dist) 

    # wall-fuild
    w_node, fg_neighbors, fg_w = solver.wall_fg(state, node_self, neighbors, w_dist)
    state["u"] = solver.vtag_u(state, v_node_self, w_neighbors, w_node, fg_neighbors, fg_w)
    
    # 更新半步压强和声速
    state["p"] = solver.fuild_p(state, B, rho_0, c_1)
    state["p"] = solver.wall_p(state, w_node, fg_neighbors, fg_w)
    state["p"] = solver.virtual_p(state, v_node_self, w_neighbors)
    state["sound"] = solver.sound(state, B, rho_0, c_1)

    # 更新半步密度和半步质量，质量没有更新，需要更新吗？
    A_s0 = solver.A_matrix(state, f_node, fwvg_neighbors, dr, dw, mask_self=False)
    state["drhodt"] = solver.change_rho(state, f_node, fwvg_neighbors, dr, dis, dw, A_s0)
    drho_0 = state["drhodt"]
    state["rho"] = state["rho"] + 0.5 * dt * state["drhodt"]
    
    # 更新半步速度
    state["mu"] = solver.mu_wlf(state, node_self, neighbors, grad_w_dist, mu_0, tau_star, n)
    state["dudt"] = solver.change_u(state, f_node, fwvg_neighbors, dis, dr, dw, H, eta, A_s0)
    du_0 = state["dudt"]
    state["u"] = state["u"] + 0.5 * dt * state["dudt"]

    # 内部和精确自由表面流体粒子的索引
    A_s1 = solver.A_matrix(state, f_node, fwvg_neighbors, dr, dw, mask_self=True)
    in_f, free, dC_i , normal = solver.free_surface(state, f_node, fwvg_neighbors, dr, dis, w, dw, A_s1, H)
    state["drdt"] = solver.shifting_r(state, in_f, free, dC_i, normal, dt, H)
    drdt_0 = state["drdt"]
    state["position"] = state["position"] + 0.5 * dt * state["drdt"]

    state["u"] = solver.vtag_u(state, v_node_self, w_neighbors, w_node, fg_neighbors, fg_w)

    # 更新压强和声速
    state["p"] = solver.fuild_p(state, B, rho_0, c_1)
    state["p"] = solver.wall_p(state, w_node, fg_neighbors, fg_w)
    state["p"] = solver.virtual_p(state, v_node_self, w_neighbors)
    state["sound"] = solver.sound(state, B, rho_0, c_1)

    # 更新密度和质量
    state["drhodt"] = solver.change_rho(state, f_node, fwvg_neighbors, dr, dis, dw, A_s0)
    drho_1 = state["drhodt"]
    state["rho"] = state["rho"] + 0.5 * dt * state["drhodt"]
    
    # 更新速度
    state["mu"] = solver.mu_wlf(state, node_self, neighbors, grad_w_dist, mu_0, tau_star, n)
    state["dudt"] = solver.change_u(state, f_node, fwvg_neighbors, dis, dr, dw, H, eta, A_s0)
    du_1 = state["dudt"]
    state["u"] = state["u"] + 0.5 * dt * state["dudt"]

    # 内部和精确自由表面流体粒子的索引
    drdt_1 = solver.shifting_r(state, in_f, free, dC_i, normal, dt, H)

    state["rho"] = rho + 0.5 * dt * (drho_0 + drho_1)
    state["u"] = u + 0.5 * dt * (du_0 + du_1)
    state["u"] = solver.vtag_u(state, v_node_self, w_neighbors, w_node, fg_neighbors, fg_w)
    state['position'] = r + 0.5 * dt * (drdt_0 + drdt_1)
    
    solver.draw(state, i)

'''
color = np.full_like(state["tag"], 'blue', dtype=object)
color[state["tag"] == 1] = 'red'
color[state["tag"] == 2] = 'green'
color[state["tag"] == 3] = 'black'
color[free] = 'orange'
plt.scatter(state['position'][:, 0], state['position'][:, 1] ,c=color,s=5)
plt.colorbar(cmap='jet')
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig(f"frames/step_{i:05d}.png", dpi=150)
'''