from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver_new import SPHSolver, Space
from fealpy.cfd.sph.particle_kernel_function_new import QuinticKernel

EPS = bm.finfo(float).eps
dx = 0.02
h = dx
n_walls = 3 #墙粒子层数
L, H = 1.0, 0.2
dx2n = dx * n_walls * 2 #墙粒子总高度
box_size = bm.array([L, H + dx2n], dtype=bm.float64) #模拟区域
rho0 = 1.0 #参考密度
V = 1.0 #估算最大流速
v0 = 10.0 #参考速度
c0 = v0 * V #声速
gamma = 1.0
p0 = (rho0 * c0**2)/ gamma #参考压力
p_bg = 5.0 #背景压力,用于计算压力p
tvf = 0 #控制是否使用运输速度
path = "./"

#时间步长和步数
T = 1.5
dt = 0.00045454545454545455
t_num = int(T / dt)

mesh = NodeMesh.from_heat_transfer_domain(dx=dx,dy=dx)
solver = SPHSolver(mesh)
space = Space()
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

mesh.nodedata["p"] = solver.tait_eos(mesh.nodedata["rho"],c0,rho0,X=p_bg)
mesh.nodedata = solver.boundary_conditions(mesh.nodedata, box_size, dx=dx)

node_self, neighbors = bm.query_point(mesh.nodedata["position"], mesh.nodedata["position"], 3*h, box_size, True, [True, True, True])

for i in range(1000):
    print("i:", i)
    mesh.nodedata["mv"] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata["tv"] = mesh.nodedata["mv"] + tvf*0.5*dt*mesh.nodedata["dtvdt"]
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0*dt*mesh.nodedata["tv"])

    r = mesh.nodedata["position"]
    node_self, neighbors = bm.query_point(mesh.nodedata["position"], mesh.nodedata["position"], 3*h, box_size, True, [True, True, True])
    
    r_i_s, r_j_s = r[neighbors], r[node_self]
    dr_i_j = bm.vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = bm.vmap(kernel)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS) # (dr/dx,dr/dy)
    grad_w_dist_norm = bm.vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    #外加速度场
    g_ext = solver.external_acceleration(mesh.nodedata["position"], box_size, dx=dx)
    
    #标记
    wall_mask = bm.where(bm.isin(mesh.nodedata["tag"], bm.array([1, 3])), 1.0, 0.0)
    fluid_mask = bm.where(mesh.nodedata["tag"] == 0, 1.0, 0.0) > 0.5

    #密度处理
    rho_summation = solver.compute_rho(mesh.nodedata["mass"], neighbors, w_dist)
    rho = bm.where(fluid_mask, rho_summation, mesh.nodedata["rho"])
    
    #计算压力和背景压力
    p = solver.tait_eos(rho,c0,rho0,X=p_bg)
    pb = solver.tait_eos(bm.zeros_like(p),c0,rho0,X=p_bg)
    
    #边界处理
    p, rho, mv, tv, T = solver.enforce_wall_boundary(mesh.nodedata, p, g_ext, neighbors, node_self, w_dist, dr_i_j, with_temperature=True)
    mesh.nodedata["rho"] = rho
    mesh.nodedata["mv"] = mv
    mesh.nodedata["tv"] = tv
    
    #计算下一步的温度导数
    T += dt * mesh.nodedata["dTdt"]
    mesh.nodedata["T"] = T
    mesh.nodedata["dTdt"] = solver.temperature_derivative(mesh.nodedata, kernel, e_s, dr_i_j, dist, neighbors, node_self, grad_w_dist_norm)

    #更新动量速度的加速度
    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(mesh.nodedata, neighbors, node_self, dr_i_j, dist, grad_w_dist_norm, p)
    mesh.nodedata["dmvdt"] = mesh.nodedata["dmvdt"] + g_ext
    mesh.nodedata["p"] = p

    #更新运输速度的加速度
    mesh.nodedata["dtvdt"] = solver.compute_tv_acceleration(mesh.nodedata, neighbors, node_self, dr_i_j, dist, grad_w_dist_norm, pb)
    
    #更新边界条件
    mesh.nodedata = solver.boundary_conditions(mesh.nodedata, box_size, dx=dx)

    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)