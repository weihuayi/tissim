from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.equation import CahnHilliard
from fealpy.cfd.simulation.fem import BDF2
from fealpy.cfd.simulation.fem import CahnHilliardModel 
from pde import RayleignTaylor
from solver import two_phase_phield_solver
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.solver import DirectSolverManager
from fealpy.solver import spsolve, cg
import psutil
import time 

bm.set_backend('pytorch')
bm.set_default_device('cuda')

pde = RayleignTaylor()
mesh = pde.init_mesh(nx=16, ny=64)
#mesh = pde.init_mesh(nx=32, ny=128)
refine_value = 0.0001
refine_number = 8

##建立solver
ns_eqaution = IncompressibleNS(pde, init_variables=False) 
ns_solver = BDF2(ns_eqaution)

ch_equation = CahnHilliard(pde, init_variables=False)
ch_solver = CahnHilliardModel(ch_equation, ns_solver.uspace.scalar_space)

solver = two_phase_phield_solver(pde, ns_solver, ch_solver) 

## 设置求解变量函数,加密初始网格
phi0 = ch_solver.space.interpolate(pde.init_interface)
mesh = solver.init_interface_refine(phi0, refine_value, refine_number)

pspace, phispace, uspace = solver.update_space(mesh)
pde.epsilon = 0.08 * bm.sqrt(bm.array(2) * bm.min(mesh.entity_measure('edge')))
pde.Pe = 1/(pde.epsilon * 0.001)
#pde.Pe = 1/pde.epsilon
mincellsize = bm.min(mesh.entity_measure('cell'))
#dt = 0.16*bm.sqrt(2*bm.min(mesh.entity_measure('edge')))
dt = 0.016*bm.sqrt(2*bm.min(mesh.entity_measure('edge')))
ns_solver.dt = dt
ch_solver.dt = dt

phi0 = phispace.interpolate(pde.init_interface)
phi1 = phispace.interpolate(pde.init_interface)
phi2 = phispace.function()
mu1 = phispace.function()
mu2 = phispace.function()

u0 = ns_solver.uspace.function()
u1 = ns_solver.uspace.function()
u2 = ns_solver.uspace.function()
p1 = ns_solver.pspace.function()
p2 = ns_solver.pspace.function()

mesh.nodedata['phi'] = phi1
mesh.nodedata['velocity'] = u1.reshape(2,-1).T  
fname = './' + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)



#BC = DirichletBC((ns_solver.uspace,ns_solver.pspace), gd=(pde.velocity_boundary, pde.pressure_boundary), 
#                      threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp')

#设置参数
ns_eqaution.set_coefficient('viscosity', 1/pde.Re)
ns_eqaution.set_coefficient('pressure', 1)
ch_equation.set_coefficient('mobility', 1/pde.Pe)
ch_equation.set_coefficient('interface', pde.epsilon**2)
ch_equation.set_coefficient('free_energy', 1)


for i in range(1,2000):
    # 设置参数
    print("iteration:", i)
    print("time:", i*dt)
    
    ugdof = ns_solver.uspace.number_of_global_dofs()
    phigdof = ch_solver.space.number_of_global_dofs()
    pgdof = ns_solver.pspace.number_of_global_dofs()
    
    ns_BForm = ns_solver.BForm()
    ns_LForm = ns_solver.LForm()
    ch_BFrom = ch_solver.BForm()
    ch_LForm = ch_solver.LForm()

    is_bd = ns_solver.uspace.is_boundary_dof((pde.is_ux_boundary, pde.is_uy_boundary), method='interp')
    is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
    gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
    BC = DirichletBC((ns_solver.uspace, ns_solver.pspace), gd=gd, threshold=is_bd, method='interp')
    
    t0 = time.time()
    ch_solver.update(u0, u1, phi0, phi1)
    ch_A = ch_BFrom.assembly()
    ch_b = ch_LForm.assembly()
    t1 = time.time()
    ch_x = spsolve(ch_A, ch_b, 'mumps')
    t2 = time.time()

    phi2[:] = ch_x[:phigdof]
    mu2[:] = ch_x[phigdof:]  
    
    # 更新NS方程参数
    t3 = time.time()
    rho = solver.rho(phi1) 
    @barycentric
    def body_force(bcs, index):
        result = rho(bcs, index)
        result = bm.stack((result, result), axis=-1)
        result[..., 0] = (1/pde.Fr) * result[..., 0] * 0
        result[..., 1] = (1/pde.Fr) * result[..., 1] * -1
        return result
    
    ns_eqaution.set_coefficient('time_derivative', rho)
    ns_eqaution.set_coefficient('convection', rho)
    ns_eqaution.set_coefficient('body_force', body_force)

    ns_solver.update(u0, u1)
     
    ns_A = ns_BForm.assembly()
    ns_b = ns_LForm.assembly()
    ns_A,ns_b = BC.apply(ns_A, ns_b)
    t4 = time.time() 
    ns_x = spsolve(ns_A, ns_b, 'mumps')
    t5 = time.time()

    print("CH组装时间:", t1-t0)
    print("求解CH方程时间:", t2-t1)
    print("NS组装时间:", t4-t3)
    print("求解NS方程时间:", t5-t4)
    u2[:] = ns_x[:ugdof]
    p2[:] = ns_x[ugdof:]
        
    u0[:] = u1[:]
    u1[:] = u2[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]
    
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['velocity'] = u2.reshape(2,-1).T  
    mesh.nodedata['pressure'] = p2 
    mesh.nodedata['rho'] = rho
    fname = './' + 'test_'+ str(i).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)

    #自适应加密
    data = [u0, u1, u2, p1, p2, phi0, phi1, phi2, mu1, mu2]
    mesh = solver.interface_refine(phi1, data, refine_value, refine_number, mincellsize)
    u0 = data[0]
    u1 = data[1]
    u2 = data[2]
    p1 = data[3]
    p2 = data[4]
    phi0 = data[5]
    phi1 = data[6]
    phi2 = data[7]
    mu1 = data[8]
    mu2 = data[9]
    '''
    mesh = solver.interface_coarsen(phi1, data, refine_value, refine_number)
    u0 = data[0]
    u1 = data[1]
    u2 = data[2]
    p1 = data[3]
    p2 = data[4]
    phi0 = data[5]
    phi1 = data[6]
    phi2 = data[7]
    mu1 = data[8]
    mu2 = data[9]
    '''
    pspace, phispace, uspace = solver.update_space(mesh)
    ns_solver.set.uspace = uspace 
    ns_solver.set.pspace = pspace
    ch_solver.space = phispace

    print("自由度个数",uspace.number_of_global_dofs(), phispace.number_of_global_dofs(), pspace.number_of_global_dofs())
