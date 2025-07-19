from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.equation import CahnHilliard
from fealpy.cfd.simulation.fem import BDF2, IPCS, Newton
from fealpy.cfd.simulation.fem import CahnHilliardModel 
from pde import TwoPhaseModel
from solver import two_phase_phield_solver
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC

from fealpy.solver import DirectSolverManager
from fealpy.solver import spsolve, cg
import psutil
import time 

'''
管道流
'''

bm.set_backend('pytorch') 
#bm.set_default_device('cuda')

pde = TwoPhaseModel()
mesh = pde.init_mesh(nx=400, ny=40)
dt = 0.016*bm.sqrt(2*bm.min(mesh.entity_measure('edge')))
pde.epsilon = 0.08*bm.sqrt(2*bm.min(mesh.entity_measure('edge')))
pde.Pe = 1/pde.epsilon

ns_eqaution = IncompressibleNS(pde,init_variables=False) 
ns_solver = BDF2(ns_eqaution)
ns_solver.dt = dt

phispace = ns_solver.uspace.scalar_space
solver = two_phase_phield_solver(pde) 
ch_equation = CahnHilliard(pde, init_variables=False)
ch_solver = CahnHilliardModel(ch_equation, phispace)
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

mesh.nodedata['velocity'] = u1.reshape(2,-1).T  
mesh.nodedata['phi'] = phi1
fname = './' + 'test_'+ str(1).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ugdof = ns_solver.uspace.number_of_global_dofs()
pgdof = ns_solver.pspace.number_of_global_dofs()
phigdof = phispace.number_of_global_dofs()

ns_BForm =ns_solver.BForm()
ns_LForm = ns_solver.LForm()
ch_BFrom = ch_solver.BForm()
ch_LForm = ch_solver.LForm()

BC = DirichletBC((ns_solver.uspace,ns_solver.pspace), 
                 gd=(pde.u_dirichlet, pde.pressure_boundary), 
                      threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp')

#设置参数
ns_eqaution.set_coefficient('viscosity', 100)
ns_eqaution.set_coefficient('pressure', 1)
ch_equation.set_coefficient('mobility', 1/pde.Pe)
ch_equation.set_coefficient('interface', pde.epsilon**2)
ch_equation.set_coefficient('free_energy', 1)

for i in range(1,5000):
    # 设置参数
    print("iteration:", i)
    print("time", i*dt)
    
    ch_solver.update(u0, u1, phi0, phi1)
    ch_A = ch_BFrom.assembly()
    ch_b = ch_LForm.assembly()
    ch_x = spsolve(ch_A, ch_b, 'mumps')

    phi2[:] = ch_x[:phigdof]
    mu2[:] = ch_x[phigdof:]  

    # 更新NS方程参数
    rho = solver.rho(phi1) 
    ns_eqaution.set_coefficient('time_derivative', 1000)
    ns_eqaution.set_coefficient('convection', 1000)
    ns_eqaution.set_coefficient('time_derivative', rho)
    ns_eqaution.set_coefficient('convection', rho)

    ns_solver.update(u0, u1)
     
    ns_A = ns_BForm.assembly()
    ns_b = ns_LForm.assembly()
    ns_A,ns_b = BC.apply(ns_A, ns_b)
    ns_x = spsolve(ns_A, ns_b, 'mumps')

    u2[:] = ns_x[:ugdof]
    p2[:] = ns_x[ugdof:]
    u0[:] = u1[:]
    u1[:] = u2[:]
    p1[:] = p2[:]
    
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['velocity'] = u2.reshape(2,-1).T  
    mesh.nodedata['pressure'] = p2 
    mesh.nodedata['rho'] = rho
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)
