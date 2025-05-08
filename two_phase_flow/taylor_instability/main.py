from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.equation import CahnHilliard
from fealpy.cfd.simulation.fem import BDF2
from fealpy.cfd.simulation.fem import CahnHilliardModel 
from pde import RayleignTaylor
from solver import two_phase_phield_solver
from fealpy.decorator import barycentric
from fealpy.solver import spsolve
from fealpy.fem import DirichletBC
import psutil


bm.set_backend('pytorch')

dt = 0.00125*bm.sqrt(bm.array(2))
pde = RayleignTaylor()
#mesh = pde.init_mesh(nx=128, ny=512)
mesh = pde.init_mesh(nx=64, ny=256)
pde.epsilon = 0.08*bm.sqrt(2*bm.min(mesh.entity_measure('edge')))

ns_eqaution = IncompressibleNS(pde,init_variables=False) 
ns_solver = BDF2(ns_eqaution)
ns_solver.dt = dt

phispace = ns_solver.uspace.scalar_space

ch_equation = CahnHilliard(pde, init_variables=False)
ch_solver = CahnHilliardModel(ch_equation, phispace)
ch_solver.dt = dt

solver = two_phase_phield_solver(pde) 

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
fname = './' + 'test_'+ str(1).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ugdof = ns_solver.uspace.number_of_global_dofs()
phigdof = phispace.number_of_global_dofs()
pgdof = ns_solver.pspace.number_of_global_dofs()

ns_BForm = ns_solver.BForm()
ns_LForm = ns_solver.LForm()
ch_BFrom = ch_solver.BForm()
ch_LForm = ch_solver.LForm()


is_bd = ns_solver.uspace.is_boundary_dof((pde.is_ux_boundary, pde.is_uy_boundary), method='interp')
is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
print(is_bd.shape)
gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
BC = DirichletBC((ns_solver.uspace,ns_solver.pspace), gd=gd, 
                      threshold=is_bd, method='interp')

#BC = DirichletBC((ns_solver.uspace,ns_solver.pspace), gd=(pde.velocity_boundary, pde.pressure_boundary), 
#                      threshold=(pde.is_u_dboundary, pde.is_p_dboundary), method='interp')

#设置参数
ns_eqaution.set_coefficient('viscosity', 1/pde.Re)
ns_eqaution.set_coefficient('pressure', 1)
ch_equation.set_coefficient('mobility', 1/pde.Pe)
ch_equation.set_coefficient('interface', pde.epsilon**2)
ch_equation.set_coefficient('free_energy', 1)


for i in range(1,2000):
    # 设置参数
    print("iteration:", i)
    print("内存占用情况:",psutil.Process().memory_info().rss / 1024 ** 2, "MB")  

    ch_solver.update(u0, u1, phi0, phi1)
    ch_A = ch_BFrom.assembly()
    ch_b = ch_LForm.assembly()
    ch_x = spsolve(ch_A, ch_b, 'mumps')
    
    phi2[:] = ch_x[:phigdof]
    mu2[:] = ch_x[phigdof:]  
    

    # 更新NS方程参数
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
    
    ns_x = spsolve(ns_A, ns_b, 'mumps')

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
    mesh.nodedata['rho'] = rho
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)
