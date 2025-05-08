from fealpy.backend import backend_manager as bm
from pde import TwoPhaseModel, OnePhaseModel
from solver import two_phase_solver
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 
from fealpy.fem import LevelSetReinitModel

output = './'
dt = 0.01

pde = TwoPhaseModel()
#pde = OnePhaseModel()
mesh = pde.mesh()
epsilon = bm.min(mesh.entity_measure('edge'))

solver = two_phase_solver(pde, epsilon, dt, q=5)
pspace, usspace, uspace = solver.update_space(mesh)

phi = usspace.interpolate(pde.init_interface) 
u0 = uspace.interpolate(pde.velocity)
p0 = pspace.interpolate(pde.pressure)
rho, eta = solver.update_function(phi)
fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
solver.output(fname, mesh, u0, p0, phi, rho, eta)
for i in range(300):
    # 加密网格
    mesh, phi, u0, p0 = solver.interface_refine(phi, u0, p0, 5)
    mesh, phi, u0, p0 = solver.interface_coarsen(phi, u0, p0, 5)
    
    pspace, usspace, uspace = solver.update_space(mesh)
    rho, eta = solver.update_function(phi)
        

    evo_BForm = solver.evo_BForm(u0, usspace)
    Evo_LForm = solver.evo_LForm(u0, phi)
    evo_A = evo_BForm.assembly()
    evo_b = Evo_LForm.assembly()
    phi[:] = spsolve(evo_A, evo_b, 'mumps') 

    fname = output + 'test_'+ str(i).zfill(10) + '.vtu'
    solver.output(fname, mesh, u0, p0, phi, rho, eta)
