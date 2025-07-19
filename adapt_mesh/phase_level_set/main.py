from fealpy.backend import backend_manager as bm
from pde import TwoPhaseModel, OnePhaseModel
from solver import two_phase_solver
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 
from fealpy.fem import LevelSetReinitModel

output = './'
dt =0.5e-2

pde = TwoPhaseModel()
#pde = OnePhaseModel()
mesh = pde.mesh()
epsilon = bm.max(mesh.entity_measure('edge'))

solver = two_phase_solver(pde, epsilon, dt, q=5)
pspace, usspace, uspace = solver.update_space(mesh)

phi = usspace.interpolate(pde.init_interface)
phi = solver.heaviside(phi)
u0 = uspace.function()
p0 = pspace.function()

rho, eta = solver.update_function(phi)
fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
solver.output(fname, mesh, u0, p0, phi, rho, eta)

for i in range(1, 5000):
    print(f"第{i}个时间步") 
    
    # 加密网格
    #mesh, phi, u0, p0 = solver.interface_refine(phi, u0, p0, 5)
    #mesh, phi, u0, p0 = solver.interface_coarsen(phi, u0, p0, 5)
    pspace, usspace, uspace = solver.update_space(mesh)
    rho, eta = solver.update_function(phi)
    us = uspace.function()
    u1 = uspace.function()
    p1 = pspace.function()

    BCu = DirichletBC(space=uspace, 
            gd=pde.velocity, 
            threshold=pde.is_u_boundary, 
            method='interp')

    BCp = DirichletBC(space=pspace, 
            gd=pde.pressure, 
            threshold=pde.is_p_boundary, 
            method='interp')

    BForm0 = solver.IPCS_BForm_0(rho, eta, uspace)
    LForm0 = solver.IPCS_LForm_0(rho, eta, u0, p0, phi, uspace)
    A0 = BForm0.assembly()
    b0 = LForm0.assembly()
    A0,b0 = BCu.apply(A0,b0)
    us[:] = spsolve(A0, b0, 'mumps')

    BForm1 = solver.IPCS_BForm_1(pspace, uspace)
    LForm1 = solver.IPCS_LForm_1(rho, us, p0)
    A1 = BForm1.assembly()
    b1 = LForm1.assembly()
    A1,b1 = BCp.apply(A1, b1)
    p1[:] = spsolve(A1, b1, 'mumps')

    BForm2 = solver.IPCS_BForm_2(rho, uspace)
    LForm2 = solver.IPCS_LForm_2(rho, us, p0, p1)
    A2 = BForm2.assembly()
    b2 = LForm2.assembly()
    A2,b2 = BCu.apply(A2,b2)
    u1[:] = spsolve(A2, b2, 'mumps')

    u0[:] = u1
    p0[:] = p1
    
    evo_BForm = solver.evo_BForm(u1, usspace)
    Evo_LForm = solver.evo_LForm(u1, phi)
    evo_A = evo_BForm.assembly()
    evo_b = Evo_LForm.assembly()
    phi[:] = spsolve(evo_A, evo_b, 'mumps')
    #phi = solver.rein_run(phi) 

    fname = output + 'test_'+ str(i).zfill(10) + '.vtu'
    solver.output(fname, mesh, u1, p1, phi, rho, eta)
    

    '''
    if i%1 == 0:
        rein_solver = LevelSetReinitModel(phi)
        print("重置前",rein_solver.check_gradient_norm_at_interface(phi, epsilon))
        re_dt = 0.05*bm.min(mesh.entity_measure('edge'))
        re_alpha = 0.0625*bm.max(mesh.entity_measure('cell')) 
        rein_solver.options.set_reinit_params(re_dt=re_dt, re_alpha=re_alpha)
        phi[:] = rein_solver.reinit_run()
        print("重置后",rein_solver.check_gradient_norm_at_interface(phi, epsilon))
    '''



