from fealpy.backend import backend_manager as bm
from pde import Bubble_Rising_PDE
from solver import GaugeUzawaTwoPhaseFlowSolver
import sympy as sp
from fealpy.decorator import cartesian, barycentric


pde = Bubble_Rising_PDE() 
mesh = pde.set_mesh(n=64)
dt = 0.0005
solver = GaugeUzawaTwoPhaseFlowSolver(
    pde=pde, 
    mesh=mesh,
    up=2,
    pp=1,
    phip=2,
    dt=dt,
    q=4
)

uh = solver.uh
uh0 = solver.uh0
ph = solver.ph
phi0 = solver.phispace.interpolate(pde.init_phi)
phi = solver.phi
us = solver.us
ps = solver.ps
s = solver.s
mesh.nodedata['phi'] = phi0
mesh.to_vtk(f"bubble_{str(0).zfill(10)}.vtu")

time = 0

for i in range(5000):    
    time += dt
    print(f"time: {time}")

    phi[:] = solver.solve_phase_field(time, phi0, uh)
    
    us[:] = solver.solve_momentum(time, phi0, uh, phi, s)
    ps[:] = solver.solve_pressure_correction(phi, us)
    
    uh[:] = solver.update_velocity(phi, us, ps)
    s[:] = solver.update_guage(s, us)
    ph[:] = solver.update_pressure(s, ps, phi)
    
    phi0[:] = phi
    rho = solver.density(phi)
    mesh.nodedata['phi'] = phi
    mesh.nodedata['rho'] = rho
    mesh.nodedata['uh'] = uh.reshape(2,-1).T
    mesh.nodedata['ph'] = ph
    mesh.to_vtk(f"bubble_{str(i+1).zfill(10)}.vtu")


