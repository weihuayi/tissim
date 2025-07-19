from fealpy.backend import backend_manager as bm
from pde import TwoPhaseFlowPDE
from solver import GaugeUzawaTwoPhaseFlowSolver
import sympy as sp
from fealpy.decorator import cartesian
def test_phase():
    pde = TwoPhaseFlowPDE()
    pde.set_mesh(n=16)
    dt = 0.00001
    mesh = pde.mesh
    solver = GaugeUzawaTwoPhaseFlowSolver(
        pde=pde, 
        mesh=mesh, 
        up=2,
        pp=1,
        phip=1,
        dt=dt,
        q=4
    )
    
    time = 0
    solver.set_initial_condition(time)
    phi0 = solver.phi0
    phi = solver.phi

    for i in range(10):  
        time += dt
        print(f"time: {time}")
        
        uh = solver.uspace.interpolate(lambda p: pde.velocity_solution(p, time))
        phi[:] = solver.solve_phase_field(time, phi0, uh)
        phi0[:] = phi
        phi_exact = solver.phispace.interpolate(lambda p: pde.phase_solution(p, time))

        @cartesian
        def phi_exact_1(p):
            return pde.phase_solution(p, time)
        
        phi_error = mesh.error(phi, phi_exact_1, power=2)
        print(f"phi_error: {phi_error}")
        

def test_momentum():
    pde = TwoPhaseFlowPDE()
    pde.set_mesh(n=16)
    mesh = pde.mesh
    dt = 0.0001
    solver = GaugeUzawaTwoPhaseFlowSolver(
        pde=pde, 
        mesh=mesh,
        up=2,
        pp=1,
        phip=1,
        dt=dt,
        q=4
    )
    
    time = 0
    solver.set_initial_condition(time)
    uh = solver.uh0
    ph = solver.ph
    us = solver.us
    ps = solver.ps
    s = solver.s

    for i in range(10):    
        time += dt
        print(f"time: {time}")

        phi0 = solver.phispace.interpolate(lambda p: pde.phase_solution(p, time-dt))
        phi = solver.phispace.interpolate(lambda p: pde.phase_solution(p, time))
        
        us[:] = solver.solve_momentum(time, phi0, uh, phi, s)
        ps[:] = solver.solve_pressure_correction(phi, us)

        uh[:] = solver.update_velocity(phi, us, ps)
        s[:] = solver.update_guage(s, us)
        ph[:] = solver.update_pressure(s, ps)

        u_exact = solver.uspace.interpolate(lambda p: pde.velocity_solution(p, time))
        uh_error = mesh.error(uh, u_exact, power=2)
        p_exact = solver.pspace.interpolate(lambda p: pde.pressure_solution(p, time))
        ph_error = mesh.error(ph, p_exact, power=2)
        print("uh_error: ", uh_error)
        print("ph_error: ", ph_error)
        

def test():
    pde = TwoPhaseFlowPDE() 
    pde.set_mesh(n=128)
    mesh = pde.mesh
    dt = 0.000001
    solver = GaugeUzawaTwoPhaseFlowSolver(
        pde=pde, 
        mesh=mesh,
        up=2,
        pp=1,
        phip=1,
        dt=dt,
        q=4
    )
    
    time = 0
    solver.set_initial_condition(time)
    uh = solver.uh0
    ph = solver.ph
    us = solver.us
    ps = solver.ps
    s = solver.s
    phi = solver.phi
    phi0 = solver.phi0

    for i in range(10):    
        time += dt
        print(f"time: {time}")

        phi[:] = solver.solve_phase_field(time, phi0, uh)
        
        us[:] = solver.solve_momentum(time, phi0, uh, phi, s)
        ps[:] = solver.solve_pressure_correction(phi, us)
        
        uh[:] = solver.update_velocity(phi, us, ps)
        s[:] = solver.update_guage(s, us)
        ph[:] = solver.update_pressure(s, ps)
        
        phi0[:] = phi

        u_exact = solver.uspace.interpolate(lambda p: pde.velocity_solution(p, time))
        uh_error = mesh.error(uh, u_exact, power=2)
        p_exact = solver.pspace.interpolate(lambda p: pde.pressure_solution(p, time))
        ph_error = mesh.error(ph, p_exact, power=2)
        #phi_exact = solver.phispace.interpolate(lambda p: pde.phase_solution(p, time))
        @cartesian
        def phi_exact(p):
            return pde.phase_solution(p, time)
        phi_error = mesh.error(phi, phi_exact, power=2)
        print("uh_error: ", uh_error)
        print("ph_error: ", ph_error)
        print("phi_error: ", phi_error)
        
        '''
        uh_error = mesh.error(uh, u_exact, power=2, celltype=True)
        ph_error = mesh.error(ph, p_exact, power=2, celltype=True)
        mesh.celldata['erroru'] = uh_error
        mesh.celldata['errorp'] = ph_error
        mesh.to_vtk(f"two_phase_flow_{time}.vtu")
        '''


def main():
    # 创建PDE问题
    pde = TwoPhaseFlowPDE()
    
    # 验证解析解的正确性
    print("验证解析解...")
    pde.verify_phase_solution()
    pde.verify_incompressibility_solution() 
    pde.verify_momentum_solution()
    
    # 获取网格
    pde.set_mesh(n=64)
    mesh = pde.mesh
    
    # 创建求解器
    T = 0.001      # 总时间
    dt = 0.0001  # 时间步长
    solver = GaugeUzawaTwoPhaseFlowSolver(
        pde=pde, 
        mesh=mesh, 
        up=2,      # 速度空间次数
        pp=1,      # 压力空间次数  
        phip=1,    # 相场空间次数
        dt=dt,     # 时间步长
        q=4        # 积分精度
    )
    
    print(f"求解器初始化完成")
    print(f"速度自由度数: {solver.uspace.number_of_global_dofs()}")
    print(f"压力自由度数: {solver.pspace.number_of_global_dofs()}")
    print(f"相场自由度数: {solver.phispace.number_of_global_dofs()}")
    
    # 设置初始条件
    t0 = 0.0
    solver.set_initial_condition(t0)
    
    # 计算初始误差
    initial_errors = solver.get_error(t0)
    print(f"\n初始误差:")
    print(f"速度误差: {initial_errors['velocity_error']:.6e}")
    print(f"压力误差: {initial_errors['pressure_error']:.6e}")
    print(f"相场误差: {initial_errors['phase_error']:.6e}")
    
    # 运行时间演化求解
    output_freq = 5  # 输出频率
    
    solver.run(T=T, output_freq=output_freq)
    
    # 计算最终误差
    final_errors = solver.get_error(T)
    print(f"\n最终误差:")
    print(f"速度误差: {final_errors['velocity_error']:.6e}")
    print(f"压力误差: {final_errors['pressure_error']:.6e}")
    print(f"相场误差: {final_errors['phase_error']:.6e}")
    

if __name__ == "__main__":
    #main()
    test()
    #test_phase()
    #test_momentum()
