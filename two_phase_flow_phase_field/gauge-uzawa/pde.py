from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh, QuadrangleMesh
import sympy as sp

class TwoPhaseFlowPDE:
    '''
    A PHASE-FIELD MODEL AND ITS NUMERICAL 
    APPROXIMATION FOR TWO-PHASE INCOMPRESSIBLE 
    FLOWS WITH DIFFERENT DENSITIES AND VISCOSITIES
    '''

    def __init__(self):
        self.box = [-1, 1, -1, 1]
        self.x, self.y, self.t = sp.symbols("x y t")
        self.mu = 1.0
        self.gamma = 0.02
        self.lam = 0.001
        self.set_mesh()
        self.eta = 4*bm.min(self.mesh.entity_measure('edge'))
        self.rho_1 = 3.0
        self.rho_2 = 1.0
        # 解析解
        self.phi_expr = 2 + sp.sin(self.t) * sp.cos(sp.pi * self.x) * sp.cos(sp.pi * self.y)
        self.rho_expr = self.phi_expr + 2
        self.u1_expr = sp.pi * sp.sin(2 * sp.pi * self.y) * sp.sin(sp.pi * self.x) ** 2 * sp.sin(self.t)
        self.u2_expr = -sp.pi * sp.sin(2 * sp.pi * self.x) * sp.sin(sp.pi * self.y) ** 2 * sp.sin(self.t)
        self.p_expr = sp.cos(sp.pi * self.x) * sp.sin(sp.pi * self.y) * sp.sin(self.t)
        self.init_force()

        # lambdify接口
        self.phi = sp.lambdify((self.x, self.y, self.t), self.phi_expr, "numpy")
        self.rho = sp.lambdify((self.x, self.y, self.t), self.rho_expr, "numpy")
        self.u1 = sp.lambdify((self.x, self.y, self.t), self.u1_expr, "numpy")
        self.u2 = sp.lambdify((self.x, self.y, self.t), self.u2_expr, "numpy")
        self.p = sp.lambdify((self.x, self.y, self.t), self.p_expr, "numpy")

     
    def init_force(self):
        x, y, t = self.x, self.y, self.t
        eta = self.eta
        gamma = self.gamma
        mu = self.mu
        u1 = self.u1_expr
        u2 = self.u2_expr
        rho = self.rho_expr
        p = self.p_expr
        phi = self.phi_expr
        lam = self.lam

        # 相场方程
        phi_t = sp.diff(phi, t)
        u_dot_grad_phi = u1 * sp.diff(phi, x) + u2 * sp.diff(phi, y)
        lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2)
        f_phi = (phi**3 - phi) / eta**2
        self.phi_force_expr = phi_t + u_dot_grad_phi - gamma * (lap_phi - f_phi)
        

        # 动量方程
        # σ(σu)_t + (μu·∇)μu + 1/2 ∇·(ρu)u - ∇·μD(u) + ∇p + (λ/γ)(φ_t + u·∇φ)∇φ = 0
        sigma = sp.sqrt(rho)
        u = sp.Matrix([u1, u2])
        grad_u = sp.Matrix([[sp.diff(u1, x), sp.diff(u1, y)], [sp.diff(u2, x), sp.diff(u2, y)]])
        D = grad_u + grad_u.T

        # σ(σu)_t
        sig_u = sigma * u
        sig_u_t = sp.Matrix([sigma*sp.diff(sig_u[0], t), sigma*sp.diff(sig_u[1], t)])

        # (rho u·∇)u
        con_u = sp.Matrix([
            rho * (u1 * sp.diff(u1, x) + u2 * sp.diff(u1, y)),
            rho * (u1 * sp.diff(u2, x) + u2 * sp.diff(u2, y))
        ])

        # 1/2 ∇·(ρu)u
        rho_u = rho * u
        div_rho_u = sp.diff(rho_u[0], x) + sp.diff(rho_u[1], y)
        half_div_rho_u_u = 0.5 * div_rho_u * u

        # ∇·μD(u)
        div_muD = sp.Matrix([
            sp.diff(mu * D[0, 0], x) + sp.diff(mu * D[1, 0], y),
            sp.diff(mu * D[0, 1], x) + sp.diff(mu * D[1, 1], y)
        ])
        laplace_u = sp.Matrix([
            mu*sp.diff(u1, x, 2) + mu*sp.diff(u1, y, 2),
            mu*sp.diff(u2, x, 2) + mu*sp.diff(u2, y, 2)
        ])

        # ∇p
        grad_p = sp.Matrix([sp.diff(p, x), sp.diff(p, y)])

        # (λ/γ)(φ_t + u·∇φ)∇φ
        phi_t = sp.diff(phi, t)
        grad_phi = sp.Matrix([sp.diff(phi, x), sp.diff(phi, y)])
        u_dot_grad_phi = u[0] * grad_phi[0] + u[1] * grad_phi[1]
        extra_term = (lam / gamma) * (phi_t + u_dot_grad_phi) * grad_phi

        self.mom_force_expr = sig_u_t + con_u + half_div_rho_u_u - div_muD + grad_p + extra_term

    @cartesian
    def phi_force(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        phi_force = sp.lambdify((self.x, self.y, self.t), self.phi_force_expr, "numpy")
        val[:] = phi_force(x, y, t)
        return val

    @cartesian
    def mom_force(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        mom_force_0 = sp.lambdify((self.x, self.y, self.t), self.mom_force_expr[0], "numpy")
        mom_force_1 = sp.lambdify((self.x, self.y, self.t), self.mom_force_expr[1], "numpy")
        val[..., 0] = mom_force_0(x, y, t)
        val[..., 1] = mom_force_1(x, y, t)
        return val


    @cartesian
    def velocity_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.u1(x, y, t)
        val[..., 1] = self.u2(x, y, t)
        return val
    
    @cartesian
    def pressure_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        val[:] = sp.lambdify((self.x, self.y, self.t), self.p_expr, "numpy")(x, y, t)
        return val
    
    @cartesian
    def phase_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        val[:] = self.phi(x, y, t)
        return val

    @cartesian
    def density_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        val[:] = self.rho(x, y, t)
        return val

    def domain(self):
        return self.box
    
    def set_mesh(self, n=128):
        box = self.box
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        self.mesh = mesh
        return mesh

    @cartesian
    def u_dirichlet(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = self.velocity_solution(p, t)
        return val

    def verify_momentum_solution(self):
        x, y, t = self.x, self.y, self.t
        mu = self.mu
        lam = self.lam
        gamma = self.gamma
        # 解析表达式
        phi = self.phi_expr
        rho = self.rho_expr
        u1 = self.u1_expr
        u2 = self.u2_expr
        p = self.p_expr
        
        # u相关
        u = sp.Matrix([u1, u2])
        u_t = sp.Matrix([sp.diff(u1, t), sp.diff(u2, t)])
        grad_u = sp.Matrix([[sp.diff(u1, x), sp.diff(u1, y)], [sp.diff(u2, x), sp.diff(u2, y)]])
        D = grad_u + grad_u.T
        
        # σ(σu)_t
        sigma = sp.sqrt(rho)
        sig_u = sigma * u
        sig_u_t = sp.Matrix([sigma*sp.diff(sig_u[0], t), sigma*sp.diff(sig_u[1], t)])
        
        # (rho u·∇)u
        con_u = sp.Matrix([
            rho * (u1 * sp.diff(u1, x) + u2 * sp.diff(u1, y)),
            rho * (u1 * sp.diff(u2, x) + u2 * sp.diff(u2, y))
        ])
        # 1/2 ∇·(rho u) u
        div_rhou = sp.diff(rho * u1, x) + sp.diff(rho * u2, y)
        half_div_rhou_u = 0.5 * div_rhou * u
        # -∇·mu D(u)
        div_muD = sp.Matrix([
            sp.diff(mu * D[0, 0], x) + sp.diff(mu * D[0, 1], y),
            sp.diff(mu * D[1, 0], x) + sp.diff(mu * D[1, 1], y)
        ])
        # ∇p
        grad_p = sp.Matrix([sp.diff(p, x), sp.diff(p, y)])
        
        # (λ/γ)(φ_t + u·∇φ)∇φ
        phi_t = sp.diff(phi, t)
        grad_phi = sp.Matrix([sp.diff(phi, x), sp.diff(phi, y)])
        u_dot_grad_phi = u[0] * grad_phi[0] + u[1] * grad_phi[1]
        extra_term = (lam / gamma) * (phi_t + u_dot_grad_phi) * grad_phi
        
        # 总残差
        mom = sig_u_t + con_u + half_div_rhou_u - div_muD + grad_p + extra_term
        mom_res = mom - self.mom_force_expr
        print('动量方程残差:', sp.simplify(mom_res))

    def verify_phase_solution(self):
        x, y, t = self.x, self.y, self.t
        u1 = self.u1_expr
        u2 = self.u2_expr
        gamma = self.gamma
        eta = self.eta
        
        # 解析表达式
        phi = self.phi_expr
        phi_t = sp.diff(phi, t)
        lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2)
        f_phi = (phi**3 - phi) / eta**2
        u_dot_grad_phi = u1 * sp.diff(phi, x) + u2 * sp.diff(phi, y)
        phase_lhs = phi_t + u_dot_grad_phi
        phase_rhs = gamma * (lap_phi - f_phi) 
        phase_res = phase_lhs - phase_rhs - self.phi_force_expr
        print('相场方程残差:', sp.simplify(phase_res))

    def verify_incompressibility_solution(self):
        x, y, t = self.x, self.y, self.t
        u1 = self.u1_expr
        u2 = self.u2_expr
        div_u = sp.diff(u1, x) + sp.diff(u2, y)
        print('不可压残差:', sp.simplify(div_u))

class Bubble_Rising_PDE:
    def __init1__(self):
        #self.d = 0.005
        self.d = 1
        self.box = [-self.d, self.d, -2*self.d, 2*self.d]
        #self.mu = 0.0011
        self.mu =  1
        self.gamma = 0.02
        self.lam = 0.001
        self.set_mesh()
        #self.eta = 4*bm.min(self.mesh.entity_measure('edge'))
        self.eta = 0.02*self.d
        self.rho_1 = 1.0
        self.rho_2 = 10.0
    
    def __init__(self):
        #self.d = 0.005
        self.d = 1
        self.box = [-self.d, self.d, -2*self.d, 2*self.d]
        
        self.gamma = 0.02
        self.lam = 0.001
        self.set_mesh()
        #self.eta = 4*bm.min(self.mesh.entity_measure('edge'))
        self.eta = 0.02*self.d
        
        self.rho_0 = 1.161
        self.rho_1 = 1.0
        self.rho_2 = 995.65/self.rho_0
        
        self.mu_1 = 1.86e-5/(self.rho_0*0.005**1.5*9.8**0.5)
        self.mu_2 = 7.977e-4/(self.rho_0*0.005**1.5*9.8**0.5)
        self.mu_bar = bm.min([self.mu_1, self.mu_2])

    @cartesian
    def init_phi(self, p):
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        val = -bm.tanh((r - 0.5*self.d)/self.eta)
        return val

    def set_mesh(self, n=128):
        box = self.box
        #mesh = TriangleMesh.from_box(box, nx=n, ny=2*n)
        mesh = QuadrangleMesh.from_box(box, nx=n, ny=2*n)
        self.mesh = mesh
        return mesh
    
    @cartesian
    def phi_force(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        return val

    @cartesian
    def mom_force(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = 0
        #val[..., 1] = -10
        val[..., 1] = -1
        return val
    
    @cartesian
    def u_dirichlet(self, p, t):
        val = bm.zeros(p.shape, dtype=bm.float64)
        return val


    
