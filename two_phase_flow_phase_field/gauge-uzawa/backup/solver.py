from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator, ScalarConvectionIntegrator
from fealpy.fem import SourceIntegrator, ViscousWorkIntegrator, GradSourceIntegrator    
from fealpy.decorator import barycentric
from fealpy.solver import spsolve
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian

class GaugeUzawaTwoPhaseFlowSolver:
    """
    A Gauge-Uzawa method based two-phase flow solver for coupled phase-field and Navier-Stokes equations.
    
    This solver uses operator splitting method to decompose the two-phase flow problem into phase-field 
    and fluid momentum equation subproblems, handling velocity-pressure coupling and phase-field-flow 
    coupling through time stepping and iterative solving.

    Parameters:
        pde(TwoPhaseFlowPDE): PDE problem class containing two-phase flow definitions including 
            phase-field equation, momentum equation, and physical parameters.
        
        mesh(Mesh): Computational mesh for discretizing the solution domain.
        
        up(int): Polynomial degree for velocity space, default is 2.
        
        pp(int): Polynomial degree for pressure space, default is 1.
        
        phip(int): Polynomial degree for phase-field space, default is 2.
        
        dt(float): Time step size, default is 0.01.
        
        q(int): Quadrature order for numerical integration, default is 4.

    Attributes:
        mesh(Mesh): Computational mesh.
        
        uspace(TensorFunctionSpace): Velocity function space (vector space).
        
        pspace(LagrangeFESpace): Pressure function space (scalar space).
        
        phispace(LagrangeFESpace): Phase-field function space (scalar space).
        
        uh(Function): Current time velocity field.
        
        ph(Function): Current time pressure field.
        
        phi(Function): Current time phase-field.

    Methods:
        assemble_phase_field_matrix(): Assemble phase-field equation coefficient matrix.
        
        assemble_momentum_matrix(): Assemble momentum equation coefficient matrix.
        
        solve_phase_field(): Solve phase-field equation subproblem.
        
        solve_momentum(): Solve momentum equation subproblem.
        
        time_step(): Execute one complete time step solution.
    """
    
    def __init__(self, pde, mesh, up=2, pp=1, phip=1, dt=0.01, q=4):
        self.pde = pde
        self.mesh = mesh
        self.dt = dt
        self.q = q
        self.time = 0.0
        
        # Create function spaces
        usspace = LagrangeFESpace(mesh, p=up)  # Velocity scalar space
        self.uspace = TensorFunctionSpace(usspace, (mesh.GD, -1))  # Velocity vector space
        self.pspace = LagrangeFESpace(mesh, p=pp)  # Pressure space
        #self.pspace = LagrangeFESpace(mesh, p=pp, ctype='D')  # Pressure space
        self.phispace = LagrangeFESpace(mesh, p=phip)  # Phase-field space
        
        # Initialize functions
        self.uh = self.uspace.function()  # Current velocity
        self.uh0 = self.uspace.function()  # Previous time velocity
        self.ph = self.pspace.function()  # Current pressure
        self.phi = self.phispace.function()  # Current phase-field
        self.phi0 = self.phispace.function()  # Previous time phase-field
        
        # Intermediate variables
        self.us = self.uspace.function()  # Intermediate velocity
        self.s = self.pspace.function()  # guage variable
        self.ps = self.pspace.function()  # Intermediate pressure
        self.rho = self.phispace.function()  # Density field

    def set_initial_condition(self, t=0.0):
        """
        Set initial conditions by interpolating analytical solutions to finite element functions.

        Parameters:
            t(float): Initial time, default is 0.0.

        Returns:
            None: This method modifies class attributes directly, no return value.
        """
        # Set initial velocity
        self.uh0[:] = self.uspace.interpolate(lambda p: self.pde.velocity_solution(p, t))
        self.uh[:] = self.uh0[:]
        
        # Set initial pressure
        self.ph[:] = self.pspace.interpolate(lambda p: self.pde.pressure_solution(p, t))
        
        # Set initial phase-field
        self.phi0[:] = self.phispace.interpolate(lambda p: self.pde.phase_solution(p, t))
        self.phi[:] = self.phi0[:]
        self.rho[:] = self.density(self.phi)

    def density(self, phi=None):
        """
        Update density field according to ρ = 0.5(ρ₁ - ρ₂)φ + 0.5(ρ₁ + ρ₂), where φ is the phase-field variable.

        Parameters:
            phi(Function): Phase-field function, default is None.

        Returns:
            Function: Density function computed from phase-field.
        """
        if phi is None:
            phi = self.phi
        tag0 = phi[:] >1
        tag1 = phi[:] < -1
        phi[tag0] = 1
        phi[tag1] = -1
        
        rho_1 = self.pde.rho_1
        rho_2 = self.pde.rho_2
        rho = phi.space.function()
        rho[:] = 0.5 * (rho_1 - rho_2) * phi[:] + 0.5 * (rho_1 + rho_2)
        return rho  
    
    def fphi(self, phi):
        """
        Calculate the phase-field force term f(φ) = (φ³ - φ)/η².

        Parameters:
            phi(Function): Phase-field function.

        Returns:
            Function: Phase-field force term evaluated at φ.
        """
        eta = self.pde.eta
        tag0 = phi[:] > 1
        tag1 = (phi[:] >= -1) & (phi[:] <= 1)
        tag2 = phi[:] < -1
        
        f = phi.space.function()
        #f[tag0] = 2/eta**2  * (phi[tag0] - 1)
        #f[tag1] = (phi[tag1]**3 - phi[tag1]) / (eta**2)
        #f[tag2] = 2/eta**2 * (phi[tag2] + 1)
        f[:] = (phi[:]**3 - phi[:]) / (eta**2)
        return f

    def solve_phase_field(self, t1=0.0, phi0=None, uh0=None):
        """
        Assemble coefficient matrix and right-hand side vector for phase-field equation.
        
        Phase-field equation: ∂φ/∂t + u·∇φ = γ(∇²φ - f(φ)) + f_{force}(φ)
        where f(φ) = (φ³ - φ)/η²

        Returns:
            tuple: Tuple containing coefficient matrix A and right-hand side vector b (A, b).
        """
        dt = self.dt
        gamma = self.pde.gamma
        eta = self.pde.eta

        if phi0 is None:
            phi0 = self.phi0            
        if uh0 is None:
            uh0 = self.uh0
        if t1 is None:
            t1 = self.time

        # Bilinear form
        bform = BilinearForm(self.phispace)    
        bform.add_integrator(ScalarMassIntegrator(coef=1.0+dt*(gamma/eta**2), q=self.q)) 
        bform.add_integrator(ScalarDiffusionIntegrator(coef=dt*gamma, q=self.q)) 
        bform.add_integrator(ScalarConvectionIntegrator(coef=dt*uh0,  q=self.q))
        A = bform.assembly()
        
        # Linear form
        lform = LinearForm(self.phispace)
        
        @barycentric
        def source_coef(bcs, index):
            phi_val = phi0(bcs, index)
            f_force = self.phispace.interpolate(lambda p: self.pde.phi_force(p, t1))
            fphi = (phi_val**3 - phi_val) / (eta**2)

            result = (1+ dt*gamma/eta**2) * phi_val
            result -= gamma * dt * fphi
            result += dt * f_force(bcs, index)
            return result
        
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        
        b = lform.assembly()
        phi_new = spsolve(A, b, solver='mumps') 
        return phi_new

    def solve_momentum(self, t1=0.0, phi0=None, uh0=None, phi=None, s=None):
        """
        Assemble coefficient matrix and right-hand side vector for momentum equation.
        
        Momentum equation: σ(σu)_t + (ρu·∇)u + 1/2∇·(ρu)u - ∇·μD(u) + ∇p + (λ/γ)(φ_t + u·∇φ)∇φ = f

        Returns:
            tuple: Tuple containing coefficient matrix A and right-hand side vector b (A, b).
        """
        if phi0 is None:
            phi0 = self.phi0            
        if uh0 is None:
            uh0 = self.uh0        
        if phi is None:
            phi = self.phi
        if s is None:
            s = self.s

        dt = self.dt
        #dt = 1
        mu = self.pde.mu
        lam = self.pde.lam
        gamma = self.pde.gamma
        rho0 = self.density(phi0)
        rho1 = self.density(phi)

        # Bilinear form
        bform = BilinearForm(self.uspace)
        
        def mass_coef(bcs, index):
            div_u_rho = bm.einsum('cqii,cq->cq', uh0.grad_value(bcs, index), rho1(bcs, index)) 
            div_u_rho += bm.einsum('cqd,cqd->cq', rho1.grad_value(bcs, index), uh0(bcs, index)) 
            result = rho1(bcs, index) + 0.5 * dt * div_u_rho
            return result

        def phiphi_mass_coef(bcs, index):
            gphi0 = phi0.grad_value(bcs, index)
            gphi_gphi = bm.einsum('cqi,cqj->cqij', gphi0, gphi0)
            result = (lam * dt / gamma) * gphi_gphi
            return result

        def conv_coef(bcs, index):
            rhou = rho1(bcs, index)[..., None] * uh0(bcs, index)
            return dt * rhou

        mass_integrator = ScalarMassIntegrator(coef=mass_coef, q=self.q)
        phiphi_mass_integrator = ScalarMassIntegrator(coef=phiphi_mass_coef, q=self.q)
        #visc_integrator = ViscousWorkIntegrator(coef=2*dt*mu, q=self.q)
        visc_integrator = ScalarDiffusionIntegrator(coef=dt*mu, q=self.q)
        conv_integrator = ScalarConvectionIntegrator(coef=conv_coef,  q=self.q)

        bform.add_integrator(mass_integrator) 
        #bform.add_integrator(phiphi_mass_integrator)
        bform.add_integrator(visc_integrator)
        bform.add_integrator(conv_integrator)
        A = bform.assembly()
        
        # Linear form
        lform = LinearForm(self.uspace)
        
        @barycentric
        def momentum_source(bcs, index):
            result0 = bm.sqrt(rho0(bcs, index)[..., None]) * bm.sqrt(rho1(bcs, index)[..., None]) * uh0(bcs, index)
            result1 = dt * mu * s.grad_value(bcs, index)
            result2 = lam/gamma * (phi(bcs, index) - phi0(bcs, index))[..., None] * phi0.grad_value(bcs, index)

            result = result0 - result1 - result2
            return result
        

        @cartesian
        def f_source(p):
            return dt*self.pde.mom_force(p, t1)

        lform.add_integrator(SourceIntegrator(source=momentum_source, q=self.q))
        lform.add_integrator(SourceIntegrator(source=f_source, q=self.q))

        b = lform.assembly()

        @cartesian
        def u_dirichlet(p):
            return self.pde.u_dirichlet(p, t1)
        BC = DirichletBC(self.uspace, u_dirichlet, threshold=None, method='interp')
        A, b = BC.apply(A, b)
        u_new = spsolve(A, b, solver='mumps')
        
        return u_new

    def solve_pressure_correction(self, phi=None, us=None):
        """
        Pressure correction step, solve pressure Poisson equation.

        Returns:
            None: This method modifies class ph attribute directly, no return value.
        """
        if phi is None:
            phi = self.phi
        if us is None:
            us = self.us

        rho1 = self.density(phi)


        @barycentric
        def diff_coef(bcs, index):
            result = rho1(bcs, index)
            return 1/result
        
        bform = BilinearForm(self.pspace)
        bform.add_integrator(ScalarDiffusionIntegrator(coef=diff_coef, q=self.q))
        # 相容性条件 
        bform.add_integrator(ScalarMassIntegrator(coef=1e-10, q=self.q))
        A = bform.assembly()
        
        lform = LinearForm(self.pspace)
        
        @barycentric
        def div_source(bcs, index):
            u_grad = us.grad_value(bcs, index)
            div_u = bm.einsum('cqii->cq', u_grad)
            return div_u
        
        source_integrator = SourceIntegrator(source=div_source, q=self.q)
        lform.add_integrator(source_integrator)
        
        b = lform.assembly()
        
        # Solve
        p_correction = spsolve(A, b, solver='mumps')
        return p_correction
    

    def update_velocity(self, phi=None, us=None, ps=None):
        if phi is None:
            phi = self.phi
        if us is None:
            us = self.us
        if ps is None:
            ps = self.ps

        # update next time velocity
        bform = BilinearForm(self.uspace)
        bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=self.q))
        A = bform.assembly()

        lform = LinearForm(self.uspace)
        @barycentric
        def source_coef(bcs, index):
            result = us(bcs, index)
            rho = self.density(phi)
            result += (1/rho(bcs, index)[..., None] )* ps.grad_value(bcs, index)
            return result
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        b = lform.assembly()
        u_new = spsolve(A, b, solver='mumps')
        return u_new

    def update_guage(self, s0=None, us=None):
        if s0 is None:
            s0 = self.s
        if us is None:
            us = self.us
        # update next time guage variable
        bform = BilinearForm(self.pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=self.q))
        A = bform.assembly()

        lform = LinearForm(self.pspace)
        @barycentric
        def source_coef(bcs, index):
            result = s0(bcs, index) - bm.einsum('cqii->cq', us.grad_value(bcs, index))
            return result
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        b = lform.assembly()

        s_new = spsolve(A, b, solver='mumps')

        return s_new

    def update_pressure(self, s1=None, ps1=None):
        if s1 is None:
            s1 = self.s
        if ps1 is None:
            ps1 = self.ps
        # update next time pressure
        bform = BilinearForm(self.pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=self.q))
        A = bform.assembly()

        lform = LinearForm(self.pspace)
        @barycentric
        def source_coef(bcs, index):
            result = -1/self.dt * ps1(bcs, index)
            result +=  self.pde.mu * s1(bcs, index)
            return result
        lform.add_integrator(SourceIntegrator(source=source_coef, q=self.q))
        b = lform.assembly()
        p_new = spsolve(A, b, solver='mumps')


        return p_new

    def time_step(self):
        """
        Execute one complete time step solution.
        
        Using operator splitting method:
        1. Solve phase-field equation
        2. Solve momentum equation
        3. Pressure correction
        4. Velocity correction

        Returns:
            None: This method updates all field variables, no return value.
        """
        # Step 1: Solve phase-field equation
        self.phi[:] = self.solve_phase_field()
        
        # Step 2: Solve momentum equation (prediction step)
        self.us[:] = self.solve_momentum()
        
        # Step 3: Pressure correction
        self.ps[:] = self.solve_pressure_correction()
 
        # Step 4: up_date next time velocity, guage variable and pressure
        self.uh[:] = self.update_velocity()
        self.s[:] = self.update_guage()
        self.ph[:] = self.update_pressure()

        # Save previous time solutions
        self.uh0[:] = self.uh[:]
        self.phi0[:] = self.phi[:]

        print("time: ", self.time)


    def run(self, T=1.0, output_freq=10):
        """
        Run time evolution solution.

        Parameters:
            T(float): Total time, default is 1.0.
            
            output_freq(int): Output frequency, output results every this many time steps, default is 10.

        Returns:
            None: This method executes time evolution process, no return value.
        """
        step = 0
        
        # Set initial conditions
        self.set_initial_condition(t=0)
        
        print(f"Starting time evolution solution, total time: {T}, time step: {self.dt}")
        
        while self.time < T:
            step += 1
            self.time  += self.dt
            
            # Execute one time step
            self.time_step()
            
            if step % output_freq == 0:
                print(f"Time step: {step}, time: {self.time:.4f}")
                print(f"velocity_error: {self.get_error(self.time)['velocity_error']:.6e}")
                print(f"pressure_error: {self.get_error(self.time)['pressure_error']:.6e}")
                print(f"phase_error: {self.get_error(self.time)['phase_error']:.6e}")
                
        print(f"Solution completed, total time steps: {step}")

    def get_error(self, t):
        """
        Calculate errors between current numerical solution and analytical solution.

        Parameters:
            t(float): Current time.

        Returns:
            dict: Dictionary containing L2 errors for each field.
        """
        # Velocity error
        u_exact = self.uspace.interpolate(lambda p: self.pde.velocity_solution(p, t))
        u_error = self.mesh.error(self.uh, u_exact, power=2)
        
        # Pressure error
        p_exact = self.pspace.interpolate(lambda p: self.pde.pressure_solution(p, t))
        p_error = self.mesh.error(self.ph, p_exact, power=2)
        
        # Phase-field error
        phi_exact = self.phispace.interpolate(lambda p: self.pde.phase_solution(p, t))
        phi_error = self.mesh.error(self.phi, phi_exact, power=2)
        
        return {
            'velocity_error': u_error,
            'pressure_error': p_error,
            'phase_error': phi_error
        } 
    
