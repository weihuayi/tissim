from fealpy.backend import backend_manager as bm
from fealpy.fem import LevelSetLFEMModel
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.cfd import NSFEMSolver
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.decorator import cartesian,barycentric
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarConvectionIntegrator, 
                        ScalarDiffusionIntegrator, 
                        ScalarMassIntegrator,
                        SourceIntegrator,
                        ScalarSourceIntegrator,
                        PressWorkIntegrator,
                        FluidBoundaryFrictionIntegrator,
                        ViscousWorkIntegrator,
                        GradSourceIntegrator)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)

from fealpy.fem import RecoveryAlg


class two_phase_solver:
    def __init__(self, pde, epsilon, dt, q=4):
        self.epsilon = epsilon
        self.pde = pde
        self.q = q
        self.dt = dt

    def show_mesh(self,mesh):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        plt.show()

    def interface_refine(self, phi, u, p, n):
        epsilon = self.epsilon
        usspace = u.space.scalar_space
        pspace = p.space
        phispace = phi.space
        mesh = phi.space.mesh
        
        for i in range(n):
            phicell2dof = phispace.cell_to_dof()
            phic2f = phi[phicell2dof]
            
            pcell2dof = pspace.cell_to_dof()
            pc2f = p[pcell2dof]
            
            ucell2dof = usspace.cell_to_dof()
            u0 = u.reshape(mesh.GD,-1).T[...,0]
            u1 = u.reshape(mesh.GD,-1).T[...,1] 
            u0c2f = u0[ucell2dof]
            u1c2f = u1[ucell2dof]

            cellmeasure = mesh.entity_measure('cell')
            isMark = bm.abs(bm.mean(phic2f,axis=-1))<epsilon/2
            isMark = bm.logical_and(bm.mean(phic2f,axis=-1)<-0.01,isMark)
            isMark = bm.logical_and(isMark,cellmeasure>4e-5)
            data = {'phi':phic2f,'p':pc2f, 'u0':u0c2f,'u1':u1c2f} 
            option = mesh.bisect_options(data=data, disp=False)
            mesh.bisect(isMark,options=option)
            
            pspace , usspace, uspace = self.update_space(mesh)
            phi = usspace.function()
            phicell2dof = usspace.cell_to_dof()
            phi[phicell2dof.reshape(-1)] = option['data']['phi'].reshape(-1)
            
            p = pspace.function()
            pcell2dof = pspace.cell_to_dof()
            p[pcell2dof.reshape(-1)] = option['data']['p'].reshape(-1)
            
            u = uspace.function()
            ucell2dof = usspace.cell_to_dof()
            u0 = usspace.function()
            u1 = usspace.function()
            uscell2dof = usspace.cell_to_dof()
            u0[uscell2dof.reshape(-1)] = option['data']['u0'].reshape(-1)
            u1[uscell2dof.reshape(-1)] = option['data']['u1'].reshape(-1)
            u[:] = bm.stack((u0[:],u1[:]), axis=1).T.flatten()
        
        return mesh, phi, u, p

    def interface_coarsen(self, phi, u, p, n):
        epsilon = self.epsilon
        usspace = u.space.scalar_space
        pspace = p.space
        phispace = phi.space
        mesh = phi.space.mesh
        
        for i in range(n):
            phicell2dof = phispace.cell_to_dof()
            phic2f = phi[phicell2dof]
            
            pcell2dof = pspace.cell_to_dof()
            pc2f = p[pcell2dof]
            
            ucell2dof = usspace.cell_to_dof()
            u0 = u.reshape(mesh.GD,-1).T[...,0]
            u1 = u.reshape(mesh.GD,-1).T[...,1] 
            u0c2f = u0[ucell2dof]
            u1c2f = u1[ucell2dof]

            cellmeasure = mesh.entity_measure('cell')
            isMark = bm.abs(bm.mean(phic2f,axis=-1))>epsilon
            isMark = bm.logical_and(bm.mean(phic2f,axis=-1)>0.01,isMark)
            data = {'phi':phic2f,'p':pc2f, 'u0':u0c2f,'u1':u1c2f} 
            option = mesh.bisect_options(data=data, disp=False)
            mesh.coarsen(isMark,options=option)
            
            pspace , usspace, uspace = self.update_space(mesh)
            phi = usspace.function()
            phicell2dof = usspace.cell_to_dof()
            phi[phicell2dof.reshape(-1)] = option['data']['phi'].reshape(-1)
            
            p = pspace.function()
            pcell2dof = pspace.cell_to_dof()
            p[pcell2dof.reshape(-1)] = option['data']['p'].reshape(-1)
            
            u = uspace.function()
            ucell2dof = usspace.cell_to_dof()
            u0 = usspace.function()
            u1 = usspace.function()
            uscell2dof = usspace.cell_to_dof()
            u0[uscell2dof.reshape(-1)] = option['data']['u0'].reshape(-1)
            u1[uscell2dof.reshape(-1)] = option['data']['u1'].reshape(-1)
            u[:] = bm.stack((u0[:],u1[:]), axis=1).T.flatten()
        
        return mesh, phi, u, p

    def heaviside(self, phi):
        '''
        phi 距离函数
        返回函数
        '''
        epsilon = self.epsilon
        space = phi.space
        fun = space.function()
        tag = (-epsilon <= phi[:])  & (phi[:] <= epsilon)
        tag1 = phi[:] > epsilon
        fun[tag1] = 1
        fun[tag] = 0.5*(1+phi[tag]/epsilon) 
        fun[tag] += 0.5*bm.sin(bm.pi*phi[tag]/epsilon)/bm.pi
        return fun 
    
    def update_space(self, mesh, up=2, pp=1):
        '''
        更新空间
        '''
        usspace = LagrangeFESpace(mesh, p=up)
        uspace = TensorFunctionSpace(usspace, (mesh.GD,-1))
        pspace = LagrangeFESpace(mesh, p=pp)
        return pspace, usspace, uspace
    
    def update_function(self, phi):
        epsilon = self.epsilon
        H = self.heaviside(phi)
        rho = phi.space.function()
        eta = phi.space.function()
        rho[:] = self.pde.gas.rho/self.pde.liquid.rho + \
                (1 - self.pde.gas.rho/self.pde.liquid.rho) * H[:]
        eta[:] = self.pde.gas.eta/self.pde.liquid.eta + \
                (1 - self.pde.gas.eta/self.pde.liquid.eta) * H[:]
        return rho, eta
    
    def output(self, fname, mesh, u, p, phi, rho, mu): 
        mesh.nodedata['velocity'] = u.reshape(mesh.GD,-1).T
        mesh.nodedata['pressure'] = p
        mesh.nodedata['rho'] = rho
        mesh.nodedata['surface'] = phi
        mesh.nodedata['mu'] = mu
        mesh.to_vtk(fname=fname)
    

    def delta_fun(self, phi):
        fun = phi.space.function()
        epsilon = self.epsilon
        tag = (-epsilon <= phi[:])  & (phi[:] <= epsilon)
        fun[tag] = 1/(2*epsilon) * (1 + bm.cos(bm.pi*phi[tag]/epsilon))
        return fun
 
    def interface_normal(self, phi):
        fun = RecoveryAlg().grad_recovery(phi)
        fun[:] = fun[:] / bm.linalg.norm(fun[:], axis=-1)[..., bm.newaxis]
        return fun


    def IPCS_BForm_0(self, rho, eta, uspace, threshold=None):
        Re = self.pde.Re
        q = self.q
        dt = self.dt

        Bform = BilinearForm(uspace)
        
        @barycentric
        def coef_m(bcs, index):
            return rho(bcs, index) / dt
        M = ScalarMassIntegrator(coef=coef_m, q=q)

        
        @barycentric
        def coef_f(bcs, index):
            return -eta(bcs, index) / self.pde.Re
        F = FluidBoundaryFrictionIntegrator(coef=coef_f, q=q, threshold=threshold)
        
        @barycentric
        def coef_vm(bcs, index):
            return 2*eta(bcs, index) / self.pde.Re
        VW = ViscousWorkIntegrator(coef=coef_vm, q=q)
        Bform.add_integrator(VW)
        Bform.add_integrator(F)
        Bform.add_integrator(M)
        return Bform

    def IPCS_LForm_0(self, rho, eta, u0, p0, phi, pthreshold=None):
        dt = self.dt
        Re = self.pde.Re
        We = self.pde.We
        q = self.q
        uspace = u0.space
        mesh = uspace.mesh
        gd = mesh.GD

        Lform = LinearForm(uspace)
        
        def coef(bcs, index):
            result = 1/dt*rho(bcs, index)[..., bm.newaxis]*u0(bcs, index)
            result -= rho(bcs, index)[..., bm.newaxis] * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            
            delta = self.delta_fun(phi)
            normal = self.interface_normal(phi)
            kappa = normal.grad_value(bcs, index)[...,0,0]+normal.grad_value(bcs, index)[...,1,1]
            #result += 1/self.pde.We * kappa[..., bm.newaxis] * delta(bcs, index)[..., bm.newaxis] * phi.grad_value(bcs, index)
            return result
        ipcs0_lform_SI = SourceIntegrator(q=q)
        ipcs0_lform_SI.source = coef
        Lform.add_integrator(ipcs0_lform_SI)
        
        def G_coef(bcs, index):
            I = bm.eye(gd)
            result = bm.repeat(p0(bcs,index)[...,bm.newaxis], gd, axis=-1)
            result = bm.expand_dims(result, axis=-1) * I
            return result
        ipcs0_lform_GSI = GradSourceIntegrator(q=q)
        ipcs0_lform_GSI.source = G_coef
        Lform.add_integrator(ipcs0_lform_GSI)
        

        def B_coef(bcs, index):
            result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), mesh.face_unit_normal(index=index))
            return result

        ipcs0_lform_BSI = BoundaryFaceSourceIntegrator(q=q, threshold=pthreshold)
        ipcs0_lform_BSI.source = B_coef
        Lform.add_integrator(ipcs0_lform_BSI)
        return Lform


    def IPCS_BForm_1(self, pspace, uspace):
        dt = self.dt
        q = self.q

        Bform = BilinearForm(pspace)
        D = ScalarDiffusionIntegrator(coef=1, q=q)
        Bform.add_integrator(D)
        return Bform 
    
    def IPCS_LForm_1(self, rho, us, p0):
        pspace = p0.space
        dt = self.dt
        q = self.q

        def coef(bcs, index=None):
            result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
            result *= rho(bcs, index)
            return result
        Lform = LinearForm(pspace)
        ipcs1_lform_SI = SourceIntegrator(q=q)
        ipcs1_lform_SI.source = coef
        Lform.add_integrator(ipcs1_lform_SI) 
        
        def grad_coef(bcs, index=None):
            result = p0.grad_value(bcs, index)
            return result

        ipcs1_lform_GSI = GradSourceIntegrator(q=q)
        ipcs1_lform_GSI.source = grad_coef
        Lform.add_integrator(ipcs1_lform_GSI)
        return Lform
    

    def IPCS_BForm_2(self, rho, uspace):
        q = self.q

        Bform = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=rho, q=q)
        Bform.add_integrator(M)
        return Bform

    def IPCS_LForm_2(self, rho, us, p0, p1):
        uspace = us.space
        dt = self.dt
        q = self.q

        Lform = LinearForm(uspace)
        
        def coef(bcs, index):
            result = rho(bcs, index)[..., bm.newaxis] * us(bcs, index)
            result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
            return result

        ipcs2_lform_SI = SourceIntegrator(q=q)
        ipcs2_lform_SI.source = coef
        Lform.add_integrator(ipcs2_lform_SI)
        return Lform


    def evo_BForm(self, u, phispace):
        dt = self.dt
        Bform = BilinearForm(phispace)
        
        @barycentric
        def con_coef(bcs, index=None):
            result = u(bcs, index)
            return 0.5 * dt * result
        
        evo_CN_con = ScalarConvectionIntegrator(q=self.q)
        evo_CN_con.coef = con_coef
        Bform.add_integrator(ScalarMassIntegrator(q=self.q))
        Bform.add_integrator(evo_CN_con)
        return Bform

    def evo_LForm(self, u, phi0):
        phispace = phi0.space
        dt = self.dt
        Lform = LinearForm(phispace)
        
        @barycentric
        def source_coef(bcs, index=None):
            gradphi = phi0.grad_value(bcs, index) 
            uu = u(bcs, index)
            result = phi0(bcs, index) - 0.5 * dt * bm.einsum('...i, ...i -> ...', gradphi, uu)
            return result
        
        evo_CN_source = ScalarSourceIntegrator(q=self.q)
        evo_CN_source.source = source_coef
        Lform.add_integrator(evo_CN_source)
        return Lform
    
    
        
