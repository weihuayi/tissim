from fealpy.backend import backend_manager as bm

class two_phase_phield_solver:
    def __init__(self, pde):
        self.pde = pde

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
    
    def show_mesh(self,mesh):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        plt.show()

    
    def rho(self, phi):
        result = phi.space.function()
        result[:] = (self.pde.rho_up - self.pde.rho_down)/2 * phi[:]
        result[:] += (self.pde.rho_up + self.pde.rho_down)/2 
        return result
    

