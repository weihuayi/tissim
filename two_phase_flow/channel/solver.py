from fealpy.backend import backend_manager as bm

class two_phase_phield_solver:
    def __init__(self, pde):
        self.pde = pde
    
    def rho(self, phi):
        result = phi.space.function()
        result[:] = (self.pde.rho_left - self.pde.rho_right)/2 * phi[:]
        result[:] += (self.pde.rho_left + self.pde.rho_right)/2 
        return result
    

