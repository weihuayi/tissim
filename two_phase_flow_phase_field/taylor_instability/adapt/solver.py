from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

class two_phase_phield_solver:
    def __init__(self, pde, ns_solver, interface_solver):
        self.ns_solver = ns_solver
        self.interface_solver = interface_solver
        self.pde = pde
    
    def interface_refine(self, phi, func_data, value, maxit, mincellmeasure):

        bcs = bm.array([[1/3,1/3,1/3]], dtype=bm.float64)
        mesh = phi.space.mesh
        option_data = {}
        k = 0

        for i in range(maxit):
            for func in func_data:
                space = func.space
                if space.__class__.__name__ == 'TensorFunctionSpace':
                   gd = space.shape[0]
                   cell2dof = space.scalar_space.cell_to_dof()
                   for j in range(gd):
                       option_data[f'tensor{k}'] =  func.reshape(gd,-1).T[...,j][cell2dof]
                       k += 1

                else:
                    cell2dof = space.cell_to_dof()
                    option_data[f'scalar{k}'] = func[cell2dof]
                    k += 1
             
            grad_norm_phi = bm.linalg.norm(phi.grad_value(bcs),axis=-1)[:,0]
            is_Marked =  bm.abs(grad_norm_phi) > value
            cellmeasure = mesh.entity_measure('cell')
            is_Marked = bm.logical_and(is_Marked,cellmeasure>mincellmeasure)
            if bm.sum(is_Marked) == 0:
                return mesh
            
            option_data['phi'] = phi[phi.space.cell_to_dof()]
            option = mesh.bisect_options(data=option_data, disp=False)
            mesh.bisect(is_Marked, options=option)
            
            k = 0
            for i,func in enumerate(func_data):
                space = func.space
                if space.__class__.__name__ == 'TensorFunctionSpace':
                    gd = space.shape[0]
                    space = LagrangeFESpace(mesh, p=space.p)
                    cell2dof = space.cell_to_dof()
                    result = []
                    for j in range(gd):
                        scalar_fun  = space.function()
                        scalar_fun[cell2dof.reshape(-1)] = option['data'][f'tensor{k}'].reshape(-1)
                        result.append(scalar_fun[:])
                        k += 1
                    space = TensorFunctionSpace(space, (mesh.GD,-1))
                    func_data[i] = space.function()
                    func_data[i][:] = bm.stack(result, axis=1).T.flatten()

                else:
                    space = LagrangeFESpace(mesh, p=space.p)
                    func_data[i] = space.function()
                    cell2dof = space.cell_to_dof()
                    func_data[i][cell2dof.reshape(-1)] = option['data'][f'scalar{k}'].reshape(-1)                
                    k += 1
            phi = LagrangeFESpace(mesh, p=phi.space.p).function()
            phi[phi.space.cell_to_dof().reshape(-1)] = option['data']['phi'].reshape(-1)
        return mesh

    def interface_coarsen(self, phi, func_data, value, maxit):
        bcs = bm.array([[1/3,1/3,1/3]], dtype=bm.float64)
        mesh = phi.space.mesh
        option_data = {}
        
        for i in range(maxit):
            k = 0
            for func in func_data:
                space = func.space
                if space.__class__.__name__ == 'TensorFunctionSpace':
                   gd = space.shape[0]
                   cell2dof = space.scalar_space.cell_to_dof()
                   for j in range(gd):
                       option_data[f'tensor{k}'] =  func.reshape(gd,-1).T[...,j][cell2dof]
                       k += 1

                else:
                    cell2dof = space.cell_to_dof()
                    option_data[f'scalar{k}'] = func[cell2dof]
                    k += 1
            grad_norm_phi = bm.linalg.norm(phi.grad_value(bcs),axis=-1)[:,0]
            is_Marked =  bm.abs(grad_norm_phi) < value
            cellmeasure = mesh.entity_measure('cell')
            if bm.sum(is_Marked) == 0:
                print("no coarsen")
                return mesh
            option = mesh.bisect_options(data=option_data, disp=False)
            option_data['phi'] = phi[phi.space.cell_to_dof()]
            option = mesh.bisect_options(data=option_data, disp=False)
            mesh.coarsen(is_Marked,options=option)
            
            k = 0
            for i,func in enumerate(func_data):
                space = func.space
                if space.__class__.__name__ == 'TensorFunctionSpace':
                    gd = space.shape[0]
                    space = LagrangeFESpace(mesh, p=space.p)
                    cell2dof = space.cell_to_dof()
                    result = []
                    for j in range(gd):
                        scalar_fun  = space.function()
                        scalar_fun[cell2dof.reshape(-1)] = option['data'][f'tensor{k}'].reshape(-1)
                        result.append(scalar_fun[:])
                        k += 1
                    space = TensorFunctionSpace(space, (mesh.GD,-1))
                    func_data[i] = space.function()
                    func_data[i][:] = bm.stack(result, axis=1).T.flatten()
                else:
                    space = LagrangeFESpace(mesh, p=space.p)
                    func_data[i] = space.function()
                    cell2dof = space.cell_to_dof()
                    func_data[i][cell2dof.reshape(-1)] = option['data'][f'scalar{k}'].reshape(-1)                
                    k += 1
            phi = LagrangeFESpace(mesh, p=phi.space.p).function()
            phi[phi.space.cell_to_dof().reshape(-1)] = option['data']['phi'].reshape(-1)
        return mesh
        

    def init_interface_refine(self, phi, refine_value, maxit):
        uspace = self.ns_solver.uspace 
        pspace = self.ns_solver.pspace
        phispace = phi.space
        mesh = phispace.mesh
        
        bcs = bm.array([[1/3,1/3,1/3]], dtype=bm.float64)
        for i in range(maxit):
            grad_norm_phi = bm.linalg.norm(phi.grad_value(bcs),axis=-1)[:,0]
            is_Marked =  bm.abs(grad_norm_phi) > refine_value

            phicell2dof = phispace.cell_to_dof()
            phic2f = phi[phicell2dof]
            data = {'phi':phic2f}
            
            option = mesh.bisect_options(data=data, disp=False)
            mesh.bisect((is_Marked), options=option)
            
            phispace = LagrangeFESpace(mesh, p=phispace.p)
            self.interface_solver.space = phispace
            phi = phispace.function()
            phicell2dof = phispace.cell_to_dof()
            phi[phicell2dof.reshape(-1)] = option['data']['phi'].reshape(-1)
        return mesh

    def update_space(self, mesh):
        '''
        更新空间
        '''
        up = self.ns_solver.uspace.p
        pp = self.ns_solver.pspace.p
        
        usspace = LagrangeFESpace(mesh, p=up)
        uspace = TensorFunctionSpace(usspace, (mesh.GD,-1))
        pspace = LagrangeFESpace(mesh, p=pp)
        self.ns_solver.set.uspace = uspace
        self.ns_solver.set.pspace = pspace
        self.interface_solver.space = usspace
        return pspace, usspace, uspace
    
    def update_function(self, phi):
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
    

