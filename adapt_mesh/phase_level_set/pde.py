from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian,barycentric


class TwoPhaseModel:
    '''
    Injection model
    杨斌新论文
    '''
    class liquid:
        def __init__(self):
            self.rho = 1
            self.eta = 1000
            self.theta = 28.35e-3 #界面张力系数

    class gas:
        def __init__(self):
            self.rho = 0.01
            self.eta = 1 
    
    class dimensionless:
        def __init__(self):
            self.L = 1
            self.U = 1

    def __init__(self, eps=1e-10):
        self.eps = eps
        self.gas = self.gas()
        self.liquid = self.liquid()
        self.dimensionless = self.dimensionless() 

    @property
    def Re(self):
        '''
        流体的雷诺数
        '''
        result = self.liquid.rho * self.dimensionless.U * \
                self.dimensionless.L / self.liquid.eta
        return result 

    @property
    def We(self):
        '''
        流体的韦伯数
        '''
        result = self.liquid.rho * self.dimensionless.U**2 * \
                self.dimensionless.L / self.liquid.theta
        return result


    def domain(self):
        '''
        单位m
        '''
        domain = [0, 10, 0, 1]
        return domain

    def mesh(self, nx=100, ny=10):
        '''
        生成网格
        nx, ny: 网格数目
        '''
        mesh = TriangleMesh.from_box(self.domain(), nx, ny)
        return mesh
    
    @cartesian
    def init_interface(self, p):
        '''
        初始化界面
        '''
        x = p[...,0]
        y = p[...,1]
        #val =  (x-h) - 2*h*y*(h-y)/h**2
        val =  0.5 - x
        return val

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = (10-x) 
        return val

    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 10.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        #tag_left = bm.abs(p[..., 0]) < self.eps
        #return tag_left | tag_up | tag_down 
        return tag_up | tag_down 






class OnePhaseModel:
    class liquid:
        def __init__(self):
            self.rho = 1
            self.eta = 1
            self.theta = 28.35e-3 #界面张力系数

    class gas:
        def __init__(self):
            self.rho = 1
            self.eta = 1
    
    class dimensionless:
        def __init__(self):
            self.L = 1
            self.U = 1

    def __init__(self, eps=1e-10):
        self.eps = eps
        self.gas = self.gas()
        self.liquid = self.liquid()
        self.dimensionless = self.dimensionless() 

    @property
    def Re(self):
        '''
        流体的雷诺数
        '''
        result = self.liquid.rho * self.dimensionless.U * \
                self.dimensionless.L / self.liquid.eta
        return result 

    @property
    def We(self):
        '''
        流体的韦伯数
        '''
        result = self.liquid.rho * self.dimensionless.U**2 * \
                self.dimensionless.L / self.liquid.theta
        return result


    def domain(self):
        '''
        单位m
        '''
        domain = [0, 1, 0, 1]
        return domain

    def mesh(self, nx=16, ny=16):
        '''
        生成网格
        nx, ny: 网格数目
        '''
        mesh = TriangleMesh.from_box(self.domain(), nx, ny)
        return mesh
    
    @cartesian
    def init_interface(self, p):
        '''
        初始化界面
        '''
        x = p[...,0]
        y = p[...,1]
        #val =  (x-h) - 2*h*y*(h-y)/h**2
        val =  0.5-x
        return val

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = 8*(1-x) 
        return val

    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down 
