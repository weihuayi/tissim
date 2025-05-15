from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian,barycentric


class TwoPhaseModel:
    def __init__(self, eps=1e-10):
        self.rho_left = 10
        self.rho_right = 1
        self.Re = 1
        self.epsilon = 0.01
        self.Pe = 1/self.epsilon
        self.eps = eps    

    def domain(self):
        '''
        单位m
        '''
        domain = [0, 4, 0, 1]
        return domain

    def init_mesh(self, nx=512, ny=128):
        '''
        生成网格
        nx, ny: 网格数目
        '''
        mesh = TriangleMesh.from_box(self.domain(), nx, ny)
        self.mesh = mesh
        return mesh
    
    @cartesian
    def init_interface(self, p):
        '''
        初始化界面
        '''
        x = p[...,0]
        y = p[...,1]
        val =  bm.tanh((x-0.2)/(bm.sqrt(bm.array(2))*self.epsilon))
        return val
    
    @cartesian
    def velocity_boundary(self, p):
        '''
        边界速度
        '''
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure_boundary(self, p):
        x = p[..., 0]
        val = (4-x) 
        return val

    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 4) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        tag_left = bm.abs(p[..., 0]) < self.eps
        return tag_left | tag_up | tag_down 
