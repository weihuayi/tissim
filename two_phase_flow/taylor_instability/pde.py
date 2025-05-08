from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian,barycentric


class RayleignTaylor:
    def __init__(self, eps=1e-10):
        self.rho_up = 3
        self.rho_down = 1
        self.Re = 3000
        self.Fr = 1
        self.epsilon = 0.01
        self.Pe = 1/self.epsilon
        self.eps = eps    

    def domain(self):
        '''
        单位m
        '''
        domain = [0, 1, 0, 4]
        return domain

    def init_mesh(self, nx=128, ny=512):
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
        val =  bm.tanh((y-2-0.1*bm.cos(bm.pi*2*x))/(bm.sqrt(bm.tensor(2))*self.epsilon))
        return val
    
    @cartesian
    def velocity_boundary(self, p):
        '''
        边界速度
        '''
        result = bm.zeros_like(p, dtype=bm.float64)
        return result
    
    @cartesian
    def pressure_boundary(self, p):
        '''
        边界压力
        '''
        result = bm.zeros_like(p[..., 0], dtype=bm.float64)
        return result

    @cartesian
    def is_p_boundary(self, p):
        result = bm.zeros_like(p[..., 0], dtype=bool)
        return result

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
        tag_right = bm.abs(p[..., 0] - 1) < self.eps
        return tag_up | tag_down | tag_left | tag_right 
    
    @cartesian
    def is_uy_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
        tag_right = bm.abs(p[..., 0] - 1) < self.eps
        return tag_up | tag_down | tag_left | tag_right 
    
    @cartesian
    def is_ux_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 4.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down  
