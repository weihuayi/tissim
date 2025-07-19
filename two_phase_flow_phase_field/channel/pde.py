from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian,barycentric
import matplotlib.pyplot as plt
import numpy as np


class TwoPhaseModel:
    def __init__(self, eps=1e-10):
        self.rho_right = 1000
        self.rho_left = 10
        self.Re = 100
        self.epsilon = 0.01
        self.Pe = 1/self.epsilon
        self.eps = eps    

    def domain(self):
        '''
        单位m
        '''
        domain = [0, 10, 0, 1]
        return domain

    def init_mesh(self, nx=200, ny=20):
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
        val =  bm.tanh((x-0.1)/(bm.sqrt(bm.array(2))*self.epsilon))
        return val
    
    @cartesian
    def velocity_boundary(self, p):
        '''
        边界速度
        '''
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        value[...,0] = 5
        return value
    
    @cartesian
    def pressure_boundary(self, p):
        x = p[..., 0]
        val = (10-x) 
        return val

    @cartesian
    def u_dirichlet(self, p):
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        
        # 在入口边界设置速度
        is_inlet = self.is_inlet_boundary(p)
        value[is_inlet, 0] = 0.8*y[is_inlet]*(1-y[is_inlet])  # ux = 5
        value[is_inlet, 1] = 0  # uy = 0
        
        # 其他边界速度为0
        is_other = ~is_inlet
        value[is_other, 0] = 0  # ux = 0
        value[is_other, 1] = 0  # uy = 0
        
        return value
        

    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 4) < self.eps
        #result = tag_left | tag_right
        result = bm.zeros_like(p[...,0], dtype=bm.bool)
        return result

    @cartesian
    def is_wall_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_inlet = (p[..., 1] >= 0.2) & (p[..., 1] <= 0.75)
        return (tag_left & ~tag_inlet) | tag_up | tag_down

    @cartesian
    def is_inlet_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_y = (p[..., 1] >= 0.2) & (p[..., 1] <= 0.75)
        #return tag_left & tag_y
        return tag_left

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        tag_left = bm.abs(p[..., 0]) < self.eps
        return tag_up | tag_down | tag_left


    @cartesian
    def is_outlet_boundary(self, p):
        tag_right = bm.abs(p[..., 0] - 10.0) < self.eps
        return tag_right


def plot_boundaries(model):
    mesh = model.init_mesh()  # 生成网格
    node = mesh.node  # 所有节点坐标
    # 判断每个节点属于哪个边界
    is_inlet = model.is_inlet_boundary(node)
    is_outlet = model.is_outlet_boundary(node)
    is_wall = model.is_wall_boundary(node)

    plt.figure(figsize=(12, 2))
    plt.scatter(node[is_inlet, 0], node[is_inlet, 1], c='blue', label='Inlet')
    plt.scatter(node[is_outlet, 0], node[is_outlet, 1], c='red', label='Outlet')
    plt.scatter(node[is_wall, 0], node[is_wall, 1], c='green', label='Wall')
    plt.legend()
    plt.title("边界可视化")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# 用法示例
if __name__ == "__main__":
    model = TwoPhaseModel()
    plot_boundaries(model)
    