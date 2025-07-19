import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext
from fealpy.decorator import barycentric

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.decorator import barycentric,cartesian

from fealpy.pde.navier_stokes_equation_2d import ChannelFlowWithLevelSet as PDE

# 参数设置
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解NS方程
        """)

parser.add_argument('--udegree',
        default=2, type=int,
        help='运动有限元空间的次数, 默认为 2 次.')

parser.add_argument('--pdegree',
        default=1, type=int,
        help='压力有限元空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=20, type=int,
        help='单元尺寸')

parser.add_argument('--nt',
        default=500, type=int,
        help='时间剖分段数，默认剖分 5000 段.')

parser.add_argument('--T',
        default=3 ,type=float,
        help='演化终止时间, 默认为 5')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

parser.add_argument('--step',
        default=5, type=int,
        help='隔多少步输出一次')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
ns = args.ns
output = args.output

alpha = 0.625/100
step = args.step
rho_gas = 1
rho_melt = 1 
mu_gas = 0.001
mu_melt = 0.05 



# 网格,空间,函数
udim = 2
h = 1
domain = [0,4*h,0,h]
mesh = MF.boxmesh2d([domain[0],domain[1],domain[2],domain[3]], 4*ns, ns)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
eps = 1e-12

@cartesian
def is_outflow_boundary(p):
    return np.abs(p[..., 0] - domain[1]) < eps

@cartesian
def is_inflow_boundary(p):
    return np.abs(p[..., 0] - domain[0]) < eps

@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 1] - domain[3]) < eps) | \
           (np.abs(p[..., 1] - domain[2]) < eps)

@cartesian
def u_inflow_dirichlet(p):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros(p.shape)
    value[...,0] = 2*h*y*(h-y)/(h**2)
    value[...,1] = 0
    return value


def dist(p):
    x = p[...,0]
    y = p[...,1]
    val =  (x-h) - 2*h*y*(h-y)/h**2
    return val

def changemu(s0,epp=0.1):
    H = s0.copy()
    tag_m = s0 > epp
    tag_g = s0 < -epp
    tag_mild = np.logical_and(~tag_m,~tag_g)
    H[tag_m] = 1
    H[tag_g] = 0
    H[tag_mild] = (1/2+s0/(2*epp)+np.sin(np.pi*s0/epp)/(2*np.pi))[tag_mild]
    H = mu_melt+(mu_gas-mu_melt)*H
    return H

def changerho(s0,epp=0.1):
    H = s0.copy()
    tag_m = s0 > epp
    tag_g = s0 < -epp
    tag_mild = np.logical_and(~tag_m,~tag_g)
    H[tag_m] = 1
    H[tag_g] = 0
    H[tag_mild] = (1/2+s0/(2*epp)+np.sin(np.pi*s0/epp)/(2*np.pi))[tag_mild]
    H = rho_melt+(rho_gas-rho_melt)*H
    return H

mesh.grad_lambda()
uspace = LagrangeFiniteElementSpace(mesh,p=udegree)

phi0 = uspace.interpolation(dist)
## 初始网格
## 加密
for i in range(5):
    cell2dof = mesh.cell_to_ipoint(udegree)
    phi0c2f = phi0[cell2dof]
    isMark = np.abs(np.mean(phi0c2f,axis=-1))< 0.05
    data = {'phi0':phi0c2f} 
    option = mesh.bisect_options(data=data)
    mesh.bisect(isMark,options=option)

    uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
    cell2dof = uspace.cell_to_dof()
    phi0 = uspace.function()
    phi0[cell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)


phi0 = uspace.interpolation(dist)

mu = changemu(phi0)
rho = changerho(phi0)
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)

u0 = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)
ips = uspace.interpolation_points()
ipsu0 = u_inflow_dirichlet(ips)

u0[phi0>=0] = 0
u0[phi0<0] = ipsu0[phi0<0]

p0 = pspace.function()
p1 = pspace.function()

fname = output + 'test_.vtu'
mesh.nodedata['velocity'] = u0
mesh.nodedata['mu'] = mu
mesh.nodedata['phi'] = phi0
mesh.nodedata['rho'] = rho
fname = output + 'test_0000000000.vtu'
mesh.to_vtk(fname=fname)
ctx = DMumpsContext()
ctx.set_silent()

def rein(phi0,dt=0.0001):
    phi1 = uspace.function()
    phi2 = uspace.function()
    phi1[:] = phi0
    A = uspace.mass_matrix()
    SS = uspace.stiff_matrix()
    ctx.set_centralized_sparse(A)
    E0 = 1e10
    cont = 0

    @barycentric
    def signp(bcs):
        val0 = phi0(bcs)
        grad = phi1.grad_value(bcs)
        val1 = np.sqrt(np.sum(grad**2, -1))

        val = (1 - val1)*np.sign(val0)  
        return val
     
    for i in range(100):
        print("i = ", i)

        b = dt*uspace.source_vector(signp)
        b += A@phi1
        b -= dt*alpha*(SS@phi1)
        ctx.set_rhs(b)
        ctx.run(job=6)
        phi2[:] = b

        E = uspace.integralalg.error(phi2, phi1)
        print("相邻两次迭代误差:", E)
        
        if E < 2e-4:
            break
        
        if E0 < E:
            fail = 1
            print("求解发散!", cont)
            break
        
        cont += 1
        E0 = E
        phi1[:] = phi2
        
        #val1 = np.sqrt(np.sum(phi1.grad_value(bc)**2, axis=-1))
    return phi1

for i in range(0, nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
    
    mesh.grad_lambda()
    uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
    pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
    # 更新
    ucell2dof = uspace.cell_to_dof()
    pcell2dof= pspace.cell_to_dof()
    
    ugdof = mesh.number_of_global_ipoints(udegree)
    pgdof = mesh.number_of_global_ipoints(pdegree)
    gdof = pgdof+2*ugdof


    M = mesh.cell_phi_phi_matrix(udegree, udegree)
    M = mesh.construct_matrix(udegree, udegree, M)

    C1 = mesh.cell_gphix_phi_matrix(udegree, pdegree)
    C2 = mesh.cell_gphiy_phi_matrix(udegree, pdegree)
    C1 = mesh.construct_matrix(udegree, pdegree, C1)
    C2 = mesh.construct_matrix(udegree, pdegree, C2)

    xx = np.zeros(gdof, np.float64)

    is_u_bdof = uspace.is_boundary_dof()
    is_uin_bdof = uspace.is_boundary_dof(threshold = is_inflow_boundary)
    is_uout_bdof = uspace.is_boundary_dof(threshold = is_outflow_boundary)
    is_p_bdof = pspace.is_boundary_dof(threshold = is_outflow_boundary)

    is_u_bdof[is_uout_bdof] = False 

    ipoint = uspace.interpolation_points()[is_uin_bdof]
    uinfow = u_inflow_dirichlet(ipoint)
    xx[0:ugdof][is_uin_bdof] = uinfow[:,0]
    xx[ugdof:2*ugdof][is_uin_bdof] = uinfow[:,1]

    isBdDof = np.hstack([is_u_bdof,is_u_bdof,is_p_bdof])
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)

    AA = uspace.mass_matrix()
    SS = uspace.stiff_matrix()
        

    #组装左端矩阵
    S = mesh.cell_stiff_matrix(udegree, udegree ,udegree ,mu)
    S = mesh.construct_matrix(udegree,udegree,S)
    cu0x = u0[..., 0]
    cu0y = u0[..., 1]
     
    D1 = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, c3=cu0x)
    D2 = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, c3=cu0y)
    D1 = mesh.construct_matrix(udegree, udegree, D1)
    D2 = mesh.construct_matrix(udegree, udegree, D2)
     
    A = bmat([[1/dt*M + S+D1+D2, None, -C1],\
            [None, 1/dt*M + S +D1+D2, -C2],\
            [C1.T, C2.T, None]], format='csr')
    #组装右端向量
    fb1 = np.hstack((M@u0[:,0],M@u0[:,1])).T
     
    b = 1/dt*fb1
    b = np.hstack((b,[0]*pgdof))
    
    ## 边界条件处理
    A = T@A + Tbd
    b[isBdDof] = xx[isBdDof]
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    #levelset
    cu1x = np.array(u1[...,0])
    cu1y = np.array(u1[...,1])
    D2x = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, c3=cu1x) 
    D2y = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, c3=cu1y) 
    D2x = mesh.construct_matrix(udegree, udegree, D2x)
    D2y = mesh.construct_matrix(udegree, udegree, D2y)
    D2 = D2x+D2y
    D = M  + dt/2*D2

    b4 = M@phi0 - dt/2*D2@phi0
     
    ctx.set_centralized_sparse(D)
    x = b4.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    phi0[:] = x 
    
    for j in range(5): 
        phi0c2f = phi0[ucell2dof]
        u1xc2f = u1[:,0][ucell2dof]
        u1yc2f = u1[:,1][ucell2dof]
        p1c2f = p1[pcell2dof]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))<0.05
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)>-0.01,isMark)
        isMark = np.logical_and(isMark,cellmeasure>4e-5)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.bisect(isMark,options=option)

        uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
        pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
        ucell2dof = uspace.cell_to_dof()
        pcell2dof = pspace.cell_to_dof()
        phi0 = uspace.function()
        u1 = uspace.function(dim=2)
        p1 = pspace.function()
        phi0[ucell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[:,0][ucell2dof.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[:,1][ucell2dof.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[pcell2dof.reshape(-1)] = option['data']['p1'].reshape(-1)
        rho = changerho(phi0)
        mu = changemu(phi0)
    
    
    #重新粗化
    for j in range(5):
        phi0c2f = phi0[ucell2dof]
        u1xc2f = u1[:,0][ucell2dof]
        u1yc2f = u1[:,1][ucell2dof]
        p1c2f = p1[pcell2dof]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))>0.05
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)<0.01,isMark)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.coarsen(isMark,options=option)

        uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
        pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
        ucell2dof = uspace.cell_to_dof()
        pcell2dof = pspace.cell_to_dof()
        phi0 = uspace.function()
        u1 = uspace.function(dim=2)
        p1 = pspace.function()
        phi0[ucell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[:,0][ucell2dof.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[:,1][ucell2dof.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[pcell2dof.reshape(-1)] = option['data']['p1'].reshape(-1)
        rho = changerho(phi0)
        mu = changemu(phi0)
    
    if i%step == 0:
        phi0 = rein(phi0)
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['velocity'] = u1
        mesh.nodedata['pressure'] = p1
        mesh.nodedata['mu'] = mu 
        mesh.nodedata['rho'] = rho
        mesh.nodedata['phi'] = phi0
        mesh.to_vtk(fname=fname) 
    
    u0 = uspace.function(dim=2) 
    p0 = pspace.function() 
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()

ctx.destroy()
