#========== import pks==========
#
# ==============================

import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
# from skimage.transform import resize

@ jit
def elementMatVec(a, b, phi):
    """ elementary stiffness matrix and load vector for a single element,
    Args:
        a: half of the length of the element
        b: half of the width of the element
        phi: angle of the element, normally 90 degree
    Returns:
        keLambda: elementary stiffness matrix for lambda
        keMu: elementary stiffness matrix for mu
        feLambda: elementary load vector for lambda
        feMu: elementary load vector for mu
        feAlpha: elementary load vector for alpha
    """

    CMu = jnp.diag(jnp.array([2, 2, 1]))
    CLambda = jnp.zeros((3, 3))
    CLambda = CLambda.at[0:2, 0:2].set(1)
    # Two Gauss points in both directions
    xx = jnp.array([-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)])
    yy = xx
    ww = jnp.array([1, 1])

    # Initialize
    keLambda = jnp.zeros((8, 8))
    keMu = jnp.zeros((8, 8))
    feLambda = jnp.zeros((8, 3))
    feMu = jnp.zeros((8, 3))
    feAlpha = jnp.zeros((8, 1))

    L = jnp.zeros((3, 4))
    L = L.at[0, 0].set(1)
    L = L.at[1, 3].set(1)
    L = L.at[2, 1:3].set(1)
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            # Integration point
            x = xx[ii]
            y = yy[jj]

            # Differentiated shape functions
            dNx = 1 / 4 * jnp.array([-(1 - y), (1 - y), (1 + y), -(1 + y)])
            dNy = 1 / 4 * jnp.array([-(1 - x), -(1 + x), (1 + x), (1 - x)])

            # Jacobian
            J = jnp.matmul(
                jnp.vstack((dNx, dNy)),
                jnp.array([[-a, a, a + 2 * b / jnp.tan(phi * jnp.pi / 180), 2 * b / jnp.tan(phi * jnp.pi / 180) - a],
                    [-b, -b, b, b]]).T)
        
            detJ = J.at[0,0].get() * J.at[1,1].get() - J.at[0,1].get() * J.at[1,0].get()
            invJ = 1 / detJ * jnp.array([[J.at[1,1].get(), -J.at[0,1].get()], [-J.at[1,0].get(), J.at[0,0].get()]])

            # Weight factor at this point
            weight = ww.at[ii].get() * ww.at[jj].get() * detJ

            # Strain-displacement matrix "matlab code G = [invJ zeros(2); zeros(2) invJ];"
            G = jnp.vstack([jnp.hstack([invJ,jnp.zeros((2,2))]),jnp.hstack([jnp.zeros((2,2)),invJ])])
            dN = jnp.zeros((4, 8))    
            dN = dN.at[0,0::2].set(dNx)
            dN = dN.at[1,0::2].set(dNy)
            dN = dN.at[2,1::2].set(dNx)
            dN = dN.at[3,1::2].set(dNy)
            B = jnp.matmul(jnp.matmul(L,G),dN)
            # Element matrices
            keLambda = keLambda + weight * jnp.matmul(jnp.matmul(B.T, CLambda), B)
            keMu = keMu + weight * jnp.matmul(jnp.matmul(B.T, CMu), B)
            # Element load
            feLambda = feLambda + weight * jnp.matmul(jnp.matmul(B.T, CLambda),jnp.diag(jnp.array([1,1,1])) )
            feMu = feMu + weight * jnp.matmul(jnp.matmul(B.T, CMu),jnp.diag(jnp.array([1,1,1])) )
            feAlpha = feAlpha + weight * jnp.matmul(jnp.matmul(B.T, CMu),(jnp.array([1,1,0]).reshape(-1,1,order='F')) )

    return keLambda, keMu, feLambda, feMu, feAlpha




def edofMat_func(nelx,nely):
    """ Element degree of freedom matrix
    Args:
        nelx: number of elements in x direction
        nely: number of elements in y direction
    Returns:
        edofMat: element degree of freedom matrix (note for global dof, the pbcs will be forced to applied 'applied_pbcs' function)
    """
    nel = nelx * nely
    nodenrs = jnp.reshape(jnp.arange(0,(1+nelx)*(1+nely)),(1+nely,1+nelx),order='F') # note starting from 0
    edofVec = jnp.reshape(2*nodenrs[0:-1,0:-1]+2,(nel,1),order='F')
    edofMat = jnp.repeat(edofVec, 8, axis=1)
    edofMat = edofMat + jnp.repeat(jnp.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1])[:, jnp.newaxis], nel, axis=1).T
    return edofMat





# @jit
# def applied_pbcs(nelx,nely):
#     """ Applied periodic boundary conditions
#     Args:
#         nelx: number of elements in x direction
#         nely: number of elements in y direction
#     Returns:
#         dofVector: degree of freedom vector (the perdoic bouary will be applied in main code edofMat = dofVector[edofMat])
#     """
#     nn = (1+nelx)*(1+nely) # total number of nodes
#     nnP = (nelx)*(nely)   # Total number of unique nodes, for pbcs
#     list1 = jnp.arange(0,nnP);
#     # nnPArray_v0 = jnp.reshape(jnp.arange(0,nnP), (nely, nelx), order='F')
#     nnPArray_v0 = jnp.reshape(list1, (nely, nelx), order='F')
#     nnPArray_v1 = jnp.vstack((nnPArray_v0, nnPArray_v0[0,:]))
#     nnPArray = jnp.hstack((nnPArray_v1, nnPArray_v1[:,0][:,jnp.newaxis]))
#     dofVector = jnp.zeros((2*nn))
#     dofVector = dofVector.at[0::2].set(2*nnPArray.flatten(order='F'))
#     dofVector = dofVector.at[1::2].set(2*nnPArray.flatten(order='F')+1)
#     return dofVector




def applied_pbcs(nelx, nely):
    """Applied periodic boundary conditions
    Args:
        nelx: number of elements in x direction
        nely: number of elements in y direction
    Returns:
        dofVector: degree of freedom vector (the periodic boundary will be applied in the main code edofMat = dofVector[edofMat])
    """
    nn = (1 + nelx) * (1 + nely)  # total number of nodes
    nnP = nelx * nely  # Total number of unique nodes, for pbcs
    nnPArray_v0 = np.reshape(np.arange(0, nnP), (nely, nelx), order='F')
    nnPArray_v1 = np.vstack((nnPArray_v0, nnPArray_v0[0, :]))
    nnPArray = np.hstack((nnPArray_v1, nnPArray_v1[:, 0][:, np.newaxis]))
    dofVector = np.zeros((2 * nn))
    dofVector[0::2] = 2 * nnPArray.flatten(order='F')
    dofVector[1::2] = 2 * nnPArray.flatten(order='F') + 1
    dofVector = jnp.array(dofVector).astype(int)
    return dofVector

class InitialDefineVar():
    """ Pre-compute variables, meshes, mat, etc,optimization parameters,etc.
    Args:
        nelx: number of elements in x direction
        nely: number of elements in y direction
        E0: Young's modulus of the material
        nu: Poisson's ratio of the material
        penal: penalization power
        rmin: filter radius
        volfrac: volume fraction
        
    """
    def __init__(self) -> None:
        pass

    def HomoVar(self):
        self.nelx = 30
        self.nely = 30
        self.E = jnp.array([1.e-9, 1., 2.])
        self.nu = jnp.array([0.3, 0.3, 0.3])
        self.cte = jnp.array([0.3, 0.3, 0.3])
        self.q = 3#  number of materials
        self.penal = 3.0 # penalization power
    def OptimVar(self): 
        self.penal = 3.0 # penalization power
        self.rmin = 1.5 # filter radius
        self.volfrac = 0.5 # volume fraction
        self.q = 3 # number of materials


@ jit
def obj_and_selected_grad(f,x):
    """ compute the objective value (as function of input, which is a vector) and the gradient of obj w.r.t to input at ''indeces_v''
    Args:
        f: the objective function,e.g f(x), 
        x: inputs, or design variable
        indix_v: index of inputs x, which is intested for gradient information, a vector 
    Returns:
        obj: objective function
        dobj_selected_grad: seleted gradient to 'x', dependent on the indix_v
    """
    # x = x.reshape(-1,1,order = 'F')
    print('call the obj_and_selected_grad')
    # obj = f(x)
    # print(obj)
    # dobj_selected_grad = jnp.take(jax.grad(f(x),indix_v))

    # return obj 


# @jit
def make_bitmap(p, nx, ny, alpha):
    """ make bitmap for visualization
    Args: 
        p: number of materials
        nx: number of elements in x direction
        ny: number of elements in y direction
        alpha: the design variable, a vector of size (nx*ny, p)
    Returns:
        I: bitmap
    """
    alpha = alpha.reshape(nx * ny, p, order='F');
    alpha = np.array(alpha)
    color = np.array([[1, 0, 0], [0, 0, 1],[1, 1, 1], [0, 1, 0], [0, 0, 0]])
    I = np.zeros((nx * ny, 3))
    for j in range(p):
        I[:, :3] += alpha[:, j, None] * color[j, :3]
    I = resize(I.reshape(ny, nx, 3,order = 'F'), (ny * 10, nx * 10), order=1)
    # I = I.reshape(ny, nx, 3,order='F')
    return I


def vmap_liner_solver(A,b):
    return jax.jit(jnp.linalg.solve(A,b))



        



