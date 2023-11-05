import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse import coo_matrix,bmat

def computeFilter1(nelx,nely, rmin):
    """ Linear  density Filter Method, smooth the field along nlex*nely domain 
    Args:   
        nelx: number of elements in x direction
        nely: number of elements in y direction
        rmin: filter radius
    Returns:
        H: filter matrix
        Hs: sum of filter matrix
    """
    H = np.zeros((nelx*nely,nelx*nely));
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = (i1)*nely+j1;
            imin = max(i1-(np.ceil(rmin)-1),0.);
            imax = min(i1+(np.ceil(rmin)),nelx);
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1-(np.ceil(rmin)-1),0.);
                jmax = min(j1+(np.ceil(rmin)),nely);
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2*nely+j2;
                    H[e1, e2] = max(0.,rmin-\
                                       np.sqrt((i1-i2)**2+(j1-j2)**2));

    # Hs = np.sum(H,1);
    Hs=H.sum(1)
    
    
    return H, Hs.reshape(-1,1,order = 'F');


def computeFilter(nx, ny, rmin):
    """ from my top alternativing active phase code, it works well, selected this first
    Args:
        nx: number of elements in x direction
        ny: number of elements in y direction
        rmin: filter radius
    """
    nfilter=int(nx*ny*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nx):
        for j in range(ny):
            row=i*ny+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),ny))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*ny+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nx*ny,nx*ny)).tocsr()	
    Hs=H.sum(1)
    return H, Hs
# @ jax.jit
def applySensitivityFilter(filter_method,dJ,x,dv,H,Hs):
    """ Apply sensitivity filter
    """
    if filter_method == 1:
        dJ = H @ (x*dJ).reshape(-1,1,order='F')/Hs/(jnp.maximum(1e-3,x)).reshape(-1,1,order='F')
        dJ  = jnp.minimum(dJ,0).flatten('F')
    elif filter_method == 2:
        da = (H @ ((x*dJ).reshape(-1,1,order='F'))/Hs)
        dJ = da.flatten('F')
        # dJ = H @ np.reshape(x*dJ,-1,1,order='F')/Hs
        dv = H @ (x*dv).reshape(-1,1,order='F')/Hs
        # dv = H @ np.reshape(x*dv,-1,1,order='F')/Hs

        # dJ = dJ.flatten('F')
        # print('operator2_dJshape',dJ.shape)
        dv = dv.flatten('F')
    # print('dJshape',dJ.flatten('F').shape)
    return dJ,dv

def OC(filter_method,x,dJdx,dv,l,u,vf,H,Hs):
    """ Optimality Criteria
    Args:
        x: design variables
        dJdx: derivative of objective function
        l: lower bound
        u: upper bound
        vf: volume fraction constraint
    Returns:
        x_new: updated variables
    """

    x_old = x.copy()


    # l = l.copy()
    # u = u.copy()


    l1 = 0.0
    l2 = 1.e9
    # move = 0.1
    while (l2-l1) > 1e-9:
        lmid = float(0.5*(l2+l1))
        x_new = np.maximum(l,np.minimum(u,(x_old)*jnp.sqrt(jnp.abs(-dJdx/dv)/lmid)))


        if np.mean(x_new) - vf > 0:
            l1 = lmid
        else:
            l2 = lmid

    # if filter_method == 1:
    #     x_update = x_new
    #     print('xnewshape',x_new.shape)
    # else: # filter_method == 2
    #     x_update = H @ x_new.reshape(-1,1,order = 'F') /Hs
    #     x_update = x_update.flatten('F')
    #     print('x_updateshape',x_update.shape,'xnewshape',x_new.shape)

    change =np.max(np.abs(x_new-x_old));

    return x_new,change


        


def inti_guess(nx,ny,v,p,type=None):
    """ different initialization will come out different result
    Args:
        nelx: number of elements in x direction
        nely: number of elements in y direction
        v: volume fraction
        p: penalization
        type: initialization type
    Returns:
        alpha: initial design variables
    """
    alpha = np.zeros((nx*ny,p))

    def centrally_symmetric_init(x):
        '''
        Ensure the symmetry of the initial guess
        '''
        nx, ny = x.shape
        for i in range(nx):
            for j in range(ny):
                x[nx-i-1, ny-j-1] = x[i, j]
        return x

    if type == 0 or None: # default uniform  initial guess
        x = np.ones((nx,ny))
        print('*************************************')
        print('type=0,uniform initial guess')
        print('*************************************')



    elif type == 1:  #  intial guess with central hole
        x = np.ones((nx, ny))
        center_x, center_y = (nx-1) / 2,  (ny-1) / 2
        for i in range(ny):
            for j in range(nx):
                if (np.sqrt((i - center_x)**2 + (j - center_y)**2)) < (min(nx,ny)/6.):
                    x[j, i] = 1/2.
        # x = centrally_symmetric_init(x)
        
        # x = np.ones((nx,ny))
        # for i in range(ny):
        #     for j in range(nx):
        #         ii = i+1
        #         jj = j+1
        #         if (np.sqrt((ii-nx/2.)**2+(jj-ny/2.)**2)) < (min(nx,ny)/6.):
        #             x[j,i] = 1/2.
        # x = centrally_symmetric_init(x)
        print('**************************************')
        print('type=1,initial guess with central hole')
        print('**************************************')


    elif type ==2: # 4 holes at 4 boundaries
        x = np.ones((nx,ny))
        center_x, center_y = (nx-1) / 2,  (ny-1) / 2

        for i in range(ny):
            for j in range(nx):
                if ((np.sqrt((i - center_x)**2 + (j - 0.)**2)) < (min(nx,ny)/6.)) or \
                   ((np.sqrt((i - center_x)**2 + (j - ny+1)**2)) < (min(nx,ny)/6.)) or \
                     ((np.sqrt((i - 0.)**2 + (j - center_y)**2)) < (min(nx,ny)/6.)) or \
                        ((np.sqrt((i - (nx-1))**2 + (j - center_y)**2)) < (min(nx,ny)/6.)):
                    x[j, i] = 1/4.  
        print('************************************************')
        print('type=2, intial guess 4 holes at 4 boundaries')
        print('************************************************')
        
    elif type ==3:
        x = np.ones((nx,ny))
        lx = np.floor(nx/10).astype(int)
        ly =  np.floor(ny/2).astype(int)
        # ly =  np.floor(ny/10*9).astype(int)

        x[:lx+1, :ly+1] = 0.8
        x[:ly+1, nx-lx-1:] = 0.8
        x[ny-ly-1:, :lx+1] = 0.8
        x[ny-lx-1:, nx-ly-1:] = 0.8

        # x[:lx+1, :ly+1] = 0.8

        # # Rotating this rectangle 90 degrees and placing it in the bottom-left corner
        # x[ny-ly-1:, :lx+1] = 0.8

        # # Rotating again and placing it in the bottom-right corner
        # x[ny-lx-1:, ny-ly-1:] = 0.8

        # # Final 90-degree rotation and placing it in the top-right corner
        # x[:ly+1, nx-lx-1:] = 0.8





    x_mean = np.mean(x)
    alpha= np.zeros((nx*ny,p))
    for ii in range(p-1): # solid phase
        alpha[:,ii] =v[ii] /x_mean*x.flatten('F')
    alpha[:,-1] = 1 - np.sum(alpha[:,:-1],axis=1)



    # normalize
    return alpha

                


