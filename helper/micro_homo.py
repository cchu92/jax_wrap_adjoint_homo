import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import helper.utils as utils
from functools import partial


class Homogenize:
    """ 2d multi-material thermal-mechancial homogenization
    """
    # def __init__(self,nelx,nely,E,nu,cte,penal,q,ft,out_iter=None):
    def __init__(self,Mesh,Mat):
        """ initialize the class
        Args:
            Mesh: mesh size dictionary: {'nelx','nely'} 
            Mat: material propertis dictionary {'E','nu','cte'}
            DynamicArgs: parameters define the objective function 
            DynamicArgs = {'out_iter','ft','penal'}
                    out_iter: loop iteration out side for top_aapo 
                    ft: logic
                    penal: penalty factor to density 
        Returns: 
            CH: homogenized elasticity tensor
            beta_H: homogenized thermal stress tensor
        """
        self.Mesh = Mesh
        self.Mat = Mat
        self.nelx,self.nely = Mesh['nelx'],Mesh['nely']
        self.E,self.nu,self.alpha_val,self.q = Mat['E'],Mat['nu'],Mat['cte'],Mat['q']
        self.lx, self.ly = 1., 1. # length of the unit cell
        self.nelx, self.nely = self.nelx,self.nely
        self.dx, self.dy = self.lx / self.nelx, self.ly / self.nely # length of the element
        self.lambda_val = Mat['E']* Mat['nu'] /((1 + Mat['nu']) * (1 - 2 * Mat['nu'])) # plain strain  model
        self.mu_val = Mat['E']/ (2 * (1 + self.nu))
        self.lambda_val = 2*self.lambda_val*self.mu_val/(self.lambda_val+2*self.mu_val) # plain-stress model
        self.phi = 90. # degrees
        # they are 8x8 matrices, may not cost too much memory
        self.keLambda, self.keMu, self.feLambda, self.feMu, self.feAlpha = utils.elementMatVec(self.dx/2, self.dy/2, self.phi)

        self.nel = self.nelx*self.nely # total number of elements
        edofMat_full = utils.edofMat_func(self.nelx, self.nely)
        dofVector = utils.applied_pbcs(self.nelx,self.nely)
        self.edofMat = dofVector.at[edofMat_full].get().astype(int)
        nnP = (self.nelx)*(self.nely) # number of nodes on periodic boundary(elimate rigth and botton nodes)
        self.ndof = 2*nnP # # Total number of degrees of freedom

    




        # penal = q # for the penalty method
    @partial(jit, static_argnums=(0,))
    def K_idx_f_idx(self):
        """
        Args: None
        Returns:
            iK,jK,iF,jF: index of the global stiffness matrix and force vector
        """
        
        iK = jnp.kron(self.edofMat,jnp.ones((8,1))).T.flatten(order='F').astype(int)
        jK = jnp.kron(self.edofMat,jnp.ones((1,8))).T.flatten(order='F').astype(int)
        iF = jnp.tile(self.edofMat,4).T.flatten(order = 'F').astype(int)
        jF = jnp.tile(jnp.hstack((jnp.zeros(8),1*jnp.ones(8),2*jnp.ones(8),3*jnp.ones(8))),self.nel).astype(int)
        return iK,jK,iF,jF
    
    @partial(jit, static_argnums=(0,))
    def ther_mech_homo(self,x,penal):
        """
        Args: x, density based design variables
             penal, penalty factor to the desingis

        TODO: ........
        """
        # penalty varible, I prefer directly use penal
        # penal = penal
        if x.shape != (self.nelx*self.nely*self.q,1):
            raise ValueError('the shape of x is not correct, shoudl be n*1,now it is {}'.format(x.shape))


        
        @jit 
        def mat_interpolate_power(x):
            """ material interpolation
            Args:
                x: design variables (nel*q,1)
                ft: the logical operator for choosing the objective function

            Returns:
                lambda_x: powered interpolated lambda,(nelx,nely)
                mu_x: powered interpolated mu (nelx,nely)
                alpha_x: linear interpolated alpha,(nelx,nely)
            """
            x = x.reshape(self.nel,self.q,order='F')

            # power interpolation
            lambda_x = self.lambda_val[0] * x.at[:,0].get()**penal + self.lambda_val[1] * x.at[:,1].get()**penal + self.lambda_val[2] * x.at[:,2].get()**penal

            # linear interpolation, not used in this code
            # lambda_x = (self.lambda_val[0]) * x.at[:,0].get() + (self.lambda_val[1]) * x.at[:,1].get() + (self.lambda_val[2]) * x.at[:,2].get()
 
            mu_x = self.mu_val[0] * x.at[:,0].get()**penal + self.mu_val[1] * x.at[:,1].get()**penal + self.mu_val[2] * x.at[:,2].get()**penal 
            # mu_x = self.mu_val[0] * x.at[:,0].get() + self.mu_val[1] * x.at[:,1].get() + self.mu_val[2] * x.at[:,2].get()

            alpha_x = self.alpha_val[0]*x.at[:,0].get()**penal  + self.alpha_val[1]*x.at[:,1].get()**penal + self.alpha_val[2]*x.at[:,2].get()**penal 

            
            lambda_x = lambda_x.reshape(self.nely,self.nelx,order='F')
            mu_x = mu_x.reshape(self.nely,self.nelx,order='F')
            alpha_x = alpha_x.reshape(self.nely,self.nelx,order='F')
            return lambda_x, mu_x, alpha_x
        lambda_x, mu_x, alpha_x = mat_interpolate_power(x)
   
    #====================================================================
    
    ## % ========== code snippet 2.2  assemble stiffness matrix 'K' and loading 'F', function test======
        @jit
        def assemble_K_F(lambda_x,mu_x,alpha_x):
            """ assemble stiffness matrix 'K' and loading 'F'
            Args:
                lambda_x: powered interpolated lambda
                mu_x: powered interpolated mu
                alpha_x: linear interpolated alpha

            Returns:
                K: stiffness matrix
                F: loading vector
            """
            iK,jK,iF,jF = self.K_idx_f_idx()
            # assemble 'K'
            sK = self.keLambda.reshape(-1,1,order='F') @ lambda_x.reshape(-1,1,order='F').T + self.keMu.reshape(-1,1,order='F') @ mu_x.reshape(-1,1,order='F').T
            K = jnp.zeros((self.ndof,self.ndof))
            K = K.at[(iK,jK)].add(sK.flatten('F'))
            
            # assemble 'F'
            sF_1 = self.feLambda.reshape(-1,1,order='F') @ (lambda_x.reshape(-1,1,order='F').T) + self.feMu.reshape(-1,1,order='F') @ (mu_x.reshape(-1,1,order='F').T)
            sF_2 = self.feAlpha.reshape(-1,1,order='F')  @ ((alpha_x.reshape(-1,1,order='F') * (lambda_x.reshape(-1,1,order='F') + mu_x.reshape(-1,1,order='F'))).T)
            
            sF = jnp.vstack((sF_1, sF_2))
            F = jnp.zeros((self.ndof,4))
            F = F.at[(iF,jF)].add(sF.flatten('F'))
            return K, F
        K, F = assemble_K_F(lambda_x,mu_x,alpha_x)


        @jit
        def linear_solver_kuf(K,F):
            """ linear solver for KU = F
            Args:
                K: stiffness matrix
                F: loading vector
            Returns:
                chi: displacement cresponed to the mechanical loading test
                gamma: displacement cresponed to the thermal loading test
            """
            chi = jnp.zeros((self.ndof,4))
            chi = chi.at[2:self.ndof,:].set(jnp.linalg.solve(K.at[2:self.ndof,2:self.ndof].get(), F.at[2:self.ndof,:].get()))
            
            # chi = chi.at[2:self.ndof,:].set( \
            #     jax.scipy.linalg.solve\
            #     (K.at[2:self.ndof,2:self.ndof].get(), \
            #      F.at[2:self.ndof,:].get(),\
            #         assume_a = 'sym'))
            # chi = chi.at[2:self.ndof,:].set( \
            #     jax.scipy.linalg.solve\
            #     (K.at[2:self.ndof,2:self.ndof].get(), \
            #      F.at[2:self.ndof,:].get()))
            
            gamma = chi.at[:,3].get()
            return chi, gamma
        # print(f"\n solve this linear system use jax.scipy.linalg.solve with dense matrix....")
        chi, gamma = linear_solver_kuf(K,F)

        @jit
        def eps_ele_macro(alpha_x):
            """ compute macro mechanical strain, noted, elementary macro thermal strain test is desgin dependent
            Args:
                alpha_x: linear interpolated alpha
            Returns:
                chi0: elementary macro mechanical strain
                gamma0:  elementary macro thermal strain
            """

            # chi0 = zeros(nel, 8, 3);
            # macro mechanical strain
            chi0 = jnp.zeros((self.nel, 8, 3))
            chi0_e = jnp.zeros((8,4))
            # ke = keMu + keLambda; % Here the exact ratio does not matter, because
            ke = self.keMu + self.keLambda #{8x8}
            fe = self.feMu + self.feLambda #{8x3}
            fe = jnp.hstack((fe, 2*self.feAlpha)) #{8x4}

            # chi0_e([3 5:end],:) = ke([3 5:end],[3 5:end])\fe([3 5:end],:);
            ke_index = jnp.array([2,4,5,6,7])
            chi0_e = chi0_e.at[ke_index,:].set(jnp.linalg.solve(ke[ke_index,:][:,ke_index], fe[ke_index,:]))
            # chi0_e.at[[2,4,5,6,7],:].set(jnp.linalg.solve(ke[[2,4,6,7],:][:,[2,4,6,7]], fe[[2,4,6,7],:]))
            # gamma0 =alpha(:)*chi0_e(:,4)';
            # macro thermla strain 
            gamma0 = alpha_x.reshape(-1,1,order='F') @ chi0_e.at[:,3].get().reshape(1,-1,order='F') # {nel x 8}
            epsilon0_11 = jnp.kron(chi0_e[:,0].flatten('F'),jnp.ones((self.nel,1))) # {nel x 8}
            chi0 = chi0.at[:,:,0].set(epsilon0_11)
            epsilon0_22 = jnp.kron(chi0_e[:,1].flatten('F'),jnp.ones((self.nel,1))) # {nel x 8}
            chi0=chi0.at[:,:,1].set(epsilon0_22)
            epsilon0_33 = jnp.kron(chi0_e[:,2].flatten('F'),jnp.ones((self.nel,1))) # {nel x 8}
            chi0=chi0.at[:,:,2].set(epsilon0_33)
            return chi0,gamma0
        chi0, gamma0 = eps_ele_macro(alpha_x)
    
        @jit
        def average_homo(lambda_x,mu_x,\
                        chi0,gamma0,\
                        chi,gamma):
            """ 
            Args: lambda_x, mu_x: (design depedent material proepteis)
                    chi0,gamma0: test elemetary strain(gamma0 is design dependent, chi0 is not)
                    chi,gamma: pbcs solution (both are design depedent)
            Returns:
                    CH: homogenize elstiscity
                    beta_H: thermal stress tensor
                
            """

            CH = jnp.zeros((3,3)) # homogenized elasticity tensor
            beta_H = jnp.zeros((3,1)) # homogenized plasticity tensor
            cellVolume = self.lx*self.ly

            for ii in range(3):
                for jj in range(3):
                    vi = chi0.at[:,:,ii].get()- chi.at[(self.edofMat+(ii)*self.ndof)%self.ndof,\
                                            ((self.edofMat+(ii)*self.ndof)//self.ndof)].get()
                    vj = chi0.at[:,:,jj].get()- chi.at[(self.edofMat+(jj)*self.ndof)%self.ndof,\
                                            ((self.edofMat+(jj)*self.ndof)//self.ndof)].get()
                    sumLambda = jnp.matmul(vi,self.keLambda)*vj
                    sumMu = jnp.matmul(vi,self.keMu)*vj

                    # logs, 5th July, reshape order F is requried
                    sumLambda = jnp.sum(sumLambda,1).reshape(self.nely,self.nelx,order='F')
                    sumMu = jnp.sum(sumMu,1).reshape(self.nely,self.nelx,order='F')
                    #  CH(i,j) = 1/cellVolume*sum(sum(lambda.*sumLambda + mu.*sumMu));
                    CH_v = 1/cellVolume*jnp.sum(jnp.sum(lambda_x*sumLambda + mu_x*sumMu))
                    CH = CH.at[(ii,jj)].set(CH_v)

                    # homogenized on theta
                    vt = gamma0 - gamma.at[self.edofMat].get()
                    vti = vi
                    sumLambda_vt  = jnp.matmul(vt,self.keLambda)*vi
                    sumMu_vt = jnp.matmul(vt,self.keMu)*vi

                    sumLambda_vt = jnp.sum(sumLambda_vt,1).reshape(self.nely,self.nelx,order='F')
                    sumMu_vt = jnp.sum(sumMu_vt,1).reshape(self.nely,self.nelx,order='F')

                    betaH_v = 1/cellVolume*jnp.sum(jnp.sum(lambda_x*sumLambda_vt + mu_x*sumMu_vt))
                    beta_H = beta_H.at[(ii,0)].set(betaH_v)
        
            return CH, beta_H
        CH,beta_H = average_homo(lambda_x,mu_x, chi0,gamma0,chi,gamma)
        
        return CH,beta_H # for debug



def test_Homogenize():
    import jax
    import jax.numpy as jnp
    from jax import random
    import numpy as np
    from jax import jit, value_and_grad
    import utils
    import utils_OPT
    import numpy as np
    from micro_homo import Homogenize
    from functools import partial

    """ test conditional 'homo', 
    Args: 
         ft: logical operator, if 'True', selecte homotenized cte as target, if 'False' selected elastic matrix
    Return:
        obj: objective function (depednent on condition 'ft')

    """
    E = jnp.array([1.,1.,1.e-3])
    nu = jnp.array([0.3,0.3,0.3])
    cte = jnp.array([ 1,10,1.e-3])
    v= jnp.array([0.2,0.2,0.6]) # volume fraction constraint for each phase

    nelx = 41
    nely = 41;
    penal = 3;
    q=3;
    Mesh = {'nelx': nelx, 'nely': nely}
    Mat = {'E': E, 'nu': nu, 'cte': cte, 'q': q}
    forward_pre = Homogenize(Mesh,Mat)

    x_inputs = np.load('x_negatice_cte.npy')
    ch_,beta_ = forward_pre.ther_mech_homo(x_inputs.reshape(-1,1,order = 'F'),penal)
    cte_ = jnp.linalg.solve(ch_,beta_)
    cte_str = np.array2string(cte_, separator=', ')
    ch_str = np.array2string(ch_, separator=', ')
    print(cte_)
    print(ch_)
    # save cte and CH to file 
    filename = 'save_data.txt'
    with open(filename,'w') as file:
        file.write("cte =\n")
        file.write(cte_str)
        file.write("\n ch =\n")
        file.write(ch_str)

    



# test_Homogenize()

    
        
        