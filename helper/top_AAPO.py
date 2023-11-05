import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from jax import jit, value_and_grad
import helper.utils as utils
import helper.utils_OPT as utils_OPT 
import numpy as np
from helper.micro_homo import Homogenize
# from micro_homo_sparse import Homogenize
from functools import partial


class top_inner_a_b_phases:

    """ inner topology optimization of only 'a' and 'b' ativated phase, the reaming phases are constant
    """
    # load pks

    # def __init__(self,forward_pred,Opt_var):
    def __init__(self,forward_pred):
        """
        Args:
            forward_pred: forward predicition probelm 
        """

        nelx,nely = forward_pred.Mesh['nelx'],forward_pred.Mesh['nely']
        self.E,self.nu,self.cte,self.q = forward_pred.Mat['E'],forward_pred.Mat['nu'],forward_pred.Mat['cte'],forward_pred.Mat['q']
        self.nel = nelx*nely
        self.nelx = nelx
        self.nely = nely
        self.forward_pred = forward_pred
        self.objectivHandle = self.obj_and_selected_grad
    @partial(jit,static_argnums=(0,)) 
    def J_active_fns(self,x,ft=None,out_iter=None,penal=None,l_power=None):
        """with applied AAPO, the objective fuction is related to 'ft', 'out_iter'
        thus, the gradient is also related to 'ft', 'out_iter', and with additial 'a'(represent active phases)
        Args:
            x: design variables
            ft: 'True' or 'False', True: minimize cte, False: minimize pr
            out_iter: the iteration of outer loop, relate to function 'fp'
            penal: penalization factor (the default value is 3.)
        Returns:
            J: conditional objective function, ft or fp
        """
        # homogenization process 
        CH,betaH = self.forward_pred.ther_mech_homo(x.reshape(-1,1,order='F'),penal)

        @jax.jit
        def ft_wrap(CH,beta_H):
            '''' 
            minimize the cte 
            Args: 
                CH: homogenize elastic tenor
                beta_H: homogenized thermal stress tensor
            Returns:
                obj_fp: sum of cte 
            ''' 
            cte = jnp.linalg.solve(CH,beta_H)
            obj_ft = jnp.sum((cte.at[0:2].get()))
            # obj = jnp.sum(cte)
            return obj_ft
        # @partial(jit,static_argnums=(0,)) 
        # @jax.jit
        def fp_wrap(CH,out_iter,l_power=None):

            ''' minimize pr, as function of out_iter
            '''
            l = 0.7
            l = 0.8
            l = 0.6
            l = 0.75
            obj_fp = (CH.at[0,1].get() + CH.at[1,0].get()) /2- l_power**(out_iter+1)*(CH.at[0,0].get()+CH.at[1,1].get())
            # obj_fp = (CH.at[0,1].get()) - l**(out_iter+1)*(CH.at[0,0].get()+CH.at[1,1].get())
            return obj_fp  
        

        # conditionally choose 'ft' and 'fp'
        J = jax.lax.cond(ft, lambda x: ft_wrap(CH,betaH),lambda x: fp_wrap(CH,out_iter,l_power),None)
        return J
    
    @partial(jit,static_argnums=0)
    def obj_and_selected_grad(self,x,ft=None,out_iter=None,penal=None,a=None,l_power=None):
        """ with input design variable 'x', calculate the conditional objective and gradient w.r.t seltected phases 
        Args:
            x: inputs design variable
            ft: logic True for thermal obj, False for mechanical obj
            out_iter: outer iteration
            penal: penalization factor
            a: selected phase
        Returns:
            J: conditional objective value 
            dJ: conditional gradient w.r.t selected phases 
        """
        indices_ = (jnp.arange(0,self.nelx*self.nely) + a*(self.nelx*self.nely)).astype(int)
        # print(f'\n outer iteratin is {out_iter}')
        
        J = self.J_active_fns(x,ft,out_iter,penal,l_power)
        dJ = jnp.take(jax.grad(lambda x: self.J_active_fns(x,ft,out_iter,penal,l_power))(x),indices_)
        return J,dJ

   
    def TO_inner_update_x_a(self,x,Iter_var_dynamic):

        """ update x_a, x_b = x_ab - x_a
        Note for me: a, b are fixed varaible in this loop
        thus, 'a','b','l','u' 'x_ab', are fixed 
        'x_a' 'x_b' are updated acccordingly
        Args:
            Iter_var_dynamic = {'iter_out':out loop iteration,'ft':logic operators,'a': selcted a phase, 'b': selected b phase}
        """
        # TODO: MMA optmization algorigthm
        x = jnp.array(x)
        out_iter = Iter_var_dynamic['out_iter']
        ft = Iter_var_dynamic['ft']
        a = Iter_var_dynamic['a']
        b = Iter_var_dynamic['b']
        rmin = Iter_var_dynamic['rmin']
        penal = Iter_var_dynamic['penal']
        l_power = Iter_var_dynamic['l']

        
        itermax_ft = Iter_var_dynamic['itermax_ft']
        
        a_selected = a
        b_selected = b
        self.v = Iter_var_dynamic['volfrac']
        H,Hs = utils_OPT.computeFilter(self.nelx,self.nely, rmin)
        self.objectivHandle = self.obj_and_selected_grad # return (obj,dobj)
        x = x.reshape(self.nel,self.q, order ='F')
        
        self.x_ab = jnp.sum(x.at[:,[a_selected,b_selected]].get(),axis=1) # it is constant
        # lower and up bounds, upp bound on 'x_ab'
        r = np.ones(self.nelx*self.nely)
        move = 0.05
        # move = 0.05
        # move = 0.08
        for k in range(self.q):
            if k != a_selected and k != b_selected:
                r = r - x[:,k]
        l = np.maximum(0,x.at[:,a_selected].get()-move) 
        u = np.minimum(r,x.at[:,a_selected].get()+move)
        loop_inner = 0 
        # itermax_ft =8; # it is dynamic now 
        itermax_fp =1;
        # itermax_fp =2;
        # iter_max_in = jax.lax.cond(ft,lambda x: itermax_ft,lambda x: itermax_fp,x)
        iter_max_in = jax.lax.cond(ft, lambda _: itermax_ft, lambda _: itermax_fp, operand=None)
        # print(f'\n ft is {ft}, iter_max_in is {iter_max_in}')
        while loop_inner < iter_max_in:
            J,dJdxa = self.objectivHandle(x.reshape(-1,1,order ='F'),ft,out_iter,penal,a_selected,l_power)
            # FILTER GRADIENT 

            dva = np.ones(self.nelx*self.nely)
            filter_method = 1;
            dJdxa,dva = utils_OPT.applySensitivityFilter(filter_method, # fileter method
                                                         np.array(dJdxa),# sensitivity w.r.t objective function
                                                         np.array(x.at[:,a_selected].get()), # design variable
                                                         dva, # sensitivity w.r.t volume constraint
                                                         H,Hs)


            x_a_update,change = utils_OPT.OC(filter_method, # filter method
                                             x.at[:,a_selected].get(),
                                             dJdxa, # sensitivity w.r.t objective function
                                             dva, # sensitivity w.r.t volume constraint
                                             l,u, # lower and upper bounds
                                             self.v[a_selected], # volume constraint
                                             H,Hs)
            
            if filter_method  ==2:
                x_a_update = (H @ x_a_update.reshape(-1,1,order='F')/Hs).flatten('F')
            # x_a_update is numpy array
            x_b_update = r - x_a_update
            x = x.at[:,a_selected].set(jnp.array(x_a_update))
            x = x.at[:,b_selected].set(jnp.array(x_b_update))
            loop_inner  = loop_inner+1
            
        return x,change,J

 





            




      
