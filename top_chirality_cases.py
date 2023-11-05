

# only with chirality intial guess test

# impor pks

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from jax import jit, value_and_grad
from functools import partial
from matplotlib import pyplot as plt
import time


# import user defined functions
import helper.utils as utils
import helper.utils_OPT as utils_OPT
import numpy as np
from helper.micro_homo import Homogenize
from helper.top_AAPO import top_inner_a_b_phases



# CANDIATE MATS VECTORS
E = jnp.array([1.,1.,1.e-3])
nu = jnp.array([0.3,0.3,0.3])
cte = jnp.array([ 1,10,1.e-3])
v= jnp.array([0.2,0.1,0.7]) # volume fraction constraint for each phase

# mesh size
nelx,nely,q = 90,90,int(3)
Mesh = {'nelx': nelx, 'nely': nely}
Mat = {'E': E, 'nu': nu, 'cte': cte, 'q': q}

forward_pre = Homogenize(Mesh,Mat)
J_dJ_AAPO = top_inner_a_b_phases(forward_pre)


def opi_micro_strucure(penal,v,type_initial_guess,rmin,l_power,itermax_ft):
    ''' top_aapo w.r.t differet hyperparamemtes sets
    Args:
        penal: penalty factor
        type_initial_guess: 1,2,3 intial guess
        rmin: filter radius 
        l_power: C11  + l^l_power *C22
        itermax_ft: max iteration steps to minize cte 
    returns:
        x: the final topology optimizaiotn 
    '''
    # intial guess 
    # rmin =rmin
    tol_f = 0.07
    x = utils_OPT.inti_guess(nelx,nely,np.array(v),q,type=type_initial_guess)
    iter_out = 0
    iter_out_max = 30
    change = 1 
    while iter_out<iter_out_max:
        iter_out +=1
        for a in range(q):
            for b in range(a+1,q):
                if b == q-1: # volid is the last phase, now we have volid phase, minimize mechancial problem
                    ft = False; # minimize mechancial problem
                else: 
                    ft = True # both seltected phase are solid,  minimize cte
                Iter_var_dynamic = {'a':a, 'b':b,'rmin':rmin,'ft':ft,'out_iter':iter_out,'penal':penal,'volfrac':v,'l':l_power,'itermax_ft':itermax_ft}

                x,change,J = J_dJ_AAPO.TO_inner_update_x_a(x,Iter_var_dynamic)[0:3]
                if (change<tol_f or (iter_out-1)%100 ==0)  and rmin>1.2:
                    tol_f = 0.99*tol_f
                    rmin = 0.99*rmin
                    rmin = 0.99*rmin
        print('iter',iter_out)
    return x


# %% tuning hyperparameters to generate diverse designs

initial_guess_list = [int(3)]
l_list = [0.8]# 0.7 is not enough, make it 0.74
v_list = [jnp.array([0.2,0.2,0.6])]
rmin_list = [6]
penalt_list = [3.]
itermax_ft_list = [int(4),int(5),int(6),int(7)]
# itermax_ft_list = [int(4)]


coutnpy = 0
path_to_save = '/s/dep/sms/w/FHG_Programmierbare_Materialien/PlainDutchWeave/chuc_2023_04/JAX_GPU/code/ProgMat_v1/collec_data/c4/chirality/'

for penalt in penalt_list:
    for l_power in l_list:
        for itermax_ft in itermax_ft_list:
            x = opi_micro_strucure(penal = penalt,
                                v=jnp.array([0.1,0.2,0.7]),
                                type_initial_guess = int(3),
                                rmin = 3.,l_power = l_power,
                                itermax_ft = itermax_ft)
            file = 'rve_opt_' + str(coutnpy) + '.npy'
            np.save(path_to_save+file,x)
            coutnpy +=1
        



