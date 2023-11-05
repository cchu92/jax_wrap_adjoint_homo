import numpy as np
import matplotlib.pyplot as plt


def make_bitmap2(p, nx, ny, alpha):
    # material color, [red;blue;white;green;black]
    alpha = alpha.reshape(nx * ny, p, order='F')
    color = np.array([[1, 0, 0], [0, 0, 1],[1, 1, 1], [0, 1, 0], [0, 0, 0]])
    # '1,0,0 red' '0,0,1 blue' '1,1,1 white ' '0,1,0' green,'0,0,0' black
    I = np.zeros((nx * ny, 3))
    for j in range(p):
        I[:, :3] += alpha[:, j, None] * color[j, :p]
    I = I.reshape(ny, nx, 3, order='F')
    return I

def mirror_x(x,nelx,nely,q):
    def mirror_xp(xp,nelx,nely):
        xp = xp.reshape(nelx,nely,order='F')
        xp_periodic = np.hstack((xp,np.fliplr(xp)))
        xp_periodic = np.vstack((xp_periodic,np.flipud(xp_periodic)))
        return xp_periodic.flatten('F')

    x = x.reshape(nelx*nely,q,order='F')
    mirror_x = np.zeros((nelx*nely*4,q))
    for ii in range(q):
        mirror_x[:,ii] = mirror_xp(x[:,ii],nelx,nely)
    return mirror_x

def heaviside_projection(x, beta, eta=None):
    eta = 0.;  # default value
    return 1 / (1 + np.exp(-beta * (x - eta)))
def brute_force_bibary_convert(x):
    mask = x == np.max(x, axis=1)[:, None]
    return mask.astype(int)


nelx,nely=80,80
nelx,nely=90,90
# nelx,nely=40,40

win_path = '../../collec_data/c4/chirality'
# win_path = '../../collec_data/c4/sym'
for ii in range(32):
    plt.subplot(8,4,ii+1)
    file = win_path+'/rve_opt'+'_'+str(ii)+'.npy'
    x_perioic = np.load(file,allow_pickle=True)
    print(x_perioic.shape)
    I = make_bitmap2(3, nelx, nely, x_perioic)
    plt.imshow(I)
plt.show()


# use image display
# new_dimension = 360
# import cv2
# for ii in range(4):
#     plt.subplot(1,4,ii+1)
#     file = win_path+'/rve_opt'+'_'+str(ii)+'.npy'
#     x = np.load(file,allow_pickle=True).reshape(90,90,3,order='F')
#     x = cv2.resize(x, dsize=(new_dimension, new_dimension), interpolation=cv2.INTER_CUBIC)
#     plt.imshow(x)
#     plt.axis('off')
# plt.show()
