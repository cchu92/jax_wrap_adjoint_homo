import numpy as np
import matplotlib.pyplot as plt

path_load_npy = 'D:/Dropbox/Dropbox/Fraunhofer_Cluster/JAX_GPU/code/ProgMat_v1/collec_data/c3_chirality/'
path_save_img = 'D:/Dropbox/Dropbox/Fraunhofer_Cluster/JAX_GPU/code/ProgMat_v1/collec_image/c3_chirality/binary/'
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


nelx,nely=90,90
# nelx,nely=60,60
nb_data = 800
for i in range(nb_data):
    x = np.load(path_load_npy+'rve_opt_'+str(i)+'.npy',allow_pickle=True)

    # binary images
    mask = x == np.max(x, axis=1)[:, None]
    x = mask.astype(int)


    I = make_bitmap2(3, nelx, nely, x)
    # clear the plot
    plt.clf()
    # axis off
    plt.axis('off')
    plt.imshow(I)

    # save image
    # plt.savefig(path_save_img_non_binary+'/rve_opt_nonbinary'+str(i)+'.png')
    plt.savefig(path_save_img+'rve_opt'+str(i)+'.png')

    # print(i)
