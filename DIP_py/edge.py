import numpy as np
from basic_funcs import convlution


def guassian_kernel(sigma,size):
    """return a size*size guassian kernel with sigma

    Args:
        sigma:
        size:
    return:
        guassian kernel
     
    """
    k=size//2

    out=np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            out[i,j]=1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-k)**2)/2*sigma**2)
    
    return out



def non_maximum_suppression(mag,theta):
    """make non maximum suppression

    Args:
        mag: magnitude of pic sqrt(dx**2+dy**2)
        theta: theta map of pic value is range from 0-360
    return:
        out: result after the operation
    
    """

    H, W = mag.shape

    theta = np.floor((theta + 22.5) / 45) * 45

    out=np.zeros((H-1,W-1))
    for i in range(1,H-1):
        for j in range(1,W-1):
            if theta[i,j]==0 or theta[i,j]==180:
                if mag[i,j]>mag[i-1,j] and mag[i,j]>mag[i+1,j]:
                    out[i,j]=mag[i,j]
                else:
                    out[i,j]=0
            
            if theta[i,j]==45 or theta[i,j]==225:
                if mag[i,j]>mag[i+1,j+1] and mag[i,j]>mag[i-1,j-1]:
                    out[i,j]=mag[i,j]
                else:
                    out[i,j]=0
            if theta[i,j]==90 or theta[i,j]==270:
                if mag[i,j]>mag[i,j+1] and mag[i,j]>mag[i,j-1]:
                    out[i,j]=mag[i,j]
                else:
                    out[i,j]=0
            
            if theta[i,j]==135 or theta[i,j]==315:
                if mag[i,j]>mag[i-1,j+1] and mag[i,j]>mag[i+1,j-1]:
                    out[i,j]=mag[i,j]
                else:
                    out[i,j]=0
    return out


def double_threshold(img,high,low):
    """
    
    """
    strong_edge=img>high
    weak_edge=(low<=img)& (img<=high)
    return strong_edge,weak_edge


def edge_Linking(img,strong_edge,weak_edge):
    """
    img
    
    
    """
    lst=np.argwhere(strong_edge==True).tolist()
    out=np.zeros_like(img)
    out[strong_edge]=1
    H,W=out.shape
 

    while lst:
        m,n=lst.pop(0)
        
        neib_8=[]

        for i in range(-1,2):
            for j in range(-1,2):
                if 0<=m+i<=H-1 and 0<=n+i<=W-1:
                    if i==0 and j==0:
                        continue
                    neib_8.append([m+i,n+j])
       
        
        for i in range(len(neib_8)):
            if weak_edge[neib_8[i][0],neib_8[i][1]]==True:

                ############# 将变为strong的edge weak 标志置0  ##############
                weak_edge[neib_8[i][0],neib_8[i][1]]=False

                # print(neib_8[i])
                # weak 被分类为 strong 后要进行同样的操作
                lst.append(neib_8[i])
                # print(len(lst))
                out[neib_8[i][0],neib_8[i][1]]=1

    return out   

def find_edge(img,high,low,sigma=1,size=5):
    g_f_5=guassian_kernel(sigma,size)
    img_filted=convlution(img,g_f_5)
    canny_h=np.array([-0.5,0,0.5])
    canny_h=canny_h.reshape((1,-1))
    canny_v=np.array([[0.5],[0],[-0.5]])

    dx=convlution(img_filted,canny_h)
    dy=convlution(img_filted,canny_v)

    mag=np.sqrt(dx**2+dy**2)
    theta=np.arctan2(dx,dy)*180/np.pi%360

    img_supp=non_maximum_suppression(mag,theta)


    s,w=double_threshold(img_supp,high,low)

    res=edge_Linking(img_supp,s,w)

    return res

