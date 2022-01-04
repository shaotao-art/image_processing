import numpy as np
from matplotlib import pyplot as plt
from skimage import io ,color
from skimage.transform import resize

def read_img(path):
    img=io.imread(path)
    img=img.astype(np.float64)/255

    return img


def display_img(img):
    io.imshow(img)


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad a 3d image.

    Args:
        image: numpy array of shape (H, W, 3). 
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).
    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    out=np.pad(image,pad_width=[(pad_width,pad_width),(pad_height,pad_height)],mode='constant')

    return out

def convlution(image, kernel):
    """ implementation of convolution filter.
    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.


    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.
    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    if len(image.shape)==3:
        print(len(image.shape))
        Hi, Wi,_ = image.shape
        Hk, Wk,_= kernel.shape
    if len(image.shape)==2:
        Hi, Wi = image.shape
        Hk, Wk = kernel.shape

    out = np.zeros((Hi, Wi))

    # pad image
    image = zero_pad(image, Wk//2, Hk//2)

    print(image.shape)
    for m in range(Hi):
        for n in range(Wi):
                out[m, n] =  np.sum(image[m: m+Hk, n: n+Wk] * kernel)


    return out

def correlation(image,kernel):
    """correlation between 3d image and kernel

    Args:
        images: numpy array of shape (Hi,Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.
    Returns:
        out: numpy array of shape (Hi, Wi).

    """
    kernel=np.flip(kernel)
    kernel=(kernel-np.mean(kernel))/np.std(kernel)
    out=convlution(image,kernel)

    return out


def normalized_correlation(image,kernel):
    """before correaltion, narmalize the template and the accord area of img

    Args:
        images: numpy array of shape (Hi,Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.
    Returns:
        out: numpy array of shape (Hi, Wi).

    """
    Hi, Wi,_ = image.shape
    Hk, Wk,_= kernel.shape

    # normalize the template
    kernel=(kernel-np.mean(kernel))/np.std(kernel)

    out = np.zeros((Hi, Wi))

    # pad image
    image = zero_pad(image, Wk//2, Hk//2)

    for m in range(Hi):
        for n in range(Wi):
            # noramlize the accord area of img 
            temp=image[m: m+Hk, n: n+Wk,:]
            accord_area=(temp-np.mean(temp))/np.std(temp)

            out[m, n] =  np.sum(accord_area * kernel)
    
    return out
