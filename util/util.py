from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from methods.solver import phase_retrieval

def calculate_psnr(out, ref, width):
    s = ref.shape
    rot = np.rot90(out,2)
    if width is None:
        out1 = out
        rot1 = rot
    else:
        w1, w2 = width
        out1 = out[w1:-w1,w2:-w2]
        rot1 = rot[w1:-w1,w2:-w2]

    psnr = peak_signal_noise_ratio(out1, ref)
    psnr_rot = peak_signal_noise_ratio(rot1, ref)
    if(psnr_rot > psnr):
        return psnr_rot, rot1, rot
    else:
        return psnr, out1, out

def reconstructImage(image, mag, padding_key, method, steps=2000):
    assert method == 'gs' or method == 'hio' or method == 'oss',\
    'method must be \'gs\', \'hio\' or \'oss\''
    if method == 'gs':
        mode = 'gs'
        beta = 1
    elif method == 'hio':
        mode = 'hybrid'
        beta = 0.8
    elif method == 'oss':
        mode = 'oss'
        beta = 0.8
    if padding_key is None:
        width = None
    else:
        w1 = (padding_key.shape[0]-image.shape[0])//2
        w2 = (padding_key.shape[1]-image.shape[1])//2
        width = (w1,w2)

    image_recon_full = phase_retrieval(mag=mag, mask=padding_key, mode=mode,beta=beta, max_steps=steps)
    img_recon_clipped = np.clip(image_recon_full, a_min=0, a_max=1.)
    psnr, image_recon, _  = calculate_psnr(img_recon_clipped, image, width)
    return image_recon, psnr

def padImage(image,width=None):
    if width is None:
        w1 = image.shape[0]//2
        w2 = image.shape[1]//2
    else:
        w1,w2 = width
    padding_key = np.pad(np.ones(image.shape),((w1,w1),(w2,w2)))
    image_padded = np.pad(image,((w1,w1),(w2,w2)))

    return image_padded, padding_key