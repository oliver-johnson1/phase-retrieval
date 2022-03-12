import numpy as np

def phase_retrieval(mag, mask=None, threshhold=0, beta=0.8, 
                    max_steps=200, mode='hybrid', tol = 1e-5):
    """
    Code adapted from hhu-machine-learning/phase-retrieval-cgan
    
    This function implements Gerchberg-Saxton algorithm, HIO and OSS.
    Inputs:
    mag: magnitude of FFT2 of image
    mask: mask of image (for oversampling). If not specified it is set to
    an array of 1's the same size as the image.
    theshold: threshold paramter for HIO and OSS methods
    beta: beta value for HIO and OSS methods
    max_steps: the maximum number of steps to take
    mode: the mode used in the alternating projection solver
    tol: when change in error is below tol we stop iterating
    """
    
    assert beta > 0, 'step size must be a positive number'
    assert max_steps > 0, 'steps must be a positive number'
    assert mode == 'input-output' or mode == 'output-output'\
        or mode == 'hybrid' or mode == 'gs' or mode == 'oss',\
    'mode must be \'input-output\', \'output-output\' or \'hybrid\''
    
    if mask is None:
        mask = np.ones(mag.shape)

    assert mag.shape == mask.shape, 'mask and mag must have same shape'

    # Sample random phase and initialize image x
    init_phi = 2*np.pi*np.random.rand(*mag.shape)
    y_hat = mag*np.exp(1j * init_phi)
    x = np.zeros(mag.shape)

    x_prev = None

    # Make Gaussian kernel for OSS
    if mode == 'oss':
        k = np.zeros(x.shape)
        for i in range(0,k.shape[0]):
            for j in range(0,k.shape[1]):
                k[i,j] = (i-k.shape[0]//2)**2 + (j-k.shape[1]//2)**2
        k = k/np.max(k)
        alpha = np.arange(max_steps//10 + max_steps + 1,1+max_steps//10,-1)
        alphas = []
        m = mag.shape[0]*2
        maxx = max_steps//10
        j = 0
        for i in range(max_steps):
            alphas.append(max(m,maxx))
            if j==maxx:
                m = m - mag.shape[0]*2//10
                j=0
            j+=1
        alpha = alpha
        alpha = (alpha*0.5*mag.shape[0]*2).astype(int)
        alpha = np.array(alphas)/4

    converged = False
    i = 0
    while i<max_steps and not converged:        
        # Inverse fourier transform
        y = np.real(np.fft.ifft2(y_hat))

        # Previous iterate
        if x_prev is None:
            x_prev = y
        else:
            x_prev = x 

        # Updates for elements that satisfy object domain constraints
        if mode == "output-output" or mode == "hybrid" or mode == 'gs' or mode == 'oss':
            x = y
            zdd = y

        # Find elements that violate object domain constraints or are not masked
        indices = np.logical_or(np.logical_and(y < threshhold, mask), np.logical_not(mask))

        # Compute signal domain error (for all expect gs)
        if mode != 'gs':
            error = np.sum(x[indices] ** 2)

        # Updates for elements that violate object domain constraints
        if mode == "hybrid" or mode == "input-output":
            x[indices] = x_prev[indices] - beta*y[indices] 
        elif mode == "output-output":
            x[indices] = y[indices] - beta*y[indices]
        elif mode == 'gs':
            x[indices] = 0
        elif mode == 'oss':
            zdd[indices] = x_prev[indices]-beta*y[indices]
            Fdd = np.fft.fftshift(np.fft.fft2(zdd))
            W = np.exp(-0.5*((k)/alpha[i]**2))
            W = W/np.max(W)
            x[indices] = np.real(np.fft.ifft2(np.fft.ifftshift(Fdd*W))[indices])
        
        # Fourier Transform
        x_hat = np.fft.fft2(x)
        
        # compute error for GS case
        if mode == 'gs':
            error = np.mean((np.abs(x_hat) - mag) ** 2)
        
        if error < tol:
            converged = True
        
        # Satisfy Fourier Domain Constraints
        y_hat = mag*np.exp(1j*np.angle(x_hat))
        i += 1 
    return x