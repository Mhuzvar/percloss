import numpy as np
import matplotlib.pyplot as plt
import scipy

def Acurve(N=63, fs=44100, mode=0):
    """
    magnitude response usable in frequency sampling method for FIR filter design.
    arguments:
        nvals       ...output vector length
        fs          ...sampling frequency
        mode        ...response definition
                            0   ..ITU-R BS.468-4
    outputs:
        vector of magnitudes
    """
    
    """
    1. get magnitude response (interp from available data)
    3. calculate filter coeffs 0..M using formula 7.30 from book
    4. calculate the rest of the coefficients using formula 7.31 (from the book)
    """
    
    match mode:
        case 0:
            fd = (31.5, 63, 100, 200, 400, 800, 1000, 2000, 3150, 4000, 5000,
                  6300, 7100, 8000, 9000, 10000, 12500, 14000, 16000, 20000, 31500)
            rd = (-29.9, -23.9, -19.8, -13.8, -7.8, -1.9, 0, 5.6, 9, 10.5, 11.7,
                  12.2, 12, 11.4, 10.1, 8.1, 0, -5.3, -11.7, -22.2, -42.7)
        case _:
            raise Exception(f"Wrong mode '{mode}' provided!")
    M = (N-1)//2    
    
    fn = np.linspace(0,fs//2,M+1)
    fnlog = np.log10(fn[1:])
    
    fdlog = np.log10(np.asarray(fd))
    fdlog = np.insert(fdlog, 0, fdlog[0]-30)
    rdmod = np.insert(np.asarray(rd), 0, rd[0]-600)
    rnlog = np.interp(fnlog, fdlog, rdmod)

    H = np.insert(np.power(10, rnlog/20), 0, 0)

    h = np.zeros(N)

    k_ = np.linspace(1, M, M, endpoint=True)
    for n in range(M+1):
        h[n] = (1/(N))*(H[0]+2*np.sum(H[1:M+1]*np.cos((2*np.pi*k_*(n-M))/N)))
        h[(2*M)-n] = h[n]
    
    
    if False:
        plt.semilogx(fd, rd)
        plt.xlim([20, 2e4])
        plt.ylim([-50,20])
        plt.grid()
        plt.semilogx(fn[1:],rnlog)
        plt.xlim([20, 2e4])
        plt.ylim([-50,20])
        plt.grid()
        plt.show()
        plt.plot(fn,H)
        plt.xlim([20, 2e4])
        plt.grid()
        plt.show()
        plt.plot(h)
        plt.show()
        plt.semilogx(fd, rd)
        plt.semilogx(fn,10*np.log10(np.abs(np.fft.fft(h))**2)[0:M+1])
        plt.title(f"N={N}")
        plt.xlim([20, 2e4])
        plt.show()

    return h


if __name__=="__main__":
    for i in range(8, 12):
        Acurve(N=(2**i)-1)
