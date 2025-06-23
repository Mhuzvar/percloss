import numpy as np
import matplotlib.pyplot as plt
import scipy

def Acurve(nvals=1024, fs=44100, mode=0):
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
    match mode:
        case 0:
            fd = (31.5, 63, 100, 200, 400, 800, 1000, 2000, 3150, 4000, 5000,
                  6300, 7100, 8000, 9000, 10000, 12500, 14000, 16000, 20000, 31500)
            rd = (-29.9, -23.9, -19.8, -13.8, -7.8, -1.9, 0, 5.6, 9, 10.5, 11.7,
                  12.2, 12, 11.4, 10.1, 8.1, 0, -5.3, -11.7, -22.2, -42.7)
        case _:
            raise Exception(f"Wrong mode '{mode}' provided!")
    fn = np.linspace(0,22050,22050)
    rn = np.interp(fn, np.insert(np.asarray(fd), 0, 0), np.insert(np.asarray(rd),0 ,-60))
    plt.semilogx(fd, rd)
    plt.grid()
    plt.show()
    plt.semilogx(fn,rn)
    plt.show()


if __name__=="__main__":
    Acurve()
