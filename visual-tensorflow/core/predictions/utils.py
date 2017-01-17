# Utils for handling predictions
import numpy as np

def smooth(x, window_len=48, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len-1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w=eval('np.' + window + '(window_len)')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]

def DTWDistance(s1, s2, w=None):
    '''
    Calculates dynamic time warping Euclidean distance between two
    sequences. Option to enforce locality constraint for window w.
    '''
    DTW={}

    if w:
        w = max(w, abs(len(s1)-len(s2)))

        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i-w), min(len(s2), i+w)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        else:
            for j in range(len(s2)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])