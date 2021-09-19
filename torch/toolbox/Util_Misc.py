import numpy as np

#...!...!..................
def expand_dash_list(kL=['']): 
    # expand list if '-' are present
    kkL=[]
    for x in kL:
        #print('aa',x, '-' not in x)
        if '-' not in x:
            kkL.append(x) ; continue
        xL=x.split('-')
        #print('b',xL)
        for i in range(int(xL[0]),int(xL[1])+1):
            kkL.append(i)
    print('DEL:',kL,'  to ',kkL)
    return kkL

#...!...!..................
def mini_plotter(args):
    import matplotlib as mpl
    if args.noXterm:
        mpl.use('Agg')  # to plot w/o X-server
        print('Graphics disabled')
    else:
        mpl.use('TkAgg') 
        print('Graphics started, canvas will pop-out')
    import matplotlib.pyplot as plt
    return plt

#...!...!..................
def smoothF(x,window_len=20,window='hanning', verb=0):
    """smooth the data using a window with requested size.
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    assert x.ndim == 1
    assert x.size > window_len
    if window_len<3:   return x

    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    if verb: print('smooth Inp',x.shape,x.dtype,'window_len',window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    y=y[(window_len//2-1):-(window_len//2)]
        
    if verb: print('smooth Out',y.shape,y.dtype,window_len//2-1)
    y=y[:x.shape[0]] # hack to get the same out dim for odd window_len
    
    return y

