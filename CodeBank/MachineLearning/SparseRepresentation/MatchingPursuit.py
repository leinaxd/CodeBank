import numpy as np
import scipy as sp

class matchingPursuit:
    """
    Method:
        f(t) approx \hat{f}_N(t) = \sum_{n=1}^N a_n \cdot g_{\gamma_n} (t)
    Notation:
        Y                       the dataset (rows=samples, cols=features)
        x                       sparse representation with nNonZeroComponents
        x_gamma_n = a_n         NonZero entries of x
        D                       The codebook/Dictionary which compress de Y data
        g_gamma_n               gamma_n column of D
    Sources:
        https://en.wikipedia.org/wiki/Matching_pursuit?oldformat=true
        https://www.kaggle.com/code/danieleichenbaum/omp-final/edit
        https://sparse-plex.readthedocs.io/en/latest/book/pursuit/mp/algorithm.html
        https://sparse-plex.readthedocs.io/en/latest/book/pursuit/omp/algorithm.html
    """
    def __init__(self, codebook_size:int, n_components:int):
        self.codebook_size=codebook_size
        self.n_components = n_components
        self.max_iter = 10
        self.tol = 1E-3

    def _init_dict(self, Y:np.ndarray):
        """D is a normalized matrix (orthogonaly is not mandatory)"""
        if min(Y.shape) < self.codebook_size: 
            D = np.random.randn(self.codebook_size, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.codebook_size)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1, keepdims=True) #norm 2, for column vectors
        return D

    def _maxInnerProduct_slow(self, R:np.ndarray, D:np.ndarray):
        """argmax_g\in D |<R,g>|"""
        proj_val_norm = -np.inf
        proj_val = 0
        g_gamma_n = 0
        for i in range(self.codebook_size):
            g = D[[i],:]
            _proj_val = R.dot(g.T) #g.T
            _proj_val_norm = np.linalg.norm(_proj_val)
            if proj_val_norm < _proj_val_norm: 
                proj_val_norm = _proj_val_norm
                proj_val = _proj_val
                g_gamma_n = g
        return g_gamma_n, proj_val

    def _maxInnerProduct(self, R:np.ndarray, D:np.ndarray):
        """argmax_g\in D |<R,g>|"""
        M = R.dot(D.T)
        ix = np.argmax( np.linalg.norm(M, axis=0) )
        return D[[ix]], M[:,[ix]]

    def __call__(self, Y):
        """Y=f(t)"""
        R = Y #R1 = y
        D = self._init_dict(Y)
        result = {'a_n':[], 'gamma_n':[], 'normR':[]}
        for n in range(self.max_iter):
            g_gamma_n, a_n = self._maxInnerProduct(R, D)
            R = R - a_n.dot(g_gamma_n)
            normR = np.linalg.norm(R)
            result['a_n'].append(a_n)
            result['gamma_n'].append(g_gamma_n)
            result['normR'].append(normR)
            if normR <= self.tol: break #Matrix L2

        return result['a_n'], result['gamma_n'], result['normR']




#Y = (100, 2)  = Y-PHI^T z 
#D^T = (100, 10) 
#z = (10, 2)

#Y = (100, 1)  = Y-PHI^T z 
#D^T = (100, 10) 
#z = (10, 1)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodeBank.Debugging.SyntheticData import syntheticFunc
    plt.style.use('dark_background')

    expr = 'x^2+a_1*n'
    dom  = {'x':'N(0,1)','n':'N(0,1)'}

    func_1 = syntheticFunc(expr, dom)
 
    test =1
    if test == 1: 
        n_components = 3
        codebook_size = 10

        x, y = func_1(nSamples=100, a_1=0.3)
        MP=matchingPursuit(codebook_size, n_components)
        data = np.stack((x,y),-1)
        a_n, gamma_n, normR = MP(data[1])

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.plot(x,y,'o')
        ax1.set_xlabel('x~N(0,1)')
        ax1.set_ylabel(f'$f(x)={expr}$')
        # print(a_n)
        # ax2.stem(a_n)
        ax2.set_xlabel(f'n')
        ax2.set_ylabel(f'$a_n$')
        plt.show()
