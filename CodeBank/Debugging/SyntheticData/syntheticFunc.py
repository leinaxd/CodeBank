import numpy as np

class syntheticFunc:
    def __init__(self, expresion:str, domain:dict):
        self.expresion = expresion
        self.domain = domain

    def __call__(self, **kwargs):
        return self.data_1(**kwargs)


    def data_1(self, nSamples=100, a_1=0.5):
        x = np.random.randn(nSamples)
        y = x*x+a_1*np.random.randn(nSamples)
        return x,y




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    expr = 'x^2+a_1*n'
    func = syntheticFunc(expr, {'x':'N(0,1)','n':'N(0,1)'})

    x, y = func(nSamples=100, a_1=0.5)
    plt.plot(x,y,'o')
    plt.xlabel('x~N(0,1)')
    plt.ylabel(f'$f(x)={expr}$')
    plt.show()