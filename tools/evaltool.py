import numpy as np
def evalfun(num_corr,predg,goldg):
    rg = np.array(num_corr)/np.array(goldg)
    pg = np.array(num_corr)/np.array(predg)

    f1 = 2*rg*pg/(rg+pg)

    return f1
