import numpy as np



def findnearest(srchvalue,srcharray,bias=0):
    srca=srcharray.copy()
    if bias==-1: # only choose value <= to the search value
        srca[srca>0]=np.inf
    elif bias==1: # only choose value <= to the search value
        srca[srca<0]=np.inf
    srca=abs(srca)-abs(srchvalue)
    rc=srca.argmin()

    return rc

#%% test
if __name__=="__main__":
    import numpy as np
    def ff(x):
        x[x==1]=np.inf


# %%
