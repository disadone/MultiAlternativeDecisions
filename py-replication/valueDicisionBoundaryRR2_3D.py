'''
Filename: /home/flumer/Documents/papercode/MultiAlternativeDecisions/py-replication/valueDicisionBoundaryRR2_3D.py
Path: /home/flumer/Documents/papercode/MultiAlternativeDecisions/py-replication
Created Date: Sunday, August 2nd 2020, 10:00:19 am
Author: LI Xiaodong

Copyright (c) 2020 Your Company
'''
#%%
import numpy as np
aa=np.asarray
from utils import findnearest
from scipy import optimize,ndimage,signal

# %%
def max_(x):
    x_=np.stack(x)
    V=np.max(x_,axis=0)
    D=np.argmax(x_,axis=0)
    D[(D==0) & (x[0]==x[1]) & (x[1]==x[2])]=123
    D[(D==0) & (x[0]==x[1])]=12
    D[(D==1) & (x[1]==x[2])]=23
    D[(D==0) & (x[0]==x[2])]=13
    return V,D
def backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0):
    rho=rho_
    V=np.repeat(np.zeros_like(Rh[0])[np.newaxis,:,:,:],len(t),axis=0)
    D=np.repeat(np.zeros_like(Rh[0])[np.newaxis,:,:,:],len(t),axis=0)
    EVnext=np.repeat(np.zeros_like(Rh[0])[np.newaxis,:,:,:],len(t),axis=0)
    Ptrans=[None]*len(t)
    iStrans=[None]*len(t)
    V[len(t)-1],D[len(t)-1]=max_(Rh-rho*tNull)

    for iT in range(len(t)-2,-1,-1):
        EVnext[iT],Ptrans[iT],iStrans[iT]=E(V[iT+1],S,t[iT],dt,g)
        V[iT],D[iT]=max_(np.r_[Rh-rho*tNull,np.expand_dims(EVnext[iT]-(rho+c)*dt,axis=0)])
        print('%d/%d'%(iT,len(t)-1))
    V0=V[0,iS0[0],iS0[1],0]
    D[D==0]=3
    print('rho = %d\t V0 = %d\t'%(rho_,V0))
    return V0,V,D,EVnext,rho,Ptrans,iStrans
def E(V,S,t,dt,g):
    aSscale=np.abs(S[0][0,:,0])
    CR=np.zeros((3,3))
    CX=np.zeros((3,3))
    for i in range(3):
        CR[i,i]=g[i]['varR']
        CX[i,i]=g[i]['varX']
    v=np.zeros(3)
    iStrans=[[],[],[]]
    for k in range(3):
        g[k]['varRh']=g[k]['varR']*g[k]['varX']/(t*g[k]['varR']+g[k]['varX'])
        v[k]=varTrans(g[k]['varRh'],g[k]['varR'],g[k]['varX'],t,dt)
        iStrans[k]=np.where(aSscale<3*np.sqrt(v[k]))[0] # 3 sigma standard?
    
    iSs=np.meshgrid(iStrans[0],iStrans[1],iStrans[2],indexing='ij')
    Ptrans=normal3(
        aa([S[0][iSs[0],iSs[1],iSs[2]],
        S[1][iSs[0],iSs[1],iSs[2]],
        S[2][iSs[0],iSs[1],iSs[2]]]),
        aa([0,0,0]),
        aa([[v[0],0,0],[0,v[1],0],[0,0,v[2]]]))
    mgn=np.ceil(aa(Ptrans.shape)/2)
    EV=signal.fftconvolve(V,Ptrans,'same')/signal.fftconvolve(np.ones_like(V),Ptrans,'same') # normaliztion
    return EV,Ptrans,iStrans

def varTrans(varRh,varR,varX,t,dt):
    return (varR/(varR*(t+dt)+varX))**2*(varX+varRh*dt)*dt
def normal3(x,m,C):
    d=np.zeros_like(x)
    for i in range(3):
        d[i]=x[i]-m[i]
    H=-1/2*np.linalg.solve(C,np.eye(3)) # inverse of C
    prob=d[0]**2*H[0,0]+d[0]*d[1]*H[0,1]+d[0]*d[2]*H[0,2]+\
        d[1]*d[0]*H[1,0]+d[1]**2*H[1,1]+d[1]*d[2]*H[1,2]+\
        d[2]*d[0]*H[2,0]+d[2]*d[1]*H[2,1]+d[2]**2*H[2,2]
    prob=np.exp(prob)
    prob/=prob.sum()
    return prob

#%%

# def valueDecisionBoundaryRR2_3D():
Smax = 4      # Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]) Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resSL  = 15      # Grid resolution of state space
resS = 41      # Grid resolution of state space
tmax = 3       # Time limit
dt   = .05       # Time step
c    = 0       # Cost of evidene accumulation
tNull = .25     # Non-decision time + inter trial interval
g=[{'meanR':0,'varR':0,'varX':0,'varRh':0}]*3
g[0]['meanR'] = 0. # Prior mean of state (dimension 1)
g[0]['varR']  = 5. # Prior variance of state
g[0]['varX']  = 2. # Observation noise variance
g[1]['meanR'] = 0. # Prior mean of state (dimension 2)
g[1]['varR']  = 5. # Prior variance of state
g[1]['varX']  = 2. # Observation noise variance
g[2]['meanR'] = 0. # Prior mean of state (dimension 3)
g[2]['varR']  = 5. # Prior variance of state
g[2]['varX']  = 2. # Observation noise variance

t = np.arange(0,tmax+dt,dt)
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'}    

## utility function:
untilityFunc=lambda X:X
# untilityFunc=lambda X:np.tanh(X)
# untilityFunc=lambda X:np.sign(X)*np.abs(X)**0.5

## Reward rate, Average-adjusted value, Decision (finding solution):
SscaleL=np.linspace(-Smax,Smax,resSL)
S=np.meshgrid(SscaleL,SscaleL,SscaleL,indexing='xy')
iS0=[findnearest(g[0]['meanR'],SscaleL),\
    findnearest(g[1]['meanR'],SscaleL),\
        findnearest(g[2]['meanR'],SscaleL)]
Rh=aa([*map(untilityFunc,S)])
RhMax,_=max_(Rh)

V0,V,D,EVnext,rho,Ptrans,iStrans=backwardInduction(g[0]['meanR'],c,tNull,g,Rh,S,t,dt,iS0)
rho_=optimize.fsolve(
    lambda rho:backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0)[0],x0=g[0]['meanR'])[0]

## Reward rate, Average-adjusted value, Decision (high resolution):
Sscale=np.linspace(-Smax,Smax,resS)
S=np.meshgrid(Sscale,Sscale,Sscale,indexing='xy')
iS0=[findnearest(g[0]['meanR'],Sscale),findnearest(g[1]['meanR'],Sscale),findnearest(g[2]['meanR'],Sscale)]
Rh=aa([*map(untilityFunc,S)])
RhMax,_=max_(Rh)
V0,V,D,EVnext,rho,Ptrans,iStrans=backwardInduction(g[0]['meanR'],c,tNull,g,Rh,S,t,dt,iS0)



# %%
# import matplotlib.pyplot as plt

# fig=plt.figure(figsize=plt.figaspect(0.5))
# ax=fig.add_subplot(5,4,1,projection='3d')