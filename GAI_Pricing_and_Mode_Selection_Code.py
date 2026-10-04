import numpy as np
from dataclasses import dataclass, replace
from numpy.polynomial.legendre import leggauss
from scipy.optimize import differential_evolution, minimize_scalar

@dataclass(frozen=True)
class Params:
    thetaL: float = 0.2
    thetaH: float = 0.8
    gammaT: float = 1.20
    beta0: float = 1.00
    beta1: float = 0.80
    beta2: float = 0.20
    sigma0: float = 0.30
    cG: float = 0.15
    c: float = 0.30
    cm0: float = 0.10
    ct0: float = 0.05
    cp0: float = 0.10
    e0: float = 0.30
    psi: float = 0.50


def primitives(theta, p: Params):
    beta = p.beta0 + p.beta1 * theta - p.beta2 * theta**2
    sigma = p.sigma0 * (1.0 - theta)
    alpha = beta * (1.0 - sigma)
    bT = p.ct0 / theta
    cT = p.cm0 / theta
    delta = p.psi * p.e0 * theta + p.cp0 * (1.0 - theta)
    return alpha, bT, cT, delta


def _nodes(n, a, b):
    x, w = leggauss(n)
    return (x + 1.0) * (b-a)/2.0 + a, w * (b-a)/2.0


def price_bounds(p: Params):
    theta = np.linspace(p.thetaL, p.thetaH, 4001)
    alpha, bT, _, delta = primitives(theta, p)
    pTmax = max(0.0, float(np.max(p.gammaT - bT)))
    pGmax = max(0.0, float(np.max(alpha - delta)))
    return pTmax, pGmax


def evaluate(prices, p: Params, arch='H', ntheta=80, nv=80, method='gauss'):
    if method == 'gauss':
        th, wt = _nodes(ntheta, p.thetaL, p.thetaH)
        wt = wt / (p.thetaH - p.thetaL)
        v, wv = _nodes(nv, 0.0, 1.0)
    elif method == 'midpoint':
        th = p.thetaL + (np.arange(ntheta)+0.5)*(p.thetaH-p.thetaL)/ntheta
        wt = np.full(ntheta, 1.0/ntheta)
        v = (np.arange(nv)+0.5)/nv
        wv = np.full(nv, 1.0/nv)
    else:
        raise ValueError(method)
    alpha, bT, cT, delta = primitives(th, p)
    if arch == 'T':
        pT = float(prices[0])
        T = p.gammaT*v[None,:] - pT - bT[:,None]
        G = np.full_like(T, -1e12)
    elif arch == 'G':
        pG = float(prices[0])
        G = alpha[:,None]*v[None,:] - pG - delta[:,None]
        T = np.full_like(G, -1e12)
    else:
        pT, pG = map(float, prices)
        T = p.gammaT*v[None,:] - pT - bT[:,None]
        G = alpha[:,None]*v[None,:] - pG - delta[:,None]
    maxu = np.maximum(np.maximum(T, G), 0.0)
    chooseG = (G >= T) & (G > 0.0)
    chooseT = (T > G) & (T > 0.0)
    W = wt[:,None]*wv[None,:]
    DT = float(np.sum(W*chooseT))
    DG = float(np.sum(W*chooseG))
    if arch == 'T':
        profit = float(np.sum(W*chooseT*(pT-p.c-cT[:,None])))
    elif arch == 'G':
        profit = float(np.sum(W*chooseG*(pG-p.c-p.cG)))
    else:
        profit = float(np.sum(W*chooseT*(pT-p.c-cT[:,None])) + np.sum(W*chooseG*(pG-p.c-p.cG)))
    CS = float(np.sum(W*maxu))
    return {'profit':profit,'DT':DT,'DG':DG,'demand':DT+DG,'CS':CS}


def optimize(p: Params, arch='H', n=55, seed=123):
    pTmax, pGmax = price_bounds(p)
    if arch == 'T':
        res = minimize_scalar(lambda x: -evaluate([x],p,'T',n,n)['profit'], bounds=(0,pTmax), method='bounded', options={'xatol':1e-10})
        prices = [float(res.x)]
    elif arch == 'G':
        res = minimize_scalar(lambda x: -evaluate([x],p,'G',n,n)['profit'], bounds=(0,pGmax), method='bounded', options={'xatol':1e-10})
        prices = [float(res.x)]
    else:
        res = differential_evolution(lambda x: -evaluate(x,p,'H',n,n)['profit'], [(0,pTmax),(0,pGmax)], seed=seed, tol=1e-9, popsize=18, maxiter=350, polish=True, workers=1, updating='immediate')
        prices = [float(res.x[0]),float(res.x[1])]
    out = evaluate(prices,p,arch,140,140,'gauss')
    out['prices'] = prices
    return out


def optimize_all(p: Params, seed=123):
    return {a:optimize(p,a,seed=seed) for a in ['T','G','H']}


def _conditional_shares(theta, pT, pG, p: Params):
    """Exact v-measure shares and consumer surplus for one theta under H."""
    alpha,bT,cT,delta = [float(x) for x in primitives(np.array([theta]),p)]
    gamma=p.gammaT
    # breakpoints where T=0, G=0, T=G plus support endpoints
    pts=[0.0,1.0]
    tauT=(pT+bT)/gamma
    tauG=(pG+delta)/alpha
    pts += [tauT,tauG]
    if abs(gamma-alpha)>1e-12:
        pts.append((pT+bT-pG-delta)/(gamma-alpha))
    pts=sorted(set(max(0.0,min(1.0,float(x))) for x in pts))
    dT=dG=cs=0.0
    for a,b in zip(pts[:-1],pts[1:]):
        if b-a<=1e-14: continue
        m=(a+b)/2
        UT=gamma*m-pT-bT
        UG=alpha*m-pG-delta
        if UT<=0 and UG<=0:
            continue
        if UG>=UT and UG>0:
            dG += b-a
            # integral of alpha*v-pG-delta
            cs += alpha*(b*b-a*a)/2 - (pG+delta)*(b-a)
        elif UT>UG and UT>0:
            dT += b-a
            cs += gamma*(b*b-a*a)/2 - (pT+bT)*(b-a)
    return dT,dG,cs,cT


def evaluate_exact(prices, p: Params, arch='H', ntheta=120, method='gauss'):
    if method=='gauss':
        th, wt = _nodes(ntheta,p.thetaL,p.thetaH); wt=wt/(p.thetaH-p.thetaL)
    elif method=='midpoint':
        th=p.thetaL+(np.arange(ntheta)+.5)*(p.thetaH-p.thetaL)/ntheta; wt=np.full(ntheta,1/ntheta)
    else: raise ValueError(method)
    DT=DG=CS=profit=0.0
    if arch=='T':
        pT=float(prices[0])
        for t,w in zip(th,wt):
            _,bT,cT,_=[float(x) for x in primitives(np.array([t]),p)]
            q=max(0.0,min(1.0,1-(pT+bT)/p.gammaT))
            DT += w*q
            profit += w*(pT-p.c-cT)*q
            CS += w*p.gammaT*q*q/2
    elif arch=='G':
        pG=float(prices[0])
        for t,w in zip(th,wt):
            alpha,_,_,delta=[float(x) for x in primitives(np.array([t]),p)]
            q=max(0.0,min(1.0,1-(pG+delta)/alpha))
            DG += w*q
            profit += w*(pG-p.c-p.cG)*q
            CS += w*alpha*q*q/2
    else:
        pT,pG=map(float,prices)
        for t,w in zip(th,wt):
            dT,dG,cs,cT=_conditional_shares(t,pT,pG,p)
            DT += w*dT; DG += w*dG; CS += w*cs
            profit += w*((pT-p.c-cT)*dT+(pG-p.c-p.cG)*dG)
    return {'profit':float(profit),'DT':float(DT),'DG':float(DG),'demand':float(DT+DG),'CS':float(CS)}


def optimize_exact(p: Params, arch='H', n=90, seed=123):
    pTmax,pGmax=price_bounds(p)
    if arch=='T':
        res=minimize_scalar(lambda x:-evaluate_exact([x],p,'T',n)['profit'],bounds=(0,pTmax),method='bounded',options={'xatol':1e-11})
        prices=[float(res.x)]
    elif arch=='G':
        res=minimize_scalar(lambda x:-evaluate_exact([x],p,'G',n)['profit'],bounds=(0,pGmax),method='bounded',options={'xatol':1e-11})
        prices=[float(res.x)]
    else:
        res=differential_evolution(lambda x:-evaluate_exact(x,p,'H',n)['profit'],[(0,pTmax),(0,pGmax)],seed=seed,tol=1e-10,popsize=18,maxiter=400,polish=True,workers=1)
        candidates=[list(map(float,res.x))]
        # Architecture nesting check: explicitly include the two pure-mode embeddings.
        t=optimize_exact(p,'T',n=max(50,n),seed=seed); g=optimize_exact(p,'G',n=max(50,n),seed=seed)
        candidates += [[t['prices'][0],pGmax],[pTmax,g['prices'][0]]]
        prices=max(candidates,key=lambda z:evaluate_exact(z,p,'H',max(70,n))['profit'])
    out=evaluate_exact(prices,p,arch,240,'gauss'); out['prices']=list(map(float,prices))
    return out

if __name__ == '__main__':
    p=Params()
    results={a:optimize_exact(p,a) for a in ['T','G','H']}
    for a,r in results.items(): print(a,r)
