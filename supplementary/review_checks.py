#!/usr/bin/env python3
"""Independent checks added in response to the review.
Run from the project root. Numerical checks supplement the analytical proof.
No experimental data or fitted parameters are used.
"""
if not __debug__:
    raise RuntimeError("Verification requires normal Python execution; do not use -O or -OO.")

from pathlib import Path
import csv
import itertools
import math
import sys
import json
import numpy as np
from scipy.special import gammaln, expit, xlogy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_memory as v
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'supplementary' / 'data'
DATA.mkdir(exist_ok=True)


def entropy(p):
    return float(-np.sum(xlogy(p, p)) / math.log(2))


def posterior_entropy(d, e, q):
    if q <= 0 or q >= 1:
        return 0.0
    k = np.arange(d + 1, dtype=float)
    la = k * math.log(e) + (d-k) * math.log1p(-e)
    lb = (d-k) * math.log(e) + k * math.log1p(-e)
    l0 = math.log1p(-q) + la
    l1 = math.log(q) + lb
    lm = np.logaddexp(l0, l1)
    lc = gammaln(d+1)-gammaln(k+1)-gammaln(d-k+1)
    r = expit(l1-l0)
    hr = -(xlogy(r,r)+xlogy(1-r,1-r))/math.log(2)
    return float(np.dot(np.exp(lc+lm), hr))


def check_prior_maximality():
    rows=[]; count=0
    for d in (1,2,3,4,5,9):
        for e in (.05,.2,.4):
            qs=np.linspace(0,1,1001)
            vals=np.array([posterior_entropy(d,e,float(q)) for q in qs])
            uniform=posterior_entropy(d,e,.5)
            assert abs(uniform-v.HY_given_R(d,e))<2e-12
            assert np.max(vals)<=uniform+2e-12
            assert np.max(np.abs(vals-vals[::-1]))<2e-12
            assert np.max(np.diff(vals,n=2))<2e-12
            count+=len(qs)
            rows.append((d,e,uniform,float(np.max(vals)),float(np.max(np.diff(vals,n=2)))))
    with (DATA/'prior_maximality.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','epsilon','uniform_conditional_entropy_bits','maximum_on_grid_bits','largest_second_difference']);w.writerows(rows)
    return count


def check_direct_enumeration_and_biased_cycles():
    cases=0;cycles=0
    for d in (1,3,5):
        L=d+2
        states=list(itertools.product((0,1),repeat=L))
        c0=(0,)*L;c1=(1,)*d+(0,)*2
        weights=np.array([sum(z[:d]) for z in states])
        for e in (.05,.2,.4):
            a=np.array([e**sum(u!=w for u,w in zip(z,c0))*(1-e)**sum(u==w for u,w in zip(z,c0)) for z in states])
            b=np.array([e**sum(u!=w for u,w in zip(z,c1))*(1-e)**sum(u==w for u,w in zip(z,c1)) for z in states])
            assert abs(a.sum()-1)<2e-12 and abs(b.sum()-1)<2e-12
            for q0 in (.17,.5,.83):
                q=q0;heat_sum=0.;conditional_sum=0.;N=30
                for n in range(N):
                    # Vary biased thresholds and mix with a fresh source-independent coin.
                    t=n%(d+2)
                    decoder=.85*(weights>=t)+.15*((n%3)+1)/4
                    z=(1-q)*a+q*b
                    joint0=(1-q)*a;joint1=q*b
                    direct_post=entropy(np.concatenate([joint0,joint1]))-entropy(z)
                    F=posterior_entropy(d,e,q)
                    assert abs(direct_post-F)<5e-12
                    qnext=float(np.dot(z,decoder))
                    heat=entropy(z)-v.h(qnext)
                    bound=L*v.h(e)-F+v.h(q)-v.h(qnext)
                    assert abs(heat-bound)<5e-12
                    heat_sum+=heat;conditional_sum+=F
                    q=qnext;cycles+=1
                telescoped=L*v.h(e)-conditional_sum/N+(v.h(q0)-v.h(q))/N
                strengthened=L*v.h(e)-posterior_entropy(d,e,.5)+(v.h(q0)-v.h(q))/N
                assert abs(heat_sum/N-telescoped)<5e-12
                assert heat_sum/N>=strengthened-5e-12
                if q0==.5:
                    assert heat_sum/N>=L*v.h(e)-posterior_entropy(d,e,.5)-5e-12
                cases+=1
    return cases,cycles


def check_finite_asymptotics():
    e=.05 # Scoped variable; never reused for another noise grid.
    coefficient=v.h(e)/(-math.log(2*math.sqrt(e*(1-e))))
    ns=np.unique(np.round(np.geomspace(10,1e12,220)).astype(np.int64))
    rows=[];stats=[]
    for target in (.1,.5,.9):
        leading_residual=[];refined_residual=[]
        for N in ns:
            d=v.exact_distance(int(N),e,target)
            heat=d*v.h(e)-posterior_entropy(d,e,.5)
            leading=coefficient*math.log(int(N))
            refined=leading-.5*coefficient*math.log(math.log(int(N)))
            leading_residual.append(heat-leading);refined_residual.append(heat-refined)
            rows.append((e,target,int(N),d,heat,heat-leading,heat-refined))
        stats.append({'threshold':target,'targets':len(ns),'leading_residual_range':float(np.ptp(leading_residual)),'loglog_residual_range':float(np.ptp(refined_residual))})
    with (DATA/'finite_asymptotic_residuals.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['epsilon','threshold','N','distance','exact_heat_units','leading_residual_units','loglog_residual_units']);w.writerows(rows)
    # The agreement in Table 1 is not a general equality of distance criteria.
    N=3;dkl=1
    while -2*N*math.log1p(-v.omega(dkl,e))>math.log(2):dkl+=2
    exact=v.exact_distance(N,e,.5)
    assert (dkl,exact)==(1,3)
    return stats,2*v.h(e)


def main():
    count=check_prior_maximality()
    cases,cycles=check_direct_enumeration_and_biased_cycles()
    residuals,step=check_finite_asymptotics()
    result={'prior_grid_entries':count,'direct_register_cases':cases,'biased_trajectory_cycles':cycles,'epsilon_for_asymptotics':.05,'odd_distance_step_heat_units':step,'finite_asymptotic_residuals':residuals,'distance_counterexample':{'epsilon':.05,'threshold':.5,'N':3,'d_KL':1,'d_exact':3},'status':'PASS'}
    (ROOT/'supplementary/review_check_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('PASS: conditional-entropy symmetry, concavity and maximality; direct entropy identities; telescoping biased-cycle bound; finite-N diagnostics.')


if __name__=='__main__':
    main()
