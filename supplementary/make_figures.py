"""Generate manuscript figures and CSVs from exact model formulas.
Run from the project root: python supplementary/make_figures.py
Requires numpy, scipy, matplotlib. No observational data or fitted parameters.
"""
if not __debug__:
    raise RuntimeError("Verification requires normal Python execution; do not use -O or -OO.")

from pathlib import Path
import csv, math, importlib.util
from decimal import Decimal, localcontext, ROUND_FLOOR
import numpy as np
from scipy.special import gammaln, expit, xlogy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify_memory',ROOT/'supplementary/verify_memory.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
DATA=ROOT/'supplementary/data';DATA.mkdir(exist_ok=True)

def posterior_entropy(d,e):
    k=np.arange(d+1,dtype=float)
    la=k*math.log(e)+(d-k)*math.log1p(-e)
    lb=(d-k)*math.log(e)+k*math.log1p(-e)
    lw=gammaln(d+1)-gammaln(k+1)-gammaln(d-k+1)+np.logaddexp(la,lb)-math.log(2)
    q=expit(la-lb)
    ent=-(xlogy(q,q)+xlogy(1-q,1-q))/math.log(2)
    return float(np.dot(np.exp(lw),ent))

def verify_tail_prefactor():
    d=10001;errors=[]
    k=np.arange(d+1,dtype=float)
    for e in (.05,.25,.4):
        lp=gammaln(d+1)-gammaln(k+1)-gammaln(d-k+1)+k*math.log(e)+(d-k)*math.log1p(-e)
        logomega=math.log(2)+float(np.logaddexp.reduce(lp[(d+1)//2:]))
        C=-math.log(2*math.sqrt(e*(1-e)))
        A=2*math.sqrt(2/math.pi)*math.sqrt(e*(1-e))/(1-2*e)
        delta=logomega-(math.log(A)-.5*math.log(d)-C*d)
        assert abs(delta)<.01, 'tail-prefactor disagreement'
        errors.append((e,delta))
    return errors

def heat_intervals(e, threshold=.5, nmin=10, nmax=10**12):
    """Every integer transition, with 70-digit tail and threshold arithmetic."""
    intervals=[]
    with localcontext() as ctx:
        ctx.prec=70
        E=Decimal(str(e)); one=Decimal(1); logtwo=Decimal(2).ln()
        target=Decimal(str(threshold))
        def h_decimal(p):
            if p == 0 or p == 1:return Decimal(0)
            return -(p*p.ln()+(one-p)*(one-p).ln())/logtwo
        lo=Decimal(0);hi=Decimal('.5')
        for _ in range(240):
            mid=(lo+hi)/2
            if h_decimal(mid)<one-target:lo=mid
            else:hi=mid
        budget=-(one-lo-hi).ln()
        previous=0;d=1
        while previous<nmax:
            k=(d+1)//2
            mass=Decimal(math.comb(d,k))*E**k*(one-E)**(d-k)
            tail=mass
            for j in range(k,d):
                mass*=Decimal(d-j)/Decimal(j+1)*E/(one-E)
                tail+=mass
            rate=-(one-2*tail).ln()
            cap=int((budget/rate).to_integral_value(rounding=ROUND_FLOOR))
            assert Decimal(cap)*rate<=budget<Decimal(cap+1)*rate
            left=max(nmin,previous+1);right=min(nmax,cap)
            if left<=right:
                intervals.append((left,right,d,d*v.h(e)-posterior_entropy(d,e),cap))
            previous=cap;d+=2
        assert intervals[0][0]==nmin and intervals[-1][1]==nmax
        assert all(b[0]==a[1]+1 for a,b in zip(intervals,intervals[1:]))
    return intervals

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42})
e=.05;d=5;w=v.omega(d,e);n=np.arange(0,1001)
z=np.exp(n*math.log1p(-w));q=(1-z)/2
exact=1-(-(xlogy(q,q)+xlogy(1-q,1-q))/math.log(2))
kl=z*z;dob=z
fig,ax=plt.subplots(figsize=(6.6,3.8))
ax.plot(n,exact,label='Exact attainable retention',color='#176b87',lw=2)
ax.plot(n,kl,'--',label=r'Squared KL bound $(1-\omega_d)^{2N}$',color='#b85c38')
ax.plot(n,dob,':',label='Dobrushin upper bound',color='#666666')
qrec=0;rows=[]
for N in n:
    rows.append((int(N),1-v.h(qrec),float(exact[N]),float(kl[N]),float(dob[N])))
    qrec=v.conv(qrec,w/2)
assert max(abs(r[1]-r[2]) for r in rows)<2e-12
marks=rows[::50];ax.plot([r[0] for r in marks],[r[1] for r in marks], 'o',ms=3.5,mfc='white',mec='#176b87',label='Independent propagation')
ax.axhline(.5,color='black',lw=.8,alpha=.6)
ax.set(xlabel='Completed noise-correction cycles N',ylabel='Mutual information (bits)',ylim=(0,1.04),xlim=(0,1000))
ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.18);fig.tight_layout()
fig.savefig(OUT/'retention_bounds.pdf');fig.savefig(OUT/'retention_bounds.png',dpi=180);plt.close(fig)
with (DATA/'retention.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['N','independent_MI_bits','exact_MI_bits','KL_bound_bits','Dobrushin_bound_bits']);wr.writerows(rows)

fig,ax=plt.subplots(figsize=(6.6,3.8));rows=[];transition_rows=[]
ns=np.unique(np.round(np.geomspace(10,1e12,220)).astype(np.int64))
for e,color in [(.05,'#176b87'),(.2,'#b85c38'),(.4,'#6b5b95')]:
    C=-math.log(2*math.sqrt(e*(1-e)));heat=[];leading=[]
    for N in ns:
        distance=v.exact_distance(int(N),e,.5)
        Hpost=posterior_entropy(distance,e)
        Q=distance*v.h(e)-Hpost
        fano=distance*v.h(e)-v.h(v.omega(distance,e)/2)
        assert Q>=fano-1e-8 and math.isfinite(Q)
        heat.append(Q);leading.append(v.h(e)/C*math.log(int(N)))
        rows.append((e,int(N),distance,Hpost,Q,fano,leading[-1]))
    intervals=heat_intervals(e)
    edges=[intervals[0][0]]+[part[1]+1 for part in intervals]
    values=[part[3] for part in intervals]
    ax.stairs(values,edges,baseline=None,color=color,label=rf'Exact optimum, $\epsilon={e:g}$',lw=1.8)
    transition_rows.extend((e,.5,left,right,d,Q,cap) for left,right,d,Q,cap in intervals)
    for N,Q in zip(ns,heat):
        interval=next(part for part in intervals if part[0]<=int(N)<=part[1])
        assert abs(Q-interval[3])<1e-11, 'sample and integer interval disagree'
    ax.plot(ns,leading,'--',color=color,alpha=.65,lw=1)
ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='Target completed cycles N',ylabel=r'Mean correction heat $\bar Q/(k_B T\ln 2)$')
ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.18);fig.tight_layout()
fig.savefig(OUT/'optimal_heat.pdf');fig.savefig(OUT/'optimal_heat.png',dpi=180);plt.close(fig)
with (DATA/'optimal_heat.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['epsilon','N','minimal_distance','posterior_entropy_bits','exact_heat_units','Fano_lower_bound_units','leading_asymptotic_units']);wr.writerows(rows)
with (DATA/'heat_transitions.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['epsilon','threshold','first_integer_N','last_integer_N','minimal_distance','exact_heat_units','max_supported_N_for_distance']);wr.writerows(transition_rows)
for d in range(1,34):
    assert abs(posterior_entropy(d,.05)-v.HY_given_R(d,.05))<1e-11
errors=verify_tail_prefactor()
with (DATA/'tail_prefactor_check.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['epsilon','distance','log_ratio_to_asymptotic']);wr.writerows((e,10001,error) for e,error in errors)
print('PASS: figures and CSVs generated; 1001 independent retention checks; posterior entropy cross-checks passed')
print('Tail-prefactor log-ratios at odd d=10001:',errors)
print('PASS: all integer heat transitions computed at 70 digits;',len(transition_rows),'intervals')
