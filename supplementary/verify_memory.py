#!/usr/bin/env python3
"""Reproducibility checks for the standalone lifetime--heat manuscript.
Requires numpy and scipy; decimal is from the Python standard library.
All violations raise AssertionError.
"""
if not __debug__:
    raise RuntimeError("Verification requires normal Python execution; do not use -O or -OO.")

from pathlib import Path
import csv
import itertools
from decimal import Decimal, getcontext
from math import comb, log, log2, log1p, exp, floor
import platform
import scipy
from scipy.special import gammaln

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom


def h(x):
    return 0.0 if x <= 0 or x >= 1 else -(x * log2(x) + (1 - x) * log2(1 - x))


def conv(a, b):
    return a * (1 - b) + b * (1 - a)


def hinv(u):
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if h(mid) < u else (lo, mid)
    return (lo + hi) / 2


def f_gerber(u, e):
    return h(conv(e, hinv(min(max(u, 0.0), 1.0))))


def H(dist):
    p = np.array([v for v in dist.values() if v > 1e-300])
    return float(-(p * np.log2(p)).sum())


def omega(d, e):
    tail = float(binom.sf(d // 2, d, e))
    return 2 * tail + (float(binom.pmf(d // 2, d, e)) if d % 2 == 0 else 0)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------- (i)
def simulate_code(c0, c1, e, decoder, ncyc):
    """Exact simulation of a two-codeword hard-decision memory. Returns per-cycle
    tuples (M_n, Qmin_n, s_n, s_{n+1}, I(X;R'_n) - M_{n+1})."""
    L = len(c0)
    S = list(itertools.product([0, 1], repeat=L))
    code = (tuple(c0), tuple(c1))
    P = {x: {code[x]: 1.0} for x in (0, 1)}

    def stats(P):
        joint = {(x, r): 0.5 * p for x in P for r, p in P[x].items()}
        marg = {}
        for (x, r), p in joint.items():
            marg[r] = marg.get(r, 0) + p
        HR = H(marg)
        HRX = H(joint) - 1.0
        return HR, HRX, HR - HRX

    rows = []
    for _ in range(ncyc):
        HR, s, M = stats(P)
        Pn = {}
        for x in P:
            dct = {}
            for r, p in P[x].items():
                for y in S:
                    k = sum(a != b for a, b in zip(r, y))
                    dct[y] = dct.get(y, 0) + p * e**k * (1 - e) ** (L - k)
            Pn[x] = dct
        HRp, sp, Mp = stats(Pn)
        P2 = {x: {} for x in Pn}
        for x in Pn:
            for y, p in Pn[x].items():
                c = code[decoder(y)]
                P2[x][c] = P2[x].get(c, 0) + p
        HR2, s2, M2 = stats(P2)
        rows.append((M, HRp - HR2, s, s2, Mp - M2))
        P = P2
    return rows


def ents_soft(L, a, q):
    """Register R_i = Y xor N_i, N_i iid Bern(a), P(Y != X) = q, X uniform."""
    HR = HRX = 0.0
    for k in range(L + 1):
        p0 = a**k * (1 - a) ** (L - k)
        p1 = a ** (L - k) * (1 - a) ** k
        px0 = (1 - q) * p0 + q * p1
        px1 = (1 - q) * p1 + q * p0
        pr = 0.5 * (px0 + px1)
        c = comb(L, k)
        if pr > 0:
            HR -= c * pr * log2(pr)
        for px in (px0, px1):
            if px > 0:
                HRX -= c * 0.5 * px * log2(px)
    return HR, HRX, HR - HRX


def pmaj(L, b):
    return sum(comb(L, k) * b**k * (1 - b) ** (L - k) for k in range(L + 1) if k > L / 2)


def check_entropy_pumping():
    e = 0.05
    out = []
    for L in (3, 5, 7):
        rows = simulate_code((0,) * L, (1,) * L, e, lambda y, L=L: int(sum(y) > L / 2), 60)
        bounds = [L * f_gerber(s / L, e) - s2 + dl for (M, Q, s, s2, dl) in rows]
        require(all(row[1] >= bd - 1e-10 for row, bd in zip(rows, bounds)), "entropy-pumping violation")
        ratios = [row[1] / bd for row, bd in zip(rows, bounds) if bd > 1e-12]
        out.append(f"  repetition L={L}: min ratio {min(ratios):.4f}, max {max(ratios):.4f}")
    viol, tight = 0, 9.0
    for L in (1, 3, 5, 11, 21, 51, 101):
        for a in np.linspace(0, 0.48, 49):
            for q in (0.0, 0.01, 0.05, 0.2, 0.4):
                b = conv(e, a)
                _, s, _ = ents_soft(L, a, q)
                HRp, _, Mp = ents_soft(L, b, q)
                q2 = conv(q, pmaj(L, b))
                HR2, s2, M2 = ents_soft(L, a, q2)
                Q = HRp - HR2
                bound = L * f_gerber(s / L, e) - s2 + (Mp - M2)
                viol += Q < bound - 1e-9
                if bound > 1e-6:
                    tight = min(tight, Q / bound)
    require(viol == 0, "soft entropy-pumping violation")
    out.append(f"  soft re-encodings: violations {viol}, tightest ratio {tight:.4f}")
    return out


# ---------------------------------------------------------------- (ii)
def check_binary_kl(n=60, seed=1):
    getcontext().prec = 60
    rng = np.random.default_rng(seed)
    worst = -1.0
    for _ in range(n):
        a, b = rng.uniform(0.01, 0.99, 2)
        if abs(1 - a - b) < 0.02:
            continue
        out = lambda x: x * (1 - b) + (1 - x) * a
        D = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        f = lambda v: -D(out(v[0]), out(v[1])) / max(D(v[0], v[1]), 1e-300)
        best = None
        for s0 in np.linspace(0.05, 0.95, 7):
            for t0 in np.linspace(0.05, 0.95, 7):
                if abs(s0 - t0) < 1e-3:
                    continue
                r = minimize(f, [s0, t0], bounds=[(1e-9, 1 - 1e-9)] * 2, method="L-BFGS-B")
                if best is None or r.fun < best.fun:
                    best = r
        p, q = (Decimal(float(v)) for v in best.x)
        A, B = Decimal(float(a)), Decimal(float(b))
        DD = lambda p, q: p * (p / q).ln() + (1 - p) * ((1 - p) / (1 - q)).ln()
        oo = lambda x: x * (1 - B) + (1 - x) * A
        ratio = DD(oo(p), oo(q)) / DD(p, q)
        formula = (((1 - A) * (1 - B)).sqrt() - (A * B).sqrt()) ** 2
        require(ratio <= formula * (1 + Decimal("1e-8")), "KL candidate exceeds formula")
        worst = max(worst, float((ratio - formula) / formula))
    return f"  {n} random channels: max (searched candidate - formula)/formula = {worst:.2e} (should be <= 0)"


# ---------------------------------------------------------------- (iii)
def log_vertex_values(d, e):
    k = np.arange(d + 1)
    lp = (gammaln(d + 1) - gammaln(k + 1) - gammaln(d - k + 1)
          + k * log(e) + (d - k) * log1p(-e))
    # Positive-probability sums: no subtraction of a tail from one.
    lc = np.concatenate(([-np.inf], np.logaddexp.accumulate(lp)))
    lt = np.concatenate((np.logaddexp.accumulate(lp[::-1])[::-1], [-np.inf]))
    t = np.arange(d + 2)
    return np.logaddexp(0.5 * (lt[t] + lc[d + 1 - t]),
                        0.5 * (lt[d + 1 - t] + lc[t]))


def decimal_vertex_check(d, e):
    getcontext().prec = 80
    E = Decimal(str(e)); one = Decimal(1)
    pi = [(one - E) ** d]
    for k in range(d):
        pi.append(pi[-1] * Decimal(d - k) / Decimal(k + 1) * E / (one - E))
    cdf = [Decimal(0)]
    for v in pi:
        cdf.append(cdf[-1] + v)
    tail = [Decimal(0)] * (d + 2)
    for k in range(d, -1, -1):
        tail[k] = tail[k + 1] + pi[k]
    vals = [(tail[t] * cdf[d + 1 - t]).sqrt()
            + (tail[d + 1 - t] * cdf[t]).sqrt() for t in range(d + 2)]
    maj = vals[(d + 1) // 2]
    require(all(v >= maj * (one - Decimal("1e-65")) for v in vals),
            f"80-digit majority counterexample d={d}, eps={e}")


def check_majority_vertex():
    bad = tot = 0
    for d in list(range(1, 200, 2)) + [251, 301, 401, 601, 1001]:
        for e in np.concatenate([np.linspace(0.001, 0.2, 40), np.linspace(0.2, 0.499, 60)]):
            vals = log_vertex_values(d, float(e))
            maj = vals[(d + 1) // 2]
            tot += 1
            bad += float(np.min(vals)) < maj - 1e-10
    require(bad == 0, "log-domain majority-vertex violation")
    high = 0
    for d in (1, 3, 5, 31, 101, 251, 601, 1001):
        for e in (0.001, 0.05, 0.2, 0.3, 0.499):
            decimal_vertex_check(d, e); high += 1
    return f"  {tot} log-domain grid entries (10395 distinct): violations {bad}; {high} full-vertex 80-digit checks passed"


# ---------------------------------------------------------------- (iv)
def check_kl_lifetime():
    e = 0.05
    cases = [("rep3", (0, 0, 0), (1, 1, 1), lambda y: int(sum(y) >= 2)),
             ("rep5", (0,) * 5, (1,) * 5, lambda y: int(sum(y) >= 3)),
             ("rep5, biased threshold", (0,) * 5, (1,) * 5, lambda y: int(sum(y) >= 2)),
             ("L=5, d=3", (0,) * 5, (1, 1, 1, 0, 0), lambda y: int(sum(y[:3]) >= 2))]
    out = []
    for name, c0, c1, dec in cases:
        d = sum(a != b for a, b in zip(c0, c1))
        w = omega(d, e)
        rows = simulate_code(c0, c1, e, dec, 300)
        r = [max(0.0, M) / exp(2 * n * log1p(-w)) for n, (M, *_rest) in enumerate(rows)]
        require(all(v <= 1 + 1e-8 for v in r), "KL lifetime violation")
        out.append(f"  {name}: max M_N/(1-w)^(2N) = {max(r):.3f}, at completed N=299: {r[-1]:.3f}")
    return out


# ---------------------------------------------------------------- (v)
def HY_given_R(L, e):
    s = 0.0
    for k in range(L + 1):
        a = e**k * (1 - e) ** (L - k)
        b = e ** (L - k) * (1 - e) ** k
        s += comb(L, k) * 0.5 * (a + b) * h(a / (a + b))
    return s


def lifetime_rep(L, e, Ms):
    rstar = 1 - 2 * hinv(1 - Ms)
    return floor(log(rstar) / log1p(-omega(L, e)))


def exact_distance(N, e, Ms):
    target = -log(1 - 2 * hinv(1 - Ms))
    d = 1
    while -N * log1p(-omega(d, e)) > target:
        d += 1
    return d


def check_exact_uniform():
    count = 0
    for d in (1, 2, 3, 4, 5, 6):
        for e in (.05, .25, .4):
            w = omega(d, e)
            # Independent exact propagation of the binary logical channel,
            # including fair randomized decisions at ties.
            p = 0.0
            noisy_marginal = {}
            for bits in itertools.product((0, 1), repeat=d):
                k = sum(bits)
                a = e**k * (1-e)**(d-k)
                b = e**(d-k) * (1-e)**k
                noisy_marginal[bits] = (a+b)/2
                decision_error = 1.0 if 2*k > d else (.5 if 2*k == d else 0.0)
                p += a * decision_error
            direct_heat = H(noisy_marginal)-1
            posterior_heat = d*h(e)-HY_given_R(d,e)
            require(abs(direct_heat-posterior_heat) < 2e-12,
                    "enumerated heat/posterior identity mismatch")
            require(abs(2 * p - w) < 1e-13, "fair-tie overlap mismatch")
            q = 0.0
            for N in range(101):
                expected = 1 - h((1 - exp(N * log1p(-w))) / 2)
                actual = 1 - h(q)
                require(abs(actual - expected) < 2e-12, "exact uniform trajectory mismatch")
                require(actual <= exp(2 * N * log1p(-w)) + 2e-12, "squared bound mismatch")
                q = conv(q, float(p)); count += 1
            # Unfavorable symmetric decoders still satisfy the exact bound.
            for pn in ((float(p) + .5) / 2, 1 - float(p)):
                q = 0.0
                for N in range(50):
                    upper = 1 - h((1 - exp(N * log1p(-w))) / 2)
                    require(1 - h(q) <= upper + 2e-12, "suboptimal symmetric bound violation")
                    q = conv(q, pn)
    for e in (.05, .25, .4):
        for m in range(1, 11):
            require(abs(omega(2*m, e)-omega(2*m-1, e)) < 1e-12, "even-distance identity")
    for d in (3, 5, 9):
        z = exp(100 * log1p(-omega(d, .05)))
        require(abs((1-h((1-z)/2))/(z*z) - 1/(2*log(2))) < 1e-8,
                "symmetric asymptotic prefactor") if z < 1e-3 else None
    # Use correlation directly to avoid information cancellation at large N.
    z = .0001
    require(abs((1-h((1-z)/2))/(z*z)-1/(2*log(2))) < 1e-7, "prefactor limit")
    return f"  {count} exact trajectory values; even distances, eps>1/5, suboptimal symmetric maps and prefactor checks passed"


def table(e=0.05, Ms=0.5):
    lines = ["  N | KL necessary d | exact d | Fano bound | exact ideal minimum"]
    records = []
    expected = [3,5,9,11,17,25,33]
    for i, N in enumerate((10, 10**2, 10**3, 10**4, 10**6, 10**9, 10**12)):
        dkl = 1
        while -2 * N * log1p(-omega(dkl, e)) > log(1 / Ms):
            dkl += 2
        d = exact_distance(N, e, Ms)
        require(d == expected[i], "table distance regression")
        require(lifetime_rep(d, e, Ms) >= N, "lifetime achievability")
        require(d == 1 or lifetime_rep(d-1,e,Ms) < N, "distance minimality")
        posterior = HY_given_R(d, e)
        fano = d*h(e)-h(omega(d,e)/2)
        qmin = d*h(e)-posterior
        require(qmin >= fano-1e-10, "Fano bound violation")
        records.append((N, dkl, d, fano, qmin))
        lines.append(f"  {N:>13} | {dkl:>3} | {d:>3} | {fano:.9f} | {qmin:.9f}")
    data = Path(__file__).resolve().parents[1] / "supplementary" / "data"
    data.mkdir(parents=True, exist_ok=True)
    with (data / "table1.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["N", "KL_necessary_distance", "exact_minimal_distance", "Fano_bound_heat_units", "exact_heat_infimum_units"])
        writer.writerows(records)
    return lines


if __name__ == "__main__":
    print("(i) entropy-pumping bound");      print("\n".join(check_entropy_pumping()))
    print("(ii) binary KL coefficient");     print(check_binary_kl())
    print("(iii) majority vertex");          print(check_majority_vertex())
    print("(iv) KL lifetime bound");         print("\n".join(check_kl_lifetime()))
    print("(v) exact uniform lifetime"); print(check_exact_uniform())
    print("(vi) lifetime-heat table (eps=0.05, M*=0.5)"); print("\n".join(table()))

    print("PASS: all checks completed; Python", platform.python_version(), "numpy", np.__version__, "scipy", scipy.__version__)
