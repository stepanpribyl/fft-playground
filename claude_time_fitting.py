"""Time-domain fit of k exponentially damped sinusoids (variable projection + LM).

Model (t measured from the pulse start, t0 is subtracted):
    y(t) = offset + sum_i A_i * exp(-g_i * t) * cos(2*pi*f_i*t + phi_i)

    A_i   amplitude at t = 0
    f_i   frequency [Hz]
    g_i   decay rate [1/s]  (= 1/tau);  Lorentzian HWHM in Hz = g_i / (2*pi)
    phi_i phase [rad]

Variable projection: for fixed (f_i, g_i) the model is linear in
(offset, a_i, b_i) with  a_i = A_i cos(phi_i),  b_i = -A_i sin(phi_i),
so these are solved by linear least squares inside the residual function and
LM (scipy least_squares, method='lm') only searches the 2k nonlinear
parameters.  A final joint LM polish in (A, f, g, phi) yields the covariance.

Starting values: you supply the frequencies f0 (e.g. read off your FFT peaks).
Starting decay rates default to 5 / record length; pass g0 if you know better.
"""
import numpy as np
from scipy.optimize import least_squares


def _design(t, f, g, offset):
    cols = [np.ones_like(t)] if offset else []
    for fi, gi in zip(f, g):
        env = np.exp(-gi * t)
        w = 2 * np.pi * fi * t
        cols += [env * np.cos(w), env * np.sin(w)]
    return np.column_stack(cols)


def _linear_solve(t, y, f, g, offset):
    X = _design(t, f, g, offset)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X, coef


def _varpro_resid(theta, t, y, k, offset):
    f, g = theta[:k], np.abs(theta[k:])  # abs: lm has no bounds
    X, coef = _linear_solve(t, y, f, g, offset)
    return X @ coef - y


def damped_sum(t, params, k, offset=True):
    """params = [offset?, A_1, f_1, g_1, phi_1, ..., A_k, f_k, g_k, phi_k]"""
    p = np.asarray(params, float)
    y = np.full_like(t, p[0]) if offset else np.zeros_like(t)
    q = p[1:] if offset else p
    for i in range(k):
        A, f, g, phi = q[4 * i: 4 * i + 4]
        y += A * np.exp(-abs(g) * t) * np.cos(2 * np.pi * f * t + phi)
    return y


def _full_resid(p, t, y, k, offset):
    return damped_sum(t, p, k, offset) - y


def fit_damped_sinusoids(t, y, f0, g0=None, t0=0.0, offset=True, polish=True):
    """
    t, y : time axis [s] and signal
    f0   : starting frequencies [Hz] (length k)
    g0   : starting decay rates [1/s] (scalar or length k); default 5/T
    t0   : time of pulse start (samples before t0 should be cut beforehand)

    Returns dict with fitted values, 1-sigma errors, and the scipy result.
    Peaks are returned sorted by frequency.
    """
    t = np.asarray(t, float) - t0
    y = np.asarray(y, float)
    f0 = np.atleast_1d(np.asarray(f0, float))
    k = len(f0)
    T = t[-1] - t[0]
    g0 = np.full(k, 5.0 / T) if g0 is None else np.broadcast_to(np.asarray(g0, float), (k,)).copy()
    if len(t) < 4 * k + 1:
        raise ValueError("not enough samples for this many peaks")

    # --- stage 1: variable projection over (f_i, g_i) ---
    res1 = least_squares(_varpro_resid, np.r_[f0, g0], args=(t, y, k, offset),
                         method="lm", x_scale="jac")
    f, g = res1.x[:k], np.abs(res1.x[k:])
    _, coef = _linear_solve(t, y, f, g, offset)
    off = coef[0] if offset else 0.0
    ab = coef[1:] if offset else coef
    a, b = ab[0::2], ab[1::2]
    A, phi = np.hypot(a, b), np.arctan2(-b, a)

    p = ([off] if offset else []) + [v for i in range(k) for v in (A[i], f[i], g[i], phi[i])]
    p = np.array(p)
    res = res1

    # --- stage 2: joint polish in (A, f, g, phi) for covariance ---
    if polish:
        res = least_squares(_full_resid, p, args=(t, y, k, offset), method="lm",
                            x_scale="jac")
        p = res.x
    J = res.jac
    dof = max(len(y) - len(p), 1)
    s2 = 2 * res.cost / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2 if polish else np.full((len(p),) * 2, np.nan)
    except np.linalg.LinAlgError:
        cov = np.full((len(p),) * 2, np.nan)
    err = np.sqrt(np.abs(np.diag(cov)))

    o = 1 if offset else 0
    peaks = []
    for i in range(k):
        Ai, fi, gi, pi_ = p[o + 4 * i: o + 4 * i + 4]
        eA, ef, eg, ep = err[o + 4 * i: o + 4 * i + 4]
        peaks.append(dict(A=abs(Ai), f=fi, gamma=abs(gi), hwhm_hz=abs(gi) / (2 * np.pi),
                          phi=(pi_ + (np.pi if Ai < 0 else 0) + np.pi) % (2 * np.pi) - np.pi,
                          err_A=eA, err_f=ef, err_gamma=eg, err_phi=ep))
    peaks.sort(key=lambda d: d["f"])
    return dict(offset=p[0] if offset else 0.0, peaks=peaks, params=p, cov=cov,
                rms=float(np.sqrt(np.mean(res.fun ** 2))), result=res)


if __name__ == "__main__":
    # Synthetic test: same pulse, three recording lengths, rough f0 (as from an FFT).
    rng = np.random.default_rng(1)
    fs = 1000.0
    truth = [(1.0, 50.0, 1 / 2.0, 0.7), (0.6, 120.0, 1 / 1.0, -1.2)]  # A, f, gamma, phi
    for T in (5.0, 10.0, 20.0):
        t = np.arange(0, T, 1 / fs)
        y = 0.02 + sum(A * np.exp(-g * t) * np.cos(2 * np.pi * f * t + ph) for A, f, g, ph in truth)
        y += 0.02 * rng.standard_normal(t.size)
        out = fit_damped_sinusoids(t, y, f0=[50.15, 119.8])  # starts off by ~0.2 Hz
        print(f"T={T:4.0f}s  rms={out['rms']:.4f}  offset={out['offset']:.4f}")
        for pk in out["peaks"]:
            print(f"   f={pk['f']:.4f}±{pk['err_f']:.1g}  A={pk['A']:.4f}±{pk['err_A']:.1g}  "
                  f"gamma={pk['gamma']:.4f}±{pk['err_gamma']:.1g}  phi={pk['phi']:.3f}")