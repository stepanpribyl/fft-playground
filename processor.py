from function import Function
from numpy import full_like, mean, argsort
import utils
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import least_squares


def multi_lorentz_model(params, freqs):
    """
    params je ploché pole: [c, A1, f1, g1, A2, f2, g2, ...]
    kde c je šum, A je amplituda, f je frekvence, g je pološířka (tlumení)
    """
    c = params[0]
    y = full_like(freqs, c)

    # Procházíme parametry po trojicích pro každou z 5 frekvencí
    for i in range(1, len(params), 3):
        A = params[i]
        f0 = params[i + 1]
        gamma = params[i + 2]
        y += (A * gamma ** 2) / ((freqs - f0) ** 2 + gamma ** 2)
    return y

def residuals(params, freqs, observed_spectrum):
    d = multi_lorentz_model(params, freqs) - observed_spectrum
    # print(f"Sum of residuals: {sum(d)}")
    return d


"""Fitting of pulse-response data with a sum of exponentially damped sinusoids.

Model (k components, optional constant offset):

    y(t) = c0 + sum_{i=1..k} A_i * exp(-g_i * t) * cos(2*pi*f_i * t + phi_i)

The model is linear in the amplitudes/phases once the nonlinear parameters
(frequencies f_i and damping rates g_i) are fixed. Using the identity

    A*cos(w t + phi) = a*cos(w t) + b*sin(w t),   a = A*cos(phi),  b = -A*sin(phi)

the linear part is solved by least squares (variable projection, "VarPro"),
so only the 2k nonlinear parameters [f_1..f_k, g_1..g_k] are optimised, e.g.
with scipy.optimize.least_squares(_varpro_resid, theta0, method="lm", ...).

Conversion from the linear coefficients back to (A, phi):
    A = hypot(a, b),   phi = arctan2(-b, a)

Common variables (used consistently across all functions)
---------------------------------------------------------
t       : ndarray, shape (N,)
          Sample times of the pulse response (e.g. seconds). Frequencies f
          are in 1/[t] (Hz if t is in seconds), damping rates g in 1/[t].
y       : ndarray, shape (N,)
          Measured pulse-response values at times t.
k       : int
          Number of damped-sinusoid components.
f       : sequence/ndarray, length k
          Oscillation frequencies f_i (cycles per unit time, not rad).
g       : sequence/ndarray, length k
          Damping (decay) rates g_i >= 0; envelope is exp(-g_i * t).
offset  : bool
          If True a constant (DC) term is included in the model.
"""

import numpy as np


def _design(t, f, g, offset):
    """Build the linear design matrix for fixed nonlinear parameters.

    Columns are, in order: an optional column of ones (offset), then for each
    component i the pair  exp(-g_i t) * cos(2 pi f_i t)  and
    exp(-g_i t) * sin(2 pi f_i t).

    Parameters
    ----------
    t : ndarray, shape (N,)
        Sample times.
    f : sequence, length k
        Frequencies f_i.
    g : sequence, length k
        Damping rates g_i.
    offset : bool
        Include a constant column if True.

    Returns
    -------
    X : ndarray, shape (N, 2*k + int(offset))
        Design matrix. Local variables: ``cols`` is the list of columns being
        assembled, ``fi``/``gi`` the i-th frequency/damping, ``env`` the decay
        envelope exp(-gi t), and ``w`` the phase argument 2*pi*fi*t (rad).
    """
    cols = [np.ones_like(t)] if offset else []
    for fi, gi in zip(f, g):
        env = np.exp(-gi * t)
        w = 2 * np.pi * fi * t
        cols += [env * np.cos(w), env * np.sin(w)]
    return np.column_stack(cols)


def _linear_solve(t, y, f, g, offset):
    """Solve for the linear coefficients at fixed (f, g) by least squares.

    Parameters
    ----------
    t, y, f, g, offset : see module docstring.

    Returns
    -------
    X : ndarray, shape (N, 2*k + int(offset))
        Design matrix from :func:`_design`.
    coef : ndarray, shape (2*k + int(offset),)
        Least-squares coefficients ordered like the columns of X:
        [c0 (if offset), a_1, b_1, ..., a_k, b_k], where a_i multiplies the
        cosine term and b_i the sine term (A_i = hypot(a_i, b_i),
        phi_i = arctan2(-b_i, a_i)). The other outputs of ``lstsq``
        (residuals, rank, singular values) are discarded.
    """
    X = _design(t, f, g, offset)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X, coef



@utils.register_results()
def _varpro_resid(theta, t, y, k, offset):
    """Variable-projection residual vector for use with a least-squares solver.

    Only the nonlinear parameters are optimised; the linear coefficients are
    re-solved exactly at every call.

    Parameters
    ----------
    theta : ndarray, shape (2*k,)
        Nonlinear parameters [f_1, ..., f_k, g_1, ..., g_k].
        Damping rates are passed through ``abs`` because the Levenberg-
        Marquardt method ("lm") does not support bounds, so the sign of g is
        irrelevant (g_i >= 0 is enforced this way).
    t, y : ndarray, shape (N,)
        Sample times and measured data.
    k : int
        Number of components (splits theta into frequencies and dampings).
    offset : bool
        Include a constant term in the linear model.

    Returns
    -------
    resid : ndarray, shape (N,)
        Model minus data, X @ coef - y, at the optimal linear coefficients.

    Side effects
    ------------
    Appends the sum of squared residuals to the global list ``results``.
    Local variables: ``f``/``g`` are the unpacked frequencies and damping
    rates, ``X``/``coef`` the design matrix and linear solution, ``res`` the
    list of squared residuals.
    """
    f, g = theta[:k], np.abs(theta[k:])  # abs: lm has no bounds
    X, coef = _linear_solve(t, y, f, g, offset)
    return X @ coef - y


def damped_sum(t, params, k, offset=True):
    """Evaluate the sum-of-damped-cosines model in amplitude/phase form.

    y(t) = c0 + sum_i A_i * exp(-|g_i| t) * cos(2 pi f_i t + phi_i)

    Parameters
    ----------
    t : ndarray, shape (N,)
        Times at which to evaluate the model.
    params : sequence of float
        Flat parameter vector
        [c0 (only if offset), A_1, f_1, g_1, phi_1, ..., A_k, f_k, g_k, phi_k]
        where c0 is the constant offset, A_i the amplitude, f_i the frequency
        (cycles per unit time), g_i the damping rate (abs() is applied, so the
        sign is ignored) and phi_i the phase in radians.
    k : int
        Number of damped components.
    offset : bool, default True
        Whether params starts with the offset c0.

    Returns
    -------
    y : ndarray, shape (N,)
        Model values. Local variables: ``p`` is params as a float array,
        ``q`` the component part of p (offset stripped), and ``A, f, g, phi``
        the parameters of the i-th component.
    """
    p = np.asarray(params, float)
    y = np.full_like(t, p[0]) if offset else np.zeros_like(t)
    q = p[1:] if offset else p
    for i in range(k):
        A, f, g, phi = q[4 * i: 4 * i + 4]
        y += A * np.exp(-abs(g) * t) * np.cos(2 * np.pi * f * t + phi)
    return y


def _full_resid(p, t, y, k, offset):
    """Residual of the full model for a joint fit of all parameters.

    Parameters
    ----------
    p : ndarray
        Full parameter vector in the layout expected by :func:`damped_sum`.
    t, y, k, offset : see module docstring / :func:`damped_sum`.

    Returns
    -------
    resid : ndarray, shape (N,)
        damped_sum(t, p, k, offset) - y.
    """
    return damped_sum(t, p, k, offset) - y


class Processor:
    def __init__(self, function, method=0):
        self.name: str = function.name
        self.function: Function = function
        self.method: int = method
        self.output: str = ""

        self.fft_peaks: List[Dict[str, float]] = []
        self.optimized_peaks: List[Dict[str, float]] = []

        self.result_params: List[dict] = []

        self.solve_fft_peaks()

    def run(self):
        if self.method == 0:
            print("\nprocessor:Running method 0: fft mapping")
            print(f"Function {self.function.name}")
            self.fft_mapping()

        elif self.method == 1:
            print("\nprocessor: Running method 1: time mapping")
            print(f"Function {self.function.name}")
            self.fit_damped_sinusoids()

        else:
            print(f"Method {self.method} has not been implemented")

        if utils.params:
            plt.plot(utils.params[1])
            plt.show()

    def solve_fft_peaks(self):
        fft_values = self.function.fft_set

        peaks, _ = find_peaks(fft_values, distance=7, prominence=0.1)
        top_peaks = peaks[argsort(fft_values[peaks])[-3:]]
        estimated_f0s = sorted(self.function.fft_freq[top_peaks])
        raw_fft_peaks = [
            {"freq": freq, "magn": fft_values[self.function.fft_freq == freq][0]} \
            for freq in estimated_f0s
        ]
        freq_set = set()
        for peak in raw_fft_peaks:
            if peak["freq"] not in freq_set:
                self.fft_peaks.append(peak)
                freq_set.add(peak["freq"])

        if len(self.fft_peaks) == 0:
            print("There is no peak in the spectrum")
            return
        else:
            print("\nprocessor: FFT PEAKS:")
            print(f"Function {self.function.name}")
            for idx, peak in enumerate(self.fft_peaks):
                print(
                f"Mód {idx + 1}: Frekvence = {peak['freq']:6.3f} Hz | Amplituda = {peak['magn']:5.2f}")


    def fft_mapping(self):
        initial_guess = []  # [c, A1, f1, g1, ...]
        fft_values = self.function.fft_set

        # Initial guess of noise
        initial_guess.append(mean(fft_values))
        for peak in self.fft_peaks:
            # A, f, gamma
            initial_guess.append(peak["magn"])
            initial_guess.append(peak["freq"])
            initial_guess.append((self.function.fft_freq[1] - self.function.fft_freq[0]) * 5)

        # RUN LM algorithm
        # max_freq_interest = 800
        # mask = self.freqs <= max_freq_interest

        result = least_squares(
            residuals,
            initial_guess,
            method='lm',
            args=(self.function.fft_freq, fft_values),
        )
        optimized_params = result.x
        print(f"--- ANALYSIS: NALEZENÉ VLASTNÍ FREKVENCE A AMPLITUDY ---")
        # print(f"Hladina šumu (c): {optimized_params[0]:.4f}")
        for idx, i in enumerate(range(1, len(optimized_params), 3)):
            amp = optimized_params[i]
            freq = optimized_params[i + 1]
            gamma = optimized_params[i + 2]
            # self.lm_peaks[axis][freq] = amp
            print(
                f"Mód {idx + 1}: Frekvence = {freq:6.3f} Hz | Amplituda = {amp:5.4f} | Tlumení (gamma) = {gamma:5.4f}")



    def fit_damped_sinusoids(self, g0=None, t0=0, offset=False, polish=True):
        """
        t, y : time axis [s] and signal
        f0   : starting frequencies [Hz] (length k)
        g0   : starting decay rates [1/s] (scalar or length k); default 5/T
        t0   : time of pulse start (samples before t0 should be cut beforehand)

        Returns dict with fitted values, 1-sigma errors, and the scipy result.
        Peaks are returned sorted by frequency.
        """
        # t = np.asarray(t, float) - t0
        # y = np.asarray(y, float)
        t = self.function.time_set - t0
        y = self.function.data_set
        f0 = np.atleast_1d(np.asarray([peak["freq"] for peak in self.fft_peaks], float))
        k = len(f0)
        T = t[-1] - t[0]
        g0 = np.full(k, 5.0 / T) if g0 is None else np.broadcast_to(np.asarray(g0, float), (k,)).copy()
        if len(t) < 4 * k + 1:
            raise ValueError("not enough samples for this many peaks")

        # --- stage 1: variable projection over (f_i, g_i) ---
        res1 = least_squares(_varpro_resid, np.r_[f0, g0], args=(t, y, k, offset),
                             method="lm", x_scale="jac", ftol=1e-15)
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
                                x_scale="jac", ftol=1e-15)
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

            self.result_params.append(
                dict(A=abs(Ai), f=fi, gamma=abs(gi), phi=fi*180/np.pi)
            )
            print(
                f"Mód {i + 1}: Frekvence = {fi:6.3f} Hz | Amplituda = {Ai:5.4f} | Tlumení (gamma) = {gi:5.4f} | Phi = {pi_*180/np.pi}")

        peaks.sort(key=lambda d: d["f"])
        return dict(offset=p[0] if offset else 0.0, peaks=peaks, params=p, cov=cov,
                    rms=float(np.sqrt(np.mean(res.fun ** 2))), result=res)