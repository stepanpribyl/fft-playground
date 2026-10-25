"""Fit k Lorentzian peaks to an FFT magnitude spectrum with scipy.optimize.least_squares (method='lm').

Model:  y(f) = offset + sum_i A_i * g_i^2 / ((f - f0_i)^2 + g_i^2)
    A_i  : peak height
    f0_i : centre frequency
    g_i  : half width at half maximum (HWHM)

Note: method='lm' (MINPACK Levenberg-Marquardt) does NOT support bounds, so
positivity of the widths is enforced inside the model via abs(g).
It also requires  number_of_data_points >= number_of_parameters.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths


def lorentz_sum(f, params, k):
    """params = [offset, A_1, f0_1, g_1, ..., A_k, f0_k, g_k]"""
    y = np.full_like(f, params[0], dtype=float)
    for i in range(k):
        A, f0, g = params[1 + 3 * i: 4 + 3 * i]
        g = abs(g)  # keep width positive without bounds
        y += A * g**2 / ((f - f0) ** 2 + g**2)
    return y


def residuals(params, f, y, k):
    res = lorentz_sum(f, params, k) - y
    print(f"Sum of residuals: {sum(res):.4e}")
    return res


def initial_guess(f, y, k):
    """Pick the k highest peaks as starting points."""
    offset = np.median(y)
    peaks, _ = find_peaks(y)
    if len(peaks) < k:  # fall back: pad with the highest remaining bins
        extra = np.setdiff1d(np.argsort(y)[::-1], peaks)[: k - len(peaks)]
        peaks = np.concatenate([peaks, extra]).astype(int)
    peaks = peaks[np.argsort(y[peaks])[::-1][:k]]
    df = f[1] - f[0]
    widths = peak_widths(y, peaks, rel_height=0.5)[0] * df / 2  # FWHM -> HWHM
    widths = np.where(widths > 0, widths, 2 * df)
    p0 = [offset]
    for idx, w in zip(peaks, widths):
        p0 += [y[idx] - offset, f[idx], w]
    return np.array(p0, dtype=float)


def fit_lorentzians(f, y, k, p0=None):
    """Fit k Lorentzians to spectrum y(f). Returns (params, cov, result)."""
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    if p0 is None:
        p0 = initial_guess(f, y, k)
    if len(f) < len(p0):
        raise ValueError("lm needs at least as many data points as parameters")

    res = least_squares(residuals, p0, args=(f, y, k), method="lm",
                        x_scale="jac")

    # Covariance estimate from the Jacobian
    J = res.jac
    dof = max(len(y) - len(res.x), 1)
    s2 = 2 * res.cost / dof  # cost = 0.5 * sum(r^2)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x), len(res.x)), np.nan)

    params = res.x.copy()
    for i in range(k):
        params[3 + 3 * i] = abs(params[3 + 3 * i])  # report positive widths
    return params, cov, res


def fft_spectrum(signal, fs):
    """One-sided FFT magnitude spectrum."""
    n = len(signal)
    y = np.abs(np.fft.rfft(signal - np.mean(signal))) * 2 / n
    f = np.fft.rfftfreq(n, d=1 / fs)
    return f, y


if __name__ == "__main__":
    # Demo: decaying sinusoids give Lorentzian-shaped peaks in the FFT.
    rng = np.random.default_rng(0)
    fs, T = 1000.0, 20.0
    t = np.arange(0, T, 1 / fs)
    sig = (1.0 * np.exp(-t / 2.0) * np.cos(2 * np.pi * 50 * t)
           + 0.6 * np.exp(-t / 1.0) * np.cos(2 * np.pi * 120 * t)
           + 0.01 * rng.standard_normal(t.size))

    f, y = fft_spectrum(sig, fs)
    mask = f < 200  # fit region of interest
    f, y = f[mask], y[mask]

    k = 2
    params, cov, res = fit_lorentzians(f, y, k)
    err = np.sqrt(np.diag(cov))

    print(f"success={res.success}  nfev={res.nfev}  rms={np.sqrt(np.mean(res.fun**2)):.3e}")
    print(f"offset = {params[0]:.4g} +/- {err[0]:.2g}")
    for i in range(k):
        A, f0, g = params[1 + 3 * i: 4 + 3 * i]
        eA, ef0, eg = err[1 + 3 * i: 4 + 3 * i]
        print(f"peak {i+1}: A={A:.4g}±{eA:.2g}  f0={f0:.4f}±{ef0:.2g}  HWHM={g:.4g}±{eg:.2g}")

    try:
        import matplotlib.pyplot as plt
        plt.plot(f, y, label="FFT")
        plt.plot(f, lorentz_sum(f, params, k), "r", label=f"{k} Lorentzians")
        plt.xlabel("Frequency"); plt.ylabel("Magnitude"); plt.legend(); plt.show()
    except ImportError:
        pass