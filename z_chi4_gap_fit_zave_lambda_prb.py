#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot z_chi4 and z_gap with error bars, fit only the averaged value z_ave,
and add a theoretical guide line z_lambda = ln(lambda_1)/ln(lambda_2).

Here lambda_2 is chosen as the Perron-Frobenius eigenvalue of the
generalized Fibonacci substitution matrix

        M = [[n, m],
             [1, 0]],

namely

        lambda_2 = lambda_L = (n + sqrt(n^2 + 4m)) / 2.

The added guide line uses

        lambda_1 = n^p,

so that

        z_lambda = ln(lambda_1) / ln(lambda_2).

The default p = 3 tests the possible n^3-type spectral scaling.
No fitting expression is displayed inside the figure.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================
n = np.arange(1, 11, dtype=float)

z_chi4 = np.array([
    2.375, 2.431, 2.574, 2.8051, 3.1248,
    3.4697, 3.8900, 4.2230, 4.5666, 4.8945
])

err_chi4 = np.array([
    0.004, 0.001, 0.002, 0.0042, 0.0171,
    0.0313, 0.0410, 0.0671, 0.0777, 0.0839
])

z_gap = np.array([
    2.361, 2.426, 2.575, 2.8043, 3.1131,
    3.4403, 3.7763, 4.1075, 4.4309, 4.7475
])

err_gap = np.array([
    0.004, 0.001, 0.0000, 0.0050, 0.0015,
    0.0011, 0.0001, 0.0001, 0.0008, 0.0000
])

# ============================================================
# Options
# ============================================================
FIT_FROM_N = 4
USE_WEIGHTED_FIT = False
ERR_FLOOR = 1e-4
OUT_PREFIX = "z_chi4_gap_fit_zave_lambda_prb"

# Parameters for the lambda-ratio line
m_fib = 1.0
P_LAMBDA1 = 3.0
PLOT_LAMBDA_RATIO_LINE = True

# ============================================================
# Average z and linear fit
# ============================================================
z_ave = 0.5 * (z_chi4 + z_gap)
err_ave_stat = 0.5 * np.sqrt(err_chi4**2 + err_gap**2)
err_ave_method = 0.5 * np.abs(z_chi4 - z_gap)
err_ave = np.sqrt(err_ave_stat**2 + err_ave_method**2)

mask_fit = n >= FIT_FROM_N
x_fit = n[mask_fit]
y_fit = z_ave[mask_fit]

if USE_WEIGHTED_FIT:
    sigma_fit = np.maximum(err_ave[mask_fit], ERR_FLOOR)
    coef, cov = np.polyfit(x_fit, y_fit, 1, w=1.0 / sigma_fit, cov=True)
    fit_type = "weighted"
else:
    coef, cov = np.polyfit(x_fit, y_fit, 1, cov=True)
    fit_type = "unweighted"

a_fit, b_fit = coef
a_fit_err, b_fit_err = np.sqrt(np.diag(cov))

x_line = np.linspace(FIT_FROM_N, 10.0, 400)
y_line = a_fit * x_line + b_fit

# ============================================================
# Lambda-ratio guide line
# ============================================================
# Length inflation factor from the substitution matrix
lambda_2 = 0.5 * (x_line + np.sqrt(x_line**2 + 4.0 * m_fib))

# Spectral scaling ansatz: lambda_1 = n^p
lambda_1 = x_line**P_LAMBDA1

z_lambda = np.log(lambda_1) / np.log(lambda_2)

# ============================================================
# Print fit information in terminal only
# ============================================================
print("Averaged z values:")
print("  n     z_chi4     err_chi4     z_gap      err_gap      z_ave")
for ni, zc, ec, zg, eg, za in zip(n, z_chi4, err_chi4, z_gap, err_gap, z_ave):
    print(f"  {int(ni):2d}   {zc:8.4f}   {ec:8.4f}   {zg:8.4f}   {eg:8.4f}   {za:8.4f}")

print()
print(f"Linear fit to z_ave for n >= {FIT_FROM_N} ({fit_type}):")
print(f"  slope     = {a_fit:.6f} ± {a_fit_err:.6f}")
print(f"  intercept = {b_fit:.6f} ± {b_fit_err:.6f}")

print()
print("Lambda-ratio guide line:")
print(f"  m_fib = {m_fib:.1f}")
print(f"  lambda_2 = (n + sqrt(n^2 + 4m)) / 2")
print(f"  lambda_1 = n^{P_LAMBDA1:.1f}")
print("  z_lambda = ln(lambda_1) / ln(lambda_2)")

# ============================================================
# PRB-like plot style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "axes.linewidth": 1.2,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,

    "legend.frameon": False,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(3.45, 2.70), dpi=300)

# z_chi4 with error bars
ax.errorbar(
    n,
    z_chi4,
    yerr=err_chi4,
    fmt="o",
    linestyle="none",
    color="crimson",
    markerfacecolor="white",
    markeredgecolor="crimson",
    markeredgewidth=1.2,
    markersize=3.2,
    elinewidth=0.9,
    capsize=2.5,
    label=r"$z_{\chi_4}$"
)

# z_gap with error bars
ax.errorbar(
    n,
    z_gap,
    yerr=err_gap,
    fmt="x",
    linestyle="none",
    color="royalblue",
    markeredgewidth=1.2,
    markersize=3.5,
    elinewidth=0.9,
    capsize=2.5,
    label=r"$z_{\rm gap}$"
)

# Linear fit line obtained from z_ave only
ax.plot(
    x_line,
    y_line,
    linestyle="--",
    linewidth=1.3,
    color="black",
    label=r"linear fit"
)

# Lambda-ratio guide line
if PLOT_LAMBDA_RATIO_LINE:
    ax.plot(
        x_line,
        z_lambda,
        linestyle="-.",
        linewidth=1.3,
        color="darkgreen",
        label=r"$\ln\lambda_1/\ln\lambda_2$"
    )

# ============================================================
# Axes formatting
# ============================================================
ax.set_xlim(0.6, 10.4)
ax.set_ylim(2.2, 5.1)

ax.set_xticks(np.arange(1, 11, 1))
ax.set_yticks(np.arange(2.5, 5.1, 0.5))
ax.minorticks_on()

ax.set_xlabel(r"$n$", fontsize=14)
ax.set_ylabel(r"$z$", fontsize=14)

ax.tick_params(axis="both", which="major", labelsize=11)

ax.legend(
    loc="upper left",
    fontsize=8.6,
    handlelength=2.2
)

# ============================================================
# Save
# ============================================================
pdf_name = f"{OUT_PREFIX}.pdf"
png_name = f"{OUT_PREFIX}.png"

plt.savefig(pdf_name)
plt.savefig(png_name, dpi=600)
plt.close(fig)

print()
print(f"Saved: {os.path.abspath(pdf_name)}")
print(f"Saved: {os.path.abspath(png_name)}")