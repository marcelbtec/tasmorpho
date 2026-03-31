#!/usr/bin/env python3
"""Reduced rank-one latent-threshold fit to published planarian phenotype counts.

Fits a probit-threshold model with latent variance sigma=1 and immediate
threshold theta_imm=0 fixed.  The four free parameters are the latent means
for 8-OH, nigericin, and monensin, plus a shared challenge threshold theta_ch.
The 8-OH immediate and challenge counts are primary fit targets; nigericin
and monensin immediate counts set their latent positions; and the nigericin
and monensin challenge penetrances are held-out predictions.

Produces closed-form estimates, an optional Nelder-Mead cross-check (when
SciPy is available), the observed Hessian, Wald standard errors,
delta-method probability intervals, plots, a text report, and a LaTeX
snippet.

No BETSE dependency — this script is fully standalone.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize  # type: ignore
except Exception:  # pragma: no cover
    minimize = None


ND = NormalDist()


@dataclass(frozen=True)
class CountDatum:
    """A single reconstructed phenotype count from the source papers."""

    label: str
    successes: int
    trials: int
    role: str
    description: str


# Reconstructed counts used in the manuscript's reduced proof-of-principle fit.
COUNTS: Tuple[CountDatum, ...] = (
    CountDatum(
        label="8OH_immediate_DH",
        successes=143,
        trials=573,
        role="primary_fit_target",
        description="Immediate double-headed outcomes after 8-OH",
    ),
    CountDatum(
        label="8OH_challenge_DH_among_immediate_SH",
        successes=36,
        trials=155,
        role="primary_fit_target",
        description="Challenge double-headed outcomes after water re-cut of 8-OH immediate SH worms",
    ),
    CountDatum(
        label="nigericin_immediate_DH",
        successes=17,
        trials=132,
        role="condition_setting",
        description="Immediate double-headed outcomes after nigericin",
    ),
    CountDatum(
        label="monensin_immediate_DH",
        successes=11,
        trials=89,
        role="condition_setting",
        description="Immediate double-headed outcomes after monensin",
    ),
)

CONSISTENCY_CHECK = {
    "label": "8OH_challenge_DH_among_immediate_DH",
    "successes": 100,
    "trials": 100,
    "description": "Consistency check only; not used in reduced fit",
}


def phi_cdf(x: float) -> float:
    """Standard normal CDF."""
    return ND.cdf(float(x))


def phi_inv(p: float) -> float:
    """Standard normal quantile function, clamped away from 0 and 1."""
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps)
    return ND.inv_cdf(p)


def clip_prob(p: float) -> float:
    """Clamp a probability to (eps, 1-eps) for numerical safety."""
    return min(max(float(p), 1e-12), 1.0 - 1e-12)


def p_immediate(mu: float) -> float:
    """Immediate DH probability: Phi(mu) with sigma=1 and theta_imm=0."""
    return phi_cdf(mu)


def p_challenge_given_sh(mu: float, theta_ch: float) -> float:
    """Challenge DH probability among immediate SH survivors."""
    denom = phi_cdf(-mu)
    if denom <= 0.0:
        return 1.0
    num = phi_cdf(-mu) - phi_cdf(theta_ch - mu)
    return clip_prob(num / denom)


def negative_log_likelihood(params: np.ndarray) -> float:
    """Binomial negative log-likelihood for the four-parameter reduced model."""
    mu_8oh, mu_nig, mu_mon, theta_ch = map(float, params)

    probs = {
        "8OH_immediate_DH": p_immediate(mu_8oh),
        "nigericin_immediate_DH": p_immediate(mu_nig),
        "monensin_immediate_DH": p_immediate(mu_mon),
        "8OH_challenge_DH_among_immediate_SH": p_challenge_given_sh(mu_8oh, theta_ch),
    }

    total = 0.0
    for datum in COUNTS:
        p = clip_prob(probs[datum.label])
        y, n = datum.successes, datum.trials
        total -= y * math.log(p) + (n - y) * math.log(1.0 - p)
    return total


def closed_form_estimate() -> np.ndarray:
    """Compute the closed-form MLE for (mu_8oh, mu_nig, mu_mon, theta_ch)."""
    count_map = {d.label: d for d in COUNTS}

    mu_8oh = phi_inv(count_map["8OH_immediate_DH"].successes / count_map["8OH_immediate_DH"].trials)
    mu_nig = phi_inv(count_map["nigericin_immediate_DH"].successes / count_map["nigericin_immediate_DH"].trials)
    mu_mon = phi_inv(count_map["monensin_immediate_DH"].successes / count_map["monensin_immediate_DH"].trials)

    q_8oh = (
        count_map["8OH_challenge_DH_among_immediate_SH"].successes
        / count_map["8OH_challenge_DH_among_immediate_SH"].trials
    )
    theta_ch = mu_8oh + phi_inv(phi_cdf(-mu_8oh) * (1.0 - q_8oh))
    return np.array([mu_8oh, mu_nig, mu_mon, theta_ch], dtype=float)


def maybe_nelder_mead(start: np.ndarray) -> Dict[str, object]:
    """Run Nelder-Mead from a perturbed start if SciPy is available."""
    if minimize is None:
        return {
            "available": False,
            "success": False,
            "message": "scipy.optimize.minimize not available",
            "x": start.tolist(),
            "fun": float(negative_log_likelihood(start)),
        }

    result = minimize(
        negative_log_likelihood,
        x0=np.asarray(start, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000},
    )
    return {
        "available": True,
        "success": bool(result.success),
        "message": str(result.message),
        "x": np.asarray(result.x, dtype=float).tolist(),
        "fun": float(result.fun),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
    }


def finite_difference_hessian(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    step_scale: float = 1e-5,
) -> np.ndarray:
    """Symmetric finite-difference Hessian of *func* at *x*."""
    x = np.asarray(x, dtype=float)
    n = x.size
    steps = np.maximum(step_scale, step_scale * np.maximum(1.0, np.abs(x)))
    H = np.zeros((n, n), dtype=float)
    fx = func(x)

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = steps[i]
        f_pp = func(x + ei)
        f_mm = func(x - ei)
        H[i, i] = (f_pp - 2.0 * fx + f_mm) / (steps[i] ** 2)

        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = steps[j]
            f_pp = func(x + ei + ej)
            f_pm = func(x + ei - ej)
            f_mp = func(x - ei + ej)
            f_mm = func(x - ei - ej)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            H[j, i] = H[i, j]

    return H


def finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    step_scale: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference gradient of *func* at *x*."""
    x = np.asarray(x, dtype=float)
    n = x.size
    steps = np.maximum(step_scale, step_scale * np.maximum(1.0, np.abs(x)))
    g = np.zeros(n, dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = steps[i]
        g[i] = (func(x + ei) - func(x - ei)) / (2.0 * steps[i])
    return g


def delta_method_interval(
    scalar_func: Callable[[np.ndarray], float],
    xhat: np.ndarray,
    cov: np.ndarray,
    z_score: float = 1.96,
) -> Tuple[float, float, float]:
    """Approximate CI for a scalar function of the parameters via the delta method."""
    estimate = float(scalar_func(xhat))
    grad = finite_difference_gradient(scalar_func, xhat)
    variance = float(grad @ cov @ grad)
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    lo = max(0.0, estimate - z_score * se)
    hi = min(1.0, estimate + z_score * se)
    return estimate, se, lo, hi


def build_count_table() -> pd.DataFrame:
    """Assemble the reconstructed count table including the consistency check."""
    rows = []
    for datum in COUNTS:
        rows.append(
            {
                "label": datum.label,
                "successes": datum.successes,
                "trials": datum.trials,
                "proportion": datum.successes / datum.trials,
                "role": datum.role,
                "description": datum.description,
            }
        )
    rows.append(
        {
            "label": CONSISTENCY_CHECK["label"],
            "successes": CONSISTENCY_CHECK["successes"],
            "trials": CONSISTENCY_CHECK["trials"],
            "proportion": CONSISTENCY_CHECK["successes"] / CONSISTENCY_CHECK["trials"],
            "role": "consistency_check_only",
            "description": CONSISTENCY_CHECK["description"],
        }
    )
    return pd.DataFrame(rows)


def build_parameter_table(xhat: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    """Build a DataFrame of parameter estimates with Wald standard errors and 95% CIs."""
    names = ["mu_8OH", "mu_nigericin", "mu_monensin", "theta_ch"]
    labels = {
        "mu_8OH": "latent write mean for 8-OH",
        "mu_nigericin": "latent mean implied by nigericin immediate DH penetrance alone",
        "mu_monensin": "latent mean implied by monensin immediate DH penetrance alone",
        "theta_ch": "challenge threshold; immediate threshold fixed at 0",
    }
    rows = []
    ses = np.sqrt(np.diag(cov))
    for name, est, se in zip(names, xhat, ses):
        rows.append(
            {
                "parameter": name,
                "estimate": float(est),
                "std_error": float(se),
                "lower_95": float(est - 1.96 * se),
                "upper_95": float(est + 1.96 * se),
                "interpretation": labels[name],
            }
        )
    return pd.DataFrame(rows)


def build_probability_table(xhat: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    """Build a DataFrame of fitted and predicted probabilities with delta-method CIs."""
    funcs: Dict[str, Callable[[np.ndarray], float]] = {
        "p_8OH_immediate": lambda x: p_immediate(float(x[0])),
        "p_8OH_challenge_given_SH": lambda x: p_challenge_given_sh(float(x[0]), float(x[3])),
        "p_nigericin_immediate": lambda x: p_immediate(float(x[1])),
        "p_monensin_immediate": lambda x: p_immediate(float(x[2])),
        "p_nigericin_challenge_given_SH": lambda x: p_challenge_given_sh(float(x[1]), float(x[3])),
        "p_monensin_challenge_given_SH": lambda x: p_challenge_given_sh(float(x[2]), float(x[3])),
    }
    meta = {
        "p_8OH_immediate": ("fitted_target", "Reproduces 143/573 immediate DH under 8-OH"),
        "p_8OH_challenge_given_SH": (
            "fitted_target",
            "Reproduces 36/155 challenge DH among 8-OH immediate SH worms",
        ),
        "p_nigericin_immediate": (
            "condition_setting",
            "Immediate DH penetrance used only to place nigericin on the shared latent axis",
        ),
        "p_monensin_immediate": (
            "condition_setting",
            "Immediate DH penetrance used only to place monensin on the shared latent axis",
        ),
        "p_nigericin_challenge_given_SH": (
            "held_out_prediction",
            "Predicted challenge DH fraction for nigericin immediate SH survivors",
        ),
        "p_monensin_challenge_given_SH": (
            "held_out_prediction",
            "Predicted challenge DH fraction for monensin immediate SH survivors",
        ),
    }

    rows = []
    for name, func in funcs.items():
        est, se, lo, hi = delta_method_interval(func, xhat, cov)
        rows.append(
            {
                "quantity": name,
                "estimate": est,
                "std_error_delta": se,
                "lower_95": lo,
                "upper_95": hi,
                "status": meta[name][0],
                "interpretation": meta[name][1],
            }
        )
    return pd.DataFrame(rows)


def make_probability_plot(prob_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of fitted and held-out probabilities with error bars."""
    order = [
        "p_8OH_immediate",
        "p_8OH_challenge_given_SH",
        "p_nigericin_immediate",
        "p_monensin_immediate",
        "p_nigericin_challenge_given_SH",
        "p_monensin_challenge_given_SH",
    ]
    plot_df = prob_df.set_index("quantity").loc[order].copy()
    labels = {
        "p_8OH_immediate": "8-OH\nimm",
        "p_8OH_challenge_given_SH": "8-OH\nch|SH",
        "p_nigericin_immediate": "Nig\nimm",
        "p_monensin_immediate": "Mon\nimm",
        "p_nigericin_challenge_given_SH": "Nig\nch|SH",
        "p_monensin_challenge_given_SH": "Mon\nch|SH",
    }

    x = np.arange(plot_df.shape[0])
    y = plot_df["estimate"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - plot_df["lower_95"].to_numpy(dtype=float),
            plot_df["upper_95"].to_numpy(dtype=float) - y,
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)
    plt.xticks(x, [labels[k] for k in plot_df.index])
    plt.ylim(0.0, 0.35)
    plt.ylabel("Probability")
    plt.title("Reduced count-based fit and held-out transfer predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_latent_axis_plot(xhat: np.ndarray, output_path: Path) -> None:
    """Plot the latent axis with thresholds, cryptic band, and treatment positions."""
    mu_8oh, mu_nig, mu_mon, theta_ch = map(float, xhat)
    xs = np.linspace(-2.5, 0.5, 500)
    density = np.exp(-0.5 * xs**2) / math.sqrt(2.0 * math.pi)

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, density)
    plt.axvline(theta_ch, linestyle="--", linewidth=1.5)
    plt.axvline(0.0, linestyle="--", linewidth=1.5)
    plt.axvspan(theta_ch, 0.0, alpha=0.2)

    means = {"8-OH": mu_8oh, "Nig": mu_nig, "Mon": mu_mon}
    y_level = density.max() * 0.8
    for label, mu in means.items():
        plt.scatter([mu], [y_level], s=50)
        plt.text(mu, y_level, f" {label}", va="center", fontsize=9)

    plt.xlabel("Latent coordinate")
    plt.ylabel("Reference density")
    plt.title("Reduced latent axis with immediate and challenge thresholds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    xhat: np.ndarray,
    nm_result: Dict[str, object],
    hessian: np.ndarray,
    cov: np.ndarray,
    param_df: pd.DataFrame,
    prob_df: pd.DataFrame,
) -> None:
    """Write a human-readable text report of the full fit."""
    report_lines = [
        "Reduced rank-one latent-threshold fit to published planarian counts",
        "",
        "Counts used in the manuscript:",
        "  8-OH immediate DH: 143 / 573",
        "  8-OH challenge DH among immediate SH after water re-cut: 36 / 155",
        "  Nigericin immediate DH: 17 / 132",
        "  Monensin immediate DH: 11 / 89",
        "",
        "Closed-form estimates:",
        f"  mu_8OH       = {xhat[0]:.9f}",
        f"  mu_nigericin = {xhat[1]:.9f}",
        f"  mu_monensin  = {xhat[2]:.9f}",
        f"  theta_ch     = {xhat[3]:.9f}",
        "",
        "Numerical optimization check (Nelder-Mead if scipy is available):",
        json.dumps(nm_result, indent=2),
        "",
        "Observed Hessian of the negative log-likelihood at the optimum:",
        np.array2string(hessian, precision=9, suppress_small=False),
        "",
        "Approximate covariance matrix (inverse Hessian):",
        np.array2string(cov, precision=9, suppress_small=False),
        "",
        "Parameter table:",
        param_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        "",
        "Probability table:",
        prob_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        "",
        "Interpretation:",
        "This script reproduces only the reduced count-based layer from the manuscript. It does not",
        "run BETSE or the semimechanistic bridge. The held-out nigericin and monensin challenge",
        "probabilities are generated solely from the shared latent thresholds fixed by 8-OH plus",
        "their immediate DH penetrances.",
    ]
    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def write_latex_snippet(output_path: Path, xhat: np.ndarray, prob_df: pd.DataFrame) -> None:
    """Write a LaTeX paragraph with the fitted parameters for the manuscript."""
    lookup = prob_df.set_index("quantity")
    snippet = rf"""
Using the 8-OH challenge data, the reduced count-based fit gives
\begin{{equation}}
\hat\mu_{{8\mathrm{{OH}}}}={xhat[0]:.3f},
\qquad
\hat\theta_{{\mathrm{{ch}}}}={xhat[3]:.3f},
\end{{equation}}
with the immediate threshold fixed at $0$ and latent variance fixed at $1$. The immediate
depolarization datasets imply
\begin{{equation}}
\hat\mu_{{\mathrm{{Nig}}}}={xhat[1]:.3f},
\qquad
\hat\mu_{{\mathrm{{Mon}}}}={xhat[2]:.3f},
\end{{equation}}
and therefore the held-out challenge predictions
\begin{{equation}}
\hat p_{{\mathrm{{Nig}}}}^{{\mathrm{{ch|SH}}}}={lookup.loc['p_nigericin_challenge_given_SH', 'estimate']:.3f},
\qquad
\hat p_{{\mathrm{{Mon}}}}^{{\mathrm{{ch|SH}}}}={lookup.loc['p_monensin_challenge_given_SH', 'estimate']:.3f}.
\end{{equation}}
These are out-of-sample predictions under the reduced model: no nigericin or monensin challenge
data enter the calibration.
""".strip() + "\n"
    output_path.write_text(snippet, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce the manuscript's reduced count-based latent-threshold fit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_reduced_count_fit",
        help="Directory to receive CSV, report, and plot outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full reduced-fit pipeline: estimate, check, tabulate, plot, report."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    xhat = closed_form_estimate()
    nm_result = maybe_nelder_mead(start=xhat + np.array([0.05, -0.03, 0.02, -0.04], dtype=float))

    hessian = finite_difference_hessian(negative_log_likelihood, xhat)
    cov = np.linalg.inv(hessian)

    count_df = build_count_table()
    param_df = build_parameter_table(xhat, cov)
    prob_df = build_probability_table(xhat, cov)

    count_df.to_csv(output_dir / "reconstructed_count_table.csv", index=False)
    param_df.to_csv(output_dir / "reduced_fit_parameters.csv", index=False)
    prob_df.to_csv(output_dir / "reduced_fit_probabilities.csv", index=False)
    pd.DataFrame(hessian, columns=param_df["parameter"], index=param_df["parameter"]).to_csv(
        output_dir / "reduced_fit_hessian.csv"
    )
    pd.DataFrame(cov, columns=param_df["parameter"], index=param_df["parameter"]).to_csv(
        output_dir / "reduced_fit_covariance.csv"
    )

    make_probability_plot(prob_df, output_dir / "reduced_fit_probabilities.png")
    make_latent_axis_plot(xhat, output_dir / "reduced_fit_latent_axis.png")
    write_report(output_dir / "reduced_fit_report.txt", xhat, nm_result, hessian, cov, param_df, prob_df)
    write_latex_snippet(output_dir / "reduced_fit_snippet.tex", xhat, prob_df)

    optimizer_df = pd.DataFrame(
        [
            {
                "method": "closed_form",
                "mu_8OH": xhat[0],
                "mu_nigericin": xhat[1],
                "mu_monensin": xhat[2],
                "theta_ch": xhat[3],
                "negative_log_likelihood": negative_log_likelihood(xhat),
            },
            {
                "method": "nelder_mead_check",
                "mu_8OH": float(nm_result["x"][0]),
                "mu_nigericin": float(nm_result["x"][1]),
                "mu_monensin": float(nm_result["x"][2]),
                "theta_ch": float(nm_result["x"][3]),
                "negative_log_likelihood": float(nm_result["fun"]),
            },
        ]
    )
    optimizer_df.to_csv(output_dir / "optimizer_check.csv", index=False)

    print("Reduced count-based fit completed.")
    print(f"Outputs written to: {output_dir}")
    print(param_df.to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print()
    print(prob_df.to_string(index=False, float_format=lambda x: f'{x:.6g}'))


if __name__ == "__main__":
    main()
