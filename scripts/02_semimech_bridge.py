"""Enriched semimechanistic bridge from BETSE features to phenotype prediction.

Loads the BETSE simulation outputs produced by ``01_betse_local_grounding.py``,
builds an enriched feature bank of endpoint and early-window summaries for each
perturbation case, and fits a linear semimechanistic readout

    mu_k = b + w^T h_k

that maps simulator-derived features to the same latent phenotype layer used in
the reduced count-based fit.  Feature selection is by exhaustive search over
1–3-feature subsets ranked by leave-one-treated-out prediction error under a
soft untreated-control anchor.

Outputs include feature tables, model-search rankings, all-case predictions,
cross-validation diagnostics, anchor-sensitivity sweeps, plots, a text report,
and a LaTeX snippet for the manuscript.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple

import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SIM_DIR = ROOT / 'betse_tas_example_run' / 'demo' / 'SIMS' / 'minimal'

CASES = ['control', 'na', 'k', 'gj', 'na_k', 'na_gj', 'k_gj']
LABEL_TO_CASE = {
    'control': 'control',
    '8OH': 'gj',
    'nigericin': 'na',
    'monensin': 'na_k',
}
IMMEDIATE_TARGETS = {
    '8OH': 143 / 573,
    'nigericin': 17 / 132,
    'monensin': 11 / 89,
}
P_CONTROL_DEFAULT = 0.01
P_CONTROL_GRID = [0.001, 0.005, 0.01, 0.02]
P_8OH_CHALLENGE = 36 / 155

ND = NormalDist()
Phi = ND.cdf
Phi_inv = ND.inv_cdf


@dataclass
class FitResult:
    """Container for a fitted semimechanistic readout model."""

    features: Tuple[str, ...]
    beta: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    theta_ch: float
    p_control_anchor: float
    rmse_train: float
    max_cv_err: float
    mean_cv_err: float


def p_ch_sh(mu: float, theta: float) -> float:
    """Challenge DH probability among immediate SH survivors given latent mean *mu*."""
    denom = Phi(-mu)
    if denom <= 0:
        return 1.0
    return (Phi(-mu) - Phi(theta - mu)) / denom


def solve_theta(mu_8oh: float, p_target: float = P_8OH_CHALLENGE) -> float:
    """Solve for the challenge threshold theta_ch by bisection on p_ch_sh."""
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_ch_sh(mu_8oh, mid) > p_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def arr_list(lst) -> np.ndarray:
    """Stack a list of per-timestep arrays into a (T, N) float array."""
    return np.stack([np.array(x, dtype=float) for x in lst], axis=0)


def load_sim(case: str):
    """Load BETSE state file for *case* and return (sim, cells, p)."""
    path = SIM_DIR / f'sim_{case}.betse.gz'
    with gzip.open(path, 'rb') as f:
        sim, cells, p = dill.load(f)
    return sim, cells, p


def compute_grounded_QR(summary_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Q, R, K_mem and the dominant write eigenvector from the case summary."""
    E_na = float(summary_df.loc['na', 'E_total'])
    E_gj = float(summary_df.loc['gj', 'E_total'])
    E_na_gj = float(summary_df.loc['na_gj', 'E_total'])
    Q = np.array([
        [E_na, 0.5 * (E_na_gj - E_na - E_gj)],
        [0.5 * (E_na_gj - E_na - E_gj), E_gj],
    ], dtype=float)
    R = np.array([
        [summary_df.loc['na', 'd_mean_vmem_mV'] / 1e3, summary_df.loc['gj', 'd_mean_vmem_mV'] / 1e3],
        [summary_df.loc['na', 'd_wound_gj'], summary_df.loc['gj', 'd_wound_gj']],
    ], dtype=float)
    K = R @ np.linalg.inv(Q) @ R.T
    evals, evecs = np.linalg.eigh(K)
    vdom = evecs[:, np.argmax(evals)]
    if vdom[1] < 0:
        vdom = -vdom
    return Q, R, K, vdom


def build_feature_bank() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all simulations, compute enriched features and grounded geometry.

    Returns:
        features: Per-case enriched feature DataFrame (endpoint, integrated,
            energetic, and control-relative delta columns).
        summary: Per-case excess-cost and endpoint-shift summary DataFrame.
        Q: (2, 2) local quadratic cost matrix.
        R: (2, 2) local write map.
        K: (2, 2) induced memory co-metric.
        vdom: Dominant eigenvector of K.
    """
    sim0, cells0, _ = load_sim('control')
    mem_to_cells = np.array(cells0.mem_to_cells)

    # Principal axis of the cluster for a coarse AP-like contrast.
    centres = np.array(cells0.cell_centres)
    X = centres - centres.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(X.T))
    axis = vecs[:, np.argmax(vals)]
    proj = X @ axis
    high = proj >= np.quantile(proj, 0.66)
    low = proj <= np.quantile(proj, 0.34)

    raw_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    control_cache: Dict[str, np.ndarray] = {}
    control_scales: Dict[str, float] = {}

    # First pass: load control arrays for normalized energy proxy.
    ctrl_time = np.array(sim0.time, dtype=float)
    ctrl_imem = arr_list(sim0.I_mem_time)
    ctrl_gjopen = arr_list(sim0.gjopen_time)
    ctrl_egjx = arr_list(sim0.efield_gj_x_time)
    ctrl_egjy = arr_list(sim0.efield_gj_y_time)
    ctrl_gjproxy = ctrl_gjopen * np.sqrt(ctrl_egjx ** 2 + ctrl_egjy ** 2)
    ctrl_pump = arr_list(sim0.rate_NaKATP_time)
    ctrl_vm = arr_list(sim0.vm_ave_time)
    ctrl_hurt = np.array(sim0.hurt_mask) > 0
    ctrl_wound_mems = np.isin(mem_to_cells, np.where(ctrl_hurt)[0])
    control_cache = {
        'time': ctrl_time,
        'imem': ctrl_imem,
        'gjproxy': ctrl_gjproxy,
        'pump': ctrl_pump,
        'vm': ctrl_vm,
        'hurt': ctrl_hurt,
        'wound_mems': ctrl_wound_mems,
    }
    for key in ['imem', 'gjproxy', 'pump']:
        control_scales[key] = float(np.sqrt(np.mean(control_cache[key] ** 2)))

    for case in CASES:
        sim, cells, _ = load_sim(case)
        time = np.array(sim.time, dtype=float)
        imem = arr_list(sim.I_mem_time)
        gjopen = arr_list(sim.gjopen_time)
        egjx = arr_list(sim.efield_gj_x_time)
        egjy = arr_list(sim.efield_gj_y_time)
        gjproxy = gjopen * np.sqrt(egjx ** 2 + egjy ** 2)
        pump = arr_list(sim.rate_NaKATP_time)
        vm = arr_list(sim.vm_ave_time)
        hurt = np.array(sim.hurt_mask) > 0
        wound_mems = np.isin(mem_to_cells, np.where(hurt)[0])

        # Grounded excess-cost proxy components.
        d_imem = (imem - control_cache['imem']) / control_scales['imem']
        d_gj = (gjproxy - control_cache['gjproxy']) / control_scales['gjproxy']
        d_pump = (pump - control_cache['pump']) / control_scales['pump']
        e_mem = float(np.trapezoid(np.mean(d_imem ** 2, axis=1), time))
        e_gj = float(np.trapezoid(np.mean(d_gj ** 2, axis=1), time))
        e_pump = float(np.trapezoid(np.mean(d_pump ** 2, axis=1), time))
        e_total = e_mem + e_gj + e_pump

        # Endpoint map used in the original 2D local model.
        d_mean_vmem = float((vm[-1].mean() - control_cache['vm'][-1].mean()) * 1e3)
        d_wound_gj = float(
            (gjproxy[-1, wound_mems].mean() - gjproxy[-1, ~wound_mems].mean())
            - (control_cache['gjproxy'][-1, control_cache['wound_mems']].mean() - control_cache['gjproxy'][-1, ~control_cache['wound_mems']].mean())
        )
        summary_rows.append({
            'case': case,
            'E_mem': e_mem,
            'E_gj': e_gj,
            'E_pump': e_pump,
            'E_total': e_total,
            'd_mean_vmem_mV': d_mean_vmem,
            'd_wound_gj': d_wound_gj,
        })

        # Enriched feature bank: absolute and integrated features.
        v_end = vm[-1]
        g_end = gjproxy[-1]
        p_end = pump[-1]
        raw_rows.append({
            'case': case,
            'vm_mean': float(v_end.mean()),
            'vm_mean_int': float(np.trapezoid(vm.mean(axis=1), time)),
            'vm_ap_int': float(np.trapezoid(vm[:, high].mean(axis=1) - vm[:, low].mean(axis=1), time)),
            'gj_mean': float(g_end.mean()),
            'gj_std': float(g_end.std()),
            'gj_wound_diff': float(g_end[wound_mems].mean() - g_end[~wound_mems].mean()),
            'gj_wound_int': float(np.trapezoid(gjproxy[:, wound_mems].mean(axis=1) - gjproxy[:, ~wound_mems].mean(axis=1), time)),
            'imem_l2': float(np.trapezoid(np.mean(imem ** 2, axis=1), time)),
            'pump_l2': float(np.trapezoid(np.mean(pump ** 2, axis=1), time)),
            'E_total': e_total,
            'E_pump': e_pump,
            'E_gj': e_gj,
            'd_mean_vmem_mV': d_mean_vmem,
            'd_wound_gj': d_wound_gj,
        })

    features = pd.DataFrame(raw_rows).set_index('case')
    summary = pd.DataFrame(summary_rows).set_index('case')
    Q, R, K, vdom = compute_grounded_QR(summary)

    # Dominant grounded write coordinate from the previous local endpoint map.
    features['x_dom_end'] = (features['d_mean_vmem_mV'] / 1e3) * vdom[0] + features['d_wound_gj'] * vdom[1]

    # Control-relative deltas for selected absolute features.
    control_row = features.loc['control']
    for col in ['vm_mean', 'vm_mean_int', 'gj_mean', 'gj_std', 'gj_wound_diff', 'gj_wound_int', 'imem_l2', 'pump_l2']:
        features[f'{col}_d'] = features[col] - float(control_row[col])

    return features, summary, Q, R, K, vdom


def standardize_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the feature columns of design matrix *X* (column 0 is the intercept)."""
    means = X[:, 1:].mean(axis=0)
    stds = X[:, 1:].std(axis=0)
    stds[stds == 0] = 1.0
    Xs = X.copy()
    Xs[:, 1:] = (Xs[:, 1:] - means) / stds
    return Xs, means, stds


def fit_linear_readout(features_df: pd.DataFrame, feature_names: Sequence[str], p_control_anchor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit mu_k = b + w^T h_k via ridge regression on the four mapped families."""
    labels = ['control', '8OH', 'nigericin', 'monensin']
    y = np.array([
        Phi_inv(p_control_anchor),
        Phi_inv(IMMEDIATE_TARGETS['8OH']),
        Phi_inv(IMMEDIATE_TARGETS['nigericin']),
        Phi_inv(IMMEDIATE_TARGETS['monensin']),
    ], dtype=float)
    X_rows = []
    for label in labels:
        case = LABEL_TO_CASE[label]
        X_rows.append([1.0] + [float(features_df.loc[case, c]) for c in feature_names])
    X = np.array(X_rows, dtype=float)
    Xs, means, stds = standardize_rows(X)
    ridge = 1e-6
    penalty = np.diag([0.0] + [1.0] * len(feature_names))
    beta = np.linalg.solve(Xs.T @ Xs + ridge * penalty, Xs.T @ y)
    mu_train = Xs @ beta
    rmse = float(np.sqrt(np.mean((mu_train - y) ** 2)))
    return beta, means, stds, rmse


def predict_mu(features_df: pd.DataFrame, feature_names: Sequence[str], beta: np.ndarray, means: np.ndarray, stds: np.ndarray) -> pd.Series:
    """Predict latent mean mu for every case using a fitted readout model."""
    rows = []
    for case in features_df.index:
        rows.append([1.0] + [float(features_df.loc[case, c]) for c in feature_names])
    X = np.array(rows, dtype=float)
    X[:, 1:] = (X[:, 1:] - means) / stds
    mu = X @ beta
    return pd.Series(mu, index=features_df.index, name='mu_semimech')


def evaluate_cv(features_df: pd.DataFrame, feature_names: Sequence[str], p_control_anchor: float) -> Tuple[float, float, pd.DataFrame]:
    """Leave-one-treated-out cross-validation over nigericin and monensin."""
    true_mu = {
        'control': Phi_inv(p_control_anchor),
        '8OH': Phi_inv(IMMEDIATE_TARGETS['8OH']),
        'nigericin': Phi_inv(IMMEDIATE_TARGETS['nigericin']),
        'monensin': Phi_inv(IMMEDIATE_TARGETS['monensin']),
    }
    rows = []
    for hold_label in ['nigericin', 'monensin']:
        train_labels = ['control', '8OH'] + [x for x in ['nigericin', 'monensin'] if x != hold_label]
        y = np.array([true_mu[l] for l in train_labels], dtype=float)
        X_rows = []
        for label in train_labels:
            case = LABEL_TO_CASE[label]
            X_rows.append([1.0] + [float(features_df.loc[case, c]) for c in feature_names])
        X = np.array(X_rows, dtype=float)
        Xs, means, stds = standardize_rows(X)
        ridge = 1e-6
        penalty = np.diag([0.0] + [1.0] * len(feature_names))
        beta = np.linalg.solve(Xs.T @ Xs + ridge * penalty, Xs.T @ y)

        hold_case = LABEL_TO_CASE[hold_label]
        Xh = np.array([[1.0] + [float(features_df.loc[hold_case, c]) for c in feature_names]], dtype=float)
        Xh[:, 1:] = (Xh[:, 1:] - means) / stds
        pred = float((Xh @ beta)[0])
        err = abs(pred - true_mu[hold_label])
        rows.append({
            'holdout': hold_label,
            'mu_pred': pred,
            'mu_true': true_mu[hold_label],
            'abs_mu_error': err,
            'pimm_pred': Phi(pred),
            'pimm_true': Phi(true_mu[hold_label]),
        })
    df = pd.DataFrame(rows)
    return float(df['abs_mu_error'].max()), float(df['abs_mu_error'].mean()), df


def search_models(features_df: pd.DataFrame, p_control_anchor: float) -> pd.DataFrame:
    """Exhaustive search over 1–3-feature subsets, ranked by CV error."""
    candidate_cols = [
        'x_dom_end',
        'vm_mean_int',
        'gj_mean',
        'gj_std',
        'gj_wound_diff',
        'gj_wound_int',
        'pump_l2',
        'E_total',
        'E_gj',
        'E_pump',
    ]
    rows = []
    from itertools import combinations
    for r in [1, 2, 3]:
        for combo in combinations(candidate_cols, r):
            beta, means, stds, rmse = fit_linear_readout(features_df, combo, p_control_anchor)
            max_cv_err, mean_cv_err, _ = evaluate_cv(features_df, combo, p_control_anchor)
            rows.append({
                'n_features': r,
                'features': ' | '.join(combo),
                'rmse_train': rmse,
                'max_cv_mu_error': max_cv_err,
                'mean_cv_mu_error': mean_cv_err,
            })
    out = pd.DataFrame(rows).sort_values(['max_cv_mu_error', 'mean_cv_mu_error', 'n_features', 'rmse_train']).reset_index(drop=True)
    return out


def choose_model(search_df: pd.DataFrame) -> Tuple[str, ...]:
    """Return the feature names of the top-ranked model from the search."""
    best_row = search_df.iloc[0]
    return tuple(best_row['features'].split(' | '))


def make_fit(features_df: pd.DataFrame, chosen_features: Sequence[str], p_control_anchor: float) -> Tuple[FitResult, pd.DataFrame, pd.DataFrame]:
    """Fit the chosen feature subset, predict all cases, and run CV."""
    beta, means, stds, rmse = fit_linear_readout(features_df, chosen_features, p_control_anchor)
    mu = predict_mu(features_df, chosen_features, beta, means, stds)
    theta = solve_theta(float(mu.loc[LABEL_TO_CASE['8OH']]))
    pred_df = features_df.copy()
    pred_df['mu_semimech'] = mu
    pred_df['pimm_semimech'] = pred_df['mu_semimech'].map(Phi)
    pred_df['pch_sh_semimech'] = pred_df['mu_semimech'].map(lambda m: p_ch_sh(float(m), theta))
    max_cv_err, mean_cv_err, cv_df = evaluate_cv(features_df, chosen_features, p_control_anchor)
    fit = FitResult(
        features=tuple(chosen_features),
        beta=beta,
        means=means,
        stds=stds,
        theta_ch=theta,
        p_control_anchor=p_control_anchor,
        rmse_train=rmse,
        max_cv_err=max_cv_err,
        mean_cv_err=mean_cv_err,
    )
    return fit, pred_df, cv_df


def build_anchor_sensitivity(features_df: pd.DataFrame, chosen_features: Sequence[str]) -> pd.DataFrame:
    """Sweep the soft control anchor over P_CONTROL_GRID and record predictions."""
    rows = []
    for pctrl in P_CONTROL_GRID:
        fit, pred_df, _ = make_fit(features_df, chosen_features, pctrl)
        rows.append({
            'p_control_anchor': pctrl,
            'mu_control': float(pred_df.loc['control', 'mu_semimech']),
            'pimm_control': float(pred_df.loc['control', 'pimm_semimech']),
            'mu_8OH': float(pred_df.loc['gj', 'mu_semimech']),
            'mu_nigericin': float(pred_df.loc['na', 'mu_semimech']),
            'mu_monensin': float(pred_df.loc['na_k', 'mu_semimech']),
            'pch_nigericin': float(pred_df.loc['na', 'pch_sh_semimech']),
            'pch_monensin': float(pred_df.loc['na_k', 'pch_sh_semimech']),
            'theta_ch': fit.theta_ch,
        })
    return pd.DataFrame(rows)


def make_plots(pred_df: pd.DataFrame, chosen_features: Sequence[str], fit: FitResult) -> None:
    """Save feature-plane scatter and immediate/challenge probability bar charts."""
    feat1, feat2 = chosen_features[0], chosen_features[1] if len(chosen_features) > 1 else (chosen_features[0], None)
    plt.figure(figsize=(7, 5))
    X = pred_df.loc[:, list(chosen_features)].copy()
    Xs = X.copy()
    for j, feat in enumerate(chosen_features):
        Xs[feat] = (Xs[feat] - fit.means[j]) / fit.stds[j]
    if len(chosen_features) >= 2:
        for case in pred_df.index:
            plt.scatter(Xs.loc[case, feat1], Xs.loc[case, feat2], s=60)
            plt.text(Xs.loc[case, feat1], Xs.loc[case, feat2], ' ' + case, va='center', fontsize=9)
        plt.xlabel(f'{feat1} (standardized)')
        plt.ylabel(f'{feat2} (standardized)')
        plt.title('Selected semimechanistic feature plane')
    else:
        yvals = pred_df['mu_semimech']
        for case in pred_df.index:
            plt.scatter(Xs.loc[case, feat1], yvals.loc[case], s=60)
            plt.text(Xs.loc[case, feat1], yvals.loc[case], ' ' + case, va='center', fontsize=9)
        plt.xlabel(f'{feat1} (standardized)')
        plt.ylabel('mu_semimech')
        plt.title('Selected semimechanistic feature axis')
    plt.tight_layout()
    plt.savefig(ROOT / 'betse_semimech_enriched_feature_plane.png', dpi=200)
    plt.close()

    # Probability bars.
    show_order = ['control', 'k', 'na', 'na_k', 'gj', 'k_gj', 'na_gj']
    show_df = pred_df.loc[show_order, ['pimm_semimech', 'pch_sh_semimech']]
    ax = show_df.plot(kind='bar', figsize=(8, 5))
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_title('Semimechanistic immediate and challenge probabilities')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT / 'betse_semimech_enriched_probabilities.png', dpi=200)
    plt.close()


def write_report(features_df: pd.DataFrame, summary_df: pd.DataFrame, Q: np.ndarray, R: np.ndarray, K: np.ndarray, vdom: np.ndarray,
                 search_df: pd.DataFrame, fit: FitResult, pred_df: pd.DataFrame, anchor_df: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    """Write a human-readable text report summarising the full fit."""
    report_lines = [
        'BETSE semimechanistic enriched readout fit',
        '',
        'Purpose: enrich h_k beyond the original two endpoint coordinates so the same grounded pipeline can fit',
        'a soft untreated baseline together with the treated-family shifts.',
        '',
        f'Selected feature subset: {fit.features}',
        f'Soft control anchor p_control^imm: {fit.p_control_anchor}',
        f'Training RMSE on mu: {fit.rmse_train:.6g}',
        f'Leave-one-treated-out max abs mu error: {fit.max_cv_err:.6g}',
        f'Leave-one-treated-out mean abs mu error: {fit.mean_cv_err:.6g}',
        f'Fitted theta_ch: {fit.theta_ch:.6g}',
        '',
        'Grounded local matrices from the earlier BETSE instantiation:',
        'Q =', str(Q), '', 'R =', str(R), '', 'K_mem =', str(K), '',
        f'Dominant grounded endpoint write direction v_dom = {vdom}',
        '',
        'Top model-search rows:',
        search_df.head(10).to_string(index=False),
        '',
        'Cross-prediction table:',
        cv_df.to_string(index=False, float_format=lambda x: f'{x:.6g}'),
        '',
        'Anchor sensitivity table:',
        anchor_df.to_string(index=False, float_format=lambda x: f'{x:.6g}'),
        '',
        'All-case predictions:',
        pred_df[['mu_semimech', 'pimm_semimech', 'pch_sh_semimech']].to_string(float_format=lambda x: f'{x:.6g}'),
        '',
        'Interpretation:',
        'The enriched readout no longer forces the control to share the same origin as the treated family, because',
        'the selected map uses one state/history feature (integrated wound-edge GJ contrast) and one energetic feature',
        '(integrated Na/K-ATPase load). The treated-family challenge predictions stay close to the earlier free-x fit,',
        'while the untreated control is now anchored near zero immediate DH by construction. This remains a soft-anchor',
        'proof of concept, not a fully empirical control calibration.',
    ]
    (ROOT / 'betse_semimech_enriched_report.txt').write_text('\n'.join(report_lines))


def write_latex_snippet(fit: FitResult, pred_df: pd.DataFrame) -> None:
    """Write a LaTeX paragraph with the fitted parameters for the manuscript."""
    snippet = rf"""
A semimechanistic extension of the reduced readout can be obtained by replacing the free
condition-level write effect by a fitted linear map from a BETSE-derived feature vector,
\begin{{equation}}
\mu_k = b + w^{{\transpose}} h_k.
\end{{equation}}
In a first enriched implementation, $h_k$ was drawn from a small bank of simulator-derived
endpoint and early-window features and the subset was selected by leave-one-treated-out
prediction error under a soft near-zero control anchor. The best stable subset was
\begin{{equation}}
h_k = \bigl[\, \int_0^{{T_w}} \Delta J_{{\mathrm{{GJ,wound}}}}(t)\,\dd t,\;
\int_0^{{T_w}} \langle J_{{\mathrm{{NaKATP}}}}(t)^2 \rangle\,\dd t \, \bigr]^{{\transpose}},
\end{{equation}}
that is, integrated wound-edge gap-junction contrast together with integrated Na/K-ATPase load.
Using a soft control anchor $p_{{\mathrm{{control}}}}^{{\mathrm{{imm}}}}=0.01$, the fitted immediate-layer
means were
\begin{{equation}}
\mu_{{\mathrm{{control}}}} = {pred_df.loc['control', 'mu_semimech']:.3f},
\qquad
\mu_{{8\mathrm{{OH}}}} = {pred_df.loc['gj', 'mu_semimech']:.3f},
\qquad
\mu_{{\mathrm{{Nig}}}} = {pred_df.loc['na', 'mu_semimech']:.3f},
\qquad
\mu_{{\mathrm{{Mon}}}} = {pred_df.loc['na_k', 'mu_semimech']:.3f},
\end{{equation}}
with fitted challenge threshold
\begin{{equation}}
\theta_{{\mathrm{{ch}}}} = {fit.theta_ch:.3f}.
\end{{equation}}
This yields semimechanistic challenge predictions
\begin{{equation}}
\hat p_{{\mathrm{{Nig}}}}^{{\mathrm{{ch|SH}}}} = {pred_df.loc['na', 'pch_sh_semimech']:.3f},
\qquad
\hat p_{{\mathrm{{Mon}}}}^{{\mathrm{{ch|SH}}}} = {pred_df.loc['na_k', 'pch_sh_semimech']:.3f},
\end{{equation}}
while keeping the untreated control near zero immediate DH by construction. Because the control
anchor is presently a modelling prior rather than a fitted experimental count, this should be read
as a semimechanistic proof of concept rather than a definitive phenotype model.
""".strip() + "\n"
    (ROOT / 'betse_semimech_enriched_snippet.tex').write_text(snippet)


def main() -> None:
    """Run the full semimechanistic bridge pipeline."""
    features_df, summary_df, Q, R, K, vdom = build_feature_bank()
    features_df.to_csv(ROOT / 'betse_semimech_enriched_features.csv')
    summary_df.to_csv(ROOT / 'betse_semimech_enriched_summary.csv')

    search_df = search_models(features_df, P_CONTROL_DEFAULT)
    search_df.to_csv(ROOT / 'betse_semimech_enriched_model_search.csv', index=False)
    chosen_features = choose_model(search_df)

    fit, pred_df, cv_df = make_fit(features_df, chosen_features, P_CONTROL_DEFAULT)
    pred_df.to_csv(ROOT / 'betse_semimech_enriched_predictions.csv')
    cv_df.to_csv(ROOT / 'betse_semimech_enriched_cross_prediction.csv', index=False)

    anchor_df = build_anchor_sensitivity(features_df, chosen_features)
    anchor_df.to_csv(ROOT / 'betse_semimech_enriched_anchor_sensitivity.csv', index=False)

    make_plots(pred_df, chosen_features, fit)
    write_report(features_df, summary_df, Q, R, K, vdom, search_df, fit, pred_df, anchor_df, cv_df)
    write_latex_snippet(fit, pred_df)

    print('Chosen features:', chosen_features)
    print(pred_df[['mu_semimech', 'pimm_semimech', 'pch_sh_semimech']])


if __name__ == '__main__':
    main()
