"""Local BETSE grounding of the TAS effort metric G.

Builds minimal BETSE configurations from a template, runs a control
simulation plus six single- and pairwise-perturbation cases (Na, K, GJ
and their combinations), and computes the grounded local geometry used
in the manuscript:

  - Normalized excess-cost proxy components (E_mem, E_gj, E_pump).
  - Local quadratic cost matrix Q_chi.
  - Local write map R_chi (endpoint shifts in mean Vmem and wound-edge
    GJ contrast).
  - Induced memory co-metric K_mem = R Q^{-1} R^T.

Must be run from the ``scripts/`` directory because output paths are
relative to the current working directory.
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

ROOT = Path('betse_tas_example_run/')
DEMO = ROOT / 'demo'
BASE_SIM = DEMO / 'sim_config.yml'
MINIMAL = DEMO / 'minimal_control.yml'


def run(cmd: list[str], cwd: Path) -> None:
    """Execute *cmd* as a subprocess in *cwd*, raising on failure."""
    print('RUN', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_sim(path: Path):
    """Load a BETSE ``.betse.gz`` state file and return (sim, cells, p)."""
    with gzip.open(path, 'rb') as f:
        sim, cells, p = dill.load(f)
    return sim, cells, p


def arr_list(lst) -> np.ndarray:
    """Stack a list of per-timestep arrays into a (T, N) float array."""
    return np.stack([np.array(x, dtype=float) for x in lst], axis=0)


def build_minimal_configs() -> None:
    """Derive minimal control and perturbation YAML configs from the template.

    Reads ``sim_config.yml``, strips it down to a short-window, small-grid,
    no-extracellular, no-network configuration, and writes
    ``minimal_control.yml`` plus one ``case_<name>.yml`` per perturbation.
    """
    yaml = YAML()
    with BASE_SIM.open() as f:
        cfg = yaml.load(f)

    cfg['world options']['world size'] = 75e-6
    cfg['world options']['cell radius'] = 5.0e-6
    cfg['world options']['lattice disorder'] = 0.15
    cfg['world options']['mesh refinement']['refine mesh'] = False
    cfg['general options']['comp grid size'] = 15
    cfg['general options']['simulate extracellular spaces'] = False
    cfg['init time settings']['time step'] = 0.02
    cfg['init time settings']['total time'] = 0.2
    cfg['init time settings']['sampling rate'] = 0.02
    cfg['sim time settings']['time step'] = 5e-4
    cfg['sim time settings']['total time'] = 0.02
    cfg['sim time settings']['sampling rate'] = 0.001
    cfg['general network']['implement network'] = False

    for key in [
        'change Na mem', 'change K mem', 'change Cl mem', 'change Ca mem',
        'block gap junctions', 'block NaKATP pump', 'apply pressure',
        'change temperature', 'change K env', 'change Cl env', 'change Na env',
    ]:
        cfg[key]['event happens'] = False

    for key in ['change Na mem', 'change K mem']:
        cfg[key]['change start'] = 0.0
        cfg[key]['change finish'] = 0.01
        cfg[key]['change rate'] = 0.001
        cfg[key]['apply to'] = ['Base']
        cfg[key]['multiplier'] = 1.4

    cfg['block gap junctions']['change start'] = 0.0
    cfg['block gap junctions']['change finish'] = 0.01
    cfg['block gap junctions']['change rate'] = 0.001
    cfg['block gap junctions']['random fraction'] = 30
    cfg['apply external voltage']['event happens'] = False
    cfg['cutting event']['event happens'] = True

    cfg['init file saving']['directory'] = 'INITS/minimal'
    cfg['init file saving']['worldfile'] = 'world_control.betse.gz'
    cfg['init file saving']['file'] = 'init_control.betse.gz'
    cfg['sim file saving']['directory'] = 'SIMS/minimal'
    cfg['sim file saving']['file'] = 'sim_control.betse.gz'

    with MINIMAL.open('w') as f:
        yaml.dump(cfg, f)

    cases = {
        'na': {'change Na mem': {'event happens': True, 'multiplier': 1.3}},
        'k': {'change K mem': {'event happens': True, 'multiplier': 1.3}},
        'gj': {'block gap junctions': {'event happens': True, 'random fraction': 20}},
        'na_k': {
            'change Na mem': {'event happens': True, 'multiplier': 1.3},
            'change K mem': {'event happens': True, 'multiplier': 1.3},
        },
        'na_gj': {
            'change Na mem': {'event happens': True, 'multiplier': 1.3},
            'block gap junctions': {'event happens': True, 'random fraction': 20},
        },
        'k_gj': {
            'change K mem': {'event happens': True, 'multiplier': 1.3},
            'block gap junctions': {'event happens': True, 'random fraction': 20},
        },
    }

    for name, changes in cases.items():
        with MINIMAL.open() as f:
            case_cfg = yaml.load(f)
        case_cfg['sim file saving']['file'] = f'sim_{name}.betse.gz'
        for section, vals in changes.items():
            for k, v in vals.items():
                case_cfg[section][k] = v
        with (DEMO / f'case_{name}.yml').open('w') as f:
            yaml.dump(case_cfg, f)


def analyze() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load all simulation outputs and compute the grounded local geometry.

    For each perturbation case, computes control-normalized excess-cost
    proxy components (membrane current, GJ flux, Na/K-ATPase pump) and
    two endpoint shift coordinates (mean Vmem in mV, wound-edge GJ
    contrast).  From the ``na``, ``gj``, and ``na_gj`` cases, assembles
    the local quadratic cost matrix Q, write map R, and induced memory
    co-metric K_mem = R Q^{-1} R^T.

    Returns:
        df: Per-case summary DataFrame.
        Q: (2, 2) local quadratic cost matrix.
        R: (2, 2) local write map.
        K: (2, 2) induced memory co-metric.
    """
    sim_dir = DEMO / 'SIMS' / 'minimal'
    files = {
        'control': 'sim_control.betse.gz',
        'na': 'sim_na.betse.gz',
        'k': 'sim_k.betse.gz',
        'gj': 'sim_gj.betse.gz',
        'na_k': 'sim_na_k.betse.gz',
        'na_gj': 'sim_na_gj.betse.gz',
        'k_gj': 'sim_k_gj.betse.gz',
    }
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for name, fn in files.items():
        sim, cells, p = load_sim(sim_dir / fn)
        time = np.array(sim.time, dtype=float)
        imem = arr_list(sim.I_mem_time)
        gjopen = arr_list(sim.gjopen_time)
        egjx = arr_list(sim.efield_gj_x_time)
        egjy = arr_list(sim.efield_gj_y_time)
        gjproxy = gjopen * np.sqrt(egjx ** 2 + egjy ** 2)
        pump = arr_list(sim.rate_NaKATP_time)
        vm = arr_list(sim.vm_ave_time)
        hurt = np.array(sim.hurt_mask) > 0
        wound_mems = np.isin(np.array(cells.mem_to_cells), np.where(hurt)[0])
        data[name] = {
            'time': time,
            'imem': imem,
            'gjproxy': gjproxy,
            'pump': pump,
            'vm': vm,
            'wound_mems': wound_mems,
        }

    ctrl = data['control']
    scales = {k: np.sqrt(np.mean(ctrl[k] ** 2)) for k in ['imem', 'gjproxy', 'pump']}

    rows = []
    for name in files:
        d_imem = (data[name]['imem'] - ctrl['imem']) / scales['imem']
        d_gj = (data[name]['gjproxy'] - ctrl['gjproxy']) / scales['gjproxy']
        d_pump = (data[name]['pump'] - ctrl['pump']) / scales['pump']
        t = ctrl['time']
        e_mem = np.trapezoid(np.mean(d_imem ** 2, axis=1), t)
        e_gj = np.trapezoid(np.mean(d_gj ** 2, axis=1), t)
        e_pump = np.trapezoid(np.mean(d_pump ** 2, axis=1), t)
        vm_shift = (data[name]['vm'][-1].mean() - ctrl['vm'][-1].mean()) * 1e3
        wgj_shift = (
            (data[name]['gjproxy'][-1, data[name]['wound_mems']].mean() - data[name]['gjproxy'][-1, ~data[name]['wound_mems']].mean())
            - (ctrl['gjproxy'][-1, ctrl['wound_mems']].mean() - ctrl['gjproxy'][-1, ~ctrl['wound_mems']].mean())
        )
        rows.append({
            'case': name,
            'E_mem': e_mem,
            'E_gj': e_gj,
            'E_pump': e_pump,
            'E_total': e_mem + e_gj + e_pump,
            'd_mean_vmem_mV': vm_shift,
            'd_wound_gj': wgj_shift,
        })

    df = pd.DataFrame(rows)
    E_na = float(df.loc[df.case == 'na', 'E_total'].iloc[0])
    E_gj = float(df.loc[df.case == 'gj', 'E_total'].iloc[0])
    E_na_gj = float(df.loc[df.case == 'na_gj', 'E_total'].iloc[0])
    Q = np.array([
        [E_na, 0.5 * (E_na_gj - E_na - E_gj)],
        [0.5 * (E_na_gj - E_na - E_gj), E_gj],
    ])
    R = np.array([
        [df.loc[df.case == 'na', 'd_mean_vmem_mV'].iloc[0] / 1e3, df.loc[df.case == 'gj', 'd_mean_vmem_mV'].iloc[0] / 1e3],
        [df.loc[df.case == 'na', 'd_wound_gj'].iloc[0], df.loc[df.case == 'gj', 'd_wound_gj'].iloc[0]],
    ])
    K = R @ np.linalg.inv(Q) @ R.T
    return df, Q, R, K


def make_outputs(df: pd.DataFrame, Q: np.ndarray, R: np.ndarray, K: np.ndarray) -> None:
    """Write summary CSV, endpoint-map and energy-decomposition plots, and a text report."""
    df.to_csv(ROOT / 'betse_summary.csv', index=False)

    plot_df = df[df.case != 'control'].copy()
    plt.figure(figsize=(7, 5))
    for _, r in plot_df.iterrows():
        plt.scatter(r['d_mean_vmem_mV'], r['d_wound_gj'], s=60)
        plt.text(r['d_mean_vmem_mV'], r['d_wound_gj'], ' ' + r['case'], fontsize=9, va='center')
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel('Endpoint shift: mean Vmem (mV)')
    plt.ylabel('Endpoint shift: wound-edge GJ proxy (a.u.)')
    plt.title('BETSE local endpoint map for perturbation cases')
    plt.tight_layout()
    plt.savefig(ROOT / 'betse_hidden_space.png', dpi=200)
    plt.close()

    plot_df2 = plot_df[['case', 'E_mem', 'E_gj', 'E_pump']].copy().set_index('case')
    ax = plot_df2.plot(kind='bar', stacked=True, figsize=(8, 5))
    ax.set_ylabel('Relative excess cost (normalized units)')
    ax.set_title('Decomposition of BETSE-derived excess-cost proxy')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ROOT / 'betse_energy_components.png', dpi=200)
    plt.close()

    report = [
        'BETSE local TAS example (single-cut, early-window)',
        '',
        'Generated by 01_betse_local_grounding.py',
        '',
        'Q =',
        str(Q),
        '',
        'R =',
        str(R),
        '',
        'K_mem =',
        str(K),
        '',
        'Case summary:',
        df.to_string(index=False, justify='left', float_format=lambda x: f'{x:.6g}'),
    ]
    (ROOT / 'betse_report.txt').write_text('\n'.join(report))


def main() -> None:
    """Run the full pipeline: generate configs, run BETSE, analyze, write outputs."""
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    run(['betse', '--headless', 'config', str(BASE_SIM.relative_to(ROOT))], cwd=ROOT)
    build_minimal_configs()
    run(['betse', '--headless', 'seed', 'minimal_control.yml'], cwd=DEMO)
    run(['betse', '--headless', 'init', 'minimal_control.yml'], cwd=DEMO)
    run(['betse', '--headless', 'sim', 'minimal_control.yml'], cwd=DEMO)
    for case in ['na', 'k', 'gj', 'na_k', 'na_gj', 'k_gj']:
        run(['betse', '--headless', 'sim', f'case_{case}.yml'], cwd=DEMO)
    df, Q, R, K = analyze()
    make_outputs(df, Q, R, K)
    print('Done. Outputs written to', ROOT)


if __name__ == '__main__':
    main()
