# tasmorpho

Reproducibility code for:

> **Hidden regenerative state in planarians: A geometric model of bioelectric memory using Tangential Action Spaces**
>
> Marcel Blattner, Applied AI Research Lab, Lucerne University of Applied Sciences and Arts
>
> *bioRxiv* preprint (forthcoming)

## Installation

Requires Python 3.10+ and (for scripts 01–02) `betse` on `PATH`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib dill ruamel.yaml scipy betse
```

`scipy` is optional — script 03 uses it for a Nelder–Mead cross-check but runs without it.

## Usage

All scripts live in `scripts/`. Scripts 01 and 02 must be run in sequence; script 03 is standalone.

```bash
cd scripts

# Step 1 — Local BETSE grounding
python 01_betse_local_grounding.py

# Step 2 — Semimechanistic bridge
python 02_semimech_bridge.py

# Step 3 — Reduced count-based fit (independent of steps 1–2)
python 03_reduced_count_fit.py
```

## Repository structure

```
scripts/
├── 01_betse_local_grounding.py          # runs BETSE, computes Q, R, K_mem
├── 02_semimech_bridge.py                # enriched semimechanistic readout
├── 03_reduced_count_fit.py              # reduced latent-threshold fit
└── betse_tas_example_run/               # reference outputs + BETSE configs
    ├── betse_report.txt
    ├── betse_summary.csv
    ├── betse_hidden_space.png
    ├── betse_energy_components.png
    └── demo/
        ├── sim_config.yml               # BETSE template config
        ├── minimal_control.yml          # minimal control config used in the paper
        ├── case_{na,k,gj,na_k,na_gj,k_gj}.yml
        ├── geo/
        └── extra_configs/
```

## Scripts

### `01_betse_local_grounding.py`

Builds minimal BETSE configs from the template, runs one control plus six perturbation cases, and computes the grounded local geometry: excess-cost proxy components, $Q_\chi$, $R_\chi$, and $K_{\mathrm{mem}}^{(\chi)}$.

| Case | Perturbation |
|------|-------------|
| `na` | 1.3× Na membrane permeability |
| `k` | 1.3× K membrane permeability |
| `gj` | 20% random gap-junction block |
| `na_k` | combined Na + K |
| `na_gj` | combined Na + GJ |
| `k_gj` | combined K + GJ |

### `02_semimech_bridge.py`

Loads simulation outputs from script 01, builds a feature bank, and fits the semimechanistic readout $\mu_k = b + w^\top h_k$ under a soft control anchor ($p_{\mathrm{control}}^{\mathrm{imm}} = 0.01$). The selected features are integrated wound-edge gap-junction contrast and integrated Na/K-ATPase load.

| Experimental family | In-silico case |
|----|------|
| 8-OH | `gj` |
| nigericin | `na` |
| monensin | `na_k` |

### `03_reduced_count_fit.py`

Fits a reduced rank-one latent-threshold model to published phenotype counts with $\sigma_\chi = 1$ and $\theta_{\mathrm{imm}} = 0$ fixed. Computes closed-form estimates, Hessian, Wald standard errors, and delta-method intervals. No BETSE dependency.

## Expected outputs

### Reduced fit (script 03)

| Dataset | Count | Role |
|---------|-------|------|
| 8-OH immediate DH | 143 / 573 | primary fit target |
| 8-OH challenge DH among immediate SH | 36 / 155 | primary fit target |
| 8-OH challenge DH among immediate DH | 100 / 100 | consistency check |
| Nigericin immediate DH | 17 / 132 | condition-setting |
| Monensin immediate DH | 11 / 89 | condition-setting |

| Parameter | Estimate |
|-----------|----------|
| $\hat\mu_{8\mathrm{OH}}$ | $-0.676 \pm 0.057$ |
| $\hat\theta_{\mathrm{ch}}$ | $-0.484 \pm 0.068$ |
| $\hat\mu_{\mathrm{Nig}}$ | $-1.132 \pm 0.139$ |
| $\hat\mu_{\mathrm{Mon}}$ | $-1.157 \pm 0.171$ |
| $\hat p_{\mathrm{Nig}}^{\mathrm{ch\|SH}}$ | $0.149$ [0.082, 0.216] **held-out** |
| $\hat p_{\mathrm{Mon}}^{\mathrm{ch\|SH}}$ | $0.145$ [0.071, 0.218] **held-out** |

### Semimechanistic bridge (script 02)

| Parameter | Estimate |
|-----------|----------|
| $\mu_{\mathrm{control}}$ | $-2.326$ |
| $\mu_{8\mathrm{OH}}$ | $-0.676$ |
| $\mu_{\mathrm{Nig}}$ | $-1.139$ |
| $\mu_{\mathrm{Mon}}$ | $-1.151$ |
| $\hat\theta_{\mathrm{ch}}$ | $-0.484$ |
| $\hat p_{\mathrm{Nig}}^{\mathrm{ch\|SH}}$ | $0.148$ |
| $\hat p_{\mathrm{Mon}}^{\mathrm{ch\|SH}}$ | $0.146$ |

## References

- Durant et al. (2017). *Biophysical Journal*, 112(10), 2231–2243.
- Durant et al. (2019). *Bioinformatics*, 35(24), 5287–5294.
- Pietak & Levin (2017). *J. R. Soc. Interface*, 14(134), 20170425.

## Citation

```bibtex
@article{Blattner2026tasmorpho,
  author  = {Blattner, Marcel},
  title   = {Hidden regenerative state in planarians: A geometric model
             of bioelectric memory using Tangential Action Spaces},
  journal = {bioRxiv},
  year    = {2026},
  note    = {Preprint}
}
```

## License

See [LICENSE](LICENSE).
