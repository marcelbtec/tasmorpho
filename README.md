# Hidden regenerative state in planarians — reproducibility repository

Reproducibility code for the manuscript:

> **Hidden regenerative state in planarians: A geometric model of bioelectric memory using Tangential Action Spaces**
> Marcel Blattner, Applied AI Research Lab, Lucerne University of Applied Sciences and Arts.
> bioRxiv preprint (forthcoming).

This repository reproduces the three quantitative layers used in the manuscript:

1. **Reduced count-based latent-threshold fit** to published planarian phenotype counts.
2. **Local BETSE grounding of the TAS effort metric** \(G\) for one cut geometry and a small perturbation family.
3. **Semimechanistic bridge** from BETSE-derived features to the same latent phenotype layer used in the reduced fit.

Taken together, these layers support the manuscript's claim that the open-path Tangential Action Spaces (TAS) framework is:

- **calibratable** from published phenotype counts,
- **groundable** in a local electrodiffusive model, and
- **compatible** with a simple semimechanistic readout that recovers the treated-family ordering and transfer scale.

## Scope and intended interpretation

This repository does **not** implement a full mechanistic model of planarian regeneration.
It reproduces a **local, proof-of-concept, early-window electrodiffusive example** plus the reduced statistical layers used in the paper.

Important scope limitations:

- The BETSE example is a **single-cut, local perturbation study**, not a whole-animal regeneration model.
- The committed minimal BETSE runs are **BETSE-based** and **do not run a gene regulatory network**.
- The semimechanistic bridge uses an **illustrative mapping** from experimental families to local in-silico perturbation classes.
- The untreated baseline in the semimechanistic layer is held near zero immediate DH with a **soft control anchor** rather than a fitted experimental control count.
- The reduced count-based fit uses counts reconstructed from rounded percentages reported in the source papers (Durant et al. 2017; Durant et al. 2019).

Use this repository as a reproducibility package for the paper's **proof-of-concept layers**, not as a definitive biological simulator.

## Source data

The phenotype counts used in the reduced fit are reconstructed from rounded percentages and sample sizes reported in:

- **Durant et al. (2017).** Long-term, stochastic editing of regenerative anatomy via targeting endogenous bioelectric gradients. *Biophysical Journal*, 112(10), 2231–2243.
- **Durant et al. (2019).** Instructing cells and networks with combined chemical and bioelectric signals. *Bioinformatics*, 35(24), 5287–5294.

The BETSE/BIGR electrodiffusive framework used for the local in-silico grounding is described in:

- **Pietak & Levin (2017).** Bioelectric gene and reaction networks: computational modelling of genetic, biochemical, and bioelectrical dynamics in pattern regulation. *Journal of The Royal Society Interface*, 14(134), 20170425.

## Repository layout

```
README.md
scripts/
├── 01_betse_local_grounding.py
├── 02_semimech_bridge.py
├── 03_reduced_count_fit.py
└── betse_tas_example_run/
    ├── betse_report.txt
    ├── betse_summary.csv
    ├── betse_hidden_space.png
    ├── betse_energy_components.png
    └── demo/
        ├── sim_config.yml
        ├── minimal_control.yml
        ├── case_na.yml
        ├── case_k.yml
        ├── case_gj.yml
        ├── case_na_k.yml
        ├── case_na_gj.yml
        ├── case_k_gj.yml
        ├── geo/
        └── extra_configs/
```

The three Python scripts are the core of the repository:

| Script | Manuscript layer | Manuscript section | Purpose |
|--------|-----------------|-------------------|---------|
| `01_betse_local_grounding.py` | Layer 2 | §4.4 and Appendix | Runs BETSE, computes grounded local geometry |
| `02_semimech_bridge.py` | Layer 3 | §4.4 | Enriched semimechanistic readout from BETSE features |
| `03_reduced_count_fit.py` | Layer 1 | §4.3 | Reduced latent-threshold fit from published counts |

Scripts are numbered by **execution order**, not by manuscript layer number.
Script 03 is fully standalone and can be run at any time.
Script 02 depends on the simulation outputs of script 01.

### `01_betse_local_grounding.py`

- Wipes and recreates `betse_tas_example_run/`, then runs `betse --headless config` to generate the template configuration (`sim_config.yml`) and supporting files (geometry SVGs, extra config templates).
- Builds the **minimal BETSE control and perturbation configs** from that template.
- Runs BETSE in headless mode through the sequence: `config` → `seed` → `init` → `sim`.
- Runs one control plus six perturbation cases (`control`, `na`, `k`, `gj`, `na_k`, `na_gj`, `k_gj`).
- Computes the grounded local objects: excess-cost proxy components, \(Q_\chi\), \(R_\chi\), \(K_{\mathrm{mem}}^{(\chi)}\).
- Writes outputs under `betse_tas_example_run/`:
  - `betse_summary.csv`
  - `betse_report.txt`
  - `betse_hidden_space.png`
  - `betse_energy_components.png`
  - `.betse.gz` state files (under `demo/SIMS/minimal/`)
  - Generated YAML configs (under `demo/`)

The committed files in `betse_tas_example_run/` are reference outputs from a prior run.
Running this script regenerates them from scratch.

### `02_semimech_bridge.py`

- Loads the `.betse.gz` simulation outputs produced by script 01.
- Rebuilds the grounded feature bank and the local matrices \(Q_\chi\), \(R_\chi\), \(K_{\mathrm{mem}}^{(\chi)}\).
- Searches low-dimensional feature subsets by leave-one-treated-out cross-validation.
- Fits the enriched semimechanistic readout \(\mu_k = b + w^\top h_k\) under a soft control anchor, where the best stable two-feature subset is integrated wound-edge gap-junction contrast together with integrated Na/K-ATPase load.
- Writes outputs alongside the script in `scripts/`:
  - `betse_semimech_enriched_features.csv`
  - `betse_semimech_enriched_summary.csv`
  - `betse_semimech_enriched_model_search.csv`
  - `betse_semimech_enriched_predictions.csv`
  - `betse_semimech_enriched_cross_prediction.csv`
  - `betse_semimech_enriched_anchor_sensitivity.csv`
  - `betse_semimech_enriched_report.txt`
  - `betse_semimech_enriched_snippet.tex`
  - `betse_semimech_enriched_feature_plane.png`
  - `betse_semimech_enriched_probabilities.png`

### `03_reduced_count_fit.py`

- Reproduces the manuscript's **reduced rank-one latent-threshold fit** from the reconstructed published counts.
- Fixes \(\sigma_\chi = 1\) and \(\theta_{\mathrm{imm}} = 0\) to set the latent scale, as described in §4.3.
- Computes the closed-form estimates and, when SciPy is available, checks them against a Nelder–Mead optimization.
- Computes the observed Hessian, approximate covariance, Wald standard errors, and delta-method intervals.
- Writes outputs to `scripts/outputs_reduced_count_fit/` by default (override with `--output-dir`):
  - `reconstructed_count_table.csv`
  - `reduced_fit_parameters.csv`
  - `reduced_fit_probabilities.csv`
  - `reduced_fit_hessian.csv`
  - `reduced_fit_covariance.csv`
  - `reduced_fit_probabilities.png`
  - `reduced_fit_latent_axis.png`
  - `reduced_fit_report.txt`
  - `reduced_fit_snippet.tex`
  - `optimizer_check.csv`

### BETSE configuration files

All configuration files live under `scripts/betse_tas_example_run/demo/`.
They are generated by script 01; the committed copies are reference outputs.

- **`sim_config.yml`** — Rich template generated by `betse config`. Includes settings that are **not** the ones finally used in the minimal manuscript example.
- **`minimal_control.yml`** — Actual control configuration for the manuscript's **minimal local BETSE run**, derived from the template by script 01.
- **`case_na.yml`**, **`case_k.yml`**, **`case_gj.yml`**, **`case_na_k.yml`**, **`case_na_gj.yml`**, **`case_k_gj.yml`** — Minimal perturbation configs derived from `minimal_control.yml`.

The minimal manuscript run differs from the richer template in these ways:

- extracellular spaces are turned **off**,
- the general network is turned **off**,
- initialization and simulation windows are **very short**,
- computation grid size is reduced,
- only a small perturbation family is used.

The `geo/` subdirectory contains SVG cell/worm geometry files and `extra_configs/` contains auxiliary GRN/metabolism templates; neither is used in the minimal manuscript run.

## Dependencies

### Per-script requirements

| Package | `01_betse_local_grounding` | `02_semimech_bridge` | `03_reduced_count_fit` |
|---------|:-:|:-:|:-:|
| `numpy` | required | required | required |
| `pandas` | required | required | required |
| `matplotlib` | required | required | required |
| `dill` | required | required | — |
| `ruamel.yaml` | required | — | — |
| `betse` CLI | required | — | — |
| `scipy` | — | — | optional |

- **`betse`** must be installed and available on `PATH`. It is invoked only by script 01. Script 02 reads the `.betse.gz` state files directly with `dill`/`gzip`; it does not call the `betse` CLI.
- **`scipy`** is used in script 03 for a Nelder–Mead optimization cross-check. The script runs without it using the closed-form solution only.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib dill ruamel.yaml scipy betse
```

If you use Conda or Mamba, install the same package set there.

## Run order

Scripts 01 and 02 must be run in order. Script 03 is standalone and can be run at any time.

```
01_betse_local_grounding.py  ──▶  02_semimech_bridge.py
                                       (reads .betse.gz files from 01)

03_reduced_count_fit.py      (independent; no BETSE dependency)
```

Script 01 uses a cwd-relative output path (`Path('betse_tas_example_run/')`), so it must be run from inside `scripts/`. Scripts 02 and 03 resolve paths relative to their own location and can be invoked from anywhere.

### 1. Local BETSE grounding (script 01)

```bash
cd scripts
python 01_betse_local_grounding.py
```

This step wipes and recreates `betse_tas_example_run/`, generates BETSE configs, runs the control and six perturbation simulations, and computes the grounded local geometry.

### 2. Enriched semimechanistic bridge (script 02)

```bash
python scripts/02_semimech_bridge.py
```

This step reads the BETSE simulation outputs from step 1, builds the enriched feature bank, searches feature subsets, fits the semimechanistic readout, and writes transfer predictions and diagnostics.

### 3. Reduced count-based fit (script 03)

```bash
python scripts/03_reduced_count_fit.py
```

Or with a custom output directory:

```bash
python scripts/03_reduced_count_fit.py --output-dir my_output_dir
```

This step is purely statistical — it fits the latent-threshold model to published counts and produces parameter estimates, uncertainties, and held-out predictions.

## Manuscript mapping

### Layer 1 — Reduced count-based fit (§4.3)

Reproduced by `scripts/03_reduced_count_fit.py`.

The reduced model fixes \(\sigma_\chi = 1\) and \(\theta_{\mathrm{imm}} = 0\) and maximizes a binomial log-likelihood for the remaining parameters. It uses these reconstructed counts:

| Dataset | Count | Role |
|---------|-------|------|
| 8-OH immediate DH | 143 / 573 | primary fit target |
| 8-OH challenge DH among immediate SH after water re-cut | 36 / 155 | primary fit target |
| 8-OH challenge DH among immediate DH after water re-cut | 100 / 100 | consistency check (justifies setting \(p_k^{\mathrm{ch\|DH}} = 1\)) |
| Nigericin immediate DH | 17 / 132 | condition-setting |
| Monensin immediate DH | 11 / 89 | condition-setting |
| Nigericin / monensin challenge DH among immediate SH | not used | held-out prediction |

Expected key outputs:

- \(\hat\mu_{8\mathrm{OH}} \approx -0.676 \pm 0.057\)
- \(\hat\theta_{\mathrm{ch}} \approx -0.484 \pm 0.068\)
- \(\hat\mu_{\mathrm{Nig}} \approx -1.132 \pm 0.139\) (condition-setting only)
- \(\hat\mu_{\mathrm{Mon}} \approx -1.157 \pm 0.171\) (condition-setting only)
- \(\hat p_{\mathrm{Nig}}^{\mathrm{ch|SH}} \approx 0.149\) \[0.082, 0.216\] (held-out prediction)
- \(\hat p_{\mathrm{Mon}}^{\mathrm{ch|SH}} \approx 0.145\) \[0.071, 0.218\] (held-out prediction)

The nigericin and monensin challenge penetrances are **out-of-sample predictions**: no re-challenge data for those treatments enter the calibration. This layer corresponds to the paper's **count-based reference fit** (§4.3, Table 2).

### Layer 2 — Grounded local BETSE example (§4.4 and Appendix)

Reproduced by `scripts/01_betse_local_grounding.py`.

This layer instantiates a **local version of the physiological-state effort metric** \(G\) in relative biophysical units and extracts example local write geometry. Deviations in transmembrane current, a wound-edge gap-junction flux proxy, and Na/K-ATPase activity define a normalized excess-cost proxy. Finite-difference perturbations then yield explicit local matrices:

- a local quadratic cost matrix \(Q_\chi\),
- a local write map \(R_\chi\),
- the induced memory co-metric \(K_{\mathrm{mem}}^{(\chi)}\).

In this illustrative case, the dominant cheap-write direction is concentrated almost entirely in wound-edge gap-junction contrast rather than in cluster-mean \(V_{\mathrm{mem}}\) displacement. Because the current, gap-junction, and pump blocks are normalized by baseline RMS scales, the anisotropy is normalization-dependent and should be read as one explicit local instantiation rather than a uniquely identified biological decomposition.

This layer corresponds to the paper's **illustrative grounded in-silico bridge**.

### Layer 3 — Enriched semimechanistic bridge (§4.4)

Reproduced by `scripts/02_semimech_bridge.py`.

This layer replaces the free condition-level write effect by a fitted linear map from a BETSE-derived feature vector (manuscript eq. 50):

\[\mu_k = b + w^\top h_k\]

The feature subset was selected by leave-one-treated-out latent prediction error. The best stable two-feature subset is (manuscript eq. 52):

\[h_k = \bigl[\, \textstyle\int_0^{T_w} \Delta J_{\mathrm{GJ,wound}}(t)\,\mathrm{d}t,\; \int_0^{T_w} \langle J_{\mathrm{NaKATP}}(t)^2 \rangle\,\mathrm{d}t \,\bigr]^\top\]

that is, integrated wound-edge gap-junction contrast together with integrated Na/K-ATPase load.

The committed proof-of-concept perturbation mapping is:

- `8OH -> gj` (gap-junction block case)
- `nigericin -> na` (Na-permeability case)
- `monensin -> na_k` (mixed Na/K case)

The soft control anchor used in the default run is \(p_{\mathrm{control}}^{\mathrm{imm}} = 0.01\).

Expected key outputs (manuscript eqs. 62–64):

- \(\mu_{\mathrm{control}} \approx -2.326\)
- \(\mu_{8\mathrm{OH}} \approx -0.676\)
- \(\mu_{\mathrm{Nig}} \approx -1.139\)
- \(\mu_{\mathrm{Mon}} \approx -1.151\)
- \(\hat\theta_{\mathrm{ch}} \approx -0.484\)
- \(\hat p_{\mathrm{Nig}}^{\mathrm{ch|SH}} \approx 0.148\)
- \(\hat p_{\mathrm{Mon}}^{\mathrm{ch|SH}} \approx 0.146\)

Leave-one-treated-out prediction errors remain below 0.014 latent units. Sweeping the soft control anchor over \(p_{\mathrm{control}}^{\mathrm{imm}} \in [0.001, 0.02]\) changes the treated-family challenge predictions only at the third decimal place.

This layer corresponds to the paper's **semimechanistic compatibility check** (§4.4, Table 3), not independent validation.

## Minimal BETSE perturbation family

The minimal local BETSE example uses the following perturbation family around a common control:

| Case | Perturbation |
|------|-------------|
| `na` | 1.3× Na membrane permeability |
| `k` | 1.3× K membrane permeability |
| `gj` | 20% random gap-junction block |
| `na_k` | combined Na + K |
| `na_gj` | combined Na + GJ |
| `k_gj` | combined K + GJ |

These are the perturbation classes used to estimate the local grounded geometry and to construct the semimechanistic bridge.

## Expected interpretation of the outputs

### Grounded local geometry

The grounded BETSE layer should be interpreted as showing:

- one explicit local route from biophysical variables to an effective cost metric,
- example matrices \(Q_\chi\), \(R_\chi\), and \(K_{\mathrm{mem}}^{(\chi)}\),
- a dominant cheap-write direction concentrated in wound-edge gap-junction contrast.

### Reduced fit

The reduced count-based fit should be interpreted as showing:

- a shared hidden-state geometry with
  - an immediate threshold (fixed at \(\theta_{\mathrm{imm}} = 0\)),
  - a lower challenge threshold (\(\hat\theta_{\mathrm{ch}} \approx -0.484\)),
  - and a reduced cryptic band between them,
- held-out transfer predictions for nigericin and monensin challenge penetrance near 15%.

### Semimechanistic bridge

The enriched bridge should be interpreted as showing:

- that a grounded simulator-derived feature subset (integrated wound-edge GJ contrast and Na/K-ATPase load) can recover the same treated-family ordering,
- and similar transfer predictions,
- without introducing free per-condition write amplitudes.

