# PCA_withbasin and PCA_QMMM_withbasin

Two scripts for asking whether the acylenzyme (AEC) ensemble of each class A β-lactamase is
**pre-organised towards the tetrahedral-intermediate (TI) ensemble**, and whether carbapenemases
(KPC-2, NMC-A, SFC-1, SME-1) are better pre-organised than the carbapenem-inhibited enzymes
(BlaC, CTX-M-16, SHV-1, TEM-1).

| Script | Purpose |
|---|---|
| `PCA_withbasin.py` | Per enzyme: reproduces the RS/TS Cα PCA from `PCA.py`, defines the **TI high-density region (HDR)**, describes what the protein looks like there (hydrogen bonds to meropenem, oxyanion hole, omega loop, hinge, hydration), and measures how often AEC frames look like it (the **reactive complex**). `--aggregate` compares the groups. |
| `PCA_QMMM_withbasin.py` | Add-on: scores KPC-2 deacylation **QM/MM transition-state frames** against each enzyme's TI-HDR signature, so the TS can be compared with the reactive complex of all eight enzymes. Does not modify `PCA_withbasin.py`. |

## What the results can and cannot say

* The MD analysis measures **pre-organisation**: overlap of the AEC ensemble with the TI ensemble and
  its active-site pattern. This is the sense of "transition state-like conformation (TLC)" used in the
  ACS Catal. 2024 paper (14, 4379): an enzyme conformation defined from the TS/TI-bound ensemble.
* The MD analysis alone does **not** show that those conformations resemble the deacylation transition
  state. `PCA_QMMM_withbasin.py` is the link: it asks whether QM/MM TS geometries satisfy the same
  TI-derived signature. State clearly in the manuscript which claims rest on the TI proxy and which on QM/MM.
* With 4 vs 4 enzymes the smallest possible exact-permutation p-value is 0.029. Report effect sizes with
  the per-enzyme points, not only p-values.

## Files that must sit together

```
PCA_withbasin.py                 main script
PCA_QMMM_withbasin.py            QM/MM add-on
omega_loop_analysis.py           (or OmegaLoop.py)   feature extraction, landmarks, group tests
loop_cluster_structures.py       (or OmegaLoopRepresentative.py)   dihedral signature helpers
ti_basin_ca_pca.py               optional: adds OVL/BC overlap coefficients to the summary
```

Python 3.8+ with `numpy pandas scipy scikit-learn matplotlib MDAnalysis biopython`.
(The Biopython "Bio.Application deprecated" warning is harmless.)

## Input layout

One folder per state per enzyme, as for `PCA.py`:

```
<enzyme>/            acylenzyme (RS):  system.parm7, Run1/ Run2/ Run3/ ... each with prod.nc (and md_dry.nc if you have it)
../TI/<enzyme>/      tetrahedral intermediate (TS): same layout
```

* The PCA frames come from `--dry-traj` (default `{dir}/Run{run}/md_dry.nc`). **If you have no `md_dry.nc`, use the
  solvated file for both**: `--dry-traj "{dir}/Run{run}/prod.nc"`.
* The hydrogen-bond and hydration features need the **solvated** trajectory (`--wet-top`, `--wet-traj`).
* `{dir}` is replaced by the RS or TS folder. **Both wet patterns must contain `{dir}`**, otherwise the TI ensemble
  would silently read the RS files (the script stops if they do not).
* `--ts_top` is relative to the TS folder (`system.parm7`). A path that exists from the working folder also works.
* The script checks that dry and wet frames correspond (Cα RMSD between matching frames) and stops if not.
  Set `--wet-per-dry` if the wet file has more frames per dry frame.
* `--equil` is the number of **frames** discarded from each run; the default (50000) is only right if your
  trajectories are that long. Set it to what you want to discard (0 keeps everything).

## PCA_withbasin.py

### What it does, per enzyme

1. **PCA exactly as `PCA.py`.** Cα atoms (`protein and name CA`, `--trim_termini` residues removed from each end),
   frames `[equil::stride]` of every run, all frames superposed on frame 0 of RS Run1 (same result as
   `trans.fit_rot_trans`), RS and TS pooled, mean removed, covariance diagonalised, projection `X @ eigvec`.
2. **TI HDR.** The region of the PC1/PC2 plane containing `--pct` % of the TI frames (2D Gaussian KDE).
   Several percentiles may be given (`--pct 90 50`); the first is used for the profile.
3. **Features** (from the wet trajectory, for exactly the PCA frames):
   * active site: Glu166–Lys73, Glu166–Asn170, Arg164–Asp179, Lys73–Ser130, Glu166 to the acyl/TI carbon,
     oxyanion-hole distances to Ser70 N and residue-237 N (`oxy_N70`, `oxy_N237`), loop Rg, loop–Ser130 distance
   * ligand: hydrogen bonds to the meropenem part of the acyl/TI residue (totals, water vs protein, `O71_hbond`,
     `O31/O32` carboxylate, and every `hb_<ligand atom>|<position>` pair above `--min-hb-occ`), `N4_O71` distance,
     hinge-region Cα centre of mass to ligand centre of mass (`hinge_COM_lig`), hydroxyethyl tail dihedral `dih_tail`
     (C7–C25–C61–O61; override with `--tail-atoms`), waters within 4 Å of the ligand
   * omega loop: φ/ψ/χ1 for Ambler 164–179 (`loop_dih_*`; switch off with `--no-omega-loop-features`)
   * deacylating-water features are **excluded**, because that water is not present in the TI
4. **Numbering.** Partner residues are labelled in Ambler numbering anchored on the **nearest** catalytic landmark
   (Lys73, Ser130, Glu166, Lys234), so an insertion elsewhere does not shift a label (Asn132 stays 132 in SME-1/NMC-A).
   Labels are position-only (`hb_O71|237bb`); the residue types seen are in `hb_partner_residues.csv`, with protonation
   variants merged (GLH→GLU, LYN→LYS, ASH→ASP, HID/HIE/HIP→HIS). The anchors are printed for each run: check them.
5. **Profile.** TI-HDR frames vs all other frames (Cohen's d; angles are compared circularly).
6. **Signature.** Up to `--sig-n` features that characterise the TI HDR:
   * mandatory first: the oxyanion contacts (`oxy_N70`, and `hb_O71|237bb` or `oxy_N237`)
   * then features with |d| ≥ 0.25 (see the code; an older docstring says 0.5), skipping near-duplicates
     (|r| > `--sig-corr`), Glu166 hydrogen bonds other than hydroxyl-O6/Asn132 with Glu166, and O71 hydrogen bonds
     other than at positions 70/237
   * interval per feature: central 90 % of the TI-HDR values for distances (5–95 %), interquartile range for `N4_O71`,
     circular 10–90 % around the circular mean for dihedrals, target value for yes/no features
   * a frame **matches** if it satisfies at least `--match-frac` of the signature features
7. **Reactive complex = in the TI HDR AND matching the signature.** Reported for AEC frames, with the TI-HDR frames
   themselves as the positive control (and `_rel_control` ratios).
8. Also: convex-hull and covariance-trace ratios (TI over AEC; the TLC criterion is a more restricted landscape),
   fraction of TI frames inside the AEC HDR, OVL/BC overlap (if `ti_basin_ca_pca.py` is present), representative frames.

### Run it

One enzyme, from its parent folder:

```bash
python PCA_withbasin.py --enzyme BlaC --rs BlaC --ts ../TI/BlaC \
    --rs_top system.parm7 --ts_top system.parm7 \
    --runs 3 --stride 100 --equil 0 \
    --dry-traj "{dir}/Run{run}/prod.nc" \
    --wet-top "{dir}/system.parm7" --wet-traj "{dir}/Run{run}/prod.nc" \
    --match-frac 0.5 --outdir BlaC_hdr
```

Repeat for each of the eight enzymes (`--enzyme` is one of KPC2 NMCA SFC1 SME1 BlaC CTXM16 SHV1 TEM1), then combine:

```bash
python PCA_withbasin.py --aggregate KPC2_hdr NMCA_hdr SFC1_hdr SME1_hdr BlaC_hdr CTXM16_hdr SHV1_hdr TEM1_hdr \
    --outdir hdr_groups
```

Use the **same options for every enzyme** (especially `--stride`, `--equil`, `--pct`, `--sig-n`, `--match-frac`);
changing them per enzyme makes the enzymes incomparable.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `--runs` | 10 | replicas per state |
| `--stride`, `--equil` | 100, 50000 | frame stride; frames to discard per run |
| `--selection`, `--trim_termini` | `protein and name CA`, 5 | PCA atoms; residues trimmed from each end |
| `--dry-traj`, `--wet-top`, `--wet-traj` | see above | file patterns (`{dir}`, `{run}`) |
| `--wet-per-dry`, `--map-tol` | 1, 0.1 Å | dry→wet frame factor; tolerance of the frame check |
| `--pct` | 90 50 | HDR percentiles (first is primary) |
| `--sig-n`, `--sig-corr` | 6, 0.9 | signature size; duplicate-feature cut-off |
| `--match-frac` | 0.8 | fraction of signature features a frame must match |
| `--min-hb-occ` | 0.10 | minimum occupancy for a hydrogen-bond pair to become a feature |
| `--hinge-motif` | `[TKR][VT][AG][GADN][PKAH][LSR][LI]R` | regex for the hinge residues (checked before the long extraction) |
| `--tail-atoms` | `C7 C25 C61 O61` | atoms of the hydroxyethyl dihedral |
| `--glu-resid` | – | topology resid of Glu166 if the motif search fails |
| `--acyl-resname/-c/-o` | auto | acyl/TI residue name and carbonyl atoms if detection fails |

### Output files (per enzyme folder)

| File | Contents |
|---|---|
| `summary.txt` | headline numbers, signature, top features |
| `enzyme_summary.csv` | one row per enzyme (feeds `--aggregate`) |
| `replica_metrics.csv` | per state and run: HDR fractions, signature match |
| `frames.csv` | every analysed frame: state, run, frame numbers, PC1/PC2, `in_HDR`, all features, `sig_score`, `sig_match`, `reactive_complex` |
| `hdr_feature_profile.csv` | each feature in TI-HDR, TI outside, AEC in/out of the HDR, d |
| `hdr_signature.csv` | signature features with their intervals |
| `hdr_hbond_partners.csv`, `hb_partner_residues.csv` | ligand hydrogen-bond partners and the residue types at each position |
| `loop_dihedral_signature_TI.csv`, `_AEC.csv` | omega-loop dihedral differences (in vs out of the HDR) |
| `representative_frames.csv` | replica and frame numbers (1-based column for cpptraj) |
| `pca_hdr.png`, `hdr_profile_top_features.png` | PCA plane with HDR contours; strongest features |

`--aggregate` writes `all_enzymes_summary.csv`, `group_tests.csv`, `group_summary.txt`, `group_comparison.png`,
`profile_d_by_enzyme.csv`, `profile_consistency_across_enzymes.csv`, `hbond_occupancy_TI_HDR_by_enzyme.csv`.

## PCA_QMMM_withbasin.py

### What it does

Scores QM/MM frames from the KPC-2 deacylation (e.g. the TS images of a string-method run) against **each
enzyme's own TI-HDR signature**, using the same feature code as `PCA_withbasin.py` (it imports it).

* A TS frame's **signature score** = fraction of that enzyme's signature features it satisfies, over the features
  that can be evaluated for the QM/MM system (coverage is reported).
* The same restricted score is computed for the enzyme's AEC frames, AEC reactive-complex frames and TI-HDR frames,
  so the TS value can be read against them, plus the percentile of the TS score within the AEC frames.
* Per feature: TS value vs the reactive-complex distribution (z-score, fraction of TS frames inside the interval).
* Across enzymes: Spearman correlation between each enzyme's reactive-complex fraction and its TS score, and
  carbapenemase vs carbapenem-inhibited exact tests.

### Run it

```bash
python PCA_QMMM_withbasin.py --module PCA_withbasin \
    --enzyme-dirs KPC2_hdr NMCA_hdr SFC1_hdr SME1_hdr BlaC_hdr CTXM16_hdr SHV1_hdr TEM1_hdr \
    --qmmm-top complex.parm7 --qmmm-traj "ts*.nc" \
    --ts-frames all --match-frac 0.5 --outdir qmmm_ts
```

* Choose the TS frames with one of: `--ts-frames 10 11 12` (0-based), `--ts-frames all` (every frame of the trajectory is
  a TS image; thin with `--ts-stride N`), or `--energy energy.dat [--ts-window k]` (highest-energy frame ± k,
  assumes a single barrier). With none of these the script reports the number of frames and stops.
* `--module` is the file name of the main script without `.py`.
* **`--match-frac` must equal the value used in the main-script runs**, and `--hinge-motif`/`--tail-atoms` must match too.
* `--extra-set reactant=0,1,2` scores other frame sets the same way (e.g. to see where the TS sits relative to the reactant).

### Outputs

`ts_vs_reactive_summary.csv`, `ts_vs_reactive_features.csv`, `ts_group_tests.csv`, `qmmm_frame_features.csv`,
`summary.txt`, `ts_vs_reactive_summary.png`, `ts_feature_match_heatmap.png`.

### Limits

* **No HDR test.** The HDR half of the reactive complex lives in each enzyme's own PCA plane; `PCA_withbasin.py`
  does not save the eigenvectors, so QM/MM frames cannot be placed in it. The add-on tests the active-site
  signature only. (Saving the mean structure, eigenvectors, reference positions and Cα selection per enzyme, plus a
  rerun, would add the HDR test.)
* Features involving TI-only ligand atoms (e.g. the new hydroxyl O01) or the attacking water cannot be evaluated
  for a TS; they are skipped, not counted as misses.
* The signature is enzyme-specific (different features and intervals per enzyme), so scores across enzymes are
  not on a strictly common footing.
* Ligand atom names (`O71`, `N4`, `O31/O32`, tail atoms) must be the same in the AEC, TI and QM/MM topologies.

## Typical workflow

1. For each enzyme, run `PCA_withbasin.py` with identical options; check `summary.txt` and the printed Ambler anchors.
2. `PCA_withbasin.py --aggregate ...` for the group comparison.
3. `PCA_QMMM_withbasin.py ...` with the KPC-2 QM/MM TS frames.

## Troubleshooting

| Message | Cause / fix |
|---|---|
| `FileNotFoundError ... md_dry.nc` | no dry trajectory: pass `--dry-traj "{dir}/Run{run}/prod.nc"` |
| `... has no {dir} placeholder` | the wet/dry pattern would read the RS files for TI: add `{dir}` |
| `no frames after --equil` | `--equil` (frames) is larger than the trajectory: lower it |
| `dry and wet frames do not correspond` | wrong `--wet-per-dry`, or the wet file is not the one `md_dry.nc` came from |
| `Hinge motif ... not found` | the regex does not occur in that enzyme: set `--hinge-motif` |
| `could not uniquely locate Glu166` | use `--glu-resid` |
| `numbering anchors` warning | fewer than four anchors found: labels far from them may be shifted; check the sequence |
| `possible_numbering_shifts` warning (aggregate) | the same ligand atom bonds position N in one enzyme and N+1 in another: a numbering shift, not biology |
| `give --ts-frames or --energy` (add-on) | choose the TS frames as above |
| `NCDF trajectory ... does not contain frame information` (add-on) | the file has no `frame` dimension (for example a single-structure NetCDF restart written by the string method). The add-on currently needs a standard Amber trajectory. Inspect it with `python -c "from scipy.io import netcdf_file as n; f=n('ts.nc',mmap=False); print(f.dimensions); print({k:v.shape for k,v in f.variables.items()})"` (or `ncdump -h ts.nc`) and convert, e.g. `cpptraj`: `parm complex.parm7`, `trajin ts.nc` (several image files: one `trajin` per file), `trajout ts_traj.nc netcdf`, `run`. Then use `ts_traj.nc`. If the `.top` extension is not recognised, link it as `.parm7`. |

## Checks before trusting a result

* `enzyme_summary.csv` must contain `AEC_reactive_complex` and `TIHDR_reactive_complex`. If a version of the main
  script writes that file **before** those two values are added, the aggregate step silently drops the reactive
  complex from its tests; write `enzyme_summary.csv` again after they are set. (The QM/MM add-on reads the
  fraction from `frames.csv`, so it is unaffected.)
* Look at the printed hinge match and the anchor line for every enzyme.
* Glu166/Lys73/Asp179 protonation states should be consistent across enzymes; Glu166 hydrogen bonds are excluded
  from the signature for that reason.
* Do not tune the feature set, intervals or thresholds to raise one enzyme's reactive-complex percentage. Choose them
  once, apply them to all enzymes, and run a sensitivity check (`--pct`, `--match-frac`, `--sig-n`). If an enzyme
  stays low across settings, per-feature match rates show which part of the signature it does not satisfy.
