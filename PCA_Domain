#!/usr/bin/env python3

"""
Ensemble PCA comparison of two simulation sets (default: RS vs TS), now also usable for
comparing a DOMAIN inside a full (e.g. bifunctional) protein with the same domain simulated
alone.

All original options and outputs are unchanged. With no new options the script behaves
exactly as before (same selection, trimming, alignment and PCA for both ensembles).

Domain-in-full vs domain-alone: what changes
--------------------------------------------
The two systems have different topologies and different residue numbering, so give each
system its own selection of the domain C-alpha atoms. They must select the SAME number of
atoms in the SAME order (the script checks the count and the residue names), because the
two ensembles are superposed and projected on the same coordinates.

    python PCA_domain.py \
        --rs FullEnzyme --rs_top full.parm7 --selection_rs "protein and name CA and resid 201-400" \
        --ts DomainAlone --ts_top domain.parm7 --selection_ts "protein and name CA" \
        --label_rs "Domain in full enzyme" --label_ts "Domain alone" \
        --runs 3 --stride 100 --equil 0 --trim_termini 0 --balance

Useful options for this comparison
  --selection_rs / --selection_ts     per-system atom selection (default: --selection)
  --trim_rs / --trim_ts               per-system terminal trimming (default: --trim_termini).
                                      The ends of a free domain are real termini, the ends of the
                                      same domain in the full enzyme are linker connections:
                                      trimming both ends (or none) is usually the fairest choice.
  --fit_selection_rs / _ts            atoms used only for the superposition (e.g. a rigid core of
                                      the domain); the PCA still uses the full selection. The two
                                      selections must again match atom for atom.
  --rs_traj / --ts_traj               trajectory file pattern inside each folder, {run} = run number
                                      (default Run{run}/md_dry.nc)
  --runs_rs / --runs_ts, --equil_rs / --equil_ts, --stride_rs / --stride_ts
                                      per-system run counts / frames discarded / stride
  --balance                           subsample the larger ensemble to the size of the smaller one so
                                      neither dominates the pooled PCA
  --label_rs / --label_ts             legend labels
  --outdir                            output folder (default: current folder)

Extra outputs (in addition to the original files)
  PCA_projection.npz                  projections onto the first 10 PCs, ensemble label, run and frame
                                      number of every frame
  PC_residue_contributions.tsv        per-atom PC1/PC2 contributions with the residue number and name in
                                      BOTH systems (to map between the two numberings)
  PCA_variance.txt                    variance explained by each of the first 50 PCs
"""

import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt
from MDAnalysis import transformations as trans


# ============================================================
# Inputs
# ============================================================

parser = argparse.ArgumentParser(
    description="Ensemble PCA comparison of RS and TS ensembles "
                "(or of a domain in a full protein vs the domain alone)"
)

parser.add_argument(
    "--rs",
    default="RS",
    help="Ground state directory"
)

parser.add_argument(
    "--ts",
    default="TS",
    help="Transition state directory"
)

parser.add_argument(
    "--rs_top",
    default="RS.parm7",
    help="RS topology filename"
)

parser.add_argument(
    "--ts_top",
    default="TS.parm7",
    help="TS topology filename"
)

parser.add_argument(
    "--runs",
    type=int,
    default=10,
    help="Number of replicas"
)

parser.add_argument(
    "--stride",
    type=int,
    default=100,
    help="Frame stride"
)

parser.add_argument(
    "--equil",
    type=int,
    default=50000,
    help="Frames to discard"
)

parser.add_argument(
    "--selection",
    default="protein and name CA"
)

parser.add_argument(
    "--trim_termini",
    type=int,
    default=5,
    help="Number of N- and C-terminal residues to exclude from the analysis (set to 0 to keep all residues)"
)

# ---- new options (all default to the original behaviour) ----

parser.add_argument(
    "--label_rs", default="RS",
    help="Label of the first ensemble in the plots"
)

parser.add_argument(
    "--label_ts", default="TS",
    help="Label of the second ensemble in the plots"
)

parser.add_argument(
    "--selection_rs", default=None,
    help="Atom selection for the first system (default: --selection)"
)

parser.add_argument(
    "--selection_ts", default=None,
    help="Atom selection for the second system (default: --selection)"
)

parser.add_argument(
    "--trim_rs", type=int, default=None,
    help="Terminal residues trimmed from the first selection (default: --trim_termini)"
)

parser.add_argument(
    "--trim_ts", type=int, default=None,
    help="Terminal residues trimmed from the second selection (default: --trim_termini)"
)

parser.add_argument(
    "--fit_selection_rs", default=None,
    help="Atoms used only for the superposition in the first system (default: the PCA atoms)"
)

parser.add_argument(
    "--fit_selection_ts", default=None,
    help="Atoms used only for the superposition in the second system (default: the PCA atoms)"
)

parser.add_argument(
    "--rs_traj", default="Run{run}/md_dry.nc",
    help="Trajectory pattern inside the first folder; {run} = run number"
)

parser.add_argument(
    "--ts_traj", default="Run{run}/md_dry.nc",
    help="Trajectory pattern inside the second folder; {run} = run number"
)

parser.add_argument("--runs_rs", type=int, default=None, help="Replicas of the first system (default: --runs)")
parser.add_argument("--runs_ts", type=int, default=None, help="Replicas of the second system (default: --runs)")
parser.add_argument("--equil_rs", type=int, default=None, help="Frames discarded, first system (default: --equil)")
parser.add_argument("--equil_ts", type=int, default=None, help="Frames discarded, second system (default: --equil)")
parser.add_argument("--stride_rs", type=int, default=None, help="Stride, first system (default: --stride)")
parser.add_argument("--stride_ts", type=int, default=None, help="Stride, second system (default: --stride)")

parser.add_argument(
    "--balance", action="store_true",
    help="Subsample the larger ensemble to the size of the smaller one before the PCA"
)

parser.add_argument(
    "--allow_resname_mismatch", action="store_true",
    help="Continue if the residue names of the two selections differ (e.g. different protonation names)"
)

parser.add_argument(
    "--outdir", default=".",
    help="Output folder"
)

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


def out(name):
    return os.path.join(args.outdir, name)


def pick(value, default):
    return default if value is None else value


# per-system settings (fall back to the shared options)
SET = {}

for key in ("rs", "ts"):

    SET[key] = {
        "directory": getattr(args, key),
        "topology": getattr(args, key + "_top"),
        "traj": getattr(args, key + "_traj"),
        "runs": pick(getattr(args, "runs_" + key), args.runs),
        "equil": pick(getattr(args, "equil_" + key), args.equil),
        "stride": pick(getattr(args, "stride_" + key), args.stride),
        "selection": pick(getattr(args, "selection_" + key), args.selection),
        "trim": pick(getattr(args, "trim_" + key), args.trim_termini),
        "fit_selection": getattr(args, "fit_selection_" + key),
        "label": getattr(args, "label_" + key),
    }


##### Atom selections ########

def select_pca_atoms(universe, settings):

    atoms = universe.select_atoms(
        settings["selection"]
    )

    # Exclude flexible N-/C-terminal residues from the
    # analysis. Relies on select_atoms returning atoms in
    # residue order for a single protein chain.
    if settings["trim"] > 0:
        atoms = atoms[settings["trim"]:-settings["trim"]]

    return atoms


def select_fit_atoms(universe, settings, pca_atoms):

    # default: superpose on exactly the atoms used for the PCA (original behaviour)
    if settings["fit_selection"] is None:
        return pca_atoms

    return universe.select_atoms(
        settings["fit_selection"]
    )


def traj_path(settings, run):

    return os.path.join(
        settings["directory"],
        settings["traj"].format(run=run)
    )


##### Load Trajectory ########

def load_ensemble(settings, reference_fit):

    all_coords = []
    all_runs = []
    all_frames = []
    info = None

    print("\nLoading:", settings["directory"])

    for run in range(1, settings["runs"] + 1):

        top = os.path.join(
            settings["directory"],
            settings["topology"]
        )

        traj = traj_path(settings, run)

        print(f"  Run {run}: {traj}")

        u = mda.Universe(
            top,
            traj
        )

        protein = select_pca_atoms(
            u,
            settings
        )

        fit_atoms = select_fit_atoms(
            u,
            settings,
            protein
        )

        if len(fit_atoms) != len(reference_fit):
            raise SystemExit(
                f"\nSuperposition atoms differ: {settings['label']} selects "
                f"{len(fit_atoms)} atoms but the reference selects {len(reference_fit)}. "
                f"Use --selection_*/--fit_selection_* so both systems select the same atoms."
            )

        if info is None:
            info = {
                "n_atoms": len(protein),
                "resids": protein.resids.copy(),
                "resnames": protein.resnames.copy(),
            }
        elif len(protein) != info["n_atoms"]:
            raise SystemExit(
                f"\nRun {run} of {settings['label']} selects {len(protein)} atoms, "
                f"run 1 selected {info['n_atoms']}"
            )

        # -----------------------------
        # Align trajectory on the fly
        # -----------------------------

        u.trajectory.add_transformations(
            trans.fit_rot_trans(
                fit_atoms,
                reference_fit
            )
        )

        frames = []
        frame_numbers = []

        for ts in u.trajectory[settings["equil"]::settings["stride"]]:

            frames.append(
                protein.positions.copy()
            )

            frame_numbers.append(
                ts.frame
            )

        if len(frames) == 0:
            raise SystemExit(
                f"\n{traj}: no frames left after --equil {settings['equil']} "
                f"(trajectory has {u.trajectory.n_frames} frames)"
            )

        frames = np.asarray(
            frames,
            dtype=np.float32
        )

        print(
            f"     retained frames: {len(frames)}"
        )

        all_coords.append(frames)
        all_runs.append(np.full(len(frames), run))
        all_frames.append(np.asarray(frame_numbers))

        del u

    return (
        np.concatenate(all_coords, axis=0),
        np.concatenate(all_runs),
        np.concatenate(all_frames),
        info
    )


# ============================================================
# Build a common alignment reference
# ============================================================
# RS and TS must be superposed onto the SAME fixed reference
# structure, otherwise differences between the two projected
# clouds on the PCA plane could just be an alignment artefact
# rather than real ligand-driven conformational change.
# Frame 0 of RS Run1 is used here as that fixed reference.
# This assumes the two selections contain matching atoms in the
# same order (true if it's the same protein with only the
# ligand/topology parameters differing, or - for a domain in a
# full protein vs the domain alone - if each system's selection
# picks out the same domain residues; this is checked below).

ref_top = os.path.join(SET["rs"]["directory"], SET["rs"]["topology"])
ref_traj = traj_path(SET["rs"], 1)

ref_universe = mda.Universe(ref_top, ref_traj)
ref_universe.trajectory[0]

reference = select_pca_atoms(ref_universe, SET["rs"])

reference_fit = select_fit_atoms(ref_universe, SET["rs"], reference)

# ============================================================
# Load RS and TS
# ============================================================

RS, RS_run, RS_frame, RS_info = load_ensemble(
    SET["rs"],
    reference_fit
)


TS, TS_run, TS_frame, TS_info = load_ensemble(
    SET["ts"],
    reference_fit
)


# ============================================================
# Check that the two selections describe the same atoms
# ============================================================

if RS_info["n_atoms"] != TS_info["n_atoms"]:
    raise SystemExit(
        f"\n{SET['rs']['label']} selects {RS_info['n_atoms']} atoms but "
        f"{SET['ts']['label']} selects {TS_info['n_atoms']}. The ensembles can only be compared on "
        f"the same atoms: adjust --selection_rs/--selection_ts (and --trim_rs/--trim_ts)."
    )

mismatch = np.where(RS_info["resnames"] != TS_info["resnames"])[0]

print(
    f"\nCompared atoms: {RS_info['n_atoms']}"
)

print(
    f"  {SET['rs']['label']}: residues {RS_info['resids'][0]}-{RS_info['resids'][-1]} "
    f"({RS_info['resnames'][0]}...{RS_info['resnames'][-1]})"
)

print(
    f"  {SET['ts']['label']}: residues {TS_info['resids'][0]}-{TS_info['resids'][-1]} "
    f"({TS_info['resnames'][0]}...{TS_info['resnames'][-1]})"
)

if len(mismatch) > 0:

    shown = ", ".join(
        f"{RS_info['resnames'][i]}{RS_info['resids'][i]} vs {TS_info['resnames'][i]}{TS_info['resids'][i]}"
        for i in mismatch[:8]
    )

    message = (
        f"{len(mismatch)} of {RS_info['n_atoms']} residue names differ between the two selections "
        f"(first: {shown})"
    )

    if args.allow_resname_mismatch:
        print("\nWARNING:", message)
    else:
        raise SystemExit(
            "\n" + message + ". The selections probably do not cover the same residues "
            "(check the residue ranges, tags/caps and --trim_*), or use --allow_resname_mismatch "
            "if only the names differ (e.g. HIE/HID)."
        )

print("\nRS frames:", len(RS))
print("TS frames:", len(TS))


# ============================================================
# Optional: equalise the number of frames
# ============================================================

if args.balance and len(RS) != len(TS):

    n_keep = min(len(RS), len(TS))

    def even_subsample(n_total):

        return np.linspace(
            0,
            n_total - 1,
            n_keep
        ).astype(int)

    if len(RS) > n_keep:
        idx = even_subsample(len(RS))
        RS, RS_run, RS_frame = RS[idx], RS_run[idx], RS_frame[idx]

    if len(TS) > n_keep:
        idx = even_subsample(len(TS))
        TS, TS_run, TS_frame = TS[idx], TS_run[idx], TS_frame[idx]

    print(f"\nBalanced ensembles: {len(RS)} frames each")


# ============================================================
# Convert to coordinate matrix
# ============================================================

def flatten(coords):

    return coords.reshape(
        coords.shape[0],
        coords.shape[1]*3
    )


X_RS = flatten(RS)

X_TS = flatten(TS)


X = np.concatenate(
    [
        X_RS,
        X_TS
    ]
)


labels = np.array(
    ["RS"]*len(X_RS)
    +
    ["TS"]*len(X_TS)
)



# ============================================================
# Remove mean structure
# ============================================================

mean_structure = X.mean(axis=0)

X -= mean_structure



# ============================================================
# PCA
# ============================================================

print("\nCalculating covariance matrix")

cov = np.cov(
    X,
    rowvar=False
)


print("Diagonalising")

eigval, eigvec = np.linalg.eigh(
    cov
)


order = np.argsort(
    eigval
)[::-1]


eigval = eigval[order]

eigvec = eigvec[:,order]


projection = X @ eigvec



# ============================================================
# Eigenvalue spectrum
# ============================================================

plt.figure(figsize=(6,4))

plt.semilogy(
    eigval[:50],
    marker="o"
)

plt.xlabel(
    "Principal component"
)

plt.ylabel(
    "Variance (Å$^2$)"
)

plt.tight_layout()

plt.savefig(
    out("PCA_eigenvalues.png"),
    dpi=300
)



# ============================================================
# Cumulative variance
# ============================================================

cumulative = (
    np.cumsum(eigval)
    /
    np.sum(eigval)
)


plt.figure(figsize=(6,4))

plt.plot(
    cumulative[:50],
    marker="o"
)

plt.xlabel(
    "Principal component"
)

plt.ylabel(
    "Cumulative variance"
)

plt.tight_layout()

plt.savefig(
    out("PCA_cumulative_variance.png"),
    dpi=300
)


with open(out("PCA_variance.txt"), "w") as handle:

    handle.write("PC\tVariance\tFraction\tCumulative\n")

    for i in range(min(50, len(eigval))):

        handle.write(
            f"{i + 1}\t{eigval[i]:.6g}\t{eigval[i] / eigval.sum():.6f}\t{cumulative[i]:.6f}\n"
        )



# ============================================================
# PC1-PC2 projection
# ============================================================

RS_mask = labels=="RS"
TS_mask = labels=="TS"


plt.figure(figsize=(6,6))


plt.scatter(
    projection[RS_mask,0],
    projection[RS_mask,1],
    s=2,
    alpha=0.25,
    label=SET["rs"]["label"]
)


plt.scatter(
    projection[TS_mask,0],
    projection[TS_mask,1],
    s=2,
    alpha=0.25,
    label=SET["ts"]["label"]
)


plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    out("PCA_RS_TS_projection.png"),
    dpi=300
)


n_pc_saved = min(10, projection.shape[1])

np.savez(
    out("PCA_projection.npz"),
    projection=projection[:, :n_pc_saved],
    ensemble=labels,
    run=np.concatenate([RS_run, TS_run]),
    frame=np.concatenate([RS_frame, TS_frame]),
    label_rs=SET["rs"]["label"],
    label_ts=SET["ts"]["label"]
)



# ============================================================
# Residue contributions
# ============================================================

n_atoms = len(
    RS[0]
)


pc1 = eigvec[:,0].reshape(
    n_atoms,
    3
)

pc2 = eigvec[:,1].reshape(
    n_atoms,
    3
)


pc1_contribution = np.linalg.norm(
    pc1,
    axis=1
)


pc2_contribution = np.linalg.norm(
    pc2,
    axis=1
)



np.savetxt(
    out("PC1_residue_contribution.dat"),
    pc1_contribution
)

np.savetxt(
    out("PC2_residue_contribution.dat"),
    pc2_contribution
)


with open(out("PC_residue_contributions.tsv"), "w") as handle:

    handle.write(
        "index\tresid_rs\tresname_rs\tresid_ts\tresname_ts\tPC1\tPC2\n"
    )

    for i in range(n_atoms):

        handle.write(
            f"{i}\t{RS_info['resids'][i]}\t{RS_info['resnames'][i]}\t"
            f"{TS_info['resids'][i]}\t{TS_info['resnames'][i]}\t"
            f"{pc1_contribution[i]:.6f}\t{pc2_contribution[i]:.6f}\n"
        )



plt.figure(figsize=(8,3))

plt.plot(
    pc1_contribution
)

plt.xlabel(
    "Residue index"
)

plt.ylabel(
    "PC1 contribution"
)

plt.tight_layout()

plt.savefig(
    out("PC1_residue_contribution.png"),
    dpi=300
)



print("\nFinished PCA")
