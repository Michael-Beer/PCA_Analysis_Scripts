#!/usr/bin/env python3
"""
QM/MM transition-state frames (KPC-2 deacylation) vs the reactive complex of every enzyme.

This is an ADD-ON: it changes nothing in PCA_withbasin_*.py. It imports that script's own
feature code (so the QM/MM frames get exactly the same features) and reads the files it
already writes in each enzyme folder:
    enzyme_summary.csv, hdr_signature.csv, frames.csv
Keep it in the same folder as the main script and omega_loop_analysis.py.

What is compared
----------------
Each enzyme's reactive complex = AEC frames inside the TI HDR AND matching that enzyme's
TI-HDR signature (active-site / hinge / loop features with intervals from hdr_signature.csv).
The QM/MM TS frames get the same features (partner residues are labelled in Ambler numbers,
so 'hb_O71|237bb' means the same thing in every enzyme) and are scored against EACH
enzyme's signature:
  * signature score = fraction of that enzyme's signature features the TS frame satisfies,
    over the features that can be evaluated for a QM/MM frame (coverage reported)
  * the same restricted score for the enzyme's own AEC frames, AEC reactive-complex frames
    and TI-HDR frames, so the TS value can be read against them
  * percentile of the TS score within the enzyme's AEC frames
  * per feature: TS value vs the reactive-complex distribution (z, inside the interval?)
Then the TS match is compared with the reactive-complex fraction across the 8 enzymes
(Spearman, and carbapenemase vs carbapenem-inhibited exact tests).

What it cannot do without changing the main script
--------------------------------------------------
The HDR part of the reactive complex lives in each enzyme's own PCA plane, and the main script
does not save the eigenvectors, so QM/MM frames cannot be placed in the PC1/PC2 plane here.
This add-on therefore tests the active-site signature part only. (To add the HDR test, the main
script would have to save mean structure, eigenvectors, reference positions and C-alpha
selection per enzyme: a few lines, plus a rerun.)

Features that cannot be evaluated for the TS are skipped, not counted as misses: anything
involving a TI-only ligand atom (e.g. the new hydroxyl O01), and any feature whose column
cannot be built for the QM/MM topology. The attacking water is a separate molecule at the TS,
so water-derived features are not evaluated.

Example
-------
    python qmmm_ts_vs_reactive.py --module PCA_withbasin_1cc64c \
        --enzyme-dirs KPC2_hdr NMCA_hdr SFC1_hdr SME1_hdr BlaC_hdr CTXM16_hdr SHV1_hdr TEM1_hdr \
        --qmmm-top kpc2_qmmm.parm7 --qmmm-traj "kpc2_qmmm_ts*.nc" \
        --ts-frames 10 11 12 --match-frac 0.5 --outdir qmmm_ts
    (or --energy energy.dat [--ts-window 2] to take the highest-energy frame +/- 2 frames)

For a series of Amber restart snapshots instead of an existing trajectory, use:
    --qmmm-restarts frame_000.rst7 frame_001.rst7 frame_002.rst7
The add-on converts these ordered restart files to a temporary multi-frame DCD in --outdir.
"""
import argparse
import glob
import importlib
import re
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", default="PCA_withbasin_1cc64c",
                    help="module name of the main script (file name without .py)")
    ap.add_argument("--enzyme-dirs", nargs="+", required=True, help="output folders of the main script")
    ap.add_argument("--qmmm-top", required=True)
    qsrc = ap.add_mutually_exclusive_group(required=True)
    qsrc.add_argument("--qmmm-traj", help="QM/MM trajectory file(s), glob allowed")
    qsrc.add_argument("--qmmm-restarts", nargs="+",
                       help="ordered Amber restart files (.rst7/.restrt etc.); converted to a temporary DCD")
    ap.add_argument("--ts-frames", nargs="+", default=None,
                    help="0-based TS frame indices, or the word 'all' if every frame of the QM/MM "
                         "trajectory is a TS frame (e.g. a string-method ts.nc)")
    ap.add_argument("--ts-stride", type=int, default=1, help="with 'all': use every n-th frame")
    ap.add_argument("--energy", default=None, help="energy per frame (last column); highest = TS")
    ap.add_argument("--ts-window", type=int, default=0, help="frames either side of the highest-energy frame")
    ap.add_argument("--extra-set", action="append", default=[],
                    help="NAME=i,j,k : another labelled set of frames (e.g. reactant=0,1,2)")
    ap.add_argument("--match-frac", type=float, required=True,
                    help="MUST equal the --match-frac used in the main-script run")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="warn if fewer than this fraction of an enzyme's signature can be evaluated")
    ap.add_argument("--glu-resid", type=int, default=None)
    ap.add_argument("--hinge-motif", default="[TKR][VT][AG][GADN][PKAH][LSR][LI]R",
                    help="must equal the main-script value")
    ap.add_argument("--tail-atoms", nargs=4, default=None)
    ap.add_argument("--acyl-resname", default=None)
    ap.add_argument("--acyl-c", default=None)
    ap.add_argument("--acyl-o", default=None)
    ap.add_argument("--no-omega-loop-features", dest="omega_loop_features", action="store_false",
                    default=True, help="use if the main run used --no-omega-loop-features")
    ap.add_argument("--outdir", default="qmmm_ts")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Optional conversion of a series of Amber restart files to a multi-frame DCD
# ----------------------------------------------------------------------------
def prepare_qmmm_source(args, outdir):
    """Return the trajectory file list, converting restart snapshots if requested."""
    if args.qmmm_restarts:
        restart_files = []
        for item in args.qmmm_restarts:
            matches = sorted(glob.glob(item))
            restart_files.extend(matches if matches else ([item] if Path(item).is_file() else []))
        # Preserve the supplied order; expand globs in sorted order and remove duplicate paths.
        restart_files = list(dict.fromkeys(str(Path(f)) for f in restart_files))
        if not restart_files:
            raise SystemExit("No files found for --qmmm-restarts")

        dcd_path = Path(outdir) / "qmmm_restarts_converted.dcd"
        first = mda.Universe(args.qmmm_top, restart_files[0])
        n_atoms = first.atoms.n_atoms
        try:
            with mda.Writer(str(dcd_path), n_atoms=n_atoms) as writer:
                for i, rst in enumerate(restart_files):
                    u = first if i == 0 else mda.Universe(args.qmmm_top, rst)
                    if u.atoms.n_atoms != n_atoms:
                        raise SystemExit(
                            f"Restart {rst} has {u.atoms.n_atoms} atoms; expected {n_atoms} from {args.qmmm_top}"
                        )
                    writer.write(u.atoms)
        except Exception as exc:
            if isinstance(exc, SystemExit):
                raise
            raise SystemExit(
                "Could not convert the restart files to DCD. Check that each file is a readable "
                f"Amber restart compatible with --qmmm-top. Details: {exc}"
            ) from exc
        print(f"Converted {len(restart_files)} restart files to {dcd_path}")
        args.qmmm_traj = str(dcd_path.resolve())

    trajs = sorted(glob.glob(args.qmmm_traj)) if args.qmmm_traj else []
    if not trajs:
        raise SystemExit(f"No files match {args.qmmm_traj}")
    return trajs


# ----------------------------------------------------------------------------
# Features for the QM/MM frames (same construction as the main script)
# ----------------------------------------------------------------------------
def qmmm_features(hp, args, frames_by_set):
    ola = hp.ola
    trajs = sorted(glob.glob(args.qmmm_traj))
    if not trajs:
        raise SystemExit(f"No files match {args.qmmm_traj}")
    u0 = mda.Universe(args.qmmm_top, *trajs)
    prot0, seq0 = ola.protein_seq(u0)
    roles, loop = ola.find_ref_landmarks(prot0, seq0, args.glu_resid)
    ref = {"seq": seq0, "roles": roles, "loop": loop,
           "resnames": {k: prot0[v].resname for k, v in roles.items()}}
    ns = types.SimpleNamespace(stride=1, frames=None, acyl_resname=args.acyl_resname,
                               acyl_c=args.acyl_c, acyl_o=args.acyl_o)
    acyl = ola.find_acyl(u0, ns)
    if acyl is None:
        raise SystemExit("No acyl-enzyme residue (Ser OG bonded to the ligand carbon) in the QM/MM topology; "
                         "use --acyl-resname/--acyl-c/--acyl-o")
    lig_names = {a.name for a in acyl["C"][0].residue.atoms}
    n_traj = u0.trajectory.n_frames
    allf = sorted({f for fs in frames_by_set.values() for f in fs})
    if allf[-1] >= n_traj or allf[0] < 0:
        raise SystemExit(f"frame {allf[-1]} outside the QM/MM trajectory ({n_traj} frames)")
    ns.frames = allf
    r = ola.analyse_replica("QMMM", 0, args.qmmm_top, trajs, ref, ns)
    g = r["geom"].copy()
    g = g[[c for c in g.columns if not c.startswith("wat")]]
    if args.omega_loop_features:
        dih, meta = np.asarray(r["dih"]), list(r["dih_meta"])
        cols = {}
        for j, item in enumerate(meta):
            low = str(item).lower().replace("χ1", "chi1").replace("χ", "chi")
            tors = next((t for t in ("phi", "psi", "chi1") if t in low), None)
            if tors is None:
                continue
            pos = [int(x) for x in re.findall(r"(?<!\d)(\d{3})(?!\d)", low) if 164 <= int(x) <= 179]
            if pos and f"loop_dih_{tors}_{pos[0]}" not in cols:
                cols[f"loop_dih_{tors}_{pos[0]}"] = dih[:, j]
        if cols:
            g = pd.concat([g.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    c, h = hp.contact_features(args.qmmm_top, trajs, allf, ref, args, ns)
    hb = {}
    for (la, pa), grp in (h.groupby(["lig_atom", "partner"]) if len(h) else []):
        v = np.zeros(len(allf))
        v[grp["fi"].values] = 1.0
        hb[f"hb_{la}|{pa}"] = v
    F = pd.concat([g.reset_index(drop=True), c.reset_index(drop=True), pd.DataFrame(hb)], axis=1)
    F.insert(0, "frame", allf)
    return F, lig_names


def lig_atom_available(feature, lig_names):
    m = re.match(r"hb_(.+?)\|", feature)
    if not m:
        return True
    lab = m.group(1)
    return (lab in lig_names) if lab != "O31/O32" else bool({"O31", "O32"} & lig_names)


# ----------------------------------------------------------------------------
# Signature evaluation (mirrors the matching in the main script)
# ----------------------------------------------------------------------------
def circ_in(x, lo, hi):
    return ((x - lo) % 360.0) <= ((hi - lo) % 360.0)


def evaluate(df, spec, lig_names=None):
    """Matrix (frames x signature features): 1 match, 0 miss, NaN not evaluable."""
    out = {}
    for r in spec.itertuples():
        f = r.feature
        if f not in df.columns:
            if lig_names is not None and f.startswith("hb_") and lig_atom_available(f, lig_names):
                x = np.zeros(len(df))                     # atom exists, bond never seen: a genuine 0
            else:
                out[f] = np.full(len(df), np.nan)
                continue
        else:
            x = df[f].to_numpy(float)
        if bool(r.binary):
            m = (x == r.low)
        elif "dih" in f.lower():
            m = circ_in(x, r.low, r.high)
        else:
            m = (x >= r.low) & (x <= r.high)
        out[f] = np.where(np.isfinite(x), m.astype(float), np.nan)
    return pd.DataFrame(out, index=df.index)


def row_score(M):
    with np.errstate(all="ignore"):
        return np.nanmean(M.to_numpy(float), axis=1)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    hp = importlib.import_module(args.module)
    ola = hp.ola
    ENZ, COL, group_of = ola.ENZYMES, ola.COLOURS, ola.group_of
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- prepare trajectory source and choose QM/MM frames ---------------------------------
    sets = {}
    trajs_ = prepare_qmmm_source(args, out)
    n_traj = mda.Universe(args.qmmm_top, *trajs_).trajectory.n_frames
    if args.ts_frames and [str(x).lower() for x in args.ts_frames] == ["all"]:
        sets["TS"] = list(range(0, n_traj, args.ts_stride))
        print(f"Using all {len(sets['TS'])} frames of the QM/MM trajectory as TS frames")
    elif args.ts_frames:
        sets["TS"] = [int(x) for x in args.ts_frames]
    elif args.energy:
        e = np.loadtxt(args.energy)
        e = e[:, -1] if e.ndim > 1 else e
        ts = int(np.nanargmax(e))
        sets["TS"] = [i for i in range(ts - args.ts_window, ts + args.ts_window + 1) if 0 <= i < len(e)]
        print(f"Highest-energy frame {ts} (assumes a single barrier); TS set = {sets['TS']}")
    else:
        raise SystemExit(f"The QM/MM trajectory has {n_traj} frames. Say which are the transition state: "
                         f"--ts-frames 10 11 12 (0-based), --ts-frames all (every frame is a TS image, "
                         f"e.g. a string-method ts.nc; add --ts-stride N to thin), or --energy file "
                         f"[--ts-window k] to take the highest-energy frame +/- k.")
    for s_ in args.extra_set:
        name, fr = s_.split("=", 1)
        sets[name] = [int(x) for x in fr.split(",")]
    QF, lig_names = qmmm_features(hp, args, sets)
    QF.to_csv(out / "qmmm_frame_features.csv", index=False)

    # --- per enzyme -----------------------------------------------------------------------------
    rows, per_feat = [], []
    heat = {}
    for d in args.enzyme_dirs:
        d = Path(d)
        summ = pd.read_csv(d / "enzyme_summary.csv").iloc[0]
        enz = hp._canonical_enzyme_name(summ["enzyme"]) if hasattr(hp, "_canonical_enzyme_name") else summ["enzyme"]
        spec = pd.read_csv(d / "hdr_signature.csv")
        spec = spec[spec["enzyme"].astype(str) == str(summ["enzyme"])] if "enzyme" in spec else spec
        F = pd.read_csv(d / "frames.csv")
        if "reactive_complex" not in F.columns:
            raise SystemExit(f"{d}/frames.csv has no reactive_complex column; rerun the final main script")

        for name, fr in sets.items():
            Q = QF[QF.frame.isin(fr)].reset_index(drop=True)
            MQ = evaluate(Q, spec, lig_names)
            ok = [f for f in spec.feature if MQ[f].notna().all()]
            coverage = len(ok) / max(len(spec), 1)
            if coverage < args.min_coverage:
                print(f"  WARNING {enz}: only {len(ok)}/{len(spec)} signature features evaluable for the QM/MM frames")
            spec_ok = spec[spec.feature.isin(ok)]
            sq = row_score(MQ[ok]) if ok else np.full(len(Q), np.nan)

            # the same restricted signature on the enzyme's own frames
            MF = evaluate(F, spec_ok)
            sF = row_score(MF) if ok else np.full(len(F), np.nan)
            aec = (F.state == "AEC").to_numpy()
            rc = F.reactive_complex.to_numpy(bool)
            tihdr = ((F.state == "TI") & F.in_HDR.astype(bool)).to_numpy()
            row = {"enzyme": enz, "group": group_of(enz), "set": name, "n_frames": len(Q),
                   "n_signature": len(spec), "n_evaluable": len(ok), "coverage": coverage,
                   # taken from frames.csv: enzyme_summary.csv from the main script does not contain it
                   "AEC_reactive_complex": float(F.loc[aec, "reactive_complex"].astype(float).mean()),
                   "TS_score_mean": float(np.nanmean(sq)) if len(sq) else np.nan,
                   "TS_score_min": float(np.nanmin(sq)) if len(sq) else np.nan,
                   "TS_match_frac": float(np.nanmean(sq >= args.match_frac)) if len(sq) else np.nan,
                   "AEC_all_score": float(np.nanmean(sF[aec])),
                   "AEC_reactive_complex_score": float(np.nanmean(sF[aec & rc])) if (aec & rc).any() else np.nan,
                   "TI_HDR_score": float(np.nanmean(sF[tihdr])) if tihdr.any() else np.nan,
                   "TS_percentile_in_AEC": float(np.nanmean([(sF[aec] <= v).mean() for v in sq])) if ok else np.nan}
            rows.append(row)

            # per feature: where does the TS sit relative to the reactive-complex distribution?
            if name == "TS":
                for r in spec_ok.itertuples():
                    xr = F.loc[aec & rc, r.feature].to_numpy(float)
                    xr = xr[np.isfinite(xr)]
                    xt = Q[r.feature].to_numpy(float) if r.feature in Q else np.zeros(len(Q))
                    if "dih" in r.feature.lower() and len(xr):
                        mu = np.rad2deg(np.arctan2(np.sin(np.deg2rad(xr)).mean(), np.cos(np.deg2rad(xr)).mean()))
                        wrap = lambda v: (v - mu + 180.0) % 360.0 - 180.0
                        z = (np.nanmean(wrap(xt)) - np.nanmean(wrap(xr))) / (np.nanstd(wrap(xr)) or np.nan)
                    else:
                        z = (np.nanmean(xt) - xr.mean()) / (xr.std() or np.nan) if len(xr) else np.nan
                    frac = float(np.nanmean(MQ[r.feature]))
                    per_feat.append({"enzyme": enz, "feature": r.feature, "binary": r.binary,
                                     "interval_low": r.low, "interval_high": r.high,
                                     "TS_mean": float(np.nanmean(xt)),
                                     "reactive_complex_mean": float(xr.mean()) if len(xr) else np.nan,
                                     "reactive_complex_sd": float(xr.std()) if len(xr) else np.nan,
                                     "TS_z_vs_reactive_complex": z, "TS_frac_in_interval": frac})
                    heat.setdefault(enz, {})[r.feature] = frac
    summ_df = pd.DataFrame(rows)
    summ_df.to_csv(out / "ts_vs_reactive_summary.csv", index=False)
    pd.DataFrame(per_feat).to_csv(out / "ts_vs_reactive_features.csv", index=False)

    # --- across enzymes ---------------------------------------------------------------------------------
    ts = summ_df[summ_df.set == "TS"].set_index("enzyme").reindex([e for e in ENZ if e in set(summ_df.enzyme)])
    lines = ["TS frames scored against each enzyme's TI-HDR signature "
             f"(restricted to features evaluable for the QM/MM frames; match = score >= {args.match_frac}):",
             ts[["group", "n_evaluable", "n_signature", "AEC_reactive_complex", "TS_score_mean", "TS_match_frac",
                 "AEC_all_score", "AEC_reactive_complex_score", "TI_HDR_score",
                 "TS_percentile_in_AEC"]].round(3).to_string()]
    ok_ = ts[["AEC_reactive_complex", "TS_score_mean"]].dropna()
    if len(ok_) >= 4:
        rho, p = spearmanr(ok_.AEC_reactive_complex, ok_.TS_score_mean)
        lines.append(f"\nSpearman, AEC reactive-complex fraction vs TS signature score across {len(ok_)} enzymes: "
                     f"rho = {rho:.2f}, p = {p:.3f} (small n: descriptive)")
    tests = ola.table_tests(ts, ["TS_score_mean", "TS_match_frac", "TS_percentile_in_AEC",
                                 "AEC_reactive_complex"])
    tests.to_csv(out / "ts_group_tests.csv", index=False)
    lines += ["\nGroup tests (carbapenemase vs carbapenem-inhibited, enzyme level, exact permutation):",
              tests.round(3).to_string(index=False) if len(tests) else "  (needs >= 2 enzymes per group)"]
    extra = summ_df[summ_df.set != "TS"]
    if len(extra):
        lines += ["\nOther QM/MM frame sets, same scoring:",
                  extra.pivot_table(index="enzyme", columns="set", values="TS_score_mean").round(3).to_string()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print("\n" + text)

    # --- figures ---------------------------------------------------------------------------------------------------
    present = list(ts.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c, ttl in zip(axes, ("AEC_reactive_complex", "TS_score_mean"),
                          ("AEC reactive-complex fraction (main run)",
                           "QM/MM TS: signature score vs this enzyme's signature")):
        for i, e in enumerate(present):
            ax.bar(i, ts.loc[e, c], color=COL[group_of(e)])
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45)
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ts_vs_reactive_summary.png", dpi=200)
    plt.close(fig)
    if heat:
        H = pd.DataFrame(heat).reindex(columns=[e for e in present if e in heat])
        fig, ax = plt.subplots(figsize=(1.0 * H.shape[1] + 3, 0.3 * H.shape[0] + 2))
        im = ax.imshow(np.ma.masked_invalid(H.to_numpy(float)), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(H.shape[1]))
        ax.set_xticklabels(H.columns, rotation=45)
        ax.set_yticks(range(H.shape[0]))
        ax.set_yticklabels(H.index, fontsize=7)
        fig.colorbar(im, label="fraction of TS frames inside that enzyme's interval")
        fig.tight_layout()
        fig.savefig(out / "ts_feature_match_heatmap.png", dpi=200)
        plt.close(fig)
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()

