#!/usr/bin/env python3
"""
What does the protein look like in the TI high-density region (HDR) of the
RS/TS PCA, and how often does the AEC ensemble look like that?

Run once per enzyme, in the same layout as your PCA.py (RS/ and TS/ folders), then
once with --aggregate to compare carbapenemases with non-carbapenemases.

Space (identical to PCA.py)
---------------------------
  * selection 'protein and name CA', first/last --trim_termini residues removed
  * frames [equil::stride] of every Run<n>/md_dry.nc (RS = acylenzyme, TS = TI)
  * every frame superposed on frame 0 of RS Run1 (same result as
    trans.fit_rot_trans), RS and TS pooled, mean removed, covariance
    diagonalised (np.linalg.eigh), projection = X @ eigvec
HDR = the region of the PC1/PC2 plane containing --pct % of the TI frames (2D
Gaussian KDE). Several percentiles can be given (--pct 90 50); the first one
is used for the profile.

Profile (needs the SOLVATED trajectory: --wet-top/--wet-traj, same frames)
--------------------------------------------------------------------------
For exactly the frames used in the PCA, from the wet trajectory:
  * omega-loop conformation: phi/psi/chi1 for Ambler 164-179, loop Rg, loop
    COM-Ser130 distance
  * active site: Glu166-Lys73, Glu166-Asn170, Arg164-Asp179, Lys73-Ser130,
    Glu166 to the acyl/TI carbon, oxyanion-hole distances (Ser70 N, residue 237 N)
  * hydrogen bonds to the ligand (meropenem part of the acyl/TI residue):
    totals, by water/protein, and per (ligand atom | partner residue) pair
  * Ligand geometry/positioning: N4-O71 distance and hinge-region C-alpha COM
to ligand COM distance (--hinge-resids)
  * hydration: waters within 4 A of the ligand, within 5 A of the acyl carbon,
    within 3.5 A of Glu166 O, plus the Bürgi-Dunitz/bridging-water metrics
Compared between TI-HDR frames and all other frames (Cohen's d, descriptive).
The HDR "signature" is the top --sig-n features with |d| >= 0.5; a frame matches
a feature if it lies in the central 90 % of the TI-HDR values (binary features:
the TI-HDR majority value), and "looks like the TI HDR" if it matches at least
--match-frac of the signature features. This is reported for AEC frames, with
the TI-HDR frames themselves as the positive control. The signature is
independent of position in the PCA plane, so it gives a mechanistic measure of
pre-organisation as well as the geometric (in-HDR) one.
Also reported: your hull-area comparison, OVL/BC overlap of the RS and TS
projections (needs ti_basin_ca_pca.py next to this script), representative
frames.

Dry/wet frame correspondence: wet frame = dry frame * --wet-per-dry. The script
compares C-alpha coordinates of corresponding frames and stops if they differ
(wrong factor or differently written files).

Example (one enzyme, in its folder)
-----------------------------------
    python ti_hdr_profile.py --enzyme KPC2 --rs RS --ts TS --rs_top RS.parm7 --ts_top TS.parm7 \
        --runs 10 --stride 100 --equil 50000 \
        --wet-top "{dir}/system.parm7" --wet-traj "{dir}/Run{run}/prod.nc" --outdir KPC2_hdr
Then:
    python ti_hdr_profile.py --aggregate KPC2_hdr NMCA_hdr SFC1_hdr SME1_hdr \
        BlaC_hdr CTXM16_hdr SHV1_hdr TEM1_hdr --outdir hdr_groups
"""
import argparse
import importlib
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.lib.distances import capped_distance, calc_angles, calc_dihedrals, distance_array
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import(*names, required=True):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            pass
    if required:
        raise ImportError(f"None of {names} found; keep this script next to the omega analysis scripts")
    return None


ola = _import("omega_loop_analysis", "OmegaLoop")
lcs = _import("loop_cluster_structures", "OmegaLoopRepresentative")
tbc = _import("ti_basin_ca_pca", required=False)
ENZYMES, COLOURS, group_of = ola.ENZYMES, ola.COLOURS, ola.group_of

SER_ATOMS = {"N", "H", "H1", "H2", "H3", "CA", "HA", "CB", "HB1", "HB2", "HB3", "OG", "HG",
             "C", "O", "OXT"}
BACKBONE = {"N", "H", "CA", "HA", "C", "O", "OXT"}
CONT_COLS = ["E166_K73", "E166_N170", "R164_D179", "K73_S130", "E166_acylC", "K73_acylOG",
             "oxy_N70", "oxy_N237", "loop_Rg", "loop_COM_S130"]


def sub(a, n):
    return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]

import re

def find_hinge_resids(u, motif, quiet=False):
    """Return topology residue numbers for the first regex match to a one-letter motif."""
    if not motif:
        return []
    prot, seq = ola.protein_seq(u)
    m = re.search(motif, seq)
    if m is None:
        raise SystemExit(f"Hinge motif {motif!r} not found in protein sequence: {seq}")
    start, end = m.start(), m.end()
    resids = [prot[i].resid for i in range(start, end)]
    if not quiet:
        print(f"Hinge motif {motif!r} matched sequence positions "
              f"{start + 1}-{end} ({m.group()}); residues {resids}")
    return resids
# ----------------------------------------------------------------------------
# PCA, as in PCA.py
# ----------------------------------------------------------------------------
def ca_atoms(u, args):
    ag = u.select_atoms(args.selection)
    if args.trim_termini > 0:
        ag = ag[args.trim_termini:-args.trim_termini]
    return ag


def fit_frames(u, ag, idx, ref_pos):
    """Frames `idx` superposed on ref_pos (== trans.fit_rot_trans(ag, reference))."""
    rc = ref_pos.mean(axis=0)
    out = []
    for f in idx:
        u.trajectory[int(f)]
        mob = ag.positions
        mc = mob.mean(axis=0)
        R, _ = rotation_matrix(mob - mc, ref_pos - rc)
        out.append(((mob - mc) @ R.T + rc).astype(np.float32))
    return np.array(out)


def resolve_top(directory, topology):
    """Topology given relative to the state folder, or as a path from the current folder."""
    for cand in (os.path.join(directory, topology), topology):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"Topology not found: tried {os.path.join(directory, topology)} and {topology}")


def check_templates(args):
    """The wet patterns must contain {dir}, otherwise TI would silently read the RS files."""
    for name in ("wet_top", "wet_traj", "dry_traj"):
        pat = getattr(args, name)
        if "{dir}" not in pat and args.rs != args.ts:
            raise SystemExit(f"--{name.replace('_', '-')} has no {{dir}} placeholder, so the TI "
                             f"ensemble would read the same files as RS. Use e.g. "
                             f"\"{{dir}}/Run{{run}}/prod.nc\" and \"{{dir}}/system.parm7\".")


def load_ensemble(directory, topology, ref_pos, args):
    coords, run_id, dry_idx = [], [], []
    for run in range(1, args.runs + 1):
        top = resolve_top(directory, topology)
        traj = args.dry_traj.format(dir=directory, run=run)
        print(f"  {directory} Run {run}: {traj}")
        u = mda.Universe(top, traj)
        ag = ca_atoms(u, args)
        idx = np.arange(args.equil, u.trajectory.n_frames, args.stride)
        if len(idx) == 0:
            raise SystemExit(f"{traj}: {u.trajectory.n_frames} frames, nothing left after "
                             f"--equil {args.equil} (set --equil to the number of frames to discard)")
        coords.append(fit_frames(u, ag, idx, ref_pos))
        run_id.append(np.full(len(idx), run))
        dry_idx.append(idx)
    return np.concatenate(coords), np.concatenate(run_id), np.concatenate(dry_idx)


# ----------------------------------------------------------------------------
# HDR
# ----------------------------------------------------------------------------
def make_hdr(Z2, pct, max_fit=4000):
    kde = gaussian_kde(sub(Z2, max_fit).T)
    thr = np.percentile(kde(sub(Z2, 4000).T), 100 - pct)
    return kde, thr


def in_hdr(model, Z2):
    kde, thr = model
    parts = np.array_split(Z2, max(1, len(Z2) // 5000 + 1))
    return np.concatenate([kde(c.T) >= thr for c in parts])


# ----------------------------------------------------------------------------
# Wet-trajectory features
# ----------------------------------------------------------------------------
def verify_frame_mapping(dry_top, dry_traj, wet_top, wet_traj, dry_idx, ref_pos, args):
    ud, uw = mda.Universe(dry_top, dry_traj), mda.Universe(wet_top, *wet_traj)
    agd, agw = ca_atoms(ud, args), ca_atoms(uw, args)
    if len(agd) != len(agw):
        raise SystemExit(f"dry and wet topologies select different numbers of C-alpha atoms "
                         f"({len(agd)} vs {len(agw)}); check --wet-top and --selection")
    worst = 0.0
    for f in sub(dry_idx, 3):
        ud.trajectory[int(f)]
        a = agd.positions
        wf = int(f) * args.wet_per_dry
        if wf >= uw.trajectory.n_frames:
            raise SystemExit(f"wet trajectory has {uw.trajectory.n_frames} frames; need frame {wf}. "
                             f"Check --wet-per-dry.")
        uw.trajectory[wf]
        b = agw.positions
        _, rmsd = rotation_matrix(a - a.mean(axis=0), b - b.mean(axis=0))
        worst = max(worst, float(rmsd))
    if worst > args.map_tol:
        raise SystemExit(f"dry and wet frames do not correspond (C-alpha RMSD up to {worst:.2f} A "
                         f"> {args.map_tol}); adjust --wet-per-dry or point --wet-traj at the file "
                         f"md_dry.nc was made from")
    return worst


RESNAME_NORM = {"GLH": "GLU", "ASH": "ASP", "LYN": "LYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
                "CYX": "CYS", "CYM": "CYS"}
TAIL_DEFAULT = [["C7"], ["C25", "C6"], ["C61", "C8"], ["O61", "O62", "O6", "O8"]]


def find_tail_atoms(res, names=None):
    """Atoms of the 6-alpha-hydroxyethyl dihedral C7-C25-C61-O61 (or --tail-atoms) in the acyl/TI residue."""
    cand = [[n] for n in names] if names else TAIL_DEFAULT
    out = []
    for opts in cand:
        hit = [a for n in opts for a in res.atoms if a.name == n]
        if not hit:
            return None
        out.append(hit[0])
    return out


def contact_features(top, trajs, frames, ref, args, ns):
    """Ligand hydrogen bonds, oxyanion contacts, hydration and ligand geometry, per frame.

    Partner residues are numbered in Ambler numbering anchored on the NEAREST catalytic
    landmark (Lys73, Ser130, Glu166, Lys234), so an insertion or deletion elsewhere in the
    sequence does not shift the label (Asn132 stays 132 in NMC-A / SME-1). Labels carry the
    position only ('132', '237bb'); the residue types seen are written to
    hb_partner_residues.csv, with protonation variants merged (GLH -> GLU, LYN -> LYS,
    ASH -> ASP, HID/HIE/HIP -> HIS).
    """
    u = mda.Universe(top, *trajs)
    prot, seq = ola.protein_seq(u)
    mapping = ola.map_to_ref(ref["seq"], seq)
    glu = prot[mapping.get(ref["roles"]["Glu166"])]
    anchors = []
    for role, amb in (("Lys73", 73), ("Ser130", 130), ("Glu166", 166), ("Lys234", 234)):
        pi = mapping.get(ref["roles"].get(role))
        if pi is not None:
            anchors.append((prot[pi].resid, amb))

    def ambler(a):
        k, amb0 = min(anchors, key=lambda t: abs(t[0] - a.resid))
        return amb0 + (a.resid - k)

    acyl = ola.find_acyl(u, ns)
    n = len(frames)
    cols = ["n_hb_lig", "n_hb_lig_water", "n_hb_lig_protein", "n_hb_O71", "O71_hbond",
            "n_hb_O31O32", "O31O32_hbond", "N4_O71", "hinge_COM_lig", "n_wat_lig4A", "dih_tail"]
    rec_cols = ["fi", "lig_atom", "partner", "resname"]
    if acyl is None:
        return pd.DataFrame(np.nan, index=range(n), columns=cols), pd.DataFrame(columns=rec_cols)
    res = acyl["C"][0].residue
    lig = res.atoms[np.array([a.name not in SER_ATOMS and not a.name.startswith("H") for a in res.atoms],
                             dtype=bool)]
    lig_acc = lig[np.array([a.name[0] in "NO" for a in lig], dtype=bool)]
    o71 = lig[np.array([a.name == "O71" for a in lig], dtype=bool)]
    n4 = lig[np.array([a.name == "N4" for a in lig], dtype=bool)]
    tail = find_tail_atoms(res, args.tail_atoms)
    if tail is None:
        print("  WARNING: hydroxyethyl dihedral atoms not found in the ligand residue; "
              "set --tail-atoms C7 <C> <C> <O>. dih_tail will be NaN.")

    def lig_atom_label(a):
        """Treat the two carboxylate oxygens O31/O32 as one equivalent acceptor."""
        return "O31/O32" if a.name in ("O31", "O32") else a.name

    hinge_resids = find_hinge_resids(u, args.hinge_motif, quiet=True)
    hinge = (u.select_atoms("protein and name CA and resid " + " ".join(map(str, hinge_resids)))
             if hinge_resids else None)
    water_names = set(ola.WATER_RESNAMES)

    # donors (H bonded to N/O) and acceptors in the system
    names = u.atoms.names.astype(str)
    first = np.array([x[0] for x in names])
    is_h = first == "H"
    b = u.bonds.indices
    m1, m2 = is_h[b[:, 0]] & ~is_h[b[:, 1]], is_h[b[:, 1]] & ~is_h[b[:, 0]]
    h_idx = np.concatenate([b[m1, 0], b[m2, 1]])
    d_idx = np.concatenate([b[m1, 1], b[m2, 0]])
    keep = np.isin(first[d_idx], ["N", "O"])
    h_idx, d_idx = h_idx[keep], d_idx[keep]
    DON, HAT = u.atoms[d_idx], u.atoms[h_idx]
    in_lig = np.isin(DON.indices, lig.indices)
    outside = ~in_lig & (DON.resindices != res.resindex)
    outD, outH = DON[outside], HAT[outside]
    ligD, ligH = DON[in_lig], HAT[in_lig]
    ACC = u.select_atoms("(name O*) or (resname HIS HID HIE HIP and name ND1 NE2)")
    ACC = ACC[ACC.resindices != res.resindex]
    water_o = u.select_atoms(f"resname {' '.join(ola.WATER_RESNAMES)} and name O OW")

    def label(a):
        if a.resname in water_names:
            return "WAT", "water", "WAT"
        return (f"{ambler(a)}{'bb' if a.name in BACKBONE else ''}", "protein",
                RESNAME_NORM.get(a.resname, a.resname))

    recs, rows = [], []
    for fi, f in enumerate(frames):
        ts = u.trajectory[int(f)]
        box = ts.dimensions
        nh = {"water": 0, "protein": 0}
        n_o71 = 0
        n_o31o32 = 0
        # ligand as acceptor
        dpos, hpos = outD.positions, outH.positions
        pr = capped_distance(lig_acc.positions, dpos, 3.0, box=box, return_distances=False)
        if len(pr):
            i, j = pr[:, 0], pr[:, 1]
            ang = np.degrees(calc_angles(dpos[j], hpos[j], lig_acc.positions[i], box=box))
            for ii, jj in zip(i[ang >= 150], j[ang >= 150]):
                lab, typ, rn = label(outD[jj])
                lig_label = lig_atom_label(lig_acc[ii])
                recs.append((fi, lig_label, lab, rn))
                nh[typ] += 1
                if lig_acc[ii].name == "O71":
                    n_o71 += 1
                if lig_label == "O31/O32":
                    n_o31o32 += 1
        # ligand as donor
        if len(ligD):
            apos = ACC.positions
            pr = capped_distance(ligD.positions, apos, 3.0, box=box, return_distances=False)
            if len(pr):
                i, j = pr[:, 0], pr[:, 1]
                ang = np.degrees(calc_angles(ligD.positions[i], ligH.positions[i], apos[j], box=box))
                for ii, jj in zip(i[ang >= 150], j[ang >= 150]):
                    lab, typ, rn = label(ACC[jj])
                    lig_label = lig_atom_label(ligD[ii])
                    recs.append((fi, lig_label, lab, rn))
                    nh[typ] += 1
                    if ligD[ii].name == "O71":
                        n_o71 += 1
                    if lig_label == "O31/O32":
                        n_o31o32 += 1
        wpos = water_o.positions
        n_l = (len(np.unique(capped_distance(lig.positions, wpos, 4.0, box=box,
                                             return_distances=False)[:, 1])) if len(wpos) else 0)

        def dist(p, q):  # minimum-image distance
            return float(distance_array(np.asarray(p, np.float32)[None, :],
                                        np.asarray(q, np.float32)[None, :], box=box)[0, 0])

        n4_o71 = dist(n4.positions[0], o71.positions[0]) if len(n4) and len(o71) else np.nan
        hinge_lig = (dist(hinge.center_of_mass(), lig.center_of_mass())
                     if hinge is not None and len(hinge) else np.nan)
        if tail is not None:
            p = [a.position[None, :] for a in tail]
            tail_dih = float(np.degrees(calc_dihedrals(p[0], p[1], p[2], p[3], box=box))[0])
        else:
            tail_dih = np.nan
        rows.append({"n_hb_lig": nh["water"] + nh["protein"], "n_hb_lig_water": nh["water"],
                     "n_hb_lig_protein": nh["protein"], "n_hb_O71": n_o71,
                     "O71_hbond": float(n_o71 > 0), "n_hb_O31O32": n_o31o32,
                     "O31O32_hbond": float(n_o31o32 > 0), "N4_O71": n4_o71,
                     "hinge_COM_lig": hinge_lig, "n_wat_lig4A": n_l, "dih_tail": tail_dih})
    return pd.DataFrame(rows), pd.DataFrame(recs, columns=rec_cols)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def is_binary(x):
    v = np.unique(x[np.isfinite(x)])
    return len(v) <= 2 and set(v) <= {0.0, 1.0}


# ----------------------------------------------------------------------------
# Per-enzyme analysis
# ----------------------------------------------------------------------------
def run_enzyme(args):
    enz = args.enzyme
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- PCA exactly as PCA.py -------------------------------------------------
    check_templates(args)
    ref_u = mda.Universe(resolve_top(args.rs, args.rs_top), args.dry_traj.format(dir=args.rs, run=1))
    ref_u.trajectory[0]
    ref_pos = ca_atoms(ref_u, args).positions.copy()
    print("Loading RS (acylenzyme) and TS (TI):")
    RS, rs_run, rs_idx = load_ensemble(args.rs, args.rs_top, ref_pos, args)
    TS, ts_run, ts_idx = load_ensemble(args.ts, args.ts_top, ref_pos, args)
    n_at = RS.shape[1]
    X = np.concatenate([RS.reshape(len(RS), -1), TS.reshape(len(TS), -1)]).astype(np.float64)
    X -= X.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(X, rowvar=False))
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    P = X @ eigvec
    Za, Zt = P[:len(RS)], P[len(RS):]
    print(f"RS frames {len(RS)}, TS frames {len(TS)}; PC1+PC2 variance "
          f"{eigval[:2].sum() / eigval.sum():.2f}")

    # --- HDR -----------------------------------------------------------------
    pcts = args.pct
    hdrs = {p: make_hdr(Zt[:, :2], p) for p in pcts}
    prim = pcts[0]
    ti_in = {p: in_hdr(hdrs[p], Zt[:, :2]) for p in pcts}
    aec_in = {p: in_hdr(hdrs[p], Za[:, :2]) for p in pcts}
    hdr_aec = make_hdr(Za[:, :2], prim)
    ti_in_aechdr = in_hdr(hdr_aec, Zt[:, :2])

    # --- wet-trajectory features ---------------------------------------------------
    wet_top = args.wet_top.format(dir=args.rs, enzyme=enz)
    u0 = mda.Universe(wet_top)
    prot0, seq0 = ola.protein_seq(u0)
    roles, loop = ola.find_ref_landmarks(prot0, seq0, args.glu_resid)
    ref = {"seq": seq0, "roles": roles, "loop": loop,
           "resnames": {k: prot0[v].resname for k, v in roles.items()}}
    ns = types.SimpleNamespace(stride=1, frames=None, acyl_resname=args.acyl_resname,
                               acyl_c=args.acyl_c, acyl_o=args.acyl_o)
    find_hinge_resids(u0, args.hinge_motif)          # fail now, not after the extraction
    find_hinge_resids(mda.Universe(args.wet_top.format(dir=args.ts, enzyme=enz)), args.hinge_motif)

    def wet_for(state_dir, state_top, run_ids, dry_idx, label):
        geoms, conts, recs, results = [], [], [], []
        top = args.wet_top.format(dir=state_dir, enzyme=enz)
        for run in np.unique(run_ids):
            m = run_ids == run
            frames = dry_idx[m] * args.wet_per_dry
            trajs = [args.wet_traj.format(dir=state_dir, run=run, enzyme=enz)]
            if run == 1:
                worst = verify_frame_mapping(resolve_top(state_dir, state_top),
                                             args.dry_traj.format(dir=state_dir, run=1),
                                             top, trajs, dry_idx[m], ref_pos, args)
                print(f"  {label}: dry/wet frame check OK (max C-alpha RMSD {worst:.3f} A)")
            ns.frames = [int(f) for f in frames]
            r = ola.analyse_replica(enz, int(run), top, trajs, ref, ns)
            c, h = contact_features(top, trajs, frames, ref, args, ns)
            if len(h):
                h["run"] = int(run)
            g = r["geom"].copy()
            g = g[[c for c in g.columns if not c.startswith("wat")]]

            # Add omega-loop backbone/side-chain torsions to the SAME per-frame feature
            # table used to build the TI-HDR signature (enabled by default). The
            # existing loop_dihedral_signature_*.csv output is descriptive only;
            # these columns make the torsions eligible for signature selection.
            if args.omega_loop_features:
                dih = np.asarray(r["dih"])
                meta = list(r["dih_meta"])
                if dih.ndim == 2 and dih.shape[0] == len(g) and dih.shape[1] == len(meta):
                    loop_cols = {}
                    for j, item in enumerate(meta):
                        label = str(item)
                        low_label = label.lower().replace("χ1", "chi1").replace("χ", "chi")
                        torsion = next((t for t in ("phi", "psi", "chi1") if t in low_label), None)
                        if torsion is None:
                            continue
                        # Metadata formats vary between versions of omega_loop_analysis;
                        # select torsions whose residue label contains an Ambler 164-179
                        # position, avoiding unrelated digits such as atom indices.
                        nums = [int(x) for x in re.findall(r"(?<!\d)(\d{3})(?!\d)", low_label)]
                        positions = [n for n in nums if 164 <= n <= 179]
                        if not positions:
                            continue
                        amb = positions[0]
                        # Keep a predictable name that the signature matcher recognises
                        # as circular (the name contains 'dih').
                        col = f"loop_dih_{torsion}_{amb}"
                        if col not in loop_cols:
                            loop_cols[col] = dih[:, j]
                    if loop_cols:
                        g = pd.concat([g.reset_index(drop=True), pd.DataFrame(loop_cols)], axis=1)
                    else:
                        print("  WARNING: --omega-loop-features was set, but no phi/psi/chi1 ")
                        print("           columns for Ambler 164-179 could be identified in dih_meta.")
                else:
                    print(f"  WARNING: unexpected dihedral array shape {dih.shape}; expected ")
                    print(f"           ({len(g)}, {len(meta)}). Omega-loop features not added for this run.")
            g["fi"] = np.arange(len(g))
            geoms.append(g)
            conts.append(c)
            recs.append(h.assign(global_offset=np.where(m)[0][0]) if len(h) else h)
            results.append({"enzyme": enz, "replica": int(run), "dih": r["dih"], "dih_meta": r["dih_meta"],
                            "geom": r["geom"], "_col": {k: i for i, k in enumerate(r["dih_meta"])},
                            "mask": m})
        return (pd.concat(geoms, ignore_index=True).drop(columns="fi"),
                pd.concat(conts, ignore_index=True), recs, results)

    print("Extracting wet-trajectory features ...")
    g_a, c_a, rec_a, res_a = wet_for(args.rs, args.rs_top, rs_run, rs_idx, "RS")
    g_t, c_t, rec_t, res_t = wet_for(args.ts, args.ts_top, ts_run, ts_idx, "TS")

    def hb_matrix(recs, n_frames):
        """Boolean (frame x 'ligatom|partner') matrix from the per-run records."""
        out_cols = {}
        for h in recs:
            if len(h) == 0:
                continue
            for (la, pa), g in h.groupby(["lig_atom", "partner"]):
                col = f"hb_{la}|{pa}"
                v = out_cols.setdefault(col, np.zeros(n_frames, bool))
                v[g["fi"].values + g["global_offset"].values] = True
        return pd.DataFrame(out_cols)

    hb_a, hb_t = hb_matrix(rec_a, len(g_a)), hb_matrix(rec_t, len(g_t))
    keep_cols = []
    for col in sorted(set(hb_a.columns) | set(hb_t.columns)):
        occ = max(hb_a[col].mean() if col in hb_a else 0,
                  hb_t[col][ti_in[prim]].mean() if col in hb_t and ti_in[prim].any() else 0)
        if occ >= args.min_hb_occ:
            keep_cols.append(col)
    allrec = pd.concat([h for h in rec_a + rec_t if len(h)], ignore_index=True) if any(len(h) for h in rec_a + rec_t) else pd.DataFrame()
    if len(allrec):
        tab = (allrec.groupby(["lig_atom", "partner", "resname"]).size().rename("n_frames").reset_index())
        tab.insert(0, "feature", "hb_" + tab.lig_atom + "|" + tab.partner)
        tab.insert(0, "enzyme", enz)
        tab[tab.feature.isin(keep_cols)].to_csv(out / "hb_partner_residues.csv", index=False)
    hb_a = hb_a.reindex(columns=keep_cols, fill_value=False).astype(float)
    hb_t = hb_t.reindex(columns=keep_cols, fill_value=False).astype(float)

    def assemble(g, c, hb, state, run, idx, Z):
        df = pd.concat([g.reset_index(drop=True), c.reset_index(drop=True), hb.reset_index(drop=True)], axis=1)
        df.insert(0, "state", state)
        df.insert(1, "run", run)
        df.insert(2, "dry_frame", idx)
        df.insert(3, "wet_frame", idx * args.wet_per_dry)
        df["PC1"], df["PC2"] = Z[:, 0], Z[:, 1]
        return df

    FA = assemble(g_a, c_a, hb_a, "AEC", rs_run, rs_idx, Za)
    FT = assemble(g_t, c_t, hb_t, "TI", ts_run, ts_idx, Zt)
    FA["in_HDR"], FT["in_HDR"] = aec_in[prim], ti_in[prim]
    F = pd.concat([FA, FT], ignore_index=True)
    
    # --- profile: TI-HDR frames vs all other frames --------------------------------
    skip = {"state", "run", "dry_frame", "wet_frame", "PC1", "PC2", "in_HDR"}
    feats = [c for c in F.columns if c not in skip and pd.api.types.is_numeric_dtype(F[c])]
    is_tihdr = ((F.state == "TI") & F.in_HDR).to_numpy()
    prof = []
    m_ti = (F.state == "TI").to_numpy()
    m_aec = (F.state == "AEC").to_numpy()
    in_h = F.in_HDR.to_numpy().astype(bool)
    for c in feats:
        x = F[c].to_numpy(float)
        if "dih" in c.lower():      # angles in degrees: circular mean/SD and d on the shifted values
            r_ = np.deg2rad(x[is_tihdr & np.isfinite(x)])
            mu = np.rad2deg(np.arctan2(np.sin(r_).mean(), np.cos(r_).mean())) if len(r_) else 0.0
            xz = (x - mu + 180.0) % 360.0 - 180.0
            mean_of = lambda m, xz=xz, mu=mu: float((np.nanmean(xz[m]) + mu + 180.0) % 360.0 - 180.0)
            sd_of = lambda m, xz=xz: float(np.nanstd(xz[m]))
            dd = cohens_d(xz[is_tihdr], xz[~is_tihdr])
        else:
            mean_of = lambda m, x=x: np.nanmean(x[m])
            sd_of = lambda m, x=x: np.nanstd(x[m])
            dd = cohens_d(x[is_tihdr], x[~is_tihdr])
        prof.append({"feature": c, "binary": is_binary(x),
                     "TI_HDR_mean": mean_of(is_tihdr), "TI_HDR_sd": sd_of(is_tihdr),
                     "TI_outside_mean": mean_of(m_ti & ~is_tihdr),
                     "AEC_in_HDR_mean": mean_of(m_aec & in_h),
                     "AEC_outside_HDR_mean": mean_of(m_aec & ~in_h),
                     "AEC_all_mean": mean_of(m_aec),
                     "d_TIHDR_vs_rest": dd})
    prof = pd.DataFrame(prof).sort_values("d_TIHDR_vs_rest", key=lambda s: -s.abs())
    prof.insert(0, "enzyme", enz)
    prof.to_csv(out / "hdr_feature_profile.csv", index=False)

    # ------------------------------------------------------------------------
    # 1. Mandatory defining features: oxyanion H-bonding to O71
    # ------------------------------------------------------------------------
    cand_ser70 = [c for c in ["oxy_N70", "hb_O71|70bb", "hb_O71|70"] if c in F.columns]
    cand_237   = [c for c in ["hb_O71|237bb", "oxy_N237"] if c in F.columns]

    mandatory = []
    if cand_ser70:
        mandatory.append(cand_ser70[0])
    if cand_237:
        mandatory.append(cand_237[0])

    chosen = list(dict.fromkeys(mandatory))
        # ------------------------------------------------------------------------
    # 2. Fill remaining slots with high-impact features (|d| >= 0.5)
    # ------------------------------------------------------------------------
    # Exclude the mandatory features so they are not evaluated twice
    trivial = {"n_hb_O71", "n_hb_O62", "n_hb_O01", "n_hb_O31", "n_hb_O32"}
    
    def is_allowed_glu166_hb(feat_name):
        f = feat_name.upper()
        is_o6_glu = ("O6" in f or "O61" in f or "O62" in f) and "166" in f
        is_asn_glu = ("132" in f or "ASN" in f) and "166" in f
        return is_o6_glu or is_asn_glu
    
    def is_excluded_feature(feat_name):
        f = feat_name.upper()
        # 1. Any Glu/Glh166 H-bonding (chemical/protonation state mismatch)
        if ("166" in f or "GLH166" in f or "GLU166" in f) and (f.startswith("HB_") or f.startswith("N_HB_")):
            if not is_allowed_glu166_hb(f):
                return True
        if ("O71" in f) and (f.startswith("HB_") or "HBOND" in f):
            # Allow only the canonical oxyanion hole features (already handled in mandatory)
            if not any(res in f for res in ["70", "237"]):
                return True
        return False
    
    cand = prof[
        (prof.d_TIHDR_vs_rest.abs() >= 0.25) &
        (~prof.feature.isin(chosen)) &
        (~prof.feature.isin(trivial)) &
        (~prof.feature.apply(is_excluded_feature))
    ]    

    for r in cand.itertuples():
        if len(chosen) >= args.sig_n:
            break
        x = F[r.feature].to_numpy(float)
        dup = False
        for c in chosen:
            y = F[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 10 and x[m].std() > 0 and y[m].std() > 0 and \
                    abs(np.corrcoef(x[m], y[m])[0, 1]) > args.sig_corr:
                dup = True
                break
        if not dup:
            chosen.append(r.feature)

    sig = prof.set_index("feature").loc[chosen].reset_index() if chosen else prof.head(0)
    spec = []
    match = pd.DataFrame(index=F.index)

    for r in sig.itertuples():
        x = F[r.feature].to_numpy(float)
        xs = x[is_tihdr]
        
        if r.binary:
            # 1. Mandatory catalytic/oxyanion contacts: ALWAYS target 1.0
            if r.feature in mandatory or "oxy" in r.feature:
                target = 1.0
            # 2. Inhibitory traps (like hb_O62|GLU166) or features broken in TI: ALWAYS target 0.0
            elif r.d_TIHDR_vs_rest < 0:
                target = 0.0
            # 3. General H-bonds enriched in the TI (d > 0): target 1.0
            elif r.feature.startswith("hb_") and r.d_TIHDR_vs_rest > 0:
                target = 1.0
            else:
                target = float(np.nanmean(xs) >= 0.5)
            lo = hi = target
            m = (x == target)
        else:
            if "dih" in r.feature.lower():
                # ------------------------------------------------------------------
                # Circular Dihedral Handling (Degrees: [-180.0, 180.0])
                # ------------------------------------------------------------------
                # 1. Filter out NaNs from TI-HDR distribution
                valid_ti = xs[np.isfinite(xs)]
                
                # 2. Circular mean of TI-HDR in radians
                rads = np.deg2rad(valid_ti)
                sin_mean = np.nanmean(np.sin(rads))
                cos_mean = np.nanmean(np.cos(rads))
                circ_mean = np.rad2deg(np.arctan2(sin_mean, cos_mean))

                # 3. Shift TI-HDR angles relative to the circular mean -> [-180, 180]
                diff_ti = (valid_ti - circ_mean + 180.0) % 360.0 - 180.0

                # 4. Compute 10th-90th percentile bounds around circular mean
                lo_shift, hi_shift = np.nanpercentile(diff_ti, [10, 90])

                # 5. Measure every frame's angular distance from the circular mean
                diff_all = (x - circ_mean + 180.0) % 360.0 - 180.0
                m = (diff_all >= lo_shift) & (diff_all <= hi_shift)

                # 6. Unshift bounds for the spec report
                lo = (circ_mean + lo_shift + 180.0) % 360.0 - 180.0
                hi = (circ_mean + hi_shift + 180.0) % 360.0 - 180.0

            elif "N4_O71" in r.feature:
                # Tightened interquartile range (25th to 75th percentile)
                lo, hi = np.nanpercentile(xs, [25, 75])
                m = (x >= lo) & (x <= hi)

            else:
                # Standard linear percentile bounds for distances / loop metrics
                lo, hi = np.nanpercentile(xs, [5, 95])
                m = (x >= lo) & (x <= hi)

        match[r.feature] = np.where(np.isfinite(x), m, False)
        spec.append({
            "enzyme": enz,
            "feature": r.feature,
            "binary": r.binary,
            "low": lo,
            "high": hi,
            "d_TIHDR_vs_rest": r.d_TIHDR_vs_rest
        })

    pd.DataFrame(spec).to_csv(out / "hdr_signature.csv", index=False)
    F["sig_score"] = match.mean(axis=1) if len(spec) else np.nan
    F["sig_match"] = (F["sig_score"] >= args.match_frac) if len(spec) else False
    
    # --- Define Reactive Complex: backbone in HDR AND active-site signature matched ---
    F["reactive_complex"] = F["in_HDR"] & F["sig_match"]
    
    # --- hydrogen-bond partner table ------------------------------------------------
    hbt = prof[prof.feature.str.startswith("hb_")][["enzyme", "feature", "TI_HDR_mean", "TI_outside_mean",
                                                      "AEC_in_HDR_mean", "AEC_outside_HDR_mean",
                                                      "AEC_all_mean"]]
    hbt.to_csv(out / "hdr_hbond_partners.csv", index=False)

    # --- loop conformation (dihedral signature, TI HDR vs rest of TI; AEC in vs out) ----
    for label, res_list, flag in (("TI", res_t, FT.in_HDR.to_numpy()), ("AEC", res_a, FA.in_HDR.to_numpy())):
        for r in res_list:
            r["_lab"] = np.where(flag[r["mask"]], 1, 0)
        d = lcs.dihedral_signature(res_list, 1, args.min_frames)
        d.to_csv(out / f"loop_dihedral_signature_{label}.csv", index=False)

    # --- per-run metrics ------------------------------------------------------------------
    rows = []
    for (st, run), g in F.groupby(["state", "run"]):
        row = {"enzyme": enz, "group": group_of(enz), "state": st, "run": int(run),
               "sig_match": g.sig_match.mean(), "sig_score": g.sig_score.mean()}
        idx_st = g.index.to_numpy() - (0 if st == "AEC" else len(FA))
        for p in pcts:
            fl = aec_in[p] if st == "AEC" else ti_in[p]
            row[f"in_HDR{p:g}"] = fl[idx_st].mean()
        row["in_AEC_HDR"] = (ti_in_aechdr[idx_st].mean() if st == "TI"
                             else in_hdr(hdr_aec, Za[idx_st, :2]).mean())
        rows.append(row)
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "replica_metrics.csv", index=False)

    def hull(Z):
        return ConvexHull(Z[:, :2]).volume

    summ = {"enzyme": enz, "group": group_of(enz), "n_RS_frames": len(RS), "n_TS_frames": len(TS),
            "pc12_variance": float(eigval[:2].sum() / eigval.sum()),
            "hull_ratio_TI_over_AEC": hull(Zt) / hull(Za),
            "cov_trace_ratio_TI_over_AEC": float(np.trace(np.cov(Zt[:, :2].T)) / np.trace(np.cov(Za[:, :2].T))),
            "sig_features": ";".join(r["feature"] for r in spec)}
    for p in pcts:
        summ[f"AEC_in_HDR{p:g}"] = float(aec_in[p].mean())
        summ[f"TI_in_HDR{p:g}"] = float(ti_in[p].mean())
    a_rep, t_rep = rep[rep.state == "AEC"], rep[rep.state == "TI"]
    summ["AEC_sig_match"] = float(a_rep.sig_match.mean())
    summ["TI_sig_match"] = float(t_rep.sig_match.mean())
    summ["TIHDR_sig_match_control"] = float(F[is_tihdr].sig_match.mean())
    summ["AEC_sig_match_rel_control"] = summ["AEC_sig_match"] / summ["TIHDR_sig_match_control"]
    summ["AEC_in_HDR_rel_control"] = summ[f"AEC_in_HDR{prim:g}"] / summ[f"TI_in_HDR{prim:g}"]
    summ["TI_in_AEC_HDR"] = float(t_rep.in_AEC_HDR.mean())
    if tbc is not None:
        o, b = tbc.overlap_2d(Za[:, :2], Zt[:, :2])
        summ["OVL_2D"], summ["BC_2D"] = o, b
    pd.DataFrame([summ]).to_csv(out / "enzyme_summary.csv", index=False)

    # --- representative frames -----------------------------------------------------------------
    reps = []
    kde, _ = hdrs[prim]
    ti_idx = np.where(ti_in[prim])[0]
    j = ti_idx[np.argmax(kde(Zt[ti_idx, :2].T))]
    reps.append(("TI frame at the HDR density maximum", "TI", ts_run[j], ts_idx[j]))
    sm = F[F.state == "AEC"].sig_match.to_numpy().astype(bool)
    ss = F[F.state == "AEC"].sig_score.to_numpy()
    aec_idx = np.where(aec_in[prim] & sm)[0]
    if len(aec_idx):
        j = aec_idx[np.argmax(ss[aec_idx])]
        reps.append(("AEC frame in the HDR matching the signature best", "AEC", rs_run[j], rs_idx[j]))
    pd.DataFrame([{"enzyme": enz, "what": w, "state": s_, "run": int(r_), "dry_frame_0based": int(f),
                   "wet_frame_1based_cpptraj": int(f) * args.wet_per_dry + 1}
                  for w, s_, r_, f in reps]).to_csv(out / "representative_frames.csv", index=False)

    # --- figures ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(Zt[:, 0], Zt[:, 1], s=3, alpha=0.3, color="tab:orange", label="TS (TI)")
    ax.scatter(Za[:, 0], Za[:, 1], s=3, alpha=0.3, color="tab:blue", label="RS (AEC)")
    allp = np.vstack([Za[:, :2], Zt[:, :2]])
    gx, gy = np.linspace(allp[:, 0].min(), allp[:, 0].max(), 90), np.linspace(allp[:, 1].min(), allp[:, 1].max(), 90)
    GX, GY = np.meshgrid(gx, gy)
    for p in pcts:
        k_, thr = hdrs[p]
        ax.contour(GX, GY, k_(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape), levels=[thr],
                   colors="darkred", linewidths=1.6 if p == prim else 0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{enz}: AEC in TI HDR{prim:g} {summ[f'AEC_in_HDR{prim:g}']:.0%}; "
                 f"signature match {summ['AEC_sig_match']:.0%}", fontsize=9)
    ax.legend(markerscale=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pca_hdr.png", dpi=250)
    plt.close(fig)

    top = prof.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.d_TIHDR_vs_rest, color="tab:orange")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d, TI-HDR frames vs all other frames")
    fig.tight_layout()
    fig.savefig(out / "hdr_profile_top_features.png", dpi=200)
    plt.close(fig)
    
    F["reactive_complex"] = F["in_HDR"] & F["sig_match"]
    summ["AEC_reactive_complex"] = float(np.nanmean(F.loc[F.state == "AEC", "reactive_complex"]))
    summ["TIHDR_reactive_complex"] = float(np.nanmean(F.loc[is_tihdr, "reactive_complex"]))

    lines = [
        f"{enz}: TI HDR{prim:g} profile",
        f"AEC frames inside the TI HDR: {summ[f'AEC_in_HDR{prim:g}']:.3f} (TI frames themselves {summ[f'TI_in_HDR{prim:g}']:.3f})",
        f"AEC frames matching the TI-HDR signature: {summ['AEC_sig_match']:.3f} (TI-HDR frames {summ['TIHDR_sig_match_control']:.3f}; match = >= {args.match_frac:.0%} of the features)",
        f"AEC reactive complex (HDR + Signature): {summ['AEC_reactive_complex']:.3f} (TI-HDR frames {summ['TIHDR_reactive_complex']:.3f})",
        "Signature features (central 90 % range of TI-HDR frames; binary: majority value):",
        pd.DataFrame(spec).round(3).to_string(index=False) if spec else "   none with |d| >= 0.5",
        "\nTop features (TI-HDR vs all other frames):",
        prof.head(12)[["feature", "TI_HDR_mean", "TI_outside_mean", "AEC_in_HDR_mean",
                       "AEC_outside_HDR_mean", "AEC_all_mean", "d_TIHDR_vs_rest"]].round(3).to_string(index=False)
    ]
    (out / "summary.txt").write_text("\n".join(lines))
    F.to_csv(out / "frames.csv", index=False)
    print("\n" + "\n".join(lines))
    print(f"\nDone. Outputs in {out.resolve()}")
# ----------------------------------------------------------------------------
# Aggregate across enzymes
# ----------------------------------------------------------------------------
def _enzyme_key(name):
    """Normalise enzyme labels for matching (e.g. 'NMC-A' -> 'nmca')."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _canonical_enzyme_name(name):
    """Map a folder/output label to the canonical name in omega_loop_analysis.

    Exact normalised matches are preferred. A shortened label such as KPC is
    accepted only when it is a unique prefix of a canonical enzyme name (KPC2).
    Unknown or ambiguous labels are left unchanged so they can be reported.
    """
    label = str(name)
    key = _enzyme_key(label)
    canonical_keys = {_enzyme_key(e): e for e in ENZYMES}
    if key in canonical_keys:
        return canonical_keys[key]
    matches = [e for e in ENZYMES if _enzyme_key(e).startswith(key)] if key else []
    if len(matches) == 1:
        return matches[0]
    return label


def _canonicalise_enzyme_column(df, source, require_unique_enzyme=False):
    """Canonicalise enzyme labels and optionally enforce one row per enzyme.

    Summary tables should contain one row per enzyme. Feature-profile tables
    intentionally contain many rows per enzyme (one per feature), so duplicate
    enzyme labels are expected there.
    """
    if "enzyme" not in df.columns:
        raise ValueError(f"{source} has no 'enzyme' column")
    original = df["enzyme"].astype(str)
    mapped = original.map(_canonical_enzyme_name)
    known = set(ENZYMES)
    unmatched = sorted(set(mapped) - known)
    if unmatched:
        print(f"WARNING: {source}: enzyme labels not matched to ENZYMES and excluded from group tests: {unmatched}")
    df = df.copy()
    df["enzyme"] = mapped
    if require_unique_enzyme:
        known_rows = df.loc[df["enzyme"].isin(known), "enzyme"]
        duplicates = known_rows.duplicated(keep=False)
        if duplicates.any():
            names = sorted(known_rows[duplicates].unique())
            raise ValueError(
                f"Multiple summary rows map to the same canonical enzyme name: {names}. "
                "Check --aggregate inputs and enzyme_summary.csv files."
            )
    return df


def run_aggregate(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summ_parts = []
    for d in args.aggregate:
        part = pd.read_csv(Path(d) / "enzyme_summary.csv")
        summ_parts.append(_canonicalise_enzyme_column(part, f"{d}/enzyme_summary.csv", require_unique_enzyme=True))
    summ = pd.concat(summ_parts, ignore_index=True)
    present = set(summ["enzyme"])
    summ = summ[summ["enzyme"].isin(ENZYMES)].set_index("enzyme").reindex([e for e in ENZYMES if e in present])
    if summ.empty:
        raise ValueError("No enzyme names matched ENZYMES. Check the enzyme labels in enzyme_summary.csv and the ENZYMES list in omega_loop_analysis.")
    summ.to_csv(out / "all_enzymes_summary.csv")
    # Test joint reactive-complex sampling as well as its component metrics.
    # AEC_reactive_complex = fraction of AEC frames both inside TI-HDR and
    # matching the TI-HDR signature; TIHDR_reactive_complex is a positive control.
    cols = [c for c in ("AEC_in_HDR90", "AEC_in_HDR50", "AEC_in_HDR_rel_control", "AEC_sig_match",
                        "AEC_sig_match_rel_control", "AEC_reactive_complex", "TIHDR_reactive_complex",
                        "TI_in_AEC_HDR", "hull_ratio_TI_over_AEC", "cov_trace_ratio_TI_over_AEC",
                        "OVL_2D", "BC_2D") if c in summ.columns]
    tests = ola.table_tests(summ, cols)
    tests.to_csv(out / "group_tests.csv", index=False)

    prof_parts = []
    for d in args.aggregate:
        part = pd.read_csv(Path(d) / "hdr_feature_profile.csv")
        prof_parts.append(_canonicalise_enzyme_column(part, f"{d}/hdr_feature_profile.csv", require_unique_enzyme=False))
    prof = pd.concat(prof_parts, ignore_index=True)
    prof = prof[prof["enzyme"].isin(ENZYMES)]
    piv = prof.pivot_table(index="feature", columns="enzyme", values="d_TIHDR_vs_rest")
    cons = pd.DataFrame({"n_enzymes": piv.notna().sum(axis=1), "median_d": piv.median(axis=1),
                         "n_same_sign_and_|d|>=0.5": [int(((np.sign(r.dropna()) == np.sign(r.median()))
                                                           & (r.dropna().abs() >= 0.5)).sum())
                                                       for _, r in piv.iterrows()]})
    cons = cons.sort_values("median_d", key=lambda s: -s.abs())
    cons.to_csv(out / "profile_consistency_across_enzymes.csv")
    piv.to_csv(out / "profile_d_by_enzyme.csv")
    hb = prof[prof.feature.str.startswith("hb_")].pivot_table(index="feature", columns="enzyme",
                                                              values="TI_HDR_mean")
    hb.to_csv(out / "hbond_occupancy_TI_HDR_by_enzyme.csv")

    lines = ["Group tests (carbapenemase vs carbapenem-inhibited; enzyme level, exact permutation):",
             summ[[c for c in cols]].round(3).to_string(), "", tests.round(3).to_string(index=False),
             "\nFeatures that characterise the TI HDR consistently across enzymes (median d over enzymes):",
             cons.head(15).round(2).to_string()]
    (out / "group_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    plot_cols = [c for c in ("AEC_in_HDR90", "AEC_sig_match", "AEC_reactive_complex") if c in summ.columns]
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(5.5 * len(plot_cols), 4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, plot_cols):
        for i, e in enumerate(summ.index):
            ax.bar(i, summ.loc[e, c], color=COLOURS[group_of(e)])
        ax.set_xticks(range(len(summ)))
        ax.set_xticklabels(summ.index, rotation=45)
        ax.set_title(f"{c} (orange = carbapenemase)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "group_comparison.png", dpi=180)
    plt.close(fig)
    print(f"\nDone. Outputs in {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", nargs="+", default=None, help="enzyme output dirs to combine")
    ap.add_argument("--enzyme", choices=ENZYMES)
    ap.add_argument("--rs", default="RS")
    ap.add_argument("--ts", default="TS")
    ap.add_argument("--rs_top", default="RS.parm7")
    ap.add_argument("--ts_top", default="TS.parm7")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--equil", type=int, default=50000)
    ap.add_argument("--selection", default="protein and name CA")
    ap.add_argument("--trim_termini", type=int, default=5)
    ap.add_argument("--dry-traj", default="{dir}/Run{run}/md_dry.nc",
                    help="trajectory used for the PCA frames (use the prod.nc pattern if you have no md_dry.nc)")
    ap.add_argument("--wet-top", default="{dir}/system.parm7", help="solvated topology; {dir} = RS/TS folder")
    ap.add_argument("--wet-traj", default="{dir}/Run{run}/prod.nc", help="solvated trajectory pattern")
    ap.add_argument("--wet-per-dry", type=int, default=1, help="wet frames per dry frame")
    ap.add_argument("--map-tol", type=float, default=0.1, help="max C-alpha RMSD (A) in the dry/wet check")
    ap.add_argument("--pct", type=float, nargs="+", default=[90.0, 50.0])
    ap.add_argument("--sig-n", type=int, default=6)
    ap.add_argument("--omega-loop-features", dest="omega_loop_features", action="store_true", default=True,
                    help="include available phi/psi/chi1 torsions for Ambler 164-179 in the TI-HDR signature (default)")
    ap.add_argument("--no-omega-loop-features", dest="omega_loop_features", action="store_false",
                    help="disable omega-loop torsions for comparison with the original signature")
    ap.add_argument("--sig-corr", type=float, default=0.9,
                    help="skip a signature feature correlated above this with a chosen one")
    ap.add_argument("--match-frac", type=float, default=0.8)
    ap.add_argument("--min-hb-occ", type=float, default=0.10)
    ap.add_argument("--min-frames", type=int, default=50)
    ap.add_argument("--glu-resid", type=int, default=None, help="topology resid of Glu166 if the motif fails")
    ap.add_argument("--acyl-resname", default=None)
    ap.add_argument("--acyl-c", default=None)
    ap.add_argument("--acyl-o", default=None)
    ap.add_argument("--outdir", default="hdr_profile")
    ap.add_argument("--tail-atoms", nargs=4, default=None, metavar="ATOM",
                    help="four atom names of the hydroxyethyl dihedral (default C7 C25 C61 O61, "
                         "with C6/C8/O62/O6 fallbacks)")
    ap.add_argument("--hinge-motif",
                default="[TKR][VT][AG][GADN][PKAH][LSR][LI]R",
                help="regex motif in one-letter amino-acid code used to identify the "
                     "hinge region; the first match is used")
    args = ap.parse_args()
    if args.aggregate:
        run_aggregate(args)
    else:
        if not args.enzyme:
            ap.error("--enzyme is required (or use --aggregate)")
        run_enzyme(args)


if __name__ == "__main__":
    main()

