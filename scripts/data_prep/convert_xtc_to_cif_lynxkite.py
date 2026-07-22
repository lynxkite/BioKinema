import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mdtraj
from biotite.structure.io import load_structure, save_structure
from joblib import Parallel, delayed
from tqdm import tqdm


HOLDOUT_TOKENS = ("A779", "POSTSTERONE")


@dataclass
class TrajectoryJob:
    trajectory_key: str
    split: str
    xtc_path: Path
    top_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LynxKite GROMACS trajectories under /data into frame-wise mmCIF files."
    )
    parser.add_argument("--data_root", type=str, default="/data", help="Root path containing date folders.")
    parser.add_argument("--outdir", type=str, default="./data_lynxkite", help="Output directory.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for conversion.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=10,
        help="Save every Nth frame from each trajectory (must be >= 1).",
    )
    parser.add_argument(
        "--target_split",
        choices=["all", "train", "test"],
        default="all",
        help="Convert all jobs or only jobs assigned to one split.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory for split manifests. Defaults to <outdir>/splits.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip writing CIF files that already exist.",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=None,
        help="Optional cap for discovered trajectory jobs (useful for smoke tests).",
    )
    parser.add_argument(
        "--allow_pdb_fallback",
        action="store_true",
        help="Allow project-level PDB fallback when no GRO/TPR topology is found.",
    )
    parser.add_argument(
        "--no_strict_topology_check",
        action="store_false",
        dest="strict_topology_check",
        help="Disable strict atom-name/order sanity check between topology and temporary PDB frame.",
    )
    parser.set_defaults(strict_topology_check=True)
    return parser.parse_args()


def pdb_to_cif(pdb_path: str, cif_path: str):
    pdb_structure = load_structure(pdb_path)
    save_structure(cif_path, pdb_structure)


def sanitize_component(text: str) -> str:
    safe = text.replace("/", "__").replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in safe)


def pick_topology(xtc_path: Path, project_dir: Path, allow_pdb_fallback: bool = False):
    parent = xtc_path.parent
    stem = xtc_path.stem
    base_stem = stem.split(".part")[0]

    # Prefer exact stream matches, then md defaults, then npt defaults.
    direct_candidates = [
        parent / f"{stem}.gro",
        parent / f"{stem}.tpr",
        parent / f"{base_stem}.gro",
        parent / f"{base_stem}.tpr",
        parent / "md.gro",
        parent / "md.tpr",
        parent / "npt2.gro",
        parent / "npt2.tpr",
        parent / "npt1.gro",
        parent / "npt1.tpr",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    if not allow_pdb_fallback:
        return None

    aligned_pdbs = sorted(project_dir.glob("*aligned*.pdb"))
    if aligned_pdbs:
        return aligned_pdbs[0]

    project_pdbs = sorted(project_dir.glob("*.pdb"))
    if project_pdbs:
        return project_pdbs[0]

    return None


def assign_split(name: str) -> str:
    name_upper = name.upper()
    for token in HOLDOUT_TOKENS:
        if token in name_upper:
            return "test"
    return "train"


def discover_jobs(data_root: Path, allow_pdb_fallback: bool = False):
    jobs = []
    skipped_no_top = []
    date_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    for date_dir in date_dirs:
        project_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])
        for project_dir in project_dirs:
            xtc_files = sorted(project_dir.rglob("*.xtc"))
            for xtc_path in xtc_files:
                if xtc_path.name.startswith("."):
                    continue
                top_path = pick_topology(xtc_path, project_dir, allow_pdb_fallback=allow_pdb_fallback)
                if top_path is None:
                    skipped_no_top.append(str(xtc_path))
                    continue

                rel_stream = xtc_path.relative_to(project_dir).with_suffix("")
                trajectory_key = "__".join(
                    [
                        sanitize_component(date_dir.name),
                        sanitize_component(project_dir.name),
                        sanitize_component(str(rel_stream)),
                    ]
                )
                split = assign_split(f"{project_dir.name}__{rel_stream}")
                jobs.append(
                    TrajectoryJob(
                        trajectory_key=trajectory_key,
                        split=split,
                        xtc_path=xtc_path,
                        top_path=top_path,
                    )
                )

    return jobs, skipped_no_top


def write_split_manifests(jobs, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    train_jobs = sorted([j for j in jobs if j.split == "train"], key=lambda x: x.trajectory_key)
    test_jobs = sorted([j for j in jobs if j.split == "test"], key=lambda x: x.trajectory_key)

    train_csv = split_dir / "lynxkite_train.csv"
    test_csv = split_dir / "lynxkite_test.csv"

    for out_csv, rows in [(train_csv, train_jobs), (test_csv, test_jobs)]:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name"])
            for job in rows:
                writer.writerow([job.trajectory_key])

    return train_csv, test_csv


def _parse_atom_site_fields(cif_path: Path):
    fields = []
    rows = []
    in_loop = False
    with open(cif_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "loop_":
                in_loop = True
                fields = []
                rows = []
                continue
            if in_loop and line.startswith("_atom_site."):
                fields.append(line.strip())
                continue
            if in_loop and fields and (line.startswith("ATOM") or line.startswith("HETATM")):
                rows.append(line.split())
                continue
            if in_loop and fields and rows and not line.startswith("_atom_site."):
                break
    return fields, rows


def build_cif_metadata_summary(cif_path: Path):
    fields, rows = _parse_atom_site_fields(cif_path)
    if not fields or not rows:
        return {
            "n_label_entity_ids": 0,
            "n_label_asym_ids": 0,
            "n_label_comp_ids": 0,
            "label_entity_ids": "",
            "label_asym_ids": "",
            "label_comp_ids": "",
        }

    field_to_idx = {name.replace("_atom_site.", ""): i for i, name in enumerate(fields)}
    entity_idx = field_to_idx.get("label_entity_id")
    asym_idx = field_to_idx.get("label_asym_id")
    comp_idx = field_to_idx.get("label_comp_id")

    entity_vals = sorted({r[entity_idx] for r in rows if entity_idx is not None and entity_idx < len(r)})
    asym_vals = sorted({r[asym_idx] for r in rows if asym_idx is not None and asym_idx < len(r)})
    comp_vals = sorted({r[comp_idx] for r in rows if comp_idx is not None and comp_idx < len(r)})

    return {
        "n_label_entity_ids": len(entity_vals),
        "n_label_asym_ids": len(asym_vals),
        "n_label_comp_ids": len(comp_vals),
        "label_entity_ids": ",".join(entity_vals[:20]),
        "label_asym_ids": ",".join(asym_vals[:20]),
        "label_comp_ids": ",".join(comp_vals[:20]),
    }


def strict_topology_sanity_check(traj):
    top_atom_names = [atom.name for atom in traj.topology.atoms]
    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        traj[0].save_pdb(temp.name)
        pdb_structure = load_structure(temp.name)
    pdb_atom_names = pdb_structure.atom_name.tolist()

    if len(top_atom_names) != len(pdb_atom_names):
        raise ValueError(
            "Topology atom count mismatch after save/load: "
            f"top={len(top_atom_names)} vs pdb={len(pdb_atom_names)}"
        )

    for idx, (top_atom, pdb_atom) in enumerate(zip(top_atom_names, pdb_atom_names)):
        if top_atom != pdb_atom:
            raise ValueError(
                "Topology atom name/order mismatch at atom index "
                f"{idx}: top_atom_name={top_atom}, pdb_atom_name={pdb_atom}"
            )


def convert_job(
    job: TrajectoryJob,
    mmcif_root: Path,
    frame_stride: int,
    skip_existing: bool,
    strict_topology_check: bool,
):
    split_dir = mmcif_root / job.split
    split_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    try:
        traj = mdtraj.load(str(job.xtc_path), top=str(job.top_path))
    except Exception as exc:
        return {
            "ok": False,
            "job": job.trajectory_key,
            "split": job.split,
            "error": f"failed to load: {exc}",
            "generated": generated_paths,
            "audit": {},
        }

    if strict_topology_check:
        try:
            strict_topology_sanity_check(traj)
        except Exception as exc:
            return {
                "ok": False,
                "job": job.trajectory_key,
                "split": job.split,
                "error": f"strict topology check failed: {exc}",
                "generated": generated_paths,
                "audit": {},
            }

    frame_id = 0
    for frame_idx in range(0, traj.n_frames, frame_stride):
        cif_path = split_dir / f"{job.trajectory_key}_{frame_id:06d}.cif"
        frame_id += 1

        if skip_existing and cif_path.exists():
            generated_paths.append(str(cif_path.resolve()))
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
                traj[frame_idx].save_pdb(temp.name)
                pdb_to_cif(temp.name, str(cif_path))
            generated_paths.append(str(cif_path.resolve()))
        except Exception as exc:
            return {
                "ok": False,
                "job": job.trajectory_key,
                "split": job.split,
                "error": f"failed on frame {frame_idx}: {exc}",
                "generated": generated_paths,
                "audit": {},
            }

    audit = {
        "n_frames": traj.n_frames,
        "n_atoms": traj.n_atoms,
        "top_ext": job.top_path.suffix.lower(),
        "topology": str(job.top_path),
    }
    if generated_paths:
        audit.update(build_cif_metadata_summary(Path(generated_paths[0])))
    else:
        audit.update(
            {
                "n_label_entity_ids": 0,
                "n_label_asym_ids": 0,
                "n_label_comp_ids": 0,
                "label_entity_ids": "",
                "label_asym_ids": "",
                "label_comp_ids": "",
            }
        )

    return {
        "ok": True,
        "job": job.trajectory_key,
        "split": job.split,
        "error": None,
        "generated": generated_paths,
        "audit": audit,
    }


def write_path_list(paths, out_path: Path):
    with open(out_path, "w") as f:
        for path in sorted(paths):
            f.write(path + "\n")


def write_metadata_audit(results, jobs_by_key, out_path: Path):
    fieldnames = [
        "job",
        "split",
        "status",
        "xtc_path",
        "top_path",
        "n_frames",
        "n_atoms",
        "top_ext",
        "n_cif_written",
        "n_label_entity_ids",
        "n_label_asym_ids",
        "n_label_comp_ids",
        "label_entity_ids",
        "label_asym_ids",
        "label_comp_ids",
        "error",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["job"]):
            job = jobs_by_key.get(r["job"])
            audit = r.get("audit", {})
            writer.writerow(
                {
                    "job": r["job"],
                    "split": r["split"],
                    "status": "ok" if r["ok"] else "error",
                    "xtc_path": str(job.xtc_path) if job else "",
                    "top_path": str(job.top_path) if job else "",
                    "n_frames": audit.get("n_frames", ""),
                    "n_atoms": audit.get("n_atoms", ""),
                    "top_ext": audit.get("top_ext", ""),
                    "n_cif_written": len(r.get("generated", [])),
                    "n_label_entity_ids": audit.get("n_label_entity_ids", ""),
                    "n_label_asym_ids": audit.get("n_label_asym_ids", ""),
                    "n_label_comp_ids": audit.get("n_label_comp_ids", ""),
                    "label_entity_ids": audit.get("label_entity_ids", ""),
                    "label_asym_ids": audit.get("label_asym_ids", ""),
                    "label_comp_ids": audit.get("label_comp_ids", ""),
                    "error": r.get("error", "") or "",
                }
            )


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    outdir = Path(args.outdir)
    mmcif_root = outdir / "mmcif"
    split_dir = Path(args.split_dir) if args.split_dir else (outdir / "splits")
    data_root = Path(args.data_root)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(mmcif_root, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    jobs, skipped_no_top = discover_jobs(data_root, allow_pdb_fallback=args.allow_pdb_fallback)
    train_csv, test_csv = write_split_manifests(jobs, split_dir)

    selected_jobs = jobs
    if args.target_split in ("train", "test"):
        selected_jobs = [j for j in selected_jobs if j.split == args.target_split]
    if args.max_jobs is not None:
        selected_jobs = selected_jobs[: args.max_jobs]

    print(f"Discovered {len(jobs)} total trajectory jobs.")
    print(f"Selected {len(selected_jobs)} trajectory jobs for conversion.")
    print(f"Split manifests: {train_csv} and {test_csv}")
    print(f"No-topology skipped trajectories: {len(skipped_no_top)}")
    print(f"PDB fallback allowed: {args.allow_pdb_fallback}")
    print(f"Strict topology check: {args.strict_topology_check}")

    if not selected_jobs:
        return

    if args.num_workers > 1:
        results = [
            r
            for r in tqdm(
                Parallel(n_jobs=args.num_workers, return_as="generator_unordered")(
                    delayed(convert_job)(
                        job,
                        mmcif_root,
                        args.frame_stride,
                        args.skip_existing,
                        args.strict_topology_check,
                    )
                    for job in selected_jobs
                ),
                total=len(selected_jobs),
                desc="Converting trajectories",
            )
        ]
    else:
        results = []
        for job in tqdm(selected_jobs, desc="Converting trajectories"):
            results.append(
                convert_job(
                    job,
                    mmcif_root,
                    args.frame_stride,
                    args.skip_existing,
                    args.strict_topology_check,
                )
            )

    errors = [r for r in results if not r["ok"]]
    train_paths = []
    test_paths = []
    for r in results:
        if r["split"] == "train":
            train_paths.extend(r["generated"])
        else:
            test_paths.extend(r["generated"])

    train_txt = split_dir / "lynxkite_train_mmcif.txt"
    test_txt = split_dir / "lynxkite_test_mmcif.txt"
    write_path_list(train_paths, train_txt)
    write_path_list(test_paths, test_txt)

    jobs_by_key = {j.trajectory_key: j for j in selected_jobs}
    audit_csv = split_dir / "lynxkite_metadata_audit.csv"
    write_metadata_audit(results, jobs_by_key, audit_csv)

    print(f"Done. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    print(f"Train mmCIF list: {train_txt}")
    print(f"Test mmCIF list: {test_txt}")
    print(f"Metadata audit: {audit_csv}")

    if errors:
        err_path = split_dir / "lynxkite_conversion_errors.txt"
        with open(err_path, "w") as f:
            for err in errors:
                f.write(f"{err['job']}\t{err['error']}\n")
        print(f"Wrote error report: {err_path}")


if __name__ == "__main__":
    main()
