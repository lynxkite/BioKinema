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
    return parser.parse_args()


def pdb_to_cif(pdb_path: str, cif_path: str):
    pdb_structure = load_structure(pdb_path)
    save_structure(cif_path, pdb_structure)


def sanitize_component(text: str) -> str:
    safe = text.replace("/", "__").replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in safe)


def pick_topology(xtc_path: Path, project_dir: Path):
    parent = xtc_path.parent
    stem = xtc_path.stem
    base_stem = stem.split(".part")[0]

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


def discover_jobs(data_root: Path):
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
                top_path = pick_topology(xtc_path, project_dir)
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


def convert_job(job: TrajectoryJob, mmcif_root: Path, frame_stride: int, skip_existing: bool):
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
            }

    return {
        "ok": True,
        "job": job.trajectory_key,
        "split": job.split,
        "error": None,
        "generated": generated_paths,
    }


def write_path_list(paths, out_path: Path):
    with open(out_path, "w") as f:
        for path in sorted(paths):
            f.write(path + "\n")


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

    jobs, skipped_no_top = discover_jobs(data_root)
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

    if not selected_jobs:
        return

    if args.num_workers > 1:
        results = [
            r
            for r in tqdm(
                Parallel(n_jobs=args.num_workers, return_as="generator_unordered")(
                    delayed(convert_job)(job, mmcif_root, args.frame_stride, args.skip_existing)
                    for job in selected_jobs
                ),
                total=len(selected_jobs),
                desc="Converting trajectories",
            )
        ]
    else:
        results = []
        for job in tqdm(selected_jobs, desc="Converting trajectories"):
            results.append(convert_job(job, mmcif_root, args.frame_stride, args.skip_existing))

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

    print(f"Done. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    print(f"Train mmCIF list: {train_txt}")
    print(f"Test mmCIF list: {test_txt}")

    if errors:
        err_path = split_dir / "lynxkite_conversion_errors.txt"
        with open(err_path, "w") as f:
            for err in errors:
                f.write(f"{err['job']}\t{err['error']}\n")
        print(f"Wrote error report: {err_path}")


if __name__ == "__main__":
    main()
