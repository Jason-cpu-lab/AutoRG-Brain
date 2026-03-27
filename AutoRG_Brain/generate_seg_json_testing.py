"""
Generate Task001 segmentation JSON files in README-compatible format:
  - case_dic.json
  - dataset.json
  - test_file.json

Rules:
1) Use all NIfTI images under --data-root.
2) Use anomaly GT first (local *_seg / *_MASK) for label2.
3) If GT does not exist, fallback to pseudo anomaly masks (*_ab_mask / *_ab).
4) label1 uses anatomy masks (*_ana_mask / *_ana), typically from pseudo outputs.
5) Only keep entries where BOTH label1 and label2 exist.
"""

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

MODAL_KEYS = ["DWI", "T1WI", "T2WI", "T2FLAIR"]  # TESTING: Only 4 modalities, exclude ADC
REQUIRED_ANA_ROOT = Path("/home/jason/autorg/inference_output/T1_priority").resolve()


def empty_ds_stats():
    return {
        "total_images_scanned": 0,
        "eligible_modal_images": 0,
        "included": 0,
        "label2_gt": 0,
        "label2_pseudo": 0,
        "label1_local": 0,
        "label1_pseudo": 0,
        "excluded_missing_label1": 0,
        "excluded_missing_label2": 0,
        "modal_counts": {k: 0 for k in MODAL_KEYS},
    }


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def strip_ana_suffix(stem: str) -> str:
    if stem.endswith("_ana_mask"):
        return stem[: -len("_ana_mask")]
    if stem.endswith("_ana"):
        return stem[: -len("_ana")]
    return stem


def norm_stem(stem: str) -> str:
    s = stem.lower()
    return re.sub(r"(_ab_mask|_ana_mask|_ab|_ana)$", "", s)


def extract_case_id(stem: str) -> str:
    s = norm_stem(stem)
    return re.sub(r"([_-])(flair|t2f|t2flair|t2w|t2wi|t2|t1n|t1wi|t1|dwi|adc)$", "", s, flags=re.I)


def detect_modal(filename_stem: str):
    n = filename_stem.lower()
    # Do not map T1CE into T1WI
    if re.search(r"(^|[_-])t1ce($|[_-])|(^|[_-])t1c($|[_-])", n):
        return None
    if "adc" in n:
        return "ADC"
    if re.search(r"(^|[_-])dwi($|[_-])", n):
        return "DWI"
    if "flair" in n or re.search(r"(^|[_-])t2f($|[_-])", n):
        return "T2FLAIR"
    if re.search(r"(^|[_-])t2($|[_-])|(^|[_-])t2w($|[_-])", n):
        return "T2WI"
    if re.search(r"(^|[_-])t1($|[_-])|(^|[_-])t1n($|[_-])", n):
        return "T1WI"
    return None


def build_pseudo_maps(pseudo_roots):
    ab_by_norm = {}
    ana_by_norm = {}
    ab_by_modal = {k: {} for k in MODAL_KEYS}
    ana_by_modal = {k: {} for k in MODAL_KEYS}
    ana_exact_by_modal = {k: {} for k in MODAL_KEYS}
    ab_by_modal_case = {k: {} for k in MODAL_KEYS}
    ana_by_modal_case = {k: {} for k in MODAL_KEYS}

    for root in pseudo_roots:
        if not root.exists():
            continue
        root_resolved = root.resolve()
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            nm = p.name.lower()
            if not (nm.endswith(".nii") or nm.endswith(".nii.gz")):
                continue

            stem = strip_nii(p.name)
            key = norm_stem(stem)
            case_id = extract_case_id(stem)
            file_modal = None

            parent_modal = p.parent.name.upper()
            if parent_modal in MODAL_KEYS:
                file_modal = parent_modal
            else:
                file_modal = detect_modal(stem)

            if "_ab_mask" in nm or nm.endswith("_ab.nii.gz") or nm.endswith("_ab.nii"):
                ab_by_norm.setdefault(key, str(p))
                if file_modal in ab_by_modal:
                    ab_by_modal[file_modal].setdefault(key, str(p))
                    if case_id:
                        ab_by_modal_case[file_modal].setdefault(case_id, str(p))
            if "_ana_mask" in nm or nm.endswith("_ana.nii.gz") or nm.endswith("_ana.nii"):
                # Enforce anatomy masks only from the curated T1_priority pseudo root.
                try:
                    is_allowed_ana_root = root_resolved == REQUIRED_ANA_ROOT or REQUIRED_ANA_ROOT in root_resolved.parents
                except Exception:
                    is_allowed_ana_root = False
                if is_allowed_ana_root:
                    ana_by_norm.setdefault(key, str(p))
                    if file_modal in ana_by_modal:
                        ana_by_modal[file_modal].setdefault(key, str(p))
                        ana_stem = strip_ana_suffix(stem)
                        ana_exact_by_modal[file_modal].setdefault(ana_stem, str(p))
                        if case_id:
                            ana_by_modal_case[file_modal].setdefault(case_id, str(p))

    return ab_by_norm, ana_by_norm, ab_by_modal, ana_by_modal, ana_exact_by_modal, ab_by_modal_case, ana_by_modal_case


def find_local_gt_for_image(img_path: Path):
    """Return anomaly GT path if present near image path."""
    parent = img_path.parent
    stem = strip_nii(img_path.name)

    # ISLES-2022: derivatives/sub-xxx/ses-xxx/sub-xxx_ses-xxx_msk.nii[.gz]
    subj, ses = infer_isles_subject_session(img_path, stem)
    if subj and ses:
        for anc in [parent] + list(parent.parents):
            deriv = anc / "derivatives"
            if deriv.exists() and deriv.is_dir():
                for c in [
                    deriv / subj / ses / f"{subj}_{ses}_msk.nii",
                    deriv / subj / ses / f"{subj}_{ses}_msk.nii.gz",
                    deriv / subj / ses / f"{subj}_{ses}_MASK.nii.gz",
                    deriv / subj / ses / f"{subj}_{ses}_mask.nii.gz",
                ]:
                    if c.exists():
                        return str(c)
                break

    # BraTS style: <case>_<modal>.nii[.gz] -> <case>_seg.nii[.gz]
    m = re.match(r"(.+?)[_-](flair|t1|t1n|t1ce|t1c|t2|t2w|t2f|dwi|adc)$", stem, flags=re.I)
    if m:
        base = m.group(1)
        for c in [
            parent / f"{base}_seg.nii.gz",
            parent / f"{base}_seg.nii",
            parent / f"{base}-seg.nii.gz",
            parent / f"{base}-seg.nii",
            parent / f"{base}_MASK.nii.gz",
            parent / f"{base}_mask.nii.gz",
        ]:
            if c.exists():
                return str(c)

    # MSLesSeg style: P57_T1.nii.gz -> P57_MASK.nii.gz
    pid = stem.split("_")[0]
    for c in [
        parent / f"{pid}_MASK.nii.gz",
        parent / f"{pid}_mask.nii.gz",
        parent / f"{pid}_seg.nii.gz",
        parent / f"{pid}_seg.nii",
    ]:
        if c.exists():
            return str(c)

    # Generic local candidates
    for c in [
        parent / f"{stem}_seg.nii.gz",
        parent / f"{stem}_seg.nii",
        parent / f"{stem}-seg.nii.gz",
        parent / f"{stem}-seg.nii",
        parent / f"{stem}_msk.nii.gz",
        parent / f"{stem}_msk.nii",
        parent / f"{stem}_MASK.nii.gz",
        parent / f"{stem}_mask.nii.gz",
    ]:
        if c.exists():
            return str(c)

    return None


def pseudo_lookup(map_dict, image_stem: str):
    key = norm_stem(image_stem)
    if key in map_dict:
        return map_dict[key]

    # Relaxed match: one key extends another (e.g. P5_T2 vs P5_T2_T1)
    for k, v in map_dict.items():
        if k.startswith(key + "_") or key.startswith(k + "_"):
            return v

    return None


def get_dataset_name(img: Path, data_root: Path):
    try:
        rel = img.relative_to(data_root)
        return rel.parts[0] if len(rel.parts) > 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def infer_isles_subject_session(img_path: Path, stem: str):
    """Infer ISLES subject/session IDs, preferring folder path over filename.

    This avoids mismatches for nested files whose filename can be strokeperf-*
    while folders correctly encode sub-strokecase*/ses-*.
    """
    subj = None
    ses = None
    for part in img_path.parts:
        if subj is None and re.fullmatch(r"sub-strokecase[^/]*", part, flags=re.I):
            subj = part
        if ses is None and re.fullmatch(r"ses-[^/]+", part, flags=re.I):
            ses = part

    if subj and ses:
        return subj, ses

    m = re.match(r"(sub-[^_]+)_(ses-[^_]+)_.*$", stem, flags=re.I)
    if m:
        return m.group(1), m.group(2)
    return None, None


def expand_lookup_stems(stems, modal):
    alias_by_modal = {
        "T1WI": ["t1", "t1n", "t1wi"],
        "T2WI": ["t2", "t2w", "t2wi"],
        "T2FLAIR": ["flair", "t2f", "t2flair"],
        "DWI": ["dwi"],
        "ADC": ["adc"],
    }
    aliases = alias_by_modal.get(modal, [])
    out = []
    seen = set()

    for s in stems:
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)

        low = s.lower()
        for a in aliases:
            for sep in ["_", "-"]:
                token = f"{sep}{a}"
                if low.endswith(token):
                    prefix = s[: -len(token)]
                    for b in aliases:
                        cand = f"{prefix}{sep}{b}"
                        if cand not in seen:
                            out.append(cand)
                            seen.add(cand)
    return out


def pseudo_lookup_candidates(map_dict, stems):
    seen = set()
    for s in stems:
        if not s or s in seen:
            continue
        seen.add(s)
        hit = pseudo_lookup(map_dict, s)
        if hit is not None:
            return hit
    return None


def build_entries(data_root: Path, pseudo_roots):
    (
        ab_map,
        ana_map,
        ab_map_by_modal,
        ana_map_by_modal,
        ana_exact_by_modal,
        ab_map_by_modal_case,
        ana_map_by_modal_case,
    ) = build_pseudo_maps(pseudo_roots)

    training = []
    # TESTING: Limit to 5 cases per modality
    cases_per_modal = {k: 0 for k in MODAL_KEYS}
    stats = {
        "total_images_scanned": 0,
        "eligible_modal_images": 0,
        "included": 0,
        "label2_gt": 0,
        "label2_pseudo": 0,
        "label1_local": 0,
        "label1_pseudo": 0,
        "excluded_missing_label1": 0,
        "excluded_missing_label2": 0,
    }
    dataset_stats = {}

    for img in data_root.rglob("*"):
        if not img.is_file():
            continue
        if not (img.name.endswith(".nii") or img.name.endswith(".nii.gz")):
            continue

        stats["total_images_scanned"] += 1
        dataset_name = get_dataset_name(img, data_root)
        if dataset_name not in dataset_stats:
            dataset_stats[dataset_name] = empty_ds_stats()
        dataset_stats[dataset_name]["total_images_scanned"] += 1

        stem = strip_nii(img.name)
        low = stem.lower()

        # Skip masks themselves
        if any(x in low for x in ["_seg", "_mask", "_ab_mask", "_ana_mask"]) or low.endswith("_ab") or low.endswith("_ana"):
            continue

        modal = detect_modal(stem)
        if modal is None:
            continue

        # For ISLES-2022, keep ADC and DWI only (exclude FLAIR/T2FLAIR)
        if dataset_name == "ISLES-2022" and modal not in {"ADC", "DWI"}:
            continue

        stats["eligible_modal_images"] += 1
        dataset_stats[dataset_name]["eligible_modal_images"] += 1

        # label2 (abnormal): GT first, then pseudo fallback
        label2 = find_local_gt_for_image(img)
        l2_from_gt = label2 is not None
        label2_source = "gt" if l2_from_gt else "pseudo"
        parent_stem = strip_nii(img.parent.name)
        subj, ses = infer_isles_subject_session(img, stem)

        if dataset_name == "ISLES-2022":
            # ISLES nested files may have perf-style filenames that don't match masks;
            # prefer folder/session identifiers for robust matching.
            lookup_stems = []
            if parent_stem != img.parent.name:
                lookup_stems.append(parent_stem)
            if subj and ses:
                lookup_stems.extend([f"{subj}_{ses}", f"{subj}_{ses}_{modal.lower()}"])
            lookup_stems.append(stem)
        else:
            lookup_stems = [stem]
            if parent_stem != img.parent.name:
                lookup_stems.append(parent_stem)
            if subj and ses:
                lookup_stems.extend([f"{subj}_{ses}", f"{subj}_{ses}_{modal.lower()}"])

        lookup_stems = expand_lookup_stems(lookup_stems, modal)

        if label2 is None:
            modal_ab_map = ab_map_by_modal.get(modal, {})
            label2 = pseudo_lookup_candidates(modal_ab_map, lookup_stems)
            if label2 is None:
                case_ids = [extract_case_id(s) for s in lookup_stems if s]
                for case_id in case_ids:
                    if case_id and case_id in ab_map_by_modal_case.get(modal, {}):
                        label2 = ab_map_by_modal_case[modal][case_id]
                        break

        # label1 (anatomy): strictly from T1_priority pseudo masks only.
        # only accept ana masks whose basename exactly matches image stem,
        # i.e., <image_stem>_ana_mask.nii.gz (or _ana.nii.gz legacy).
        label1 = ana_exact_by_modal.get(modal, {}).get(stem)
        label1_source = "pseudo" if label1 is not None else None
        if label1 is not None:
            try:
                label1_resolved = Path(label1).resolve()
                if not (label1_resolved == REQUIRED_ANA_ROOT or REQUIRED_ANA_ROOT in label1_resolved.parents):
                    label1 = None
                    label1_source = None
            except Exception:
                label1 = None
                label1_source = None

        if label1 is None:
            stats["excluded_missing_label1"] += 1
            dataset_stats[dataset_name]["excluded_missing_label1"] += 1
            continue
        if label2 is None:
            stats["excluded_missing_label2"] += 1
            dataset_stats[dataset_name]["excluded_missing_label2"] += 1
            continue


        # TESTING: Only keep up to 5 cases per modality
        if cases_per_modal[modal] < 5:
            training.append(
                {
                    "image": str(img),
                    "label1": str(label1),
                    "label2": str(label2),
                    "modal": modal,
                    "dataset": dataset_name,
                    "label1_source": label1_source,
                    "label2_source": label2_source,
                }
            )
            cases_per_modal[modal] += 1
            stats["included"] += 1
            dataset_stats[dataset_name]["included"] += 1
            dataset_stats[dataset_name]["modal_counts"][modal] += 1

            if label1_source == "local":
                stats["label1_local"] += 1
                dataset_stats[dataset_name]["label1_local"] += 1
            else:
                stats["label1_pseudo"] += 1
                dataset_stats[dataset_name]["label1_pseudo"] += 1

            if l2_from_gt:
                stats["label2_gt"] += 1
                dataset_stats[dataset_name]["label2_gt"] += 1
            else:
                stats["label2_pseudo"] += 1
                dataset_stats[dataset_name]["label2_pseudo"] += 1

    # Deduplicate by (image, modal), prefer GT-like label2 over pseudo
    best = {}
    for tr in training:
        key = (tr["image"], tr["modal"])
        l2 = tr["label2"]
        gt_score = 1 if ("_seg." in l2 or "-seg." in l2 or "_MASK" in l2 or "_mask" in l2 or "_msk." in l2) else 0
        prev = best.get(key)
        if prev is None or gt_score > prev[0]:
            best[key] = (gt_score, tr)

    training = [v[1] for v in best.values()]
    training.sort(key=lambda x: x["image"])
    return training, stats, dataset_stats


def write_outputs(training, output_dir: Path):
    case_dic = {k: [] for k in MODAL_KEYS}
    for tr in training:
        case_dic[tr["modal"]].append(strip_nii(Path(tr["image"]).name))
    for k in case_dic:
        case_dic[k] = sorted(set(case_dic[k]))

    training_clean = [
        {
            "image": t["image"],
            "label1": t["label1"],
            "label2": t["label2"],
            "modal": t["modal"],
        }
        for t in training
    ]

    dataset_json = {
        "description": "Segmentation training dataset merged from /home/jason/autorg/dataset (GT-first, pseudo fallback)",
        "labels": {"0": "background", "1": "1", "2": "2", "3": "3", "4": "4"},
        "modality": {"0": "MRI"},
        "name": "AutoRG_AllDatasets_Seg",
        "numTest": 0,
        "numTraining": len(training_clean),
        "reference": "no",
        "release": "0.0",
        "tensorImageSize": "4D",
        "test": [],
        "training": training_clean,
    }

    train_ids = sorted(set(strip_nii(Path(t["image"]).name) for t in training_clean))
    test_file = {"training": train_ids, "validation": {"test": []}}

    output_dir.mkdir(parents=True, exist_ok=True)
    case_dic_path = output_dir / "case_dic.json"
    dataset_path = output_dir / "dataset.json"
    test_file_path = output_dir / "test_file.json"

    with open(case_dic_path, "w") as f:
        json.dump(case_dic, f, indent=2)
    with open(dataset_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    with open(test_file_path, "w") as f:
        json.dump(test_file, f, indent=2)

    return case_dic_path, dataset_path, test_file_path, case_dic


def write_detailed_audit(training, stats, dataset_stats, data_root: Path, pseudo_roots, output_dir: Path):
    detailed_by_dataset = {}
    for t in training:
        ds = t["dataset"]
        detailed_by_dataset.setdefault(ds, []).append(
            {
                "image": t["image"],
                "modal": t["modal"],
                "label1": t["label1"],
                "label1_source": t["label1_source"],
                "label2": t["label2"],
                "label2_source": t["label2_source"],
            }
        )

    for ds in detailed_by_dataset:
        detailed_by_dataset[ds].sort(key=lambda x: x["image"])

    audit = {
        "data_root": str(data_root),
        "pseudo_roots_used": [str(p) for p in pseudo_roots],
        "global_summary": stats,
        "dataset_summary": dataset_stats,
        "dataset_entries": detailed_by_dataset,
    }

    audit_path = output_dir / "generation_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    return audit_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Task001 segmentation JSON files")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/jason/autorg/dataset"),
        help="Root folder containing all datasets",
    )
    parser.add_argument(
        "--pseudo-root",
        type=Path,
        action="append",
        default=[
            Path("/home/jason/autorg/inference_output/T1_priority"),
        ],
        help="Pseudo label root (can be passed multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "raw_data" / "nnUNet_raw_data" / "Task001_seg_test",
        help="Output directory for case_dic.json, dataset.json, test_file.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    training, stats, dataset_stats = build_entries(args.data_root, args.pseudo_root)
    case_dic_path, dataset_path, test_file_path, case_dic = write_outputs(training, args.output_dir)
    audit_path = write_detailed_audit(training, stats, dataset_stats, args.data_root, args.pseudo_root, args.output_dir)

    print("Generated files:")
    print(f"  - {case_dic_path}")
    print(f"  - {dataset_path}")
    print(f"  - {test_file_path}")
    print(f"  - {audit_path}")
    print("Pseudo roots used:")
    for p in args.pseudo_root:
        print(f"  - {p}")
    print("\nSummary:")
    print(json.dumps(stats, indent=2))
    print("Modal counts:")
    print({k: len(v) for k, v in case_dic.items()})
    print("\nDataset summary (included / gt / pseudo):")
    compact_ds = {
        ds: {
            "included": v["included"],
            "label2_gt": v["label2_gt"],
            "label2_pseudo": v["label2_pseudo"],
            "excluded_missing_label1": v["excluded_missing_label1"],
            "excluded_missing_label2": v["excluded_missing_label2"],
        }
        for ds, v in sorted(dataset_stats.items())
    }
    print(json.dumps(compact_ds, indent=2))


if __name__ == "__main__":
    main()