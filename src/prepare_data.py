"""
Data Preparation Script — Filename → Case ID → Label Mapping
==============================================================
Scans the feature directory structure and generates a master CSV that
the training pipeline can consume.

Feature directory structure:
    feat/
    ├── Clinical/{train,val,test}/   → filename IS the Case ID
    ├── CT/{train,val,test}/         → strip _Gr* suffix → UID → lookup in CPTAC/TCGA.csv
    ├── MRI/{train,val,test}/        → strip _Gr* suffix → UID → lookup in CPTAC/TCGA.csv
    ├── WSI/{train,val,test}/        → C-prefix: trim after last '-'; T-prefix: trim after 3rd '-'
    ├── CPTAC.csv
    ├── TCGA.csv
    └── clinical.csv

Usage:
    python prepare_data.py --feature_dir ./feat --output ./feat/master_dataset.csv
"""

import argparse
import os
import re
import sys
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ─── Case ID Extraction Functions ────────────────────────────────────────────

def extract_case_id_clinical(name: str) -> str:
    """Clinical files: the filename (without extension) IS the Case ID."""
    return os.path.splitext(name)[0]


def extract_case_id_wsi(name: str) -> str:
    """
    WSI files:
      - If starts with 'C': remove everything from the LAST dash onward
      - If starts with 'T': remove everything from the 3rd dash onward
    """
    base = os.path.splitext(name)[0]

    if base.startswith('C'):
        # Find the last dash and trim
        last_dash = base.rfind('-')
        if last_dash > 0:
            return base[:last_dash]
        return base

    elif base.startswith('T'):
        # Find the 3rd dash and trim
        parts = base.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        return base

    else:
        logger.warning("WSI file '%s' doesn't start with 'C' or 'T', using full name as Case ID", name)
        return base


def extract_uid_from_scan(name: str) -> str:
    """
    MRI/CT files: strip the '_Gr...' suffix to get the UID.
    Example: 'SOME_UID_Gr3+4' → 'SOME_UID'
    """
    base = os.path.splitext(name)[0]
    # Match _Gr followed by any characters at the end
    uid = re.sub(r'_Gr.*$', '', base)
    return uid


def build_uid_to_caseid_map(feature_dir: str) -> dict:
    """
    Build a UID → Case ID lookup from CPTAC.csv and TCGA.csv.
    Auto-detects column names for UID and Patient/Case ID.
    """
    uid_map = {}

    for csv_name in ['CPTAC.csv', 'TCGA.csv']:
        csv_path = os.path.join(feature_dir, csv_name)
        if not os.path.exists(csv_path):
            logger.warning("Reference CSV not found: %s (skipping)", csv_path)
            continue

        df = pd.read_csv(csv_path)
        columns_lower = {c.lower().strip(): c for c in df.columns}

        # Auto-detect UID column
        uid_col = None
        for candidate in ['uid', 'slide_id', 'image_id', 'filename', 'file_name', 'name', 'series instance uid']:
            if candidate in columns_lower:
                uid_col = columns_lower[candidate]
                break

        # Auto-detect Case ID / Patient ID column
        pid_col = None
        for candidate in ['case_id', 'patient_id', 'patient id', 'caseid', 'patientid',
                          'subject_id', 'subjectid', 'case id']:
            if candidate in columns_lower:
                pid_col = columns_lower[candidate]
                break

        if uid_col is None or pid_col is None:
            logger.warning(
                "Could not auto-detect columns in %s. "
                "Columns found: %s. Looking for UID + Case ID columns.",
                csv_name, list(df.columns)
            )
            # Fallback: assume first column = UID, second column = Case ID
            if len(df.columns) >= 2:
                uid_col = df.columns[0]
                pid_col = df.columns[1]
                logger.info("Fallback: using '%s' as UID, '%s' as Case ID", uid_col, pid_col)
            else:
                logger.error("Cannot parse %s — skipping", csv_name)
                continue

        logger.info("Loaded %s: UID='%s', CaseID='%s' (%d entries)",
                     csv_name, uid_col, pid_col, len(df))

        for _, row in df.iterrows():
            uid = str(row[uid_col]).strip()
            case_id = str(row[pid_col]).strip()
            uid_map[uid] = case_id

    logger.info("Total UID→CaseID mappings: %d", len(uid_map))
    return uid_map


# ─── Main Processing ─────────────────────────────────────────────────────────

def prepare_dataset(feature_dir: str, output_path: str) -> pd.DataFrame:
    """
    Scan the feature directory, extract Case IDs, look up labels,
    and generate the master CSV.
    """
    clinical_csv_path = os.path.join(feature_dir, 'clinical.csv')
    if not os.path.exists(clinical_csv_path):
        logger.error("clinical.csv not found at: %s", clinical_csv_path)
        sys.exit(1)

    # Load clinical data for label lookup
    clinical_df = pd.read_csv(clinical_csv_path)
    clinical_columns_lower = {c.lower().strip(): c for c in clinical_df.columns}

    # Find the Case ID column in clinical.csv
    clinical_id_col = None
    for candidate in ['case_id', 'caseid', 'patient_id', 'patientid', 
                      'case id', 'patient id', 'subject_id']:
        if candidate in clinical_columns_lower:
            clinical_id_col = clinical_columns_lower[candidate]
            break

    if clinical_id_col is None:
        logger.error("Cannot find Case ID column in clinical.csv. Columns: %s", list(clinical_df.columns))
        sys.exit(1)

    # Find vital_status_12 column
    vital_col = None
    for candidate in ['vital_status_12', 'vitalstatus12', 'vital_status']:
        if candidate in clinical_columns_lower:
            vital_col = clinical_columns_lower[candidate]
            break

    if vital_col is None:
        logger.error("Cannot find vital_status_12 column in clinical.csv. Columns: %s", list(clinical_df.columns))
        sys.exit(1)

    logger.info("clinical.csv loaded: CaseID='%s', Label='%s' (%d patients)",
                 clinical_id_col, vital_col, len(clinical_df))

    # Build label lookup: Case ID → vital_status_12
    label_lookup = {}
    for _, row in clinical_df.iterrows():
        cid = str(row[clinical_id_col]).strip()
        label_lookup[cid] = int(row[vital_col])

    # Build UID → Case ID map for MRI/CT
    uid_map = build_uid_to_caseid_map(feature_dir)

    # Scan feature directories
    modalities = ['Clinical', 'CT', 'MRI', 'WSI']
    splits = ['train', 'val', 'test']
    records = []

    for modality in modalities:
        for split in splits:
            split_dir = os.path.join(feature_dir, modality, split)
            if not os.path.exists(split_dir):
                logger.warning("Directory not found: %s (skipping)", split_dir)
                continue

            # List entries (could be files or folders)
            entries = sorted(os.listdir(split_dir))
            logger.info("Scanning %s/%s: %d entries", modality, split, len(entries))

            for entry in entries:
                entry_path = os.path.join(split_dir, entry)

                # Get the name to extract Case ID from
                name = entry

                # Extract Case ID based on modality
                case_id = None

                if modality == 'Clinical':
                    case_id = extract_case_id_clinical(name)

                elif modality == 'WSI':
                    case_id = extract_case_id_wsi(name)

                elif modality in ('CT', 'MRI'):
                    uid = extract_uid_from_scan(name)
                    case_id = uid_map.get(uid)
                    if case_id is None:
                        logger.warning(
                            "[%s/%s] UID '%s' (from '%s') not found in CPTAC/TCGA lookup — skipping",
                            modality, split, uid, name
                        )
                        continue

                if case_id is None:
                    logger.warning("[%s/%s] Could not extract Case ID from '%s' — skipping",
                                   modality, split, name)
                    continue

                # Look up label
                label = label_lookup.get(case_id)
                if label is None:
                    logger.warning(
                        "[%s/%s] Case ID '%s' (from '%s') not found in clinical.csv — skipping",
                        modality, split, case_id, name
                    )
                    continue

                records.append({
                    'case_id': case_id,
                    'file_name': name,
                    'Modality': modality,
                    'Split': split,
                    'vital_status_12': label,
                })

    if len(records) == 0:
        logger.error("No valid records found! Check your feature directory structure and CSV files.")
        sys.exit(1)

    # Create DataFrame and save
    master_df = pd.DataFrame(records)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("MASTER DATASET SUMMARY")
    logger.info("  Total records: %d", len(master_df))
    for mod in modalities:
        mod_df = master_df[master_df['Modality'] == mod]
        if len(mod_df) > 0:
            for sp in splits:
                sp_df = mod_df[mod_df['Split'] == sp]
                if len(sp_df) > 0:
                    alive = (sp_df['vital_status_12'] == 1).sum()
                    dead = (sp_df['vital_status_12'] == 0).sum()
                    logger.info("  %s/%s: %d entries (alive=%d, dead=%d)",
                                 mod, sp, len(sp_df), alive, dead)
    logger.info("=" * 60)

    master_df.to_csv(output_path, index=False)
    logger.info("Master CSV saved to: %s", os.path.abspath(output_path))

    return master_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate master dataset CSV from feature directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --feature_dir ./feat
  python prepare_data.py --feature_dir ./feat --output ./feat/master_dataset.csv
        """
    )
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='Path to the feature directory (containing CT/, MRI/, WSI/, Clinical/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for master CSV (default: <feature_dir>/master_dataset.csv)')
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or os.path.join(args.feature_dir, 'master_dataset.csv')
    prepare_dataset(args.feature_dir, output)


if __name__ == '__main__':
    main()
