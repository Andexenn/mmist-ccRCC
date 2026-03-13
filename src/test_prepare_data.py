"""
Test script for prepare_data.py — verifies Case ID extraction logic
====================================================================
Creates a mock feature directory structure with sample filenames,
mock CSV reference files, and validates the output master CSV.

Usage:
    python test_prepare_data.py
"""

import os
import sys
import shutil
import tempfile

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import (
    extract_case_id_clinical,
    extract_case_id_wsi,
    extract_uid_from_scan,
    prepare_dataset,
)


def test_extract_case_id_clinical():
    """Clinical: filename = Case ID"""
    assert extract_case_id_clinical("C3L-00004") == "C3L-00004"
    assert extract_case_id_clinical("C3L-00004.pt") == "C3L-00004"
    assert extract_case_id_clinical("TCGA-B0-4710") == "TCGA-B0-4710"
    print("  [PASS] Clinical Case ID extraction")


def test_extract_case_id_wsi():
    """WSI: C-prefix → trim after last dash; T-prefix → trim after 3rd dash"""
    # C-prefix: remove everything from the LAST dash onward
    assert extract_case_id_wsi("C3L-00004-21") == "C3L-00004"
    assert extract_case_id_wsi("C3L-00004-21.pt") == "C3L-00004"
    assert extract_case_id_wsi("C3N-01200-25") == "C3N-01200"

    # T-prefix: remove everything from the 3rd dash onward
    assert extract_case_id_wsi("TCGA-B0-4710-01Z") == "TCGA-B0-4710"
    assert extract_case_id_wsi("TCGA-B0-4710-01Z.pt") == "TCGA-B0-4710"
    assert extract_case_id_wsi("TCGA-CZ-5985-01A-02") == "TCGA-CZ-5985"
    print("  [PASS] WSI Case ID extraction")


def test_extract_uid_from_scan():
    """MRI/CT: strip _Gr* suffix to get UID"""
    assert extract_uid_from_scan("SOME_UID_Gr3+4") == "SOME_UID"
    assert extract_uid_from_scan("SOME_UID_Gr3+4.pt") == "SOME_UID"
    assert extract_uid_from_scan("ABC_DEF_GHI_Gr2") == "ABC_DEF_GHI"
    assert extract_uid_from_scan("NO_GRADE_SUFFIX") == "NO_GRADE_SUFFIX"
    print("  [PASS] MRI/CT UID extraction")


def test_full_pipeline():
    """End-to-end test with mock data"""
    tmpdir = tempfile.mkdtemp(prefix="mmist_test_")

    try:
        # Create directory structure
        for modality in ['Clinical', 'CT', 'MRI', 'WSI']:
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(tmpdir, modality, split), exist_ok=True)

        # Create mock Clinical entries (folders)
        os.makedirs(os.path.join(tmpdir, 'Clinical', 'train', 'C3L-00004'), exist_ok=True)
        open(os.path.join(tmpdir, 'Clinical', 'train', 'C3L-00004', 'feat.pt'), 'w').close()

        os.makedirs(os.path.join(tmpdir, 'Clinical', 'train', 'TCGA-B0-4710'), exist_ok=True)
        open(os.path.join(tmpdir, 'Clinical', 'train', 'TCGA-B0-4710', 'feat.pt'), 'w').close()

        # Create mock WSI entries (folders)
        os.makedirs(os.path.join(tmpdir, 'WSI', 'train', 'C3L-00004-21'), exist_ok=True)
        open(os.path.join(tmpdir, 'WSI', 'train', 'C3L-00004-21', 'feat.pt'), 'w').close()

        os.makedirs(os.path.join(tmpdir, 'WSI', 'train', 'TCGA-B0-4710-01Z'), exist_ok=True)
        open(os.path.join(tmpdir, 'WSI', 'train', 'TCGA-B0-4710-01Z', 'feat.pt'), 'w').close()

        # Create mock CT entries (folders)
        os.makedirs(os.path.join(tmpdir, 'CT', 'train', 'UID_001_Gr3+4'), exist_ok=True)
        open(os.path.join(tmpdir, 'CT', 'train', 'UID_001_Gr3+4', 'feat.pt'), 'w').close()

        # Create mock MRI entries (folders)
        os.makedirs(os.path.join(tmpdir, 'MRI', 'train', 'UID_002_Gr2'), exist_ok=True)
        open(os.path.join(tmpdir, 'MRI', 'train', 'UID_002_Gr2', 'feat.pt'), 'w').close()

        # Create CPTAC.csv (UID → Case ID)
        cptac_df = pd.DataFrame({
            'uid': ['UID_001', 'UID_002'],
            'case_id': ['C3L-00004', 'TCGA-B0-4710']
        })
        cptac_df.to_csv(os.path.join(tmpdir, 'CPTAC.csv'), index=False)

        # Create TCGA.csv (empty but valid)
        tcga_df = pd.DataFrame({'uid': [], 'case_id': []})
        tcga_df.to_csv(os.path.join(tmpdir, 'TCGA.csv'), index=False)

        # Create clinical.csv (Case ID → vital_status_12)
        clinical_df = pd.DataFrame({
            'case_id': ['C3L-00004', 'TCGA-B0-4710'],
            'vital_status_12': [1, 0]
        })
        clinical_df.to_csv(os.path.join(tmpdir, 'clinical.csv'), index=False)

        # Run prepare_dataset
        output_path = os.path.join(tmpdir, 'master_dataset.csv')
        result_df = prepare_dataset(tmpdir, output_path)

        # Validate results
        assert os.path.exists(output_path), "Output CSV not created"
        assert len(result_df) > 0, "Result DataFrame is empty"

        # Check Clinical entries
        cli_rows = result_df[result_df['Modality'] == 'Clinical']
        assert len(cli_rows) == 2, f"Expected 2 Clinical rows, got {len(cli_rows)}"

        # Check WSI entries
        wsi_rows = result_df[result_df['Modality'] == 'WSI']
        assert len(wsi_rows) == 2, f"Expected 2 WSI rows, got {len(wsi_rows)}"

        # Check the C-prefix WSI mapping
        c_wsi = wsi_rows[wsi_rows['file_name'] == 'C3L-00004-21']
        assert len(c_wsi) == 1, "Missing C-prefix WSI entry"
        assert c_wsi.iloc[0]['case_id'] == 'C3L-00004', f"Wrong case_id: {c_wsi.iloc[0]['case_id']}"
        assert c_wsi.iloc[0]['vital_status_12'] == 1, f"Wrong label: {c_wsi.iloc[0]['vital_status_12']}"

        # Check the T-prefix WSI mapping
        t_wsi = wsi_rows[wsi_rows['file_name'] == 'TCGA-B0-4710-01Z']
        assert len(t_wsi) == 1, "Missing T-prefix WSI entry"
        assert t_wsi.iloc[0]['case_id'] == 'TCGA-B0-4710', f"Wrong case_id: {t_wsi.iloc[0]['case_id']}"
        assert t_wsi.iloc[0]['vital_status_12'] == 0, f"Wrong label: {t_wsi.iloc[0]['vital_status_12']}"

        # Check CT entry (UID_001 → C3L-00004)
        ct_rows = result_df[result_df['Modality'] == 'CT']
        assert len(ct_rows) == 1, f"Expected 1 CT row, got {len(ct_rows)}"
        assert ct_rows.iloc[0]['case_id'] == 'C3L-00004'

        # Check MRI entry (UID_002 → TCGA-B0-4710)
        mri_rows = result_df[result_df['Modality'] == 'MRI']
        assert len(mri_rows) == 1, f"Expected 1 MRI row, got {len(mri_rows)}"
        assert mri_rows.iloc[0]['case_id'] == 'TCGA-B0-4710'

        print("  [PASS] Full pipeline test")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    print("=" * 50)
    print("Running prepare_data tests...")
    print("=" * 50)

    test_extract_case_id_clinical()
    test_extract_case_id_wsi()
    test_extract_uid_from_scan()
    test_full_pipeline()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
