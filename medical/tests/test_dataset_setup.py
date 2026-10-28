"""Dataset tooling checks using synthetic files only; never loads training code."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np
from medical.datasets import setup_lpba40 as setup


class SetupTests(unittest.TestCase):
    def test_layout_copy_and_existing_destination_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for number in range(1, 41):
                for kind in ('skullstripped', 'structure.label'):
                    data = np.ones((2, 3, 4), dtype=np.uint8)
                    nib.save(nib.Nifti1Image(data, setup.AFFINE),
                             source / f'S{number:02}.delineation.{kind}.nii.gz')
            destination = root / 'prepared'
            with patch.object(setup, 'SHAPE', (2, 3, 4)):
                self.assertEqual(setup.prepare(source, destination), 80)
                self.assertEqual(len(list((destination / 'delineation_l_norm/test').glob('*.nii.gz'))), 9)
                self.assertEqual(len(list((destination / 'delineation_l_norm/train').glob('*.nii.gz'))), 30)
                self.assertEqual(setup.digest(destination / 'delineation_l_norm/fixed.nii.gz'),
                                 setup.digest(source / 'S01.delineation.skullstripped.nii.gz'))
                with self.assertRaises(FileExistsError):
                    setup.prepare(source, destination)
            self.assertEqual(len(list(source.glob('*.nii.gz'))), 80)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('../escape', 'bad')
            with self.assertRaises(ValueError):
                setup.unpack(archive, root / 'extract')
            self.assertFalse((root / 'escape').exists())

    def test_unprepared_grid_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((2, 3, 4)), np.eye(4)), path)
            with self.assertRaisesRegex(ValueError, 'incompatible grid'):
                setup.inspect(path)


if __name__ == '__main__':
    unittest.main()
