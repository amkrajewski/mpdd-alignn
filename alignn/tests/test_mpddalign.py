"""Standalone regression tests for pretrained MPDD ALIGNN wrappers."""

from pathlib import Path
from unittest import TestCase, mock

from alignn import pretrained


def _fake_run_pretrained_models(atoms_array, outputs, mode="serial", saveGraphs=False):
    """Return deterministic payloads so the wrapper behavior can be compared."""
    return [
        {
            **out,
            "mock_pred": f"{out['name']}|{mode}|{saveGraphs}",
        }
        for out in outputs
    ]


class TestMPDDAlignNPretrainedWrappers(TestCase):
    def test_run_models_from_directory_and_structure_agree_for_sigma_phase(self):
        sigma_phase_dir = Path(__file__).resolve().parents[2] / "example.SigmaPhase"
        poscar_files = sorted(
            path
            for path in sigma_phase_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".poscar"
        )

        self.assertGreater(
            len(poscar_files),
            0,
            "Expected at least one POSCAR file under example.SigmaPhase.",
        )

        with mock.patch.object(
            pretrained,
            "_run_pretrained_models",
            side_effect=_fake_run_pretrained_models,
        ):
            directory_results = pretrained.run_models_from_directory(
                str(sigma_phase_dir),
                mode="serial",
                saveGraphs=False,
            )

            directory_by_name = {result["name"]: result for result in directory_results}
            expected_names = {path.name for path in poscar_files}

            self.assertEqual(set(directory_by_name), expected_names)

            for path in poscar_files:
                with self.subTest(structure=path.name):
                    structure_result = pretrained.run_models_from_structure(
                        str(path),
                        mode="serial",
                        saveGraphs=False,
                    )

                    self.assertEqual(len(structure_result), 1)
                    self.assertEqual(structure_result[0], directory_by_name[path.name])
