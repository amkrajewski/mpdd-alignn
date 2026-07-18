"""Standalone regression tests for pretrained MPDD ALIGNN wrappers."""

from pathlib import Path
from unittest import TestCase, mock

import pytest

from alignn import pretrained
from alignn.pretrained import _run_pretrained_models


def test_invalid_model_name_raises_error():
    """Regression test: passing invalid model names should raise a ValueError
    from `_run_pretrained_models` with the expected message.
    """
    invalid_models_list = ["fake_model_name_xyz", "another_bad_model"]

    with pytest.raises(ValueError, match="The following model names were not found in default models"):
        _run_pretrained_models(
            atoms_array=[],
            outputs=[],
            models=invalid_models_list,
        )


def _fake_run_pretrained_models(
    atoms_array, outputs, mode="serial", saveGraphs=False, models=None
):
    """Return deterministic payloads so the wrapper behavior can be compared.

    Includes models information in response for validation.
    """
    return [
        {
            **out,
            "mock_pred": f"{out['name']}|{mode}|{saveGraphs}",
            "models_used": models,  # Include for verification
        }
        for out in outputs
    ]


# Get model names from default models
_default_models = pretrained.get_default_models()
_model_names = [m["name"] for m in _default_models]
_first_model = _model_names[0] if _model_names else "model_1"
_second_model = _model_names[1] if len(_model_names) > 1 else "model_2"
_third_model = _model_names[2] if len(_model_names) > 2 else "model_3"


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


@pytest.mark.parametrize(
    "models_selection,mode,saveGraphs",
    [
        (None, "serial", False),                                    # Default: all models
        ([_first_model, _second_model], "serial", False),           # Two models
        ([_first_model], "serial", True),                           # Single model with saveGraphs
        ([], "serial", False),                                      # Empty list (no models)
        ([_third_model, _first_model], "parallel", False),          # Multiple models, parallel mode
    ],
)
class TestModelFiltering:
    """Parametrized tests for model filtering functionality."""

    def test_run_models_from_structure_with_model_filter(
        self, models_selection, mode, saveGraphs
    ):
        """Test that run_models_from_structure correctly passes and validates model filtering."""
        sigma_phase_dir = Path(__file__).resolve().parents[2] / "example.SigmaPhase"
        poscar_files = sorted(
            path
            for path in sigma_phase_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".poscar"
        )

        assert len(poscar_files) > 0, "Expected at least one POSCAR file in example.SigmaPhase"

        test_path = str(poscar_files[0])

        with mock.patch.object(
            pretrained,
            "_run_pretrained_models",
            side_effect=_fake_run_pretrained_models,
        ) as mock_run:
            results = pretrained.run_models_from_structure(
                test_path,
                mode=mode,
                saveGraphs=saveGraphs,
                models=models_selection,
            )

            # Verify the mock was called with correct parameters
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs

            # Verify models parameter was passed correctly
            assert call_kwargs["models"] == models_selection, (
                f"Expected models={models_selection}, got {call_kwargs['models']}"
            )

            # Verify other parameters
            assert call_kwargs["mode"] == mode
            assert call_kwargs["saveGraphs"] == saveGraphs

            # Verify the models information is in the returned results
            assert len(results) == 1
            assert results[0]["models_used"] == models_selection

    def test_run_models_from_directory_with_model_filter(
        self, models_selection, mode, saveGraphs
    ):
        """Test that run_models_from_directory correctly passes and validates model filtering."""
        sigma_phase_dir = Path(__file__).resolve().parents[2] / "example.SigmaPhase"

        with mock.patch.object(
            pretrained,
            "_run_pretrained_models",
            side_effect=_fake_run_pretrained_models,
        ) as mock_run:
            results = pretrained.run_models_from_directory(
                str(sigma_phase_dir),
                mode=mode,
                saveGraphs=saveGraphs,
                models=models_selection,
            )

            # Verify the mock was called with correct parameters
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs

            # Verify models parameter was passed correctly
            assert call_kwargs["models"] == models_selection, (
                f"Expected models={models_selection}, got {call_kwargs['models']}"
            )

            # Verify other parameters
            assert call_kwargs["mode"] == mode
            assert call_kwargs["saveGraphs"] == saveGraphs

            # Verify the models information is in all returned results
            for result in results:
                assert result["models_used"] == models_selection
