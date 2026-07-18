"""Minimal runtime test for the README / quickstart directory example."""

import json
from pathlib import Path
import unittest

from alignn import pretrained


def _load_reference_results():
    """Load the saved example output from the bundled JSON file."""
    reference_path = Path(__file__).with_name("readme_example_reference.json")
    return json.loads(reference_path.read_text())


class TestReadmeExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pretrained = pretrained
        cls.reference_results = _load_reference_results()
        cls.example_dir = Path(__file__).resolve().parents[2] / "example.SigmaPhase"

    def test_run_models_from_directory_matches_reference(self):
        self.pretrained.download_default_models(verbose=False)

        actual_results = self.pretrained.run_models_from_directory(
            str(self.example_dir),
            mode="serial",
        )

        self.assertEqual(len(actual_results), len(self.reference_results))
        self.assertCountEqual(
            [result["name"] for result in actual_results],
            [result["name"] for result in self.reference_results],
        )

        reference_by_name = {
            result["name"]: result for result in self.reference_results
        }
        model_names = [key for key in self.reference_results[0] if key != "name"]

        print("\nREADME regression report")

        for actual in actual_results:
            structure_name = actual["name"]
            expected = reference_by_name[structure_name]
            print(f"Test structure {structure_name}:")

            with self.subTest(structure=structure_name, check="keys"):
                try:
                    self.assertEqual(set(actual), set(expected))
                except AssertionError:
                    print(" - keys [FAIL]")
                    raise
                else:
                    print(" - keys [PASS]")

            with self.subTest(structure=structure_name, check="name"):
                try:
                    self.assertEqual(actual["name"], structure_name)
                except AssertionError:
                    print(
                        f" - name [FAIL] actual={actual['name']!r} expected={structure_name!r}"
                    )
                    raise
                else:
                    print(f" - name [PASS] {structure_name}")

            for model_name in model_names:
                expected_value = expected[model_name]
                actual_value = actual[model_name]
                with self.subTest(structure=structure_name, model=model_name):
                    try:
                        self.assertAlmostEqual(actual_value, expected_value, delta=1e-3)
                    except AssertionError:
                        print(
                            f" - {model_name} [FAIL] actual={actual_value} expected={expected_value}"
                        )
                        raise
                    else:
                        print(f" - {model_name} [PASS] {actual_value} ~= {expected_value}")


if __name__ == "__main__":
    unittest.main(verbosity=2)