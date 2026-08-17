import tempfile
import unittest
from pathlib import Path

from scripts.map_ood_aliases import map_ood_task149


class OODAliasMappingTests(unittest.TestCase):
    def test_maps_original_ood_task149_without_renaming_or_overwriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            source = dataset / "ood_test_data" / "task149_test.json"
            source.parent.mkdir(parents=True)
            source.write_text('{"instances": []}', encoding="utf-8")

            alias = map_ood_task149(dataset)

            self.assertTrue(source.is_file())
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), source.resolve())
            self.assertEqual(map_ood_task149(dataset), alias)


if __name__ == "__main__":
    unittest.main()
