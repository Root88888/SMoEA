import os
import tempfile
import unittest

from router.config import discover_serving_tasks


class ServingAssetBuildTests(unittest.TestCase):
    def test_serving_task_discovery_does_not_require_benchmark_test_files(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, "train_data")
            os.makedirs(train_dir)
            for task_id in (0, 149, 150, 9149):
                with open(
                    os.path.join(train_dir, f"task{task_id}_train.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("[]")
            with open(
                os.path.join(root, "ood_tasks.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("9149\n")

            self.assertEqual(
                discover_serving_tasks({"paths": {"dataset_dir": root}}),
                [0, 149, 150],
            )


if __name__ == "__main__":
    unittest.main()
