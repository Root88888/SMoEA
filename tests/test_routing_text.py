import json
import os
import tempfile
import unittest

import numpy as np

from router.core import Router
from router.data_io import load_task_texts


class RoutingTextTests(unittest.TestCase):
    def test_training_route_prompt_is_the_full_request_without_the_answer(self):
        with tempfile.TemporaryDirectory() as root:
            train_dir = os.path.join(root, "train_data")
            test_dir = os.path.join(root, "test_data")
            os.makedirs(train_dir)
            os.makedirs(test_dir)
            with open(
                os.path.join(train_dir, "task2_train.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "instances": [
                            {
                                "input": "Probability is...",
                                "full_prompt": (
                                    "Generate a 1-5 word title.\n\n"
                                    "Text: Probability is...\n\nTitle:Probability"
                                ),
                                "output": "Probability",
                            }
                        ]
                    },
                    f,
                )
            with open(
                os.path.join(test_dir, "task2_test.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "instances": [
                            {
                                "input": "Llamas originated...",
                                "full_prompt": (
                                    "Generate a 1-5 word title.\n\n"
                                    "Text: Llamas originated...\n\nTitle:"
                                ),
                                "output": "Llama",
                            }
                        ]
                    },
                    f,
                )
            cfg = {
                "paths": {"dataset_dir": root},
                "data": {
                    "field": "input",
                    "routing_text": "answer_free_full_prompt",
                    "doc_clip_chars": 3000,
                },
            }

            self.assertEqual(
                load_task_texts(cfg, 2),
                [
                    "Generate a 1-5 word title.\n\n"
                    "Text: Probability is...\n\nTitle:"
                ],
            )
            self.assertEqual(
                load_task_texts(cfg, 2, test=True),
                [
                    "Generate a 1-5 word title.\n\n"
                    "Text: Llamas originated...\n\nTitle:"
                ],
            )

            cfg["data"]["routing_text"] = "input"
            self.assertEqual(load_task_texts(cfg, 2), ["Probability is..."])

    def test_saved_router_records_which_text_it_was_built_from(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = {
                "paths": {"assets_dir": root},
                "data": {
                    "field": "input",
                    "routing_text": "answer_free_full_prompt",
                },
                "embedding": {"model_name": "test-embedder"},
                "thresholds": {},
            }
            router = Router(cfg)
            router.FP = np.array([[1.0]], dtype=np.float32)
            router.C = np.array([[1.0]], dtype=np.float32)
            router.owner = np.array([0])
            router.unit_of = np.array([0])
            router.cal_b1 = np.array([0])
            router.cal_margin = np.array([0.5], dtype=np.float32)
            router.id_tasks = [2]
            router.units = [[0]]
            router.k_by_task = {2: 1}
            router.merged_pairs = []
            router.lex = {"test": "index"}
            router.unit_examples = {"0": "example"}

            router.save()

            with open(
                os.path.join(root, "router_assets_meta.json"),
                encoding="utf-8",
            ) as f:
                metadata = json.load(f)
            self.assertEqual(
                metadata["routing_text"], "answer_free_full_prompt"
            )

            cfg["data"]["routing_text"] = "input"
            with self.assertRaisesRegex(ValueError, "routing_text"):
                Router.load(cfg)


if __name__ == "__main__":
    unittest.main()
