"""Small offline checks for numerical invariants and safe restarts."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_zeroconv import canny_f1, completed_pair, load_val, snapshots
from train_zeroconv import RankSampler, validate_output


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def train_args(self):
        return SimpleNamespace(output_dir=str(self.root), max_steps=6, init="zero", data_seed=0)

    def save_checkpoint(self, args):
        (self.root / "config.json").write_text(json.dumps({"args": vars(args)}))
        (self.root / "checkpoint").mkdir()
        (self.root / "checkpoint/step.json").write_text('{"step": 3}')

    def test_logs_without_checkpoint_are_rejected(self):
        (self.root / "train_log.csv").write_text("step,loss\n0,1.0\n")
        with self.assertRaisesRegex(ValueError, "no resumable checkpoint"):
            validate_output(self.train_args())

    def test_resume_allows_extension(self):
        args = self.train_args()
        self.save_checkpoint(args)
        args.max_steps = 12
        validate_output(args)

    def test_resume_rejects_changed_experiment(self):
        args = self.train_args()
        self.save_checkpoint(args)
        args.init = "default"
        with self.assertRaisesRegex(ValueError, "Run settings changed"):
            validate_output(args)

    def test_resume_rejects_earlier_target(self):
        args = self.train_args()
        self.save_checkpoint(args)
        args.max_steps = 2
        with self.assertRaisesRegex(ValueError, "already at 3"):
            validate_output(args)

    def save_evaluation(self):
        settings = {"n": 2, "steps": 20}
        saved = dict(settings=settings, n=2, f1=0.5, f1_sem=0.1, precision=0.5, recall=0.5)
        (self.root / "prompt.json").write_text(json.dumps(saved))
        np.save(self.root / "prompt.npy", np.zeros((2, 2048), dtype=np.float32))
        with (self.root / "prompt.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows([["id", "f1", "precision", "recall"],
                                         ["1", 0.5, 0.5, 0.5], ["2", 0.5, 0.5, 0.5]])
        return settings

    def test_complete_evaluation_is_reused(self):
        self.assertTrue(completed_pair(self.root, "prompt", self.save_evaluation()))

    def test_partial_json_is_not_complete(self):
        settings = self.save_evaluation()
        (self.root / "prompt.json").write_text('{"settings":')
        self.assertFalse(completed_pair(self.root, "prompt", settings))

    def test_missing_features_are_not_complete(self):
        settings = self.save_evaluation()
        (self.root / "prompt.npy").unlink()
        self.assertFalse(completed_pair(self.root, "prompt", settings))

    def test_partial_csv_is_not_complete(self):
        settings = self.save_evaluation()
        (self.root / "prompt.csv").write_text("id,f1,precision,recall\n1,0.5,0.5,0.5\n2,0.5")
        self.assertFalse(completed_pair(self.root, "prompt", settings))

    def test_changed_evaluation_settings_are_rejected(self):
        settings = self.save_evaluation()
        for change in ({"steps": 30}, {"n": 3}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Evaluation settings changed"):
                completed_pair(self.root, "prompt", {**settings, **change})

    def test_short_validation_split_is_rejected(self):
        (self.root / "meta.jsonl").write_text('{"id":"1","captions":["circle"]}\n')
        with self.assertRaisesRegex(ValueError, "Requested 2 validation images"):
            load_val(self.root, 2)

    def test_missing_snapshots_are_rejected(self):
        args = SimpleNamespace(controlnet=None, run=self.root, only=None, every=500)
        with self.assertRaisesRegex(SystemExit, "No matching snapshots"):
            snapshots(args)

    def test_same_samples_across_process_layouts_and_resume(self):
        for world in (1, 2):
            streams = [list(RankSampler(0, 3, 8, 2, world, rank)) for rank in range(world)]
            per_step = 8 // world
            for step in range(3):
                batch = [g for stream in streams for g in stream[step*per_step:(step+1)*per_step]]
                self.assertEqual(sorted(batch), list(range(step*8, (step+1)*8)))
            for rank, stream in enumerate(streams):
                resumed = list(RankSampler(1, 3, 8, 2, world, rank))
                self.assertEqual(resumed, stream[per_step:])

    def test_edge_metric_perfect_and_missing_edges(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = 255
        condition = cv2.Canny(rgb, 100, 200)
        self.assertEqual(canny_f1(rgb, condition, 2), (1.0, 1.0, 1.0))
        self.assertEqual(canny_f1(np.zeros_like(rgb), condition, 2), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
