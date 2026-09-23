"""Exercise real checkpoint loading using SDL's headless video/audio drivers."""
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import torch

import main
from engine import GameEngine
from model import ActorCritic
from training_interactive import InteractiveTrainer
from training_imitation import train_imitation
from recorder import BehaviorRecorder
from config import DEFAULT_HYPERPARAMS


@contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ExecutableCheckpoint:
    """Harmless canary: unrestricted unpickling would create a marker file."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).touch()",)


class ModelLoadingTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def loaders(self, path):
        return {
            "match": lambda: main.load_agent(str(path))[0],
            "trainee": lambda: InteractiveTrainer(
                main.screen, base_model_path=str(path)
            ).model,
            "opponent": lambda: InteractiveTrainer(
                main.screen, opponent_choice=str(path)
            ).opponent_model,
        }

    def test_bundled_models_load_and_produce_valid_actions(self):
        paths = sorted((Path(__file__).resolve().parents[1] / "saved_models").glob("*.pt"))
        self.assertTrue(paths, "Bundled checkpoints must be exercised")
        state = torch.tensor(GameEngine().get_canonical_features("A")).unsqueeze(0)
        for path in paths:
            for name, load in self.loaders(path).items():
                with self.subTest(checkpoint=path.name, loader=name):
                    model = load()
                    self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
                    action, value = model.act(state, deterministic=True)
                    self.assertIn(action, range(5))
                    self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_executable_checkpoint_is_rejected_by_every_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            path = Path(directory) / "untrusted.pt"
            torch.save(ExecutableCheckpoint(marker), path)
            for name, load in self.loaders(path).items():
                with self.subTest(loader=name):
                    with self.assertRaises(pickle.UnpicklingError):
                        load()
                    self.assertFalse(marker.exists(), "Checkpoint code executed")

    def test_loaders_explicitly_request_restricted_cpu_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(ActorCritic().state_dict(), path)
            for name, load in self.loaders(path).items():
                with self.subTest(loader=name):
                    with patch.object(torch, "load", wraps=torch.load) as spy:
                        load()
                    spy.assert_called_once_with(str(path), weights_only=True, map_location="cpu")

    def test_greedy_and_scratch_do_not_load_checkpoints(self):
        with patch.object(torch, "load", side_effect=AssertionError("Unexpected load")):
            self.assertIsNone(main.load_agent("greedy")[0])
            self.assertIsInstance(main.load_agent("scratch")[0], ActorCritic)
            trainer = InteractiveTrainer(main.screen, base_model_path="scratch")
            for _ in range(30):
                trainer.run_step()
            trainer.draw()
            self.assertEqual(len(trainer.chain_states), 30)

    def test_record_train_coach_and_reload(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            engine = GameEngine()
            recorder = BehaviorRecorder()
            for step in range(30):
                action = step % 5
                recorder.record(engine.get_canonical_features("A"), action)
                engine.step(action, 0)
            self.assertTrue(Path(recorder.save()).is_file())
            params = {**DEFAULT_HYPERPARAMS, "imitation_epochs": 1}
            with patch("training_imitation.load_hyperparams", return_value=params):
                path = train_imitation()
            trainer = InteractiveTrainer(main.screen, base_model_path=path)
            for _ in range(30):
                trainer.run_step()
            trainer.apply_feedback(5.0)
            saved = main.load_agent("saved_models/finetuned_model.pt")[0]
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(torch.equal(value, saved.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
