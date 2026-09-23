"""Regression coverage for training progress and a responsive SDL event loop."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pygame
import main
from config import DEFAULT_HYPERPARAMS
from training_imitation import train_imitation, TrainingCancelled
from test_model_loading import working_directory


class ImitationProgressTests(unittest.TestCase):
    def setUp(self):
        pygame.init()
        main.screen = pygame.display.set_mode((740, 760))
        main.font = pygame.font.Font(None, 24)
        main.title_font = pygame.font.Font(None, 34)
        pygame.event.clear()

    def test_real_training_reports_monotonic_progress_and_saves(self):
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            Path('recordings').mkdir()
            np.savez('recordings/test.npz', states=np.zeros((16, 44)),
                     move_actions=np.arange(16) % 5)
            reports = []
            params = {**DEFAULT_HYPERPARAMS, 'imitation_epochs': 2,
                      'imitation_batch_size': 8}
            with patch('training_imitation.load_hyperparams', return_value=params):
                path = train_imitation(lambda value, message: reports.append((value, message)))
            values = [value for value, _ in reports]
            self.assertEqual(values, sorted(values))
            self.assertEqual(values[0], 0)
            self.assertEqual(values[-1], 1)
            self.assertTrue(Path(path).is_file())
            self.assertTrue(any('Saving' in message for _, message in reports))

    def test_cancel_does_not_save(self):
        cancel = threading.Event()
        cancel.set()
        with patch('training_imitation.torch.save') as save:
            with self.assertRaises(TrainingCancelled):
                train_imitation(cancel_event=cancel)
            save.assert_not_called()

    def test_ui_keeps_drawing_and_handles_escape_while_worker_waits(self):
        frames = 0
        worker_finished = threading.Event()
        ui_thread = threading.get_ident()

        def slow_training(progress_callback, cancel_event):
            self.assertNotEqual(threading.get_ident(), ui_thread)
            progress_callback(0.4, 'Training in progress')
            if not cancel_event.wait(3):
                raise AssertionError('UI did not process cancellation')
            worker_finished.set()
            raise TrainingCancelled()

        def flip():
            nonlocal frames
            self.assertEqual(threading.get_ident(), ui_thread)
            frames += 1
            if frames == 3 or worker_finished.is_set():
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            if frames > 100:
                self.fail('UI failed to return after cancellation')

        with patch('main.train_imitation', side_effect=slow_training), \
                patch('main.pygame.display.flip', side_effect=flip):
            result = main.run_imitation_gui()
        self.assertEqual(result, 'Training cancelled.')
        self.assertGreaterEqual(frames, 3)

    def test_failure_is_displayed_and_can_return_to_menu(self):
        def flip():
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        with patch('main.train_imitation', side_effect=ValueError('bad log')), \
                patch('main.pygame.display.flip', side_effect=flip):
            self.assertEqual(main.run_imitation_gui(), 'Training failed: bad log')

    def test_window_close_returns_without_waiting_for_worker(self):
        stopped = threading.Event()
        def slow_training(progress_callback, cancel_event):
            cancel_event.wait(3)
            stopped.set()
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        with patch('main.train_imitation', side_effect=slow_training):
            self.assertIsNone(main.run_imitation_gui())
            self.assertTrue(stopped.wait(1))


if __name__ == '__main__':
    unittest.main()
