import os
import time
import numpy as np

class BehaviorRecorder:
    def __init__(self, mode_name="session"):
        self.states = []
        self.move_actions = []
        self.mode_name = mode_name

    def record(self, state: np.ndarray, move_act: int):
        self.states.append(state)
        self.move_actions.append(move_act)

    def save(self):
        if len(self.states) < 30:
            return None
        
        os.makedirs("recordings", exist_ok=True)
        timestamp = int(time.time())
        filename = f"recordings/{self.mode_name}_{timestamp}.npz"
        np.savez_compressed(
            filename,
            states=np.array(self.states, dtype=np.float32),
            move_actions=np.array(self.move_actions, dtype=np.int64)
        )
        print(f"[Recorder] Saved {len(self.states)} transitions to {filename}")
        self.states.clear()
        self.move_actions.clear()
        return filename
