import json
import os

MAP_WIDTH = 50.0
MAP_HEIGHT = 300.0

A_ZONE_Y = (0.0, 99.0)
DEAD_ZONE_Y = (100.0, 199.0)
B_ZONE_Y = (200.0, 299.0)

TPS = 30
PLAYER_SPEED = 25.0 / TPS
BULLET_SPEED = 50.0 / TPS
FIRE_COOLDOWN = int(0.5 * TPS)
HITBOX_RADIUS = 1.6

ARENA_X = 20
ARENA_Y = 20
SCALE_X = 5.6
SCALE_Y = 2.4
ARENA_PIXEL_W = int(MAP_WIDTH * SCALE_X)
ARENA_PIXEL_H = int(MAP_HEIGHT * SCALE_Y)
SCREEN_WIDTH = 740
SCREEN_HEIGHT = 760

COLOR_BG = (18, 22, 28)
COLOR_PANEL = (28, 34, 44)
COLOR_ARENA = (10, 12, 16)
COLOR_DEADZONE = (22, 26, 36)
COLOR_GRID = (35, 42, 54)
COLOR_PLAYER_A = (0, 210, 255)
COLOR_PLAYER_B = (255, 75, 75)
COLOR_BULLET = (255, 220, 0)
COLOR_TEXT = (230, 235, 245)
COLOR_MUTED = (140, 150, 165)
COLOR_ACCENT = (60, 200, 120)

CONFIG_FILE = "hyperparams.json"
DEFAULT_HYPERPARAMS = {
    "imitation_lr": 0.001,
    "imitation_epochs": 30,
    "imitation_batch_size": 64,
    "rl_lr": 0.0002,
    "rl_epochs_per_feedback": 3,
    "gamma": 0.96,
    "exploration_rate": 0.25,
    "exploration_temp": 1.0,  # 0.2 (greedy) to 3.0 (high exploration)
    "ppo_clip": 0.15,
    "feedback_magnitude": 5.0
}

def load_hyperparams():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_HYPERPARAMS, **json.load(f)}
        except Exception:
            return DEFAULT_HYPERPARAMS.copy()
    return DEFAULT_HYPERPARAMS.copy()

def save_hyperparams(params):
    with open(CONFIG_FILE, "w") as f:
        json.dump(params, f, indent=4)
