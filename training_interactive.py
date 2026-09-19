import os
import sys
import glob
import pygame
import torch
import torch.nn as nn
import numpy as np
from config import *
from engine import GameEngine, Bullet
from model import ActorCritic
from greedy_ai import GreedyAI

class InteractiveTrainer:
    def __init__(self, screen, base_model_path=None, opponent_choice="greedy"):
        self.screen = screen
        self.params = load_hyperparams()
        self.engine = GameEngine()
        self.font = pygame.font.Font(None, 24)

        # Agent being trained (Player A - Cyan)
        self.model = ActorCritic(state_dim=44)
        self.model_name = "Scratch Model"
        if base_model_path and os.path.exists(base_model_path):
            self.model.load_state_dict(torch.load(base_model_path), strict=False)
            self.model_name = os.path.basename(base_model_path)

        # Opponent (Player B - Red)
        self.opponent_name = "Greedy AI"
        self.opponent_model = None
        self.greedy_bot_b = GreedyAI('B')
        if opponent_choice != "greedy" and opponent_choice and os.path.exists(opponent_choice):
            self.opponent_model = ActorCritic(state_dim=44)
            self.opponent_model.load_state_dict(torch.load(opponent_choice), strict=False)
            self.opponent_model.eval()
            self.opponent_name = os.path.basename(opponent_choice)

        self.chain_states = []
        self.chain_moves = []
        self.chain_closed = False
        self.eval_prompt = False
        self.result_badge = ""

    def reset_round(self):
        self.resume_flow()

    def resume_flow(self):
        self.chain_states.clear()
        self.chain_moves.clear()
        self.chain_closed = False
        self.eval_prompt = False
        self.result_badge = ""
        self.engine.bullets.clear()
        self.engine.player_a.cooldown = FIRE_COOLDOWN
        self.engine.player_b.cooldown = FIRE_COOLDOWN

    def hard_reset_center(self):
        self.engine.reset()
        self.resume_flow()
        print("[Gym] Hard reset players back to center.")

    def trigger_manual_pause(self):
        if len(self.chain_states) >= 10:
            self.chain_closed = True
            self.eval_prompt = True
            self.result_badge = "MANUAL COACH PAUSE: RATE MOTION"

    def run_step(self):
        state_np = self.engine.get_canonical_features('A')
        state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        
        eps = self.params.get("exploration_rate", 0.25)
        move_a, _ = self.model.act(state_t, deterministic=False, exploration_rate=eps)

        self.chain_states.append(state_np)
        self.chain_moves.append(move_a)

        # Opponent action
        if self.opponent_model:
            feat_b = self.engine.get_canonical_features('B')
            tb = torch.tensor(feat_b, dtype=torch.float32).unsqueeze(0)
            move_b = self.opponent_model.act(tb, deterministic=False)[0]
        else:
            move_b = self.greedy_bot_b.get_action(self.engine)

        self.engine.step(move_a, move_b)

        if self.engine.last_hit == 'B':
            self.chain_closed = True
            self.eval_prompt = True
            self.result_badge = "AGENT SCORED HIT! [SUCCESS]"
        elif self.engine.last_hit == 'A':
            self.chain_closed = True
            self.eval_prompt = True
            self.result_badge = "AGENT WAS HIT! [FAILED DODGE]"
        elif len(self.chain_states) >= 900:
            self.chain_closed = True
            self.eval_prompt = True
            self.result_badge = "STALEMATE PAUSE (30s)"

    def apply_feedback(self, reward_val: float):
        if not self.chain_states:
            return

        # 1. TIGHT WINDOW: Capture strictly the last 25 frames (~0.8s)!
        # Eliminates the 60 frames of wall-camping pollution!
        window_size = min(len(self.chain_states), 25)
        recent_states = self.chain_states[-window_size:]
        recent_moves = self.chain_moves[-window_size:]

        states = torch.tensor(np.array(recent_states), dtype=torch.float32)
        moves = torch.tensor(np.array(recent_moves), dtype=torch.int64)

        # Decisive learning rate dedicated to human feedback
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)

        if reward_val > 0:
            # POSITIVE REWARD (+1 to +5): Direct Behavioral Injection!
            # Supervised Cross-Entropy pulls P(action) up directly and decisively.
            weight = (reward_val / 5.0)  # Scale 0.2 to 1.0
            criterion = nn.CrossEntropyLoss()
            for _ in range(12):
                logits, _ = self.model(states)
                loss = criterion(logits, moves) * weight
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            print(f"[Coach Boost] Positively reinforced {window_size} frames with {reward_val:+.1f} (12 CE epochs)")
        else:
            # NEGATIVE PENALTY (-1 to -5): Direct Suppression!
            # Pushes the probability of the penalized actions down directly.
            weight = abs(reward_val) / 5.0
            for _ in range(8):
                logits, _ = self.model(states)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(1, moves.unsqueeze(1)).squeeze(1)
                # Minimize probability of those actions
                loss = selected_log_probs.mean() * weight
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            print(f"[Coach Suppress] Suppressed {window_size} frames with {reward_val:+.1f} (8 epochs)")

        self.save_model()
        self.resume_flow()

    def discard_sequence(self):
        print(f"[Coach] Resumed without weight change.")
        self.resume_flow()

    def save_model(self):
        os.makedirs("saved_models", exist_ok=True)
        path = "saved_models/finetuned_model.pt"
        torch.save(self.model.state_dict(), path)

    def draw(self):
        pygame.draw.rect(self.screen, COLOR_ARENA, (ARENA_X, ARENA_Y, ARENA_PIXEL_W, ARENA_PIXEL_H))
        dz_top = int(ARENA_Y + DEAD_ZONE_Y[0] * SCALE_Y)
        dz_h = int((DEAD_ZONE_Y[1] - DEAD_ZONE_Y[0]) * SCALE_Y)
        pygame.draw.rect(self.screen, COLOR_DEADZONE, (ARENA_X, dz_top, ARENA_PIXEL_W, dz_h))

        for b in self.engine.bullets:
            bx = int(ARENA_X + b.x * SCALE_X)
            by = int(ARENA_Y + b.y * SCALE_Y)
            b_col = COLOR_PLAYER_A if b.owner == 'A' else COLOR_PLAYER_B
            pygame.draw.circle(self.screen, b_col, (bx, by), 4)

        pa_x = int(ARENA_X + self.engine.player_a.x * SCALE_X)
        pa_y = int(ARENA_Y + self.engine.player_a.y * SCALE_Y)
        pygame.draw.polygon(self.screen, COLOR_PLAYER_A, [(pa_x, pa_y + 8), (pa_x - 8, pa_y - 6), (pa_x + 8, pa_y - 6)])

        pb_x = int(ARENA_X + self.engine.player_b.x * SCALE_X)
        pb_y = int(ARENA_Y + self.engine.player_b.y * SCALE_Y)
        pygame.draw.polygon(self.screen, COLOR_PLAYER_B, [(pb_x, pb_y - 8), (pb_x - 8, pb_y + 6), (pb_x + 8, pb_y + 6)])

        panel_x = ARENA_X + ARENA_PIXEL_W + 30
        lines = [
            "LIVE COACHING GYM",
            "-" * 24,
            f"Trainee: {self.model_name[:16]}",
            f"Score A: {self.engine.player_a.score} | B: {self.engine.player_b.score}",
            f"Exploration: {self.params.get('exploration_rate', 0.25):.0%}",
            f"Active Ticks: {len(self.chain_states)}",
            "",
            "WHILE RUNNING:",
            "[SPACE] : PAUSE TO GRADE MANEUVER!",
            "",
            "WHEN PAUSED:",
            "[1..5]  : Reward  (+1 to +5)",
            "[6..9]  : Penalty (-1 to -4)",
            "[-]     : Penalty (-5.0)",
            "[D]     : Resume (Keep Pos)",
            "[R]     : Hard Reset Center",
            "[ESC]   : Back to Menu"
        ]
        
        if self.eval_prompt:
            lines.insert(6, f">>> {self.result_badge} <<<")

        y_offset = 25
        for line in lines:
            col = COLOR_ACCENT if ">>>" in line or "[SPACE]" in line else COLOR_TEXT
            if "WAS HIT" in line: col = COLOR_PLAYER_B
            txt = self.font.render(line, True, col)
            self.screen.blit(txt, (panel_x, y_offset))
            y_offset += 24
