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

def augment_mirror_data(states_np, moves_np):
    mirrored_states = states_np.copy()
    mirrored_moves = moves_np.copy()

    mirrored_states[:, 0] = 1.0 - mirrored_states[:, 0]
    mirrored_states[:, 3] = -mirrored_states[:, 3]

    for b_idx in range(5):
        col_rel_x = 5 + b_idx * 6
        col_vx = 5 + b_idx * 6 + 2
        mirrored_states[:, col_rel_x] = -mirrored_states[:, col_rel_x]
        mirrored_states[:, col_vx] = -mirrored_states[:, col_vx]

    for b_idx in range(3):
        col_rel_x = 35 + b_idx * 3
        mirrored_states[:, col_rel_x] = -mirrored_states[:, col_rel_x]

    mask_left = (moves_np == 1)
    mask_right = (moves_np == 2)
    mirrored_moves[mask_left] = 2
    mirrored_moves[mask_right] = 1

    aug_states = np.vstack([states_np, mirrored_states])
    aug_moves = np.concatenate([moves_np, mirrored_moves])
    return aug_states, aug_moves

class InteractiveTrainer:
    def __init__(self, screen, base_model_path=None, opponent_choice="greedy"):
        self.screen = screen
        self.params = load_hyperparams()
        self.engine = GameEngine()
        self.font = pygame.font.Font(None, 24)

        self.model = ActorCritic(state_dim=44)
        self.model_name = "Scratch Model"
        if base_model_path and os.path.exists(base_model_path):
            self.model.load_state_dict(torch.load(base_model_path), strict=False)
            self.model_name = os.path.basename(base_model_path)

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
        self.last_event_type = "manual"

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
            self.last_event_type = "manual"
            self.result_badge = "MANUAL COACH PAUSE: RATE MOTION"

    def run_step(self):
        state_np = self.engine.get_canonical_features('A')
        state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        
        eps = self.params.get("exploration_rate", 0.25)
        move_a, _ = self.model.act(state_t, deterministic=False, exploration_rate=eps)

        self.chain_states.append(state_np)
        self.chain_moves.append(move_a)

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
            self.last_event_type = "hit_scored"
            self.result_badge = "AGENT SCORED HIT! [SUCCESS]"
        elif self.engine.last_hit == 'A':
            self.chain_closed = True
            self.eval_prompt = True
            self.last_event_type = "hit_taken"
            self.result_badge = "AGENT WAS HIT! [FAILED DODGE]"
        elif len(self.chain_states) >= 900:
            self.chain_closed = True
            self.eval_prompt = True
            self.last_event_type = "timeout"
            self.result_badge = "STALEMATE PAUSE (30s)"

    def apply_feedback(self, reward_val: float):
        if not self.chain_states:
            return

        total_frames = len(self.chain_states)

        # 1. TEMPORAL TARGETING:
        # If Agent Scored a Hit: Reward the LAUNCH window (~110-150 frames ago)!
        # If Hit Taken or Manual Space: Reward the immediate recent window (~25 frames ago).
        if self.last_event_type == "hit_scored" and total_frames >= 120:
            start_f = max(0, total_frames - 150)
            end_f = total_frames - 100
            target_states = self.chain_states[start_f:end_f]
            target_moves = self.chain_moves[start_f:end_f]
            print(f"[Coach] Targeted attack launch window ({start_f} to {end_f} ticks ago)")
        else:
            window = min(total_frames, 25)
            target_states = self.chain_states[-window:]
            target_moves = self.chain_moves[-window:]

        # 2. ANTI-COWARDICE FILTER (For Positive Rewards):
        # Filter out any frame where the agent was idling or glued to the wall (X < 5 or X > 45)!
        clean_states, clean_moves = [], []
        for s, m in zip(target_states, target_moves):
            self_x = s[0] * MAP_WIDTH
            is_at_wall = (self_x < 4.5 or self_x > (MAP_WIDTH - 4.5))
            # Never reward wall-camping!
            if reward_val > 0 and (is_at_wall or m == 0):
                continue
            clean_states.append(s)
            clean_moves.append(m)

        if not clean_states:
            print("[Coach Warning] Discarded reward: Agent was idling at the wall. Only active center moves can be rewarded!")
            self.resume_flow()
            return

        # Symmetrize to prevent directional bias
        aug_states, aug_moves = augment_mirror_data(np.array(clean_states), np.array(clean_moves))
        states = torch.tensor(aug_states, dtype=torch.float32)
        moves = torch.tensor(aug_moves, dtype=torch.int64)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)

        if reward_val > 0:
            weight = (reward_val / 5.0)
            criterion = nn.CrossEntropyLoss()
            for _ in range(12):
                logits, _ = self.model(states)
                loss = criterion(logits, moves) * weight
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            print(f"[Coach Boost] Reinforced {len(clean_states)} active center frames with {reward_val:+.1f}")
        else:
            weight = abs(reward_val) / 5.0
            for _ in range(8):
                logits, _ = self.model(states)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(1, moves.unsqueeze(1)).squeeze(1)
                loss = selected_log_probs.mean() * weight
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            print(f"[Coach Suppress] Suppressed {len(clean_states)} blunder frames with {reward_val:+.1f}")

        self.save_model()
        self.resume_flow()

    def discard_sequence(self):
        print("[Coach] Resumed without weight change.")
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
