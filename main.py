import os
import sys
import glob
import queue
import threading
import pygame
import torch
from config import *
from engine import GameEngine
from model import ActorCritic
from greedy_ai import GreedyAI
from recorder import BehaviorRecorder
from training_imitation import train_imitation, TrainingCancelled
from training_interactive import InteractiveTrainer
from options import show_options_menu

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Autobattler AI Arena")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
title_font = pygame.font.Font(None, 34)

def model_browser(prompt_title="Select a Model", include_greedy=False, include_scratch=False):
    os.makedirs("saved_models", exist_ok=True)
    model_files = sorted(glob.glob("saved_models/*.pt"), key=os.path.getctime, reverse=True)

    items = []
    if include_greedy:
        items.append(("greedy", "[ RULE-BASED: Greedy AI ]"))
    if include_scratch:
        items.append(("scratch", "[ INITIALIZE: Fresh Model (Scratch) ]"))

    for f in model_files:
        items.append((f, os.path.basename(f)))

    if not items:
        return None

    selected = 0
    while True:
        screen.fill(COLOR_BG)
        title = title_font.render(prompt_title, True, COLOR_PLAYER_A)
        screen.blit(title, (50, 40))

        sub = font.render("Use UP/DOWN arrows and press ENTER to select (or ESC to cancel)", True, COLOR_MUTED)
        screen.blit(sub, (50, 85))

        y = 130
        for i, (val, display_name) in enumerate(items):
            col = COLOR_ACCENT if i == selected else COLOR_TEXT
            prefix = "-> " if i == selected else "   "
            txt = font.render(prefix + display_name, True, col)
            screen.blit(txt, (60, y))
            y += 36

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(items)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(items)
                elif event.key == pygame.K_RETURN:
                    return items[selected][0]

def load_agent(choice):
    if choice == "greedy" or choice is None:
        return None, "Greedy AI"
    if choice == "scratch":
        m = ActorCritic(state_dim=44)
        m.eval()
        return m, "Scratch Model"
    
    m = ActorCritic(state_dim=44)
    m.load_state_dict(torch.load(choice, weights_only=True, map_location="cpu"), strict=False)
    m.eval()
    return m, os.path.basename(choice)

def run_match(mode="single_player", chosen_a=None, chosen_b=None):
    engine = GameEngine()
    recorder = BehaviorRecorder(mode_name=mode)
    
    model_a, name_a = None, "Human (P1)"
    model_b, name_b = None, "Greedy AI"
    greedy_bot_a = GreedyAI('A')
    greedy_bot_b = GreedyAI('B')

    if mode == "agent_vs_agent":
        model_a, name_a = load_agent(chosen_a)
        model_b, name_b = load_agent(chosen_b)
    elif mode == "multiplayer":
        name_b = "Human (P2)"
    elif mode == "single_player":
        model_b, name_b = load_agent(chosen_b)

    running = True
    while running:
        clock.tick(TPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                recorder.save()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                recorder.save()
                return

        keys = pygame.key.get_pressed()

        # --- Player A Action (WASD) ---
        if mode in ["single_player", "multiplayer"]:
            m_a = 0
            if keys[pygame.K_a]: m_a = 1      # Left
            elif keys[pygame.K_d]: m_a = 2    # Right
            elif keys[pygame.K_s]: m_a = 3    # Advance (Down)
            elif keys[pygame.K_w]: m_a = 4    # Retreat (Up)

            recorder.record(engine.get_canonical_features('A'), m_a)
        else:
            if model_a:
                feat_a = engine.get_canonical_features('A')
                t_a = torch.tensor(feat_a, dtype=torch.float32).unsqueeze(0)
                m_a = model_a.act(t_a, deterministic=False)[0]
            else:
                m_a = greedy_bot_a.get_action(engine)

        # --- Player B Action (Arrows) ---
        if mode == "multiplayer":
            m_b = 0
            if keys[pygame.K_LEFT]: m_b = 1    # Left
            elif keys[pygame.K_RIGHT]: m_b = 2 # Right
            elif keys[pygame.K_UP]: m_b = 3    # Advance (Up)
            elif keys[pygame.K_DOWN]: m_b = 4  # Retreat (Down)

            recorder.record(engine.get_canonical_features('B'), m_b)
        else:
            if model_b:
                feat_b = engine.get_canonical_features('B')
                t_b = torch.tensor(feat_b, dtype=torch.float32).unsqueeze(0)
                m_b = model_b.act(t_b, deterministic=False)[0]
            else:
                m_b = greedy_bot_b.get_action(engine)

        engine.step(m_a, m_b)

        # Render
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_ARENA, (ARENA_X, ARENA_Y, ARENA_PIXEL_W, ARENA_PIXEL_H))
        
        dz_top = int(ARENA_Y + DEAD_ZONE_Y[0] * SCALE_Y)
        dz_h = int((DEAD_ZONE_Y[1] - DEAD_ZONE_Y[0]) * SCALE_Y)
        pygame.draw.rect(screen, COLOR_DEADZONE, (ARENA_X, dz_top, ARENA_PIXEL_W, dz_h))

        # Colored Bullets
        for b in engine.bullets:
            bx = int(ARENA_X + b.x * SCALE_X)
            by = int(ARENA_Y + b.y * SCALE_Y)
            b_color = COLOR_PLAYER_A if b.owner == 'A' else COLOR_PLAYER_B
            pygame.draw.circle(screen, b_color, (bx, by), 4)

        pa_x = int(ARENA_X + engine.player_a.x * SCALE_X)
        pa_y = int(ARENA_Y + engine.player_a.y * SCALE_Y)
        pygame.draw.polygon(screen, COLOR_PLAYER_A, [(pa_x, pa_y + 8), (pa_x - 8, pa_y - 6), (pa_x + 8, pa_y - 6)])

        pb_x = int(ARENA_X + engine.player_b.x * SCALE_X)
        pb_y = int(ARENA_Y + engine.player_b.y * SCALE_Y)
        pygame.draw.polygon(screen, COLOR_PLAYER_B, [(pb_x, pb_y - 8), (pb_x - 8, pb_y + 6), (pb_x + 8, pb_y + 6)])

        px = ARENA_X + ARENA_PIXEL_W + 25
        hud_lines = [
            f"MODE: {mode.upper()}",
            "-" * 26,
            f"Player A: {name_a[:16]}",
            f"Score: {engine.player_a.score}",
            "",
            f"Player B: {name_b[:16]}",
            f"Score: {engine.player_b.score}",
            "-" * 26,
            f"Logged Frames: {len(recorder.states)}",
            "",
            "AUTO-SHOOT ACTIVE:",
            "P1: WASD",
            "P2: Arrow Keys",
            "",
            "Auto-fires straight on cooldown",
            "",
            "[ESC] Exit & Save"
        ]
        y_off = 25
        for l in hud_lines:
            col = COLOR_ACCENT if "Player" in l else COLOR_TEXT
            screen.blit(font.render(l, True, col), (px, y_off))
            y_off += 24

        pygame.display.flip()

def run_trainer_gui(base_model=None, opponent="greedy"):
    trainer = InteractiveTrainer(screen, base_model_path=base_model, opponent_choice=opponent)
    trainer.reset_round()
    running = True

    while running:
        clock.tick(TPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                # Live Coach Pause
                if not trainer.chain_closed and event.key in [pygame.K_SPACE, pygame.K_p]:
                    trainer.trigger_manual_pause()
                # Live Coach Pause
                if not trainer.chain_closed and event.key in [pygame.K_SPACE, pygame.K_p]:
                    trainer.trigger_manual_pause()
                if trainer.eval_prompt or trainer.chain_closed:
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        val = float(event.unicode)
                        trainer.apply_feedback(val)
                    elif event.key in [pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        val = -float(int(event.unicode) - 5)
                        trainer.apply_feedback(val)
                    elif event.key == pygame.K_MINUS:
                        trainer.apply_feedback(-5.0)
                    elif event.key in [pygame.K_d, pygame.K_SPACE]:
                        trainer.discard_sequence()

        if not trainer.chain_closed:
            trainer.run_step()

        screen.fill(COLOR_BG)
        trainer.draw()
        pygame.display.flip()

def run_imitation_gui():
    """Keep all SDL work on the main thread while training runs separately."""
    updates = queue.SimpleQueue()
    cancel_event = threading.Event()

    def train():
        try:
            path = train_imitation(
                progress_callback=lambda fraction, message: updates.put(
                    ("progress", fraction, message)),
                cancel_event=cancel_event,
            )
            message = (f"Model saved: {os.path.basename(path)}" if path
                       else "No logs found to train! Play Mode 2 or 3 first.")
            updates.put(("done", 1.0 if path else 0.0, message))
        except TrainingCancelled:
            updates.put(("done", None, "Training cancelled."))
        except Exception as exc:
            updates.put(("done", None, f"Training failed: {exc}"))

    worker = threading.Thread(target=train, name="imitation-training", daemon=True)
    worker.start()
    fraction, message = 0.0, "Starting training..."
    finished = False
    while True:
        clock.tick(TPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cancel_event.set()
                return None
            if event.type == pygame.KEYDOWN:
                if finished and event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    return message
                if event.key == pygame.K_ESCAPE:
                    cancel_event.set()

        while True:
            try:
                kind, progress, message = updates.get_nowait()
            except queue.Empty:
                break
            if progress is not None:
                fraction = progress
            if kind == "done":
                finished = True

        screen.fill(COLOR_BG)
        screen.blit(title_font.render("TRAINING ON LOGS", True, COLOR_PLAYER_A), (50, 80))
        bar = pygame.Rect(50, 220, SCREEN_WIDTH - 100, 30)
        pygame.draw.rect(screen, COLOR_PANEL, bar)
        fill = bar.copy()
        fill.width = int(bar.width * max(0.0, min(1.0, fraction)))
        pygame.draw.rect(screen, COLOR_ACCENT, fill)
        screen.blit(font.render(f"{fraction:.0%}", True, COLOR_TEXT), (50, 265))
        # Wrap errors and long file names to keep the result visible.
        words = message.split()
        line, y = "", 320
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and font.size(candidate)[0] > SCREEN_WIDTH - 100:
                screen.blit(font.render(line, True, COLOR_TEXT), (50, y))
                line, y = word, y + 26
            else:
                line = candidate
        screen.blit(font.render(line, True, COLOR_TEXT), (50, y))
        hint = ("[ENTER / ESC] Back to menu" if finished else
                "Cancelling... finishing current operation" if cancel_event.is_set() else
                "[ESC] Cancel training")
        screen.blit(font.render(hint, True, COLOR_MUTED), (50, 600))
        pygame.display.flip()


def main_menu():
    selected = 0
    options = [
        "1. Real Game (Select Model A vs Model B)",
        "2. Single Player (Human vs Selectable Opponent)",
        "3. Multiplayer (Human vs Human)",
        "4. Offline Imitation Training (Train on Logs)",
        "5. Interactive Sparring Gym (Fine-Tune vs Opponent)",
        "6. Options / Hyperparameters",
        "7. Exit"
    ]
    status_msg = ""

    while True:
        screen.fill(COLOR_BG)
        title = title_font.render("AUTOBATTLER AI ARENA", True, COLOR_PLAYER_A)
        screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 40))

        y = 130
        for i, opt in enumerate(options):
            col = COLOR_ACCENT if i == selected else COLOR_TEXT
            txt = font.render(opt, True, col)
            screen.blit(txt, (80, y))
            y += 48

        if status_msg:
            st_txt = font.render(status_msg, True, COLOR_BULLET)
            screen.blit(st_txt, (80, y + 15))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selected == 0:
                        mA = model_browser("Select Model for Player A (Top / Cyan)", include_greedy=True)
                        if mA:
                            mB = model_browser("Select Model for Player B (Bottom / Red)", include_greedy=True)
                            if mB:
                                run_match("agent_vs_agent", chosen_a=mA, chosen_b=mB)
                    elif selected == 1:
                        opp = model_browser("Select Opponent for Single Player", include_greedy=True)
                        if opp:
                            run_match("single_player", chosen_b=opp)
                    elif selected == 2:
                        run_match("multiplayer")
                    elif selected == 3:
                        status_msg = run_imitation_gui()
                        if status_msg is None:
                            return
                    elif selected == 4:
                        base = model_browser("Select Model to Train / Fine-Tune", include_scratch=True)
                        if base:
                            opp = model_browser("Select Sparring Opponent", include_greedy=True)
                            if opp:
                                run_trainer_gui(base_model=base, opponent=opp)
                    elif selected == 5:
                        updated_params = show_options_menu(screen, font, title_font, clock)
                        status_msg = f"Updated: Epochs={updated_params['imitation_epochs']}, LR={updated_params['imitation_lr']}"
                    elif selected == 6:
                        return

if __name__ == "__main__":
    main_menu()
