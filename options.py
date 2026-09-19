import pygame
from config import *

PARAM_DEFS = [
    ("imitation_epochs", "Imitation Epochs", int, 5, 300, 5, "{:d}"),
    ("imitation_lr", "Imitation Learning Rate", float, 0.0001, 0.0100, 0.0002, "{:.4f}"),
    ("imitation_batch_size", "Imitation Batch Size", "choice", [32, 64, 128, 256], None, None),
    ("rl_lr", "RL Fine-Tuning LR", float, 0.00005, 0.0050, 0.00005, "{:.5f}"),
    ("rl_epochs_per_feedback", "RL Epochs per Feedback", int, 1, 10, 1, "{:d}"),
    ("gamma", "Discount Factor (Gamma)", float, 0.80, 0.99, 0.01, "{:.2f}"),
    ("exploration_rate", "Exploration Rate (Random %)", float, 0.00, 0.80, 0.05, "{:.0%}"),
    ("feedback_magnitude", "Reward Magnitude Scale", float, 1.0, 20.0, 1.0, "{:.1f}")
]

def show_options_menu(screen, font, title_font, clock):
    params = load_hyperparams()
    selected = 0
    saved_notice = ""
    
    pygame.key.set_repeat(250, 60)

    running = True
    while running:
        clock.tick(TPS)
        screen.fill(COLOR_BG)

        title = title_font.render("TRAINING HYPERPARAMETERS", True, COLOR_PLAYER_A)
        screen.blit(title, (50, 35))

        sub = font.render("UP/DOWN: Select  |  LEFT/RIGHT: Adjust  |  [ENTER/ESC]: Save & Return", True, COLOR_MUTED)
        screen.blit(sub, (50, 75))

        pygame.draw.line(screen, COLOR_GRID, (50, 105), (SCREEN_WIDTH - 50, 105), 1)

        y = 125
        for i, pdef in enumerate(PARAM_DEFS):
            key, label, ptype = pdef[0], pdef[1], pdef[2]
            val = params.get(key, DEFAULT_HYPERPARAMS.get(key, 0.20))
            is_sel = (i == selected)

            if i == 0:
                cat_txt = font.render("[ 1. OFFLINE IMITATION LEARNING ]", True, COLOR_PLAYER_A)
                screen.blit(cat_txt, (50, y))
                y += 32
            elif i == 3:
                y += 10
                cat_txt = font.render("[ 2. INTERACTIVE RL FINE-TUNING ]", True, COLOR_PLAYER_B)
                screen.blit(cat_txt, (50, y))
                y += 32

            prefix = "-> " if is_sel else "   "
            lbl_color = COLOR_ACCENT if is_sel else COLOR_TEXT
            lbl_txt = font.render(prefix + label, True, lbl_color)
            screen.blit(lbl_txt, (60, y))

            if ptype == "choice":
                val_str = f"<  {val}  >"
            elif ptype == int:
                val_str = f"<  {pdef[6].format(val)}  >"
            else:
                val_str = f"<  {pdef[6].format(val)}  >"

            val_color = COLOR_BULLET if is_sel else COLOR_TEXT
            val_txt = font.render(val_str, True, val_color)
            screen.blit(val_txt, (SCREEN_WIDTH - 240, y))

            y += 40

        pygame.draw.line(screen, COLOR_GRID, (50, y + 10), (SCREEN_WIDTH - 50, y + 10), 1)
        hint = font.render("[R] Reset Defaults  |  Exploration: 0% = Pure Model, 25-40% = Force Breakout", True, COLOR_MUTED)
        screen.blit(hint, (50, y + 22))

        if saved_notice:
            msg = font.render(saved_notice, True, COLOR_ACCENT)
            screen.blit(msg, (50, y + 50))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_hyperparams(params)
                pygame.key.set_repeat(0, 0)
                return params
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_s]:
                    save_hyperparams(params)
                    pygame.key.set_repeat(0, 0)
                    return params
                elif event.key == pygame.K_r:
                    params = DEFAULT_HYPERPARAMS.copy()
                    save_hyperparams(params)
                    saved_notice = "Reset all parameters to default values!"
                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(PARAM_DEFS)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(PARAM_DEFS)
                elif event.key == pygame.K_LEFT:
                    pdef = PARAM_DEFS[selected]
                    key, ptype = pdef[0], pdef[2]
                    if ptype == "choice":
                        choices = pdef[3]
                        idx = (choices.index(params[key]) - 1) % len(choices)
                        params[key] = choices[idx]
                    elif ptype == int:
                        params[key] = max(pdef[3], params[key] - pdef[5])
                    elif ptype == float:
                        params[key] = round(max(pdef[3], params[key] - pdef[5]), 5)
                    saved_notice = f"Updated {key} to {params[key]}"
                    save_hyperparams(params)
                elif event.key == pygame.K_RIGHT:
                    pdef = PARAM_DEFS[selected]
                    key, ptype = pdef[0], pdef[2]
                    if ptype == "choice":
                        choices = pdef[3]
                        idx = (choices.index(params[key]) + 1) % len(choices)
                        params[key] = choices[idx]
                    elif ptype == int:
                        params[key] = min(pdef[4], params[key] + pdef[5])
                    elif ptype == float:
                        params[key] = round(min(pdef[4], params[key] + pdef[5]), 5)
                    saved_notice = f"Updated {key} to {params[key]}"
                    save_hyperparams(params)

    pygame.key.set_repeat(0, 0)
    return params
