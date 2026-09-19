import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import ActorCritic
from config import load_hyperparams

def augment_mirror_data(states_np, moves_np):
    """
    Applies exact bilateral reflection symmetry:
    1. Mirrors Self X (X -> 1.0 - X)
    2. Inverts relative X distances for rival and bullets
    3. Swaps Action 1 (Left) <-> Action 2 (Right)
    """
    mirrored_states = states_np.copy()
    mirrored_moves = moves_np.copy()

    # 1. Flip Self X (Index 0: sx / MAP_WIDTH)
    mirrored_states[:, 0] = 1.0 - mirrored_states[:, 0]

    # 2. Invert Rival relative X (Index 3: rel_x / MAP_WIDTH)
    mirrored_states[:, 3] = -mirrored_states[:, 3]

    # 3. Invert Top-5 Enemy Threats (Indices 5 to 34, step 6)
    # Features per threat: [rel_x, rel_y, vx, vy, time_to_hit, exists]
    for b_idx in range(5):
        col_rel_x = 5 + b_idx * 6
        col_vx = 5 + b_idx * 6 + 2
        mirrored_states[:, col_rel_x] = -mirrored_states[:, col_rel_x]
        mirrored_states[:, col_vx] = -mirrored_states[:, col_vx]

    # 4. Invert Top-3 Outgoing Bullets (Indices 35 to 43, step 3)
    # Features per bullet: [rel_x_to_rival, rel_y_to_rival, exists]
    for b_idx in range(3):
        col_rel_x = 35 + b_idx * 3
        mirrored_states[:, col_rel_x] = -mirrored_states[:, col_rel_x]

    # 5. Swap Actions: 1 (Left) <-> 2 (Right)
    mask_left = (moves_np == 1)
    mask_right = (moves_np == 2)
    mirrored_moves[mask_left] = 2
    mirrored_moves[mask_right] = 1

    # Concatenate Original + Mirrored
    aug_states = np.vstack([states_np, mirrored_states])
    aug_moves = np.concatenate([moves_np, mirrored_moves])
    
    return aug_states, aug_moves

def train_imitation():
    params = load_hyperparams()
    records = glob.glob("recordings/*.npz")
    if not records:
        print("[Imitation] No behavior logs found in recordings/! Play Mode 2 or 3 first.")
        return None

    raw_states, raw_moves = [], []
    for r in records:
        data = np.load(r)
        raw_states.append(data['states'])
        raw_moves.append(data['move_actions'])

    base_states = np.concatenate(raw_states)
    base_moves = np.concatenate(raw_moves)

    # Apply Bilateral Mirror Augmentation
    aug_states, aug_moves = augment_mirror_data(base_states, base_moves)

    states = torch.tensor(aug_states, dtype=torch.float32)
    moves = torch.tensor(aug_moves, dtype=torch.int64)

    total_samples = len(states)
    print(f"\n[Imitation] Base frames: {len(base_states)} -> Augmented dataset: {total_samples} frames (Exact 50/50 Symmetry)")

    dataset = TensorDataset(states, moves)
    loader = DataLoader(dataset, batch_size=params["imitation_batch_size"], shuffle=True)

    model = ActorCritic(state_dim=44)
    actor_params = list(model.actor_backbone.parameters()) + list(model.move_head.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=params["imitation_lr"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = params["imitation_epochs"]
    for ep in range(epochs):
        total_loss = 0.0
        correct_move = 0

        for b_states, b_moves in loader:
            optimizer.zero_grad()
            m_logits, _ = model(b_states)
            loss = criterion(m_logits, b_moves)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_move += (torch.argmax(m_logits, dim=-1) == b_moves).sum().item()

        avg_loss = total_loss / len(loader)
        move_acc = (correct_move / total_samples) * 100.0

        if (ep + 1) % max(1, epochs // 6) == 0 or (ep + 1) == epochs:
            print(f"  Epoch [{ep+1:2d}/{epochs}] - Loss: {avg_loss:.4f} | Symmetrical Accuracy: {move_acc:.1f}%")

    os.makedirs("saved_models", exist_ok=True)
    out_path = f"saved_models/imitation_model_{int(time.time())}.pt"
    torch.save(model.state_dict(), out_path)
    torch.save(model.state_dict(), "saved_models/latest_model.pt")
    # Also save over finetuned_model.pt so all modes use the balanced baseline
    torch.save(model.state_dict(), "saved_models/finetuned_model.pt")
    print(f"[Imitation] Successfully saved balanced model to:\n  -> {out_path}\n")
    return out_path

if __name__ == "__main__":
    train_imitation()
