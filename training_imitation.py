import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import ActorCritic
from config import load_hyperparams

class TrainingCancelled(Exception):
    pass

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

def train_imitation(progress_callback=None, cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled()

    params = load_hyperparams()
    records = glob.glob("recordings/*.npz")
    if not records:
        print("[Imitation] No behavior logs found in recordings/! Play Mode 2 or 3 first.", flush=True)
        return None

    raw_states, raw_moves = [], []
    for r in records:
        data = np.load(r)
        raw_states.append(data['states'])
        raw_moves.append(data['move_actions'])

    base_states = np.concatenate(raw_states)
    base_moves = np.concatenate(raw_moves)

    # Bilateral Mirror Augmentation
    aug_states, aug_moves = augment_mirror_data(base_states, base_moves)

    states = torch.tensor(aug_states, dtype=torch.float32)
    moves = torch.tensor(aug_moves, dtype=torch.int64)

    total_samples = len(states)
    print("=" * 60, flush=True)
    print(f"[Imitation] Base frames: {len(base_states)} -> Augmented: {total_samples} frames (50/50 Symmetry)", flush=True)
    print("=" * 60, flush=True)

    dataset = TensorDataset(states, moves)
    loader = DataLoader(dataset, batch_size=params["imitation_batch_size"], shuffle=True)

    model = ActorCritic(state_dim=44)
    actor_params = list(model.actor_backbone.parameters()) + list(model.move_head.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=params["imitation_lr"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    epochs = int(params["imitation_epochs"])
    total_batches = len(loader)
    total_steps = epochs * total_batches

    epoch_history = []

    for ep in range(epochs):
        epoch_loss = 0.0
        correct_move = 0

        for batch_index, (b_states, b_moves) in enumerate(loader):
            if cancel_event is not None and cancel_event.is_set():
                raise TrainingCancelled()

            optimizer.zero_grad()
            m_logits, _ = model(b_states)
            loss = criterion(m_logits, b_moves)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_move += (torch.argmax(m_logits, dim=-1) == b_moves).sum().item()

            # Push live UI update every 4 batches
            if progress_callback and (batch_index % 4 == 0 or (batch_index + 1) == total_batches):
                step_num = ep * total_batches + (batch_index + 1)
                fraction = step_num / float(total_steps)
                curr_loss = epoch_loss / (batch_index + 1)
                curr_acc = (correct_move / float((batch_index + 1) * params["imitation_batch_size"])) * 100.0
                msg = f"Epoch {ep+1}/{epochs} | Batch {batch_index+1}/{total_batches} | Loss: {curr_loss:.4f} | Acc: {curr_acc:.1f}%"
                progress_callback(fraction, msg)

        avg_loss = epoch_loss / total_batches
        move_acc = (correct_move / total_samples) * 100.0
        epoch_history.append((avg_loss, move_acc))
        if ep % 10 ==0:
       
         print(f"  Epoch [{ep+1:3d}/{epochs:3d}] - Loss: {avg_loss:.4f} | Accuracy: {move_acc:.1f}%", flush=True)

    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled()

    os.makedirs("saved_models", exist_ok=True)
    out_path = f"saved_models/imitation_model_{int(time.time())}.pt"
    torch.save(model.state_dict(), out_path)
    torch.save(model.state_dict(), "saved_models/latest_model.pt")
    torch.save(model.state_dict(), "saved_models/finetuned_model.pt")

    initial_acc = epoch_history[0][1]
    final_acc = epoch_history[-1][1]
    gain = final_acc - initial_acc

    print("=" * 60, flush=True)
    print(f"[Training Complete] Initial Acc: {initial_acc:.1f}% -> Final Acc: {final_acc:.1f}% (+{gain:.1f}% gain)", flush=True)
    print(f"[Saved Model] -> {out_path}", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Show the learning progress in the final UI message
    if progress_callback:
        progress_callback(1.0, f"Learned: {initial_acc:.1f}% -> {final_acc:.1f}% (+{gain:.1f}% Gain) | Saved: {os.path.basename(out_path)}")

    return out_path

if __name__ == "__main__":
    train_imitation()
