import torch
import os


def save_checkpoint(log_dir, dataset_name, method_name, optimizer_state, model=None):
    """
    Saves the LSTM optimizer state and optionally the Koopman model.
    """
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # LSTM
    lstm_path = os.path.join(checkpoint_dir, f"{dataset_name}_{method_name}_lstm.pth")
    torch.save(optimizer_state, lstm_path)

    # KAE
    if model is not None and method_name == "kae":
        model_path = os.path.join(checkpoint_dir, f"{dataset_name}_{method_name}_model.pth")
        torch.save(model.state_dict(), model_path)