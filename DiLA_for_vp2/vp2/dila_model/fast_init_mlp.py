import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from einops import rearrange


class ActionMLP(nn.Module):
    def __init__(self, num_actions: int = 4, action_dim: int = 32):
        super(ActionMLP, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(num_actions, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 2
        z = self.mapping(x)
        return z


def save_checkpoint(model, optimizer, epoch, loss, save_dir=".", filename="mlp_init_weights.pth"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def train_action_mlp(model, data_loader, save_dir, num_epochs=3000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize total loss for the epoch
        num_batches = 0  # Initialize batch counter

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        average_loss = epoch_loss / num_batches

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
            save_checkpoint(model, optimizer, epoch + 1, average_loss, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a small action-to-latent MLP for VP2 adaptation.")
    parser.add_argument("--inputs", type=Path, default=Path("./data/vp2/raw_action_inputs.pt"))
    parser.add_argument("--targets", type=Path, default=Path("./data/vp2/latent_action_stats.pt"))
    parser.add_argument("--save-dir", type=Path, default=Path("./checkpoints/vp2/action_decoder"))
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.load(args.inputs).float()
    targets = torch.load(args.targets).float()

    inputs = rearrange(inputs, 'n t ... -> (n t) ...')
    inputs = inputs.flatten(start_dim=1)
    targets = rearrange(targets, 'n t ... -> (n t) ...')
    targets = targets.flatten(start_dim=1)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ActionMLP(num_actions=args.num_actions, action_dim=args.action_dim).to(device)

    train_action_mlp(
        model,
        data_loader,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )
