import os
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_history(train_history, num_epochs, ckpt_dir, seed):
    if torch.is_tensor(train_history[0]):
        train_history = [loss.cpu().detach().numpy() for loss in train_history]
    # save training curves
    plt.figure()
    plt.plot(
        np.linspace(0, num_epochs, len(train_history)), train_history, label="train"
    )
    plt.tight_layout()
    plt.legend()
    plt.title("loss")
    plt.savefig(os.path.join(ckpt_dir, f"train_seed_{seed}.png"))

    plt.close()


def sync_loss(loss, device):
    t = [loss]
    t = torch.tensor(t, dtype=torch.float64, device=device)
    dist.barrier()
    dist.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
    return t[0]


def print_args(args_dict):
    console = Console()
    table = Table(title="args", title_style="bold magenta")

    table.add_column("Argument", style="cyan", justify="left")
    table.add_column("Value", style="green", justify="left")

    for k, v in args_dict.items():
        table.add_row(str(k), str(v))

    console.print(table)
