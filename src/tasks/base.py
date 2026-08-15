"""One shared training loop for all task heads (backbone stays frozen)."""
from typing import Callable, Iterable

import torch
import torch.nn as nn
from tqdm import tqdm

StepFn = Callable[[nn.Module, tuple, torch.device], torch.Tensor]


def train_head(head: nn.Module, loader: Iterable, step_fn: StepFn,
               epochs: int, lr: float, device: torch.device, tag: str) -> nn.Module:
    """Generic loop: `step_fn(head, batch, device) -> scalar loss`."""
    head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        total, n = 0.0, 0
        for batch in tqdm(loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}", leave=False):
            loss = step_fn(head, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        print(f"[{tag}] epoch {epoch + 1}/{epochs}  loss {total / max(n, 1):.4f}")
    head.eval()
    return head
