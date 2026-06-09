from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class Adan(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["exp_avg_diff"] = torch.zeros_like(parameter)
                    state["prev_grad"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_diff = state["exp_avg_diff"]
                prev_grad = state["prev_grad"]
                state["step"] += 1

                grad_diff = grad - prev_grad
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1.0 - beta3)
                denom = exp_avg_sq.sqrt().add_(eps)
                update = (exp_avg + beta2 * exp_avg_diff) / denom

                if weight_decay != 0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
                prev_grad.copy_(grad)

        return loss


class SophiaG(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if rho <= 0:
            raise ValueError("rho must be positive.")
        defaults = {
            "lr": lr,
            "betas": betas,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["hessian"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = hessian.sqrt().add_(rho + eps)
                update = exp_avg / denom

                if weight_decay != 0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)

        return loss


class Lion(Optimizer):
    """
    Implements Lion algorithm.
    Optimized version of Adam that uses the sign of the gradient.
    Ref: https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    **kwargs: Any,
):
    parameter_list = list(params)
    normalized = name.strip().lower()
    if normalized == "adamw":
        return torch.optim.AdamW(parameter_list, lr=lr, weight_decay=weight_decay, **kwargs)
    if normalized == "adan":
        return Adan(parameter_list, lr=lr, weight_decay=weight_decay, **kwargs)
    if normalized == "sophia":
        return SophiaG(parameter_list, lr=lr, weight_decay=weight_decay, **kwargs)
    if normalized == "lion":
        return Lion(parameter_list, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError("optimizer must be one of: adamw, adan, sophia, lion.")


__all__ = [
    "Adan",
    "Lion",
    "SophiaG",
    "build_optimizer",
]
