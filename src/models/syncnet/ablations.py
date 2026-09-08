from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import Competition, SyncNet


class AttentionMedium(nn.Module):

    def __init__(self, module_dim: int, msg_dim: int) -> None:
        super().__init__()

        self.msg_proj = nn.Linear(module_dim, msg_dim)
        self.q = nn.Linear(module_dim, module_dim)
        self.k = nn.Linear(module_dim, module_dim)
        self.out_dim = msg_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:

        N = h.shape[1]
        att = torch.einsum('bnd,bmd->bnm', self.q(h), self.k(h)) / h.shape[-1] ** 0.5
        att = att.masked_fill(torch.eye(N, dtype=torch.bool, device=h.device), float('-inf')).softmax(-1)

        return torch.einsum('bnm,bmD->bnD', att, self.msg_proj(h))


class SoftmaxPhaseMedium(nn.Module):

    def __init__(self, module_dim: int, msg_dim: int, phase_dim: int) -> None:
        super().__init__()

        self.msg_proj = nn.Linear(module_dim, msg_dim)
        self.log_beta = nn.Parameter(torch.tensor(1.0))
        self.up = nn.Linear(msg_dim, msg_dim * phase_dim)
        self.out_dim = msg_dim * phase_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:

        N = h.shape[1]
        att = torch.einsum('bnd,bmd->bnm', z, z) * self.log_beta.exp()
        att = att.masked_fill(torch.eye(N, dtype=torch.bool, device=h.device), float('-inf')).softmax(-1)

        return self.up(torch.einsum('bnm,bmD->bnD', att, self.msg_proj(h)))


class ContentPhase(nn.Module):

    def __init__(self, module_dim: int, phase_dim: int) -> None:
        super().__init__()

        self.to_phase = nn.Linear(module_dim, phase_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:

        return F.normalize(self.to_phase(h), dim=-1)


class SharedCell(nn.Module):

    def __init__(self, in_dim: int, module_dim: int, n_modules: int) -> None:
        super().__init__()

        self.cell = nn.GRUCell(in_dim, module_dim)
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:

        B, N, dm = h.shape

        return self.cell(inp.reshape(B * N, -1), h.reshape(B * N, dm)).reshape(B, N, dm)


class UnseededCompetition(Competition):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.cell_bias

    def bias(self) -> float:
        return 0.0


class LinesAttention(SyncNet):
    def build_medium(self, cfg):   

        return AttentionMedium(cfg.module_dim, cfg.msg_dim)


class SoftmaxRead(SyncNet):
    def build_medium(self, cfg):   
        return SoftmaxPhaseMedium(
            cfg.module_dim, cfg.msg_dim, cfg.phase_dim
            )


class ContentAddresses(SyncNet):
    def build_dynamics(self, cfg): 
        return ContentPhase(
            cfg.module_dim, cfg.phase_dim
            )


class SharedCells(SyncNet):
    def build_cells(self, cfg):    
        return SharedCell(
            cfg.tok_dim + self.medium.out_dim, cfg.module_dim, cfg.n_modules # type: ignore 
            )


class NoBias(SyncNet):
    def build_binder(self, cfg):   

        return UnseededCompetition(
            cfg.n_modules, self.n_cells, cfg.field_ch, self.field.K,
            self.field.d, cfg.tok_dim, 'zero', 0.0,
            cfg.competition_rounds, cfg.beta
        )


ABLATIONS: dict[str | None, type[SyncNet]] = {
    None: SyncNet,
    'no_bias': NoBias,
    'shared_cells': SharedCells,
    'lines_attention': LinesAttention,
    'softmax_read': SoftmaxRead,
    'content_addresses': ContentAddresses,
}
