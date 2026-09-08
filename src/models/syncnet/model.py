from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common.img_enc import build_image_encoder
from ...core.config import ModelConfig
from ...tasks import DATASETS


@dataclass
class SyncNetConfig(ModelConfig):
    name: str = 'syncnet'
    ablation: str | None = None # None | <name of ablation>
    encoder: dict[str, Any] = field(default_factory=lambda: {'name': 'cnn', 'ch': 64})

    # the modules and their medium
    n_modules: int = 6
    phase_dim: int = 6
    t_bus: int = 8
    dt: float = 0.5
    module_dim: int = 96
    msg_dim: int = 4
    kappa_dim: int = 64
    readout_hidden: int = 128

    # the field and the competition
    field_ch: int = 64 # channels the perceptral oscillators
    field_groups: int = 16 # oscillators per spatial
    field_osc_dim: int = 4 # dim of each cell oscillator
    field_steps: int = 8 # num steps for field
    field_dt: float = 1.0
    tok_dim: int = 64
    beta: float = 8.0 # competition inverse temperature (initial, learned)
    bias_init: str = 'zero' # zero | partition | random
    bias_scale: float = 4.0
    competition_rounds: int = 3


class OscillatorField(nn.Module):

    def __init__(self, field_ch: int, groups: int, osc_dim: int, steps: int, dt: float) -> None:
        super().__init__()

        self.d = osc_dim
        self.K = groups
        self.C = osc_dim * groups
        self.T = steps
        self.dt = dt

        self.z_head = nn.Conv2d(field_ch, self.C, 1)
        self.stim = nn.Conv2d(field_ch, self.C, 1)

        self.J = nn.Conv2d(self.C, self.C, 5, padding=2, bias=False)
        nn.init.normal_(self.J.weight, std=0.02)

        self.omega_raw = nn.Parameter(torch.randn(groups, osc_dim, osc_dim) * 0.1)
        self.omega_scale = 0.1

    def normalise(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape

        return F.normalize(z.view(B, self.K, self.d, S, S), dim=2).view(B, C, S, S)

    def tangent(self, z: Tensor, v: Tensor) -> Tensor:
        B, C, S, _ = z.shape

        zg = z.view(B, self.K, self.d, S, S)
        vg = v.view(B, self.K, self.d, S, S)

        return (vg - (vg * zg).sum(2, keepdim=True) * zg).reshape(B, C, S, S)

    def rotate(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape

        A = (self.omega_raw - self.omega_raw.transpose(-1, -2)) * self.omega_scale

        return torch.einsum('kde,bkeij->bkdij', A, z.view(B, self.K, self.d, S, S)).reshape(B, C, S, S)

    def to_tokens(self, z: Tensor) -> Tensor:
        B, C, S, _ = z.shape

        return z.view(B, self.K, self.d, S * S).permute(0, 3, 1, 2) # (B, P, K, d)

    def forward(self, f: Tensor) -> Tensor:
        z = self.normalise(self.z_head(f))
        c = self.stim(f)

        for _ in range(self.T):
            z = self.normalise(z + self.dt * (self.rotate(z) + self.tangent(z, self.J(z) + c)))

        return z


class Competition(nn.Module):

    def __init__(
        self,
        n_slots: int,
        n_cells: int,
        field_ch: int,
        groups: int,
        osc_dim: int,
        tok_dim: int,
        bias_init: str,
        bias_scale: float,
        rounds: int,
        beta: float,
        ) -> None:
        super().__init__()

        self.n_slots = n_slots
        self.iters = rounds
        self.groups = groups
        self.osc_dim = osc_dim

        self.cell_bias = nn.Parameter(self.seed_bias(n_slots, n_cells, bias_init, bias_scale))
        self.anchor_mu = nn.Parameter(torch.randn(1, n_slots, groups, osc_dim))
        self.anchor_log_sigma = nn.Parameter(torch.zeros(1, n_slots, groups, osc_dim))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta)))

        self.to_slot = nn.Sequential(nn.LayerNorm(field_ch), nn.Linear(field_ch, tok_dim))

    @staticmethod
    def seed_bias(n_slots: int, n_cells: int, bias_init: str, bias_scale: float) -> Tensor:
        bias = torch.zeros(n_slots, n_cells)
        S = int(round(n_cells ** 0.5))

        if bias_init == 'partition':
            rows = max(1, int(n_slots ** 0.5))
            cols = -(-n_slots // rows)

            ys = torch.arange(S).repeat_interleave(S)
            xs = torch.arange(S).repeat(S)
            region = (ys * rows // S) * cols + (xs * cols // S)

            for k in range(min(n_slots, rows * cols)):
                bias[k, region == k] = bias_scale

        elif bias_init == 'random':
            region = torch.randint(0, n_slots, (n_cells,), generator=torch.Generator().manual_seed(0))

            for k in range(n_slots):
                bias[k, region == k] = bias_scale

        elif bias_init != 'zero':
            raise ValueError(f'unknown bias_init {bias_init!r}')

        return bias

    def anchors(self, B: int, device, dtype) -> Tensor:
        eps = torch.randn(B, self.n_slots, self.groups, self.osc_dim, device=device, dtype=dtype)

        return F.normalize(self.anchor_mu + self.anchor_log_sigma.exp() * eps, dim=-1)

    def bias(self) -> Tensor:
        return self.cell_bias

    def forward(self, feats: Tensor, Zt: Tensor) -> tuple[Tensor, Tensor]:
        B, K_f = feats.shape[0], Zt.shape[2]
        phi = self.anchors(B, feats.device, feats.dtype)

        for _ in range(self.iters):
            logits = self.log_beta.exp() * torch.einsum('bkgd,bngd->bkn', phi, Zt) / K_f + self.bias()

            reads = F.softmax(logits, dim=1) # cells choose slots
            reads = reads / (reads.sum(-1, keepdim=True) + 1e-8)

            phi = F.normalize(torch.einsum('bkn,bngd->bkgd', reads, Zt), dim=-1)
            slots = self.to_slot(torch.einsum('bkn,bnf->bkf', reads, feats))

        return slots, phi


class Bus(nn.Module):

    def __init__(self, module_dim: int, message_dim: int, phase_dim: int) -> None:
        super().__init__()

        self.msg_proj = nn.Linear(module_dim, message_dim)
        self.up = nn.Linear(message_dim, message_dim * phase_dim)
        self.out_dim = message_dim * phase_dim

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        m = self.msg_proj(h)

        bus = torch.einsum('bnD,bnd->bDd', m, z)
        r = torch.einsum('bDd,bnd->bnD', bus, z) - m

        return self.up(r / h.shape[1])


class PhaseStep(nn.Module):

    def __init__(self, n_rows: int, module_dim: int, phase_dim: int, kappa_dim: int, dt: float) -> None:
        super().__init__()

        self.dt = dt

        self.omega = nn.Parameter(torch.zeros(n_rows))
        self.K = nn.Parameter(torch.ones(n_rows, n_rows))
        self.gen_raw = nn.Parameter(0.1 * torch.randn(phase_dim, phase_dim))

        self.k_mlp = nn.Sequential(
            nn.Linear(2 * module_dim, kappa_dim),
            nn.GELU(),
            nn.Linear(kappa_dim, 1),
            nn.Tanh(),
        )
        self.stim = nn.Linear(module_dim, phase_dim)

    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        B, N, dm = h.shape

        A = self.gen_raw - self.gen_raw.t()
        A = A / (A.norm() / 2 ** 0.5 + 1e-6)
        vel = self.omega.to(z.dtype)[None, :, None] * torch.einsum('de,bne->bnd', A, z)

        pairs = torch.cat([h.unsqueeze(2).expand(B, N, N, dm), h.unsqueeze(1).expand(B, N, N, dm)], -1)
        kap = self.k_mlp(pairs).squeeze(-1)
        pull = torch.einsum('bij,bjd->bid', self.K.to(z.dtype).unsqueeze(0) * kap, z)

        tangent = lambda v: v - (v * z).sum(-1, keepdim=True) * z

        return F.normalize(z + self.dt * (vel + tangent(pull) + tangent(self.stim(h))), dim=-1)


class PrivateCells(nn.Module):

    def __init__(self, n_rows: int, in_dim: int, module_dim: int, n_modules: int) -> None:
        super().__init__()

        self.cells = nn.ModuleList(nn.GRUCell(in_dim, module_dim) for _ in range(n_rows))
        self.embeds = nn.Parameter(torch.randn(n_modules, module_dim) / module_dim ** 0.5)

    def step(self, inp: Tensor, h: Tensor) -> Tensor:
        return torch.stack([cell(inp[:, k], h[:, k]) for k, cell in enumerate(self.cells)], 1)


class SyncNet(nn.Module):
    supported_callbacks = frozenset({'sync'})

    def __init__(self, cfg: SyncNetConfig, dataset: str, answer_dim: int) -> None:
        super().__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.answer_dim = answer_dim

        spec = DATASETS[dataset]
        self.q_vocab = getattr(spec, 'VOCAB_SIZE', None)
        self.q_size = spec.QUESTION_SIZE * (self.q_vocab or 1)

        self.M = cfg.n_modules
        self.N = cfg.n_modules + 1
        self.d = cfg.phase_dim
        self.T = cfg.t_bus

        self.stem = build_image_encoder(dict(cfg.encoder), dataset)
        self.adapt = nn.Conv2d(self.stem.ch, cfg.field_ch, 1)

        S = self.stem.spatial
        self.n_cells = S * S
        self.pos_emb = nn.Parameter(0.02 * torch.randn(1, cfg.field_ch, S, S))

        self.field = OscillatorField(
            cfg.field_ch, cfg.field_groups, cfg.field_osc_dim, cfg.field_steps, cfg.field_dt
        )
        self.binder = self.build_binder(cfg)
        self.anchor_to_phase = nn.Linear(self.field.K * self.field.d, cfg.phase_dim)

        self.medium = self.build_medium(cfg)
        self.identity = self.build_cells(cfg)
        self.dynamics = self.build_dynamics(cfg)

        self.grid_gamma = nn.Linear(self.q_size, cfg.field_ch)
        self.grid_beta = nn.Linear(self.q_size, cfg.field_ch)
        self.grid_norm = nn.GroupNorm(8, cfg.field_ch, affine=True)

        self.slot_gamma = nn.Linear(self.q_size, cfg.tok_dim)
        self.slot_beta = nn.Linear(self.q_size, cfg.tok_dim)
        self.slot_norm = nn.LayerNorm(cfg.tok_dim)

        self.h_init = nn.Sequential(
            nn.Linear(self.q_size, 64), nn.GELU(), nn.Linear(64, self.M * cfg.module_dim)
        )
        self.head_init = nn.Sequential(
            nn.Linear(self.q_size, 64), nn.GELU(), nn.Linear(64, cfg.module_dim)
        )
        self.head_embed = nn.Parameter(torch.randn(1, cfg.module_dim) / cfg.module_dim ** 0.5)

        self.readout = nn.Sequential(
            nn.Linear(cfg.module_dim + self.q_size, cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(cfg.readout_hidden, answer_dim),
        )

    def build_binder(self, cfg: SyncNetConfig) -> nn.Module:
        return Competition(
            cfg.n_modules,
            self.n_cells,
            cfg.field_ch,
            self.field.K,
            self.field.d,
            cfg.tok_dim,
            cfg.bias_init,
            cfg.bias_scale,
            cfg.competition_rounds,
            cfg.beta,
        )

    def build_medium(self, cfg: SyncNetConfig) -> nn.Module:
        return Bus(cfg.module_dim, cfg.msg_dim, cfg.phase_dim)

    def build_cells(self, cfg: SyncNetConfig) -> nn.Module:
        in_dim = cfg.tok_dim + self.medium.out_dim # type: ignore

        return PrivateCells(self.N, in_dim, cfg.module_dim, cfg.n_modules) # type: ignore

    def build_dynamics(self, cfg: SyncNetConfig) -> nn.Module:
        return PhaseStep(self.N, cfg.module_dim, cfg.phase_dim, cfg.kappa_dim, cfg.dt)

    def initial_phases(self, z_slots: Tensor, B: int) -> Tensor:
        z = torch.randn(B, self.N, self.d, device=z_slots.device, dtype=z_slots.dtype)
        z = F.normalize(z, dim=-1)

        return torch.cat([z_slots, z[:, self.M:]], 1)

    def encode_question(self, questions: Tensor) -> Tensor:
        if self.q_vocab:
            return F.one_hot(questions.long(), self.q_vocab).float().flatten(-2)

        return questions.float()

    def forward(self, batch: dict) -> dict:
        q = self.encode_question(batch['questions'])
        B = q.shape[0]

        f = self.adapt(self.stem(batch['images']))
        f = f * (1 + self.grid_gamma(q))[..., None, None] + self.grid_beta(q)[..., None, None]
        f = self.grid_norm(f) + self.pos_emb

        Zt = self.field.to_tokens(self.field(f))
        X, anchors = self.binder(f.flatten(2).transpose(1, 2), Zt)

        z = self.initial_phases(F.normalize(self.anchor_to_phase(anchors.flatten(2)), dim=-1), B)
        X = self.slot_norm(X * (1 + self.slot_gamma(q)).unsqueeze(1) + self.slot_beta(q).unsqueeze(1))

        h_slots = self.h_init(q).reshape(B, self.M, self.cfg.module_dim) + self.identity.embeds
        h = torch.cat([h_slots, self.head_init(q).unsqueeze(1) + self.head_embed], 1)
        X = torch.cat([X, X.new_zeros(B, 1, X.shape[-1])], 1)

        for _ in range(self.T):
            h = self.identity.step(torch.cat([X, self.medium(h, z)], -1), h) # type: ignore
            z = self.dynamics(z, h)

        return {'logits': self.readout(torch.cat([h[:, self.M], q], -1))}