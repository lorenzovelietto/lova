"""Shared pipeline contract between mod1–mod4.

split_on_call must be identical across disassemble / lift / mutate / fixup.
If Mod2 lifts with False while Mod3 mutates assuming True, live_after keys
land on different block boundaries and the liveness gate silently misfires.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Единый контракт пайплайна.

    split_on_call=True  — блок закрывается на call (гранулярность для Mod3).
    split_on_call=False — классические basic blocks (IR).
    """
    split_on_call: bool = True

    @classmethod
    def from_legacy(cls, split_on_call: bool | None = None,
                    cfg: "PipelineConfig | None" = None) -> "PipelineConfig":
        if cfg is None:
            return cls(split_on_call=True if split_on_call is None else bool(split_on_call))
        if split_on_call is not None and cfg.split_on_call != split_on_call:
            raise ValueError(
                f"PipelineConfig.split_on_call={cfg.split_on_call} "
                f"не совпадает с split_on_call={split_on_call}"
            )
        return cfg
