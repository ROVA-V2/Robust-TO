"""
Robust-TO — Unified Tool Interface  (Eq. 4 in the paper)

    (r_j, c_j) = T_j(F_j, sq)
    c_j        = c_intrinsic_j  *  rho(F_j)
    rho(F_j)   = mean of the K = ceil(|F_j|/3) smallest values of (1 - d(f))

Every tool — selection or perception — implements this contract.  A tool's
confidence is the product of its own self-assessed certainty (c_intrinsic) and
the input reliability rho(F_j) of the frames it received, so confidence is high
only when BOTH the tool is sure AND its input frames are clean.

rho is a *conservative* (worst-K) mean: averaging only the K = ceil(n/3) least
reliable frames ensures a single clean frame cannot mask catastrophic corruption
in the others (Section 3.1; ablated against a uniform mean, which loses 3.3
accuracy points on clean UrbanVideo-Bench).  Intrinsic confidences are clipped
to [0.01, 1.0]; a failed/empty result yields c_j = 0 (pure cost at training).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional
import math
import numpy as np


# ── Normalised per-tool costs (Table 16) ──────────────────────────────────────
TOOL_COSTS: dict[str, float] = {
    # Selection tools
    "assess_quality":   0.10,
    "select_frames":    0.15,
    "retrieve_frames":  0.20,
    # Perception tools (Table 16 — full set; stubs for tools not yet implemented)
    "detect_objects":   0.50,
    "caption_frame":    0.30,
    "track_temporal":   0.70,
    "recognize_action": 0.60,
    "read_text":        0.25,
}


@dataclass
class ToolResult:
    """Structured output returned by every tool call."""
    result: Any                          # tool-specific payload
    confidence: float                    # cj ∈ [0, 1]  (Eq. 4)
    tool_name: str
    source_frames: List[int] = field(default_factory=list)  # original frame indices


def worst_k_mean(quality: np.ndarray) -> float:
    """
    ρ(F) = mean of the K=⌈|F|/3⌉ smallest values of (1 − d(f)).

    Paper (Section 3.1, Eq. 4):
        "for |F|=n, returns the mean of the K=⌈n/3⌉ smallest values of
         (1−d(f)), ensuring that a single clean frame cannot mask
         catastrophic corruption in others."
    """
    n = len(quality)
    if n == 0:
        return 0.0
    k = max(1, math.ceil(n / 3))
    # k smallest values of (1-d) = k most-corrupted frames
    return float(np.mean(np.sort(quality)[:k]))


class ToolBase(ABC):
    """
    Abstract base for all RoVid tools.

    Subclasses implement _run() which returns (raw_result, c_intrinsic).
    This base class applies the confidence modulation from Eq. 4 automatically,
    using the worst-K aggregation for ρ(F) as specified in the paper.
    """

    name: str = ""

    # ── Public entry-point ────────────────────────────────────────────────────
    def __call__(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: Optional[np.ndarray] = None,
        frame_indices: Optional[List[int]] = None,
    ) -> ToolResult:
        """
        Parameters
        ----------
        frames             : (N, H, W, 3) uint8
        sub_query          : natural-language (sub-)question
        disturbance_scores : per-frame d(fi) ∈ [0,1] from Scout's assess_quality;
                             if None, all frames are treated as clean (d=0)
        frame_indices      : original frame indices for source tracking
        """
        N = len(frames)
        if disturbance_scores is None or len(disturbance_scores) == 0:
            disturbance_scores = np.zeros(N, dtype=np.float32)

        # Eq. 4 — ρ(F) = worst-K mean of (1 − d(f)),  K = ⌈|F|/3⌉
        quality = np.clip(1.0 - np.asarray(disturbance_scores, dtype=np.float32), 0.0, 1.0)
        rho = worst_k_mean(quality) if N > 0 else 0.0

        # Run tool-specific logic
        try:
            result, c_intrinsic = self._run(frames, sub_query, disturbance_scores)
        except Exception:
            result, c_intrinsic = None, 0.0

        # Failed / empty result → zero confidence = pure cost penalty at training
        if result is None:
            c_intrinsic = 0.0

        # Clip intrinsic confidence to [0.01, 1.0] as specified in paper Section B
        c_intrinsic = float(np.clip(c_intrinsic, 0.01, 1.0)) if c_intrinsic > 0 else 0.0

        # Eq. 4
        confidence = c_intrinsic * rho
        confidence = float(np.clip(confidence, 0.0, 1.0))

        fi = frame_indices if frame_indices is not None else list(range(N))
        return ToolResult(
            result=result,
            confidence=confidence,
            tool_name=self.name,
            source_frames=fi,
        )

    # ── Subclass contract ─────────────────────────────────────────────────────
    @abstractmethod
    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ):
        """
        Returns
        -------
        result      : tool-specific output (None on failure)
        c_intrinsic : tool's self-assessed output quality ∈ [0, 1]
        """
        ...
