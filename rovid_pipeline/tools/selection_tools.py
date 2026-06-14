"""
Robust-TO — Stage 1 Selection Tools

Implements the three selection tools described in Section 3.2:

    assess_quality  : parameter-free IQA scoring per frame   (cost 0.10)
    select_frames   : reliability-relevance ranking          (cost 0.15)
    retrieve_frames : disturbance-aware retrieval from pool P (cost 0.20)

All tools share the unified (result, confidence) interface from base.py.

Alignment with the paper
-------------------------
assess_quality (Eq. 2):
    d(f_i) = mean(d_blur(f_i), d_bright(f_i), d_occl(f_i))
    Each component is min-max normalised across the video before averaging, so
    blur, brightness and occlusion contribute on a comparable scale (Section C).
    Equal averaging is exactly the paper's mean; we expose both the raw [0,1]
    per-component severities and their min-max-normalised versions.

    Component formulas (Appendix, Eqs. 11-13):
      d_blur   = 1 - min(1, Var(Laplacian) / tau_blur)         (tau_blur = 500)
      d_bright = 2 * |mu_lum - mu_ref|                          (mu_ref   = 0.5)
      d_occl   = 1 - |{p : G(p) > tau_edge}| / (H*W)            (tau_edge  = 30)
    with G the Sobel gradient magnitude.

select_frames (Eq. 3):
    s(f_i) = (1 - d(f_i)) * sim(phi(f_i), psi(q)),   f_i in F,
    where F is the set of valid frames whose reliability (1 - d) and query
    relevance (sim) both exceed their thresholds (theta_rel, theta_sim).  Frames
    outside F receive s(f_i) = 0 and are discarded, so the multiplicative score
    only ever ranks frames that are simultaneously reliable and relevant; this
    prevents a heavily corrupted but query-similar frame from entering the top-K.
    K in [4, 12] is chosen adaptively by the host VLM from query complexity.

retrieve_frames:
    The pool-P retrieval tool used during evidence acquisition (case study,
    Tab. 7): re-ranks the non-selected pool by the same reliability-relevance
    criterion to recover cleaner frames for a sub-query.
"""

from __future__ import annotations
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import ToolBase, ToolResult

SimilarityFn = Callable[[np.ndarray, str], np.ndarray]
AgentFn = Callable[[str], str]

# ── Hyper-parameters from paper (Table 15 / Appendix B) ──────────────────────
TAU_BLUR   = 500.0   # Laplacian normalisation constant  (Sec. C, Eq. 11)
MU_REF     = 0.5     # neutral luminance midpoint         (Sec. C, Eq. 12)
TAU_EDGE   = 30.0    # Sobel edge threshold               (Sec. C, Eq. 13)
THETA_REL  = 0.55    # reliability threshold for select_frames  (Table 15)
THETA_SIM  = 0.30    # relevance  threshold for select_frames   (Table 15)


# ── Disturbance sub-metrics ───────────────────────────────────────────────────

def _blur_score(frame: np.ndarray) -> float:
    """
    d_blur from Eq. 11:  d_blur = 1 − min(1, Var(∇²f) / τ_blur)

    Sharp frames have high Laplacian variance → low d_blur.
    """
    gray = _to_gray(frame)
    lap_var = _laplacian_variance(gray)
    sharpness = min(lap_var / TAU_BLUR, 1.0)
    return float(1.0 - sharpness)


def _brightness_score(frame: np.ndarray) -> float:
    """
    d_bright from Eq. 12:  d_bright = 2 * |mu_lum(f_i) - mu_ref|

    where mu_lum in [0, 1] is the mean pixel intensity (the V channel of HSV
    space; equivalent to normalised luminance for grayscale) and mu_ref = 0.5,
    so both under- and over-exposure are penalised symmetrically.
    """
    gray = _to_gray(frame)
    # Normalise to [0,1].  Clip before applying the formula to absorb the tiny
    # rounding in the luminance coefficients (0.2989+0.5870+0.1140=0.9999).
    mu_lum = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
    d = 2.0 * abs(mu_lum - MU_REF)
    return float(np.clip(d, 0.0, 1.0))


def _occlusion_score(frame: np.ndarray) -> float:
    """
    d_occl from Eq. 13:
        d_occl = 1 - |{p : G(p) > tau_edge}| / (H*W)

    where G(p) = sqrt(Gx(p)^2 + Gy(p)^2) is the Sobel gradient magnitude and
    tau_edge = 30.  Frames lacking informative edge structure (a large flat /
    occluded region) score high.  Sobel (3x3) is used rather than a plain finite
    difference for stronger, more noise-tolerant gradients, as specified.
    """
    gray = _to_gray(frame)
    H, W = gray.shape
    # Sobel kernels (3×3)
    # Gx kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Gy kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -1.0 * padded[:-2, :-2] + 1.0 * padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
        - 1.0 * padded[2:, :-2] + 1.0 * padded[2:, 2:]
    )
    gy = (
        -1.0 * padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - 1.0 * padded[:-2, 2:]
        + 1.0 * padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + 1.0 * padded[2:, 2:]
    )
    G = np.hypot(gx, gy)
    edge_pixel_count = float(np.sum(G > TAU_EDGE))
    edge_fraction = edge_pixel_count / max(H * W, 1)
    return float(np.clip(1.0 - edge_fraction, 0.0, 1.0))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    rgb = frame[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:] -
        4.0 * center
    )
    return float(np.var(lap))


# ── assess_quality ────────────────────────────────────────────────────────────

class AssessQuality(ToolBase):
    """
    Parameter-free IQA scoring per frame.

    Eq. 2:  d(f_i) = mean(d_blur(f_i), d_bright(f_i), d_occl(f_i))
            (each component min-max normalised across the video first, so the
            three contribute on a comparable scale — Section C).  Equal averaging
            of the three normalised components is exactly this mean.

    Returns
    -------
    result : dict with keys
        'disturbance_scores' : np.ndarray (N,) — per-frame d(f_i) in [0,1] (Eq. 2)
        'blur_scores'        : np.ndarray — raw per-frame d_blur   in [0,1]
        'bright_scores'      : np.ndarray — raw per-frame d_bright in [0,1]
        'occl_scores'        : np.ndarray — raw per-frame d_occl   in [0,1]
        'blur_norm'          : np.ndarray — min-max-normalised d_blur
        'bright_norm'        : np.ndarray — min-max-normalised d_bright
        'occl_norm'          : np.ndarray — min-max-normalised d_occl

    The RAW component severities are absolute (each already in [0,1] by
    construction) and are what the router uses to identify the dominant
    corruption type (Section 3.2); the min-max-normalised versions are what
    Eq. 2 averages into d(f_i).
    """

    name = "assess_quality"

    # Equal weights — averaging the three normalised components yields Eq. 2's mean.
    W_BLUR   = 1.0 / 3.0
    W_BRIGHT = 1.0 / 3.0
    W_OCCL   = 1.0 / 3.0

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        blur_scores   = np.array([_blur_score(f)       for f in frames], dtype=np.float32)
        bright_scores = np.array([_brightness_score(f) for f in frames], dtype=np.float32)
        occl_scores   = np.array([_occlusion_score(f)  for f in frames], dtype=np.float32)

        # Per-video min-max normalisation before combining (Section C)
        blur_n   = _minmax_norm(blur_scores)
        bright_n = _minmax_norm(bright_scores)
        occl_n   = _minmax_norm(occl_scores)

        scores = np.clip(
            self.W_BLUR * blur_n + self.W_BRIGHT * bright_n + self.W_OCCL * occl_n,
            0.0, 1.0
        ).astype(np.float32)

        # c_intrinsic = 1.0 (deterministic tool; no model uncertainty — Section B.1)
        c_intrinsic = 1.0

        return {
            "disturbance_scores": scores,
            "blur_scores":        blur_scores,
            "bright_scores":      bright_scores,
            "occl_scores":        occl_scores,
            "blur_norm":          blur_n.astype(np.float32),
            "bright_norm":        bright_n.astype(np.float32),
            "occl_norm":          occl_n.astype(np.float32),
        }, c_intrinsic


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]; returns zeros if range is zero."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── select_frames ─────────────────────────────────────────────────────────────

class SelectFrames(ToolBase):
    """
    Reliability-relevance frame ranking.

    Eq. 3:
        s(f_i) = (1 - d(f_i)) * sim(phi(f_i), psi(q)),   f_i in F

    where F is the set of valid frames whose reliability (1 - d) and query
    relevance (sim) BOTH exceed their thresholds (theta_rel, theta_sim).  Frames
    outside F receive s(f_i) = 0 and are discarded (not forwarded downstream).

    Restricting the multiplicative score to F is what makes the coupling robust:
    a heavily corrupted frame (1 - d small) is suppressed even when it is highly
    query-relevant, so it can never be promoted into the top-K on relevance alone.

    K is determined dynamically in [4, 12] by the host VLM (or a heuristic
    fallback) from query complexity (Section 3.2).
    """

    name = "select_frames"

    def __init__(
        self,
        k_simple: int = 4,
        k_complex: int = 12,
        complexity_threshold: int = 10,
        theta_rel: float = THETA_REL,
        theta_sim: float = THETA_SIM,
        similarity_fn: Optional[SimilarityFn] = None,
        agent_fn: Optional[AgentFn] = None,
    ):
        self.k_simple = k_simple
        self.k_complex = k_complex
        self.complexity_threshold = complexity_threshold
        self.theta_rel = theta_rel
        self.theta_sim = theta_sim
        self.similarity_fn = similarity_fn
        self.agent_fn = agent_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        if N == 0:
            return None, 0.0

        K = self._select_k(sub_query, N)

        similarities = self._compute_similarities(frames, sub_query)
        reliability  = 1.0 - disturbance_scores   # (N,)

        # Eq. 3 — valid set F: reliability AND relevance both above threshold.
        reliable_mask  = reliability  >= self.theta_rel   # 1 - d >= theta_rel
        relevant_mask  = similarities >= self.theta_sim   # sim   >= theta_sim
        eligible_mask  = reliable_mask & relevant_mask

        # s(f_i) = (1-d)*sim for f_i in F; 0 otherwise (discarded).
        scores = np.where(eligible_mask, reliability * similarities, 0.0).astype(np.float32)

        # BUG FIX: only rank and select frames that are *eligible* (score > 0).
        # Paper: "frames outside F receive s(f_i)=0 and are discarded".
        # The previous argsort over ALL frames filled top-K with ineligible (score=0)
        # frames whenever eligible count < K, contaminating the selected set with
        # corrupted frames and deflating c_intrinsic.
        eligible_indices = np.where(eligible_mask)[0]         # indices of frames in F
        eligible_scores  = scores[eligible_indices]           # their scores (all > 0)
        ranked_among_eligible = np.argsort(eligible_scores)[::-1]   # descending
        top_eligible = ranked_among_eligible[:K]               # at most K eligible frames

        selected_indices = sorted(eligible_indices[top_eligible].tolist())
        # Pool = all non-selected frames (both ineligible and eligible-but-not-top-K)
        selected_set = set(selected_indices)
        pool_indices = sorted(i for i in range(N) if i not in selected_set)

        selected_scores = eligible_scores[top_eligible]
        c_intrinsic = float(selected_scores.mean()) if len(selected_scores) > 0 else 0.0
        c_intrinsic = float(np.clip(c_intrinsic, 0.0, 1.0))

        return {
            "selected_indices": selected_indices,
            "pool_indices":     pool_indices,
            "scores":           scores.tolist(),
        }, c_intrinsic

    def _compute_similarities(self, frames: np.ndarray, sub_query: str) -> np.ndarray:
        """
        sim(φ(fi), ψ(q)) with the shared VLM backbone when available.
        Falls back to neutral score (1.0) when no encoder is provided.

        Contract: similarity_fn MUST return cosine similarities in [-1, 1].
        The shift (x+1)/2 maps them to [0, 1] for threshold comparison.
        If you pass a function that already returns [0, 1] (e.g., a random
        stub), the shift will compress the range to [0.5, 1] — which is
        harmless for threshold=0.30 but documents the expected contract.
        """
        if self.similarity_fn is None:
            return np.ones(len(frames), dtype=np.float32)

        try:
            similarities = np.asarray(self.similarity_fn(frames, sub_query), dtype=np.float32)
        except Exception:
            similarities = np.ones(len(frames), dtype=np.float32)

        if similarities.shape != (len(frames),):
            similarities = np.ones(len(frames), dtype=np.float32)

        # Shift cosine similarities from [-1,1] to [0,1]
        similarities = (similarities + 1.0) / 2.0
        return np.clip(similarities, 0.0, 1.0)

    def _select_k(self, sub_query: str, n_frames: int) -> int:
        """
        K is adapted by the LLM to query complexity (Section 3.2).
        Falls back to a word-count heuristic when no agent is available.
        K is clipped to [k_simple, k_complex] = [4, 12] as per paper.
        """
        if self.agent_fn is not None:
            prompt = (
                f"Question: {sub_query}\n"
                f"Available frames: {n_frames}\n\n"
                "Choose how many frames K should be selected for visual reasoning. "
                f"Return ONLY a JSON object like {{\"k\": {self.k_simple}}}. "
                f"Prefer values between {self.k_simple} and {min(self.k_complex, n_frames)}. "
                "Use smaller K for single-hop factual questions and larger K for temporal or multi-hop questions."
            )
            try:
                raw = self.agent_fn(prompt)
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    value = int(parsed.get("k", 0))
                    if value > 0:
                        return min(max(value, self.k_simple), min(self.k_complex, n_frames))
            except Exception:
                pass

        word_count = len(sub_query.split())
        heuristic_k = self.k_complex if word_count >= self.complexity_threshold else self.k_simple
        return min(heuristic_k, n_frames)


# ── retrieve_frames ───────────────────────────────────────────────────────────

class RetrieveFrames(ToolBase):
    """
    Disturbance-aware retrieval from pool P  (Table 17, cost 0.20).

    Given the pool P assembled by select_frames (the frames NOT promoted into
    the top-K), retrieves the frames that are most relevant AND least disturbed
    for a sub-query.  This is invoked during evidence acquisition when the
    initially selected frames yield low-confidence evidence for a sub-query, as
    illustrated in the paper's case study (Tab. 7, sq3:
    retrieve_frames(...) -> track_temporal(...)).  It is a routable selection
    tool, NOT a separate iterative loop.
    """

    name = "retrieve_frames"

    def __init__(self, top_k: int = 4, similarity_fn: Optional[SimilarityFn] = None):
        self.top_k = top_k
        self.similarity_fn = similarity_fn

    def _run(
        self,
        frames: np.ndarray,
        sub_query: str,
        disturbance_scores: np.ndarray,
    ) -> Tuple[Dict, float]:
        N = len(frames)
        if N == 0:
            return None, 0.0

        # Re-use SelectFrames ranking with fixed K (no agent needed for retrieval)
        selector = SelectFrames(
            k_simple=self.top_k,
            k_complex=self.top_k,
            similarity_fn=self.similarity_fn,
        )
        inner = selector._run(frames, sub_query, disturbance_scores)
        if inner[0] is None:
            return None, 0.0

        result_dict, c_intrinsic = inner
        return {
            "retrieved_indices": result_dict["selected_indices"],
            "scores": result_dict["scores"],
        }, c_intrinsic


# ── convenience factory ───────────────────────────────────────────────────────

def build_selection_tools(
    k_simple: int = 4,
    k_complex: int = 12,
    similarity_fn: Optional[SimilarityFn] = None,
    agent_fn: Optional[AgentFn] = None,
) -> Dict[str, ToolBase]:
    return {
        "assess_quality":  AssessQuality(),
        "select_frames":   SelectFrames(
            k_simple=k_simple,
            k_complex=k_complex,
            similarity_fn=similarity_fn,
            agent_fn=agent_fn,
        ),
        "retrieve_frames": RetrieveFrames(similarity_fn=similarity_fn),
    }
