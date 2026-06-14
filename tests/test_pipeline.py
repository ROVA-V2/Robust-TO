"""
Robust-TO — dependency-light smoke tests.

These run WITHOUT any VLM / detector checkpoints (set ROVID_SKIP_MODEL_LOAD=1)
by injecting stub agent/VLM callables, so they exercise the orchestration,
selection, confidence coupling (Eq. 4), three-tier synthesis and GRPO reward
(Eqs. 5-10) end-to-end on synthetic frames.

Run:
    ROVID_SKIP_MODEL_LOAD=1 PYTHONPATH=. python -m pytest tests/ -q
    # or, without pytest:
    ROVID_SKIP_MODEL_LOAD=1 PYTHONPATH=. python tests/test_pipeline.py
"""
import math
import os

os.environ.setdefault("ROVID_SKIP_MODEL_LOAD", "1")

import numpy as np

from rovid_pipeline.rovid_pipeline import RobustTOPipeline
from rovid_pipeline.reward import subquery_reward, per_call_reward
from rovid_pipeline.tools.base import ToolBase, worst_k_mean


# ── Stubs ─────────────────────────────────────────────────────────────────────
def _agent(prompt: str) -> str:
    p = prompt.lower()
    if "decompose" in p:
        return ('[{"sub_query":"What objects are near the intersection?","type":"spatial"},'
                '{"sub_query":"In what order do they appear?","type":"temporal"}]')
    if '"k"' in p or "how many frames" in p:
        return '{"k": 6}'
    if "minimum number" in p and "integer" in p:
        return "2"
    if "tool name" in p or "routing agent" in p:
        return "detect_objects"
    if "synthesizing evidence" in p or "<answer>" in p:
        return "<think>HIGH-tier evidence supports B.</think><answer>B</answer>"
    if "briefly describe" in p:
        return "A street intersection with vehicles."
    return "ok"


def _vlm(prompt: str, frames) -> str:
    return "B-7742-XK"


def _sim(frames, query):
    rng = np.random.default_rng(0)
    return rng.random(len(frames)).astype(np.float32)


def _video(n=24, corrupt=range(10, 14)):
    rng = np.random.default_rng(0)
    f = (rng.random((n, 48, 48, 3)) * 255).astype(np.uint8)
    for i in corrupt:
        f[i] = 3  # near-black -> high brightness disturbance
    return f


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_worst_k_mean():
    q = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])  # quality = 1 - d
    k = math.ceil(len(q) / 3)
    assert abs(worst_k_mean(q) - float(np.mean(np.sort(q)[:k]))) < 1e-9


def test_eq4_input_reliability_coupling():
    class _T(ToolBase):
        name = "detect_objects"
        def _run(self, frames, sq, d):
            return "boxes", 0.95
    t = _T()
    clean = t(np.zeros((6, 8, 8, 3), np.uint8), "q", disturbance_scores=np.zeros(6))
    dirty = t(np.zeros((6, 8, 8, 3), np.uint8), "q",
              disturbance_scores=np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]))
    assert clean.confidence > dirty.confidence  # low input reliability drags c_j down


def test_reward_subq_is_sum_and_penalises_over_decomposition():
    # Eq. 9 is a SUM; R_min-sq must strictly decrease once m > m*.
    r2 = subquery_reward(n_subqueries=2, mean_confidence=0.8, optimal_subqueries=2)
    r5 = subquery_reward(n_subqueries=5, mean_confidence=0.8, optimal_subqueries=2)
    rmin = math.exp(-0.2 * max(0, 5 - 2))
    rq = (1 - math.exp(-1.0 * 5 / 2)) * 0.8
    assert abs(r5 - (rmin + rq)) < 1e-9     # sum form, not 0.5*(...)
    assert r2 > r5                          # over-decomposition is penalised


def test_per_call_reward_cost_penalty():
    # Eq. 5: cheaper tool yields higher reward at equal confidence.
    assert per_call_reward(0.8, "read_text") > per_call_reward(0.8, "track_temporal")


def test_pipeline_end_to_end():
    pipe = RobustTOPipeline(agent_fn=_agent, vlm_fn=_vlm, estimator_fn=_agent,
                            similarity_fn=_sim)
    res = pipe.run(_video(), "Which vehicle ran the red light and its plate?",
                   ground_truth="B")
    assert res["answer"] == "B"
    assert res["info"]["selected_k"] >= 4
    # primary trajectory has exactly one perception call per sub-query
    assert res["reward"].n_calls == res["info"]["n_sub_queries"]
    assert res["reward"].R_acc == 1.0


def test_training_mode_fixed_length_trajectory():
    pipe = RobustTOPipeline(agent_fn=_agent, vlm_fn=_vlm, estimator_fn=_agent,
                            similarity_fn=_sim)
    res = pipe.run(_video(), "Which vehicle ran the red light?",
                   ground_truth="B", training=True)
    # No refinement in training mode -> one primary call per sub-query.
    assert res["reward"].n_calls == res["info"]["n_sub_queries"]


def test_select_frames_excludes_ineligible_frames():
    """
    BUG FIX regression: SelectFrames must NEVER put score=0 (ineligible)
    frames into selected_indices, even when eligible count < K.

    Paper (Eq. 3): 'frames outside F receive s(f_i)=0 and are discarded'.
    Old code used argsort over ALL frames and filled K slots regardless,
    letting corrupted frames contaminate the selected set.
    """
    from rovid_pipeline.tools.selection_tools import SelectFrames
    frames = np.zeros((5, 8, 8, 3), np.uint8)
    # frames 0,1 are heavily corrupted -> reliability = 0.1 < theta_rel=0.55
    d_scores = np.array([0.9, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)
    sel = SelectFrames(k_simple=4, k_complex=4, theta_rel=0.55, theta_sim=0.30,
                       similarity_fn=lambda f, q: np.ones(len(f), np.float32))
    result, c = sel._run(frames, "test", d_scores)
    selected = result["selected_indices"]
    # Only 3 frames are eligible; K=4 should NOT pull in the 2 ineligible frames
    assert 0 not in selected, f"Ineligible frame 0 (d=0.9) in selected: {selected}"
    assert 1 not in selected, f"Ineligible frame 1 (d=0.9) in selected: {selected}"
    assert len(selected) == 3, f"Expected 3 eligible frames, got {len(selected)}: {selected}"
    # c_intrinsic must be based only on eligible frames -> ~0.9, not deflated by 0.0
    assert c > 0.8, f"c_intrinsic deflated by ineligible frames: {c:.4f}"


def test_select_frames_pool_contains_all_non_selected():
    """Pool indices must be the exact complement of selected_indices."""
    from rovid_pipeline.tools.selection_tools import SelectFrames
    frames = np.zeros((8, 8, 8, 3), np.uint8)
    d_scores = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    sel = SelectFrames(k_simple=4, k_complex=4, theta_rel=0.55, theta_sim=0.30,
                       similarity_fn=lambda f, q: np.ones(len(f), np.float32))
    result, _ = sel._run(frames, "test", d_scores)
    selected = set(result["selected_indices"])
    pool     = set(result["pool_indices"])
    assert selected | pool == set(range(len(frames))), "selected ∪ pool != all frames"
    assert selected & pool == set(), "selected ∩ pool is not empty"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed.")
