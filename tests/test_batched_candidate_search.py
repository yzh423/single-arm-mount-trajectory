import torch

from design_optimization.batched_search import evaluate_candidate_batches, safe_candidate_batch_size


def test_batched_candidate_evaluation_preserves_global_order_and_values():
    candidates = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    def evaluate(batch, start):
        return {
            "score": batch.sum(dim=1),
            "global_index": torch.arange(start, start + len(batch), device=batch.device),
        }

    result = evaluate_candidate_batches(candidates, batch_size=4, evaluate=evaluate)

    torch.testing.assert_close(result["score"], candidates.sum(dim=1))
    torch.testing.assert_close(result["global_index"], torch.arange(10))


def test_batched_candidate_evaluation_moves_results_to_cpu():
    candidates = torch.ones((5, 2))
    result = evaluate_candidate_batches(
        candidates,
        batch_size=2,
        evaluate=lambda batch, _start: {"value": batch[:, 0]},
    )
    assert result["value"].device.type == "cpu"


def test_safe_batch_scales_with_frames_and_seed_branches():
    assert safe_candidate_batch_size(16, frames=184, seeds=10) == 16
    assert safe_candidate_batch_size(16, frames=800, seeds=10) == 3
