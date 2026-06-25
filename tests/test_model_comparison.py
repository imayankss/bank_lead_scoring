from lead_scoring.model_comparison import build_model_comparison


def test_model_comparison_outputs_ranked_models():
    comparison = build_model_comparison()

    assert len(comparison) >= 3
    assert comparison["rank"].min() == 1
    assert comparison["is_best"].any()
