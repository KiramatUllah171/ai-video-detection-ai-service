from app.services.score_mapping import aggregate_frame_scores, map_label_scores
from app.utils.errors import AiServiceError


def test_score_mapping_detects_fake_label() -> None:
    result = map_label_scores({"FAKE": 0.8, "REAL": 0.2})

    assert result.ai_score == 0.8
    assert result.real_probability == 0.2
    assert result.mapped_fake_label == "FAKE"


def test_score_mapping_detects_real_label() -> None:
    result = map_label_scores({"authentic": 0.9, "synthetic": 0.1})

    assert result.ai_score == 0.1
    assert result.real_probability == 0.9
    assert result.mapped_real_label == "authentic"


def test_score_mapping_detects_common_fake_terms() -> None:
    fake_terms = ["fake", "Fake", "FAKE", "deepfake", "AI-generated", "generated", "synthetic", "manipulated"]

    for term in fake_terms:
        result = map_label_scores({term: 0.77, "real": 0.23})
        assert result.ai_score == 0.77
        assert result.real_probability == 0.23


def test_score_mapping_detects_common_real_terms() -> None:
    real_terms = ["real", "Real", "REAL", "authentic", "genuine", "original"]

    for term in real_terms:
        result = map_label_scores({term: 0.82, "fake": 0.18})
        assert result.ai_score == 0.18
        assert result.real_probability == 0.82


def test_unclear_labels_fail_safely() -> None:
    try:
        map_label_scores({"LABEL_0": 0.7, "LABEL_1": 0.3})
    except AiServiceError as exception:
        assert exception.error_code == "AI_SERVICE_INVALID_RESPONSE"
    else:
        raise AssertionError("Expected AI_SERVICE_INVALID_RESPONSE")


def test_aggregation_uses_top_frames_and_p90() -> None:
    result = aggregate_frame_scores([0.1, 0.2, 0.3, 0.8, 0.9])

    assert result.mean_score == 0.46
    assert result.top_k_mean > result.mean_score
    assert result.p90_score > result.mean_score
    assert result.visual_score > result.mean_score
    assert result.high_frame_count == 2


def test_aggregation_rejects_empty_scores() -> None:
    try:
        aggregate_frame_scores([])
    except AiServiceError as exception:
        assert exception.error_code == "AI_SERVICE_INVALID_RESPONSE"
    else:
        raise AssertionError("Expected AI_SERVICE_INVALID_RESPONSE")
