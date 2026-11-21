import copy

import numpy as np
import pytest

from acousticase.evaluation import LinearAudioHead, binary_metrics, class_labels, validate_split
from acousticase.sound_events import events_from_scores, group_scores, load_profile, profile_hash, score_timeline


@pytest.fixture
def labels():
    return class_labels()


def frames(labels, **signals):
    scores = np.zeros((len(next(iter(signals.values()))), len(labels)))
    for label, values in signals.items():
        scores[:, labels.index(label)] = values
    return scores


def test_real_class_names_include_previously_unreachable_labels(labels):
    scores = frames(labels, **{"Accelerating, revving, vroom": [0.9]*4, "Smash, crash": [0, 0.9, 0, 0]})
    events, _ = events_from_scores(scores, labels, 2.5)
    assert {e["measurements"]["sound_group"] for e in events} >= {"engine", "impact"}
    assert all(e["evidence_type"] == "model_suggestion" for e in events)
    assert all("collision" not in e["label"] for e in events)


def test_hysteresis_joins_flicker_without_creating_events_from_weak_noise(labels):
    profile = load_profile()
    profile["onset_thresholds"]["siren"] = 0.5
    scores = frames(labels, Siren=[0, .8, .4, .8, 0, 0, .2, .2])
    events, _ = events_from_scores(scores, labels, 4.2, profile)
    assert len(events) == 1
    assert events[0]["start_s"] == pytest.approx(.48)
    assert events[0]["end_s"] == pytest.approx(2.4)
    assert events[0]["measurements"]["active_frames"] == 2


def test_separated_horns_remain_separate_and_short_horn_is_kept(labels):
    scores = frames(labels, **{"Vehicle horn, car horn, honking": [.9, 0, 0, .9]})
    events, _ = events_from_scores(scores, labels, 2.1)
    assert len(events) == 2
    assert events[0]["end_s"] < events[1]["start_s"]
    assert all(0 <= e["start_s"] <= e["peak_s"] <= e["end_s"] <= 2.1 for e in events)


def test_groups_do_not_add_correlated_probabilities_and_keep_simultaneous_sounds(labels):
    scores = frames(labels, **{"Siren": [.2]*3, "Police car (siren)": [.2]*3,
                               "Vehicle horn, car horn, honking": [.9]*3, "Engine": [.9]*3, "Vehicle": [.9]*3})
    profile = load_profile()
    profile["onset_thresholds"]["siren"] = .5
    events, info = events_from_scores(scores, labels, 2, profile)
    assert {e["measurements"]["sound_group"] for e in events} == {"horn", "engine"}
    assert info["suppressed_broad_labels"] == 1


def test_short_clip_bounds_and_score_timeline(labels):
    scores = frames(labels, **{"Vehicle horn, car horn, honking": [.9]})
    events, _ = events_from_scores(scores, labels, .25)
    assert events[0]["end_s"] == .25
    assert events[0]["peak_s"] == .25
    timeline = score_timeline(scores, labels, .25, load_profile())
    assert timeline["start_s"] == [0]
    assert timeline["end_s"] == [.25]
    assert all(len(g["scores"]) == 1 for g in timeline["groups"])


def test_cap_is_explicit_ids_unique_and_profile_changes_cache_hash(labels):
    scores = frames(labels, **{"Vehicle horn, car horn, honking": [.9,0,0,.9,0,0,.9]})
    events, info = events_from_scores(scores, labels, 4, max_events=2)
    assert len(events) == 2 and info["omitted_event_count"] == 1
    assert len({e["id"] for e in events}) == 2
    profile = load_profile()
    changed = copy.deepcopy(profile)
    changed["onset_thresholds"]["horn"] = .99
    assert profile_hash(profile) != profile_hash(changed)


def test_invalid_model_output_is_rejected(labels):
    with pytest.raises(ValueError, match="finite"):
        group_scores(np.full((2, len(labels)), np.nan), labels)
    with pytest.raises(ValueError, match="class map"):
        group_scores(np.zeros((2, 1)), ["Speech"])


def test_evaluation_rejects_source_leakage():
    config = {"train_folds": [1], "validation_folds": [4], "test_folds": [5], "categories": ["engine"]}
    rows = [{"filename": f"{fold}.wav", "fold": fold, "src_file": str(fold), "category": "engine"} for fold in (1,4,5)]
    assert validate_split(rows, config)["test"] == {5}
    rows[-1]["src_file"] = "1"
    with pytest.raises(ValueError, match="cross partitions"):
        validate_split(rows, config)


def test_binary_metrics_count_false_positives_and_misses():
    result = binary_metrics([True, True, False, False], [True, False, True, False])
    assert result == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": .5, "recall": .5, "f1": .5}


def test_portable_head_uses_saved_scaling_and_stable_softmax(tmp_path):
    path = tmp_path / "head.npz"
    np.savez(path, mean=np.zeros(4), scale=np.ones(4), weights=np.array([[1000,0,0,0], [-1000,0,0,0]]),
             bias=np.zeros(2), classes=np.array(["engine", "siren"]))
    predictions = LinearAudioHead(path).predict(np.array([[1, 0], [1, 0]]))
    assert predictions[0]["label"] == "engine"
    assert predictions[0]["model_score"] == 1
