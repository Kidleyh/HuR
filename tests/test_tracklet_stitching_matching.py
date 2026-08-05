"""Tests for global assignment, margins, chains, and source-track coverage."""

from astrolabe.scorers.video.person_tracking.schemas import TrackedDetection
from astrolabe.scorers.video.tracklet_stitching.matching import (
    build_logical_chains,
    maximum_weight_assignment,
    select_merged_edges,
)
from astrolabe.scorers.video.tracklet_stitching.schemas import CandidateEdge, StitchingConfig, Tracklet


def tracklet(track_id, start):
    detection = TrackedDetection.from_xyxy(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9,
        bbox_xyxy=[10, 10, 30, 50], image_width=100, image_height=100,
    )
    return Tracklet(track_id, [start], [detection], start, start)


def edge(source, target, score):
    return CandidateEdge(source, target, 1, score=score, decision="eligible")


def test_global_assignment_beats_pairwise_greedy_and_is_one_to_one():
    tracklets = [tracklet(1, 0), tracklet(2, 0), tracklet(3, 2), tracklet(4, 2)]
    edges = [edge(1, 3, 0.90), edge(1, 4, 0.80), edge(2, 3, 0.85)]
    selected = maximum_weight_assignment(tracklets, edges)
    assert {(item.from_track_id, item.to_track_id) for item in selected} == {(1, 4), (2, 3)}
    assert len({item.from_track_id for item in selected}) == len(selected)
    assert len({item.to_track_id for item in selected}) == len(selected)


def test_dummy_unmatched_does_not_force_edges():
    tracklets = [tracklet(1, 0), tracklet(2, 0), tracklet(3, 2)]
    edges = [edge(1, 3, 0.9)]
    selected = select_merged_edges(tracklets, edges, StitchingConfig())
    assert [(item.from_track_id, item.to_track_id) for item in selected] == [(1, 3)]


def test_low_margin_becomes_uncertain():
    tracklets = [tracklet(1, 0), tracklet(3, 2), tracklet(4, 2)]
    best, close = edge(1, 3, 0.80), edge(1, 4, 0.79)
    selected = select_merged_edges(tracklets, [best, close], StitchingConfig())
    assert selected == []
    assert best.decision == "uncertain"
    assert "ambiguous_assignment" in best.rejection_reasons


def test_chain_expansion_and_singletons_cover_all_tracks():
    tracklets = [tracklet(1, 0), tracklet(3, 3), tracklet(7, 7), tracklet(12, 1)]
    chains = build_logical_chains(tracklets, [edge(1, 3, 0.9), edge(3, 7, 0.9)])
    assert [1, 3, 7] in chains
    assert [12] in chains
    flattened = [track_id for chain in chains for track_id in chain]
    assert sorted(flattened) == [1, 3, 7, 12]
    assert len(flattened) == len(set(flattened))
