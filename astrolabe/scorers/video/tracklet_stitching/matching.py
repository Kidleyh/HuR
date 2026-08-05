"""Deterministic maximum-weight one-to-one assignment with ambiguity margins."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import CandidateEdge, StitchingConfig, Tracklet


def _margin(selected: CandidateEdge, alternatives: List[float]) -> float:
    return selected.score - max(alternatives) if alternatives else 1.0


def maximum_weight_assignment(
    tracklets: Sequence[Tracklet], edges: Sequence[CandidateEdge]
) -> List[CandidateEdge]:
    """Return the deterministic global assignment before ambiguity filtering."""
    ids = sorted(item.track_id for item in tracklets)
    if not ids:
        return []
    index = {track_id: position for position, track_id in enumerate(ids)}
    count = len(ids)
    weights = np.zeros((count, count * 2), dtype=np.float64)
    by_pair: Dict[Tuple[int, int], CandidateEdge] = {}
    for edge in edges:
        if edge.decision == "eligible":
            row, column = index[edge.from_track_id], index[edge.to_track_id]
            # Tiny stable preference only breaks exact score ties.
            weights[row, column] = edge.score + 1e-12 * (count - column)
            by_pair[(row, column)] = edge
    rows, columns = linear_sum_assignment(-weights)
    return [
        by_pair[(row, column)]
        for row, column in sorted(zip(rows.tolist(), columns.tolist()))
        if (row, column) in by_pair
    ]


def select_merged_edges(
    tracklets: Sequence[Tracklet], edges: Sequence[CandidateEdge], config: StitchingConfig
) -> List[CandidateEdge]:
    """Apply global assignment; leave low-margin selected edges uncertain."""
    assigned = maximum_weight_assignment(tracklets, edges)
    selected: List[CandidateEdge] = []
    viable = [edge for edge in edges if not edge.rejection_reasons or edge.decision == "uncertain"]
    for edge in assigned:
        outgoing = [other.score for other in viable if other.from_track_id == edge.from_track_id and other.to_track_id != edge.to_track_id]
        incoming = [other.score for other in viable if other.to_track_id == edge.to_track_id and other.from_track_id != edge.from_track_id]
        edge.outgoing_margin = _margin(edge, outgoing)
        edge.incoming_margin = _margin(edge, incoming)
        if (edge.outgoing_margin >= config.minimum_assignment_margin and
                edge.incoming_margin >= config.minimum_assignment_margin):
            edge.decision = "merged"
            selected.append(edge)
        else:
            edge.decision = "uncertain"
            edge.rejection_reasons.append("ambiguous_assignment")
    # Eligible edges not selected by the global assignment are not merges.
    selected_ids = {id(edge) for edge in selected}
    for edge in edges:
        if edge.decision == "eligible" and id(edge) not in selected_ids:
            edge.decision = "uncertain"
            edge.rejection_reasons.append("not_selected_by_global_assignment")
    return selected


def build_logical_chains(tracklets: Sequence[Tracklet], merged: Sequence[CandidateEdge]) -> List[List[int]]:
    successor = {edge.from_track_id: edge.to_track_id for edge in merged}
    predecessor = {edge.to_track_id: edge.from_track_id for edge in merged}
    by_id = {item.track_id: item for item in tracklets}
    starts = [track_id for track_id in by_id if track_id not in predecessor]
    chains: List[List[int]] = []
    visited = set()
    for start in sorted(starts, key=lambda value: (by_id[value].start_frame, value)):
        chain, current = [], start
        while current in by_id and current not in visited:
            visited.add(current)
            chain.append(current)
            current = successor.get(current, -1)
        chains.append(chain)
    if len(visited) != len(by_id):
        raise RuntimeError("Merged tracklet graph contains a cycle")
    chains.sort(key=lambda chain: (by_id[chain[0]].start_frame, min(chain)))
    return chains
