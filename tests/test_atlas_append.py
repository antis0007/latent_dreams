import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, StatePoint, TransitionEdge


def test_atlas_append_growth_and_edges():
    atlas = LatentAtlas()
    for i in range(5):
        atlas.append_points(
            [
                StatePoint(
                    state_id=f"s{i}",
                    run_id="r",
                    run_label="x",
                    basin="narrative",
                    step_idx=i,
                    preview="p",
                    committed="c",
                    coherence=0.5,
                    entropy=0.5,
                    embedding=np.random.randn(16).astype("float32"),
                )
            ]
        )
        if i:
            atlas.append_transition(TransitionEdge(f"s{i-1}", f"s{i}", "r", i, 0.2, 0.1, 1.0))
    assert len(atlas.points) == 5
    assert len(atlas.edges) == 4


def test_to_frame_tolerates_partial_cluster_labels():
    atlas = LatentAtlas()
    atlas.append_points(
        [
            StatePoint(
                state_id=f"s{i}",
                run_id="r",
                run_label="x",
                basin="narrative",
                step_idx=i,
                preview="p",
                committed="c",
                coherence=0.5,
                entropy=0.5,
                embedding=np.random.randn(16).astype("float32"),
            )
            for i in range(3)
        ]
    )
    atlas.cluster_labels = np.array([0, 1], dtype=int)

    frame = atlas.to_frame()

    assert len(frame) == 3
    assert frame.loc[2, "cluster"] == -1


def test_neighbors_handles_stale_nn_index_without_raising():
    atlas = LatentAtlas()
    for i in range(4):
        atlas.append_points(
            [
                StatePoint(
                    state_id=f"s{i}",
                    run_id="r",
                    run_label="x",
                    basin="narrative",
                    step_idx=i,
                    preview="p",
                    committed="c",
                    coherence=0.5,
                    entropy=0.5,
                    embedding=np.random.randn(16).astype("float32"),
                )
            ]
        )

    neighbors = atlas.neighbors(3, k=4)

    assert isinstance(neighbors, list)
    assert len(neighbors) <= 3


def test_append_latent_state_tracks_recurrence_and_attractor_strength():
    from gguf_dream_lab.backend.dream.state import DreamMode, LatentState

    atlas = LatentAtlas(recurrence_distance_threshold=0.25)
    base = np.zeros(8, dtype=np.float32)
    far = np.ones(8, dtype=np.float32) * 5.0

    a = LatentState(run_id="r", basin="narrative", mode=DreamMode.ENHANCED_LATENT, latent_vector=base, timestamp=10.0)
    b = LatentState(
        run_id="r",
        basin="narrative",
        mode=DreamMode.ENHANCED_LATENT,
        latent_vector=base + 0.01,
        timestamp=11.0,
    )
    c = LatentState(run_id="r", basin="narrative", mode=DreamMode.ENHANCED_LATENT, latent_vector=far, timestamp=12.0)
    d = LatentState(
        run_id="r",
        basin="narrative",
        mode=DreamMode.ENHANCED_LATENT,
        latent_vector=base + 0.02,
        timestamp=14.0,
    )

    first = atlas.append_latent_state(a, run_label="x", step_idx=0)
    second = atlas.append_latent_state(b, run_label="x", step_idx=1)
    atlas.append_latent_state(c, run_label="x", step_idx=2)
    fourth = atlas.append_latent_state(d, run_label="x", step_idx=4)

    assert first.recurrence == 1
    assert second.attractor_id == first.state_id
    assert fourth.return_count == 1
    assert fourth.visit_count == 3
    assert fourth.attractor_strength > 0.0

    ranked = atlas.candidate_attractors(basin="narrative", top_k=2)
    assert ranked[0].attractor_id == first.state_id


def test_basin_prior_stats_and_frame_include_basin_metrics():
    atlas = LatentAtlas()
    atlas.append_points(
        [
            StatePoint(
                state_id="s0",
                run_id="r0",
                run_label="run",
                basin="narrative",
                step_idx=0,
                preview="",
                committed="",
                coherence=0.7,
                entropy=0.2,
                density=0.5,
                attractor_strength=1.2,
                basin_sample_count=3,
                basin_force_magnitude=0.4,
                basin_prior_spread=0.3,
                embedding=np.ones(8, dtype=np.float32),
            )
        ]
    )

    stats = atlas.basin_prior_stats("narrative", ref_vector=np.ones(8, dtype=np.float32))
    frame = atlas.to_frame()

    assert int(stats["sample_count"]) == 1
    assert "basin_sample_count" in frame.columns
    assert float(frame.iloc[0]["basin_force_magnitude"]) == 0.4
