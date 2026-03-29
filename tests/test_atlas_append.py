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
