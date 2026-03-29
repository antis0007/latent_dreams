import numpy as np

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage, LatentAtlas, StatePoint, TransitionEdge


def test_atlas_save_load(tmp_path):
    atlas = LatentAtlas()
    atlas.append_points(
        [
            StatePoint(
                state_id="s1",
                run_id="r1",
                run_label="run",
                basin="null_prior",
                step_idx=0,
                preview="preview",
                committed="",
                coherence=0.5,
                entropy=0.8,
                embedding=np.ones(8, dtype=np.float32),
            )
        ]
    )
    atlas.append_transition(TransitionEdge("s0", "s1", "r1", 0, 0.1, 0.1, 1.0))
    store = AtlasStorage(tmp_path)
    store.save(atlas)
    loaded = store.load()
    assert len(loaded.points) == 1
    assert loaded.points[0].state_id == "s1"
    assert len(loaded.edges) == 1
