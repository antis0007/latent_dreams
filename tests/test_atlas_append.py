import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, StatePoint


def test_atlas_append_growth():
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
    assert len(atlas.points) == 5
