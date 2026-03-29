from types import SimpleNamespace
import numpy as np

from gguf_dream_lab.backend.dream.state import DreamMode, LatentState
from gguf_dream_lab.storage.session_store import SessionStore


def test_session_store_save_and_load_ticks(tmp_path):
    store = SessionStore(tmp_path)
    tick = SimpleNamespace(
        run_id="r1",
        step_idx=0,
        preview_text="preview",
        committed_text="",
        coherence=0.4,
        entropy=0.3,
        local_density=0.2,
        token_stability=0.1,
        smoothness=0.5,
        state_id="s1",
        phase="HYPNAGOGIC",
        mode="enhanced_latent",
        status="running",
    )
    store.save_ticks("r1", [tick], metadata={"x": 1})
    df = store.load_ticks("r1")
    assert len(df) == 1
    assert df.iloc[0]["state_id"] == "s1"


def test_latent_state_clone_roundtrip():
    st = LatentState(run_id="r", basin="narrative", mode=DreamMode.ENHANCED_LATENT, latent_vector=np.ones(4, dtype=np.float32))
    st2 = st.clone_with_vector(np.zeros(4, dtype=np.float32))
    assert st2.state_id != st.state_id
    assert st2.latent_vector.shape == (4,)
