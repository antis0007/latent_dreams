from __future__ import annotations

from pathlib import Path
from time import time

import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dcc, html

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import AppConfig, Basin
from gguf_dream_lab.storage.session_store import SessionStore


def create_dash_app(config: AppConfig) -> Dash:
    runtime = LlamaCppBackend(config.runtime)
    atlas_store = AtlasStorage(config.storage_dir / "atlas")
    atlas = atlas_store.load()
    controller = DreamController(runtime=runtime, atlas=atlas)
    session_store = SessionStore(config.storage_dir / "sessions")

    app = Dash(__name__)
    app.title = "GGUF Dream Lab"

    app.layout = html.Div(
        style={"backgroundColor": "#101318", "color": "#f2f4f8", "fontFamily": "Inter, Arial", "padding": "16px"},
        children=[
            html.H2("GGUF Dream Lab"),
            html.Div(id="capability", style={"marginBottom": "8px", "opacity": 0.9}),
            html.Div(id="status", style={"marginBottom": "8px", "fontWeight": 600}),
            html.Div([
                html.Label("Basin"),
                dcc.Dropdown(id="basin", options=[{"label": b.value, "value": b.value} for b in Basin], value=config.dream.basin.value),
                html.Label("Prompt"), dcc.Input(id="prompt", value=config.dream.prompt, type="text", style={"width": "100%"}),
                html.Label("Tick rate (Hz)"), dcc.Slider(id="tick-hz", min=0.5, max=8, step=0.5, value=config.dream.tick_hz),
                html.Label("Coherence threshold"), dcc.Slider(id="threshold", min=0.3, max=0.95, step=0.01, value=config.dream.coherence_threshold),
                html.Label("Noise amplitude"), dcc.Slider(id="noise", min=0.01, max=0.6, step=0.01, value=config.dream.noise_amplitude),
                html.Label("Branch count"), dcc.Slider(id="branches", min=1, max=5, step=1, value=config.dream.branch_count),
                html.Button("Start", id="start-btn"), html.Button("Pause", id="pause-btn"), html.Button("Resume", id="resume-btn"), html.Button("Stop", id="stop-btn"),
            ]),
            html.H4("Preview (provisional recall)"),
            html.Div(id="preview", style={"fontStyle": "italic", "opacity": 0.6, "minHeight": "80px"}),
            html.H4("Committed (crystallized recall)"),
            html.Div(id="committed", style={"minHeight": "80px"}),
            html.Pre(id="metrics"),
            dcc.Graph(id="latent-graph", style={"height": "500px"}),
            dcc.Slider(id="scrub-step", min=0, max=1, step=1, value=0),
            html.Div(id="selected-state"),
            dcc.Interval(id="ticker", interval=int(1000 / max(config.dream.tick_hz, 0.5)), n_intervals=0),
            dcc.Store(id="control-ack"),
        ],
    )

    @app.callback(
        Output("control-ack", "data"),
        Input("start-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("resume-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        State("prompt", "value"),
        State("basin", "value"),
        State("threshold", "value"),
        State("tick-hz", "value"),
        State("noise", "value"),
        State("branches", "value"),
        prevent_initial_call=True,
    )
    def controls(start, pause, resume, stop, prompt, basin, threshold, tick_hz, noise, branches):
        trigger = ctx.triggered_id
        if trigger == "start-btn":
            config.dream.prompt = prompt or ""
            config.dream.basin = Basin(basin)
            config.dream.coherence_threshold = float(threshold)
            config.dream.tick_hz = float(tick_hz)
            config.dream.noise_amplitude = float(noise)
            config.dream.branch_count = int(branches)
            controller.start(config.dream)
        elif trigger == "pause-btn":
            controller.pause()
        elif trigger == "resume-btn":
            controller.resume()
        elif trigger == "stop-btn":
            controller.stop()
            atlas_store.save(controller.atlas)
            session_store.save_ticks(controller.state.run_id, controller.tick_history, metadata={"dream": config.dream.model_dump()})
        return {"at": time(), "status": controller.state.status}

    @app.callback(Output("status", "children"), Input("ticker", "n_intervals"))
    def refresh_status(_):
        detail = f" ({controller.state.status_detail})" if controller.state.status_detail else ""
        return f"Status: {controller.state.status}{detail}"

    @app.callback(Output("capability", "children"), Input("ticker", "n_intervals"))
    def refresh_capability(_):
        caps = runtime.capabilities()
        return f"Mode: {caps.active_mode.value} | backend: {caps.backend_name} | capture_sites: {caps.capture_sites or ['none']}"

    @app.callback(
        Output("preview", "children"),
        Output("committed", "children"),
        Output("metrics", "children"),
        Output("latent-graph", "figure"),
        Output("selected-state", "children"),
        Input("ticker", "n_intervals"),
    )
    def refresh_stream(_):
        frame = controller.atlas.to_frame()
        if frame.empty:
            fig = px.scatter(x=[0], y=[0], title="No latent states yet")
            fig.update_layout(template="plotly_dark")
            return "", "", "No ticks yet.", fig, ""

        max_step = int(frame["step_idx"].max())
        selected = frame[frame["step_idx"] == max_step].tail(1)
        tick = controller.state.latest_tick

        fig = px.scatter(frame, x="x", y="y", color="coherence", symbol="phase", hover_data=["state_id", "basin", "step_idx", "preview", "committed", "density", "entropy"])
        traj = frame.sort_values("step_idx")
        fig.add_scatter(x=traj["x"], y=traj["y"], mode="lines", line={"width": 4, "color": "#FFFFFF"}, name="trajectory")
        if not selected.empty:
            fig.add_scatter(x=selected["x"], y=selected["y"], mode="markers", marker={"size": 16, "color": "#ff4d6d"}, name="selected")
        fig.update_layout(template="plotly_dark", title="Latent-state atlas (projection only; control stays high-dimensional)")

        selected_txt = ""
        if not selected.empty:
            row = selected.iloc[0]
            selected_txt = f"selected state={row['state_id']} phase={row['phase']} basin={row['basin']}"

        if tick is None:
            return "", "", "No ticks yet.", fig, selected_txt
        metrics = f"mode={tick.mode}\nphase={tick.phase}\ncoherence={tick.coherence:.3f}\nentropy={tick.entropy:.3f}\ndensity={tick.local_density:.3f}\nstability={tick.token_stability:.3f}"
        return tick.preview_text, tick.committed_text, metrics, fig, selected_txt


    @app.callback(Output("scrub-step", "max"), Input("ticker", "n_intervals"))
    def refresh_scrub_max(_):
        frame = controller.atlas.to_frame()
        if frame.empty:
            return 1
        return int(frame["step_idx"].max())

    return app


def run_ui(config: AppConfig, host: str = "127.0.0.1", port: int = 8050) -> None:
    app = create_dash_app(config)
    app.run(host=host, port=port, debug=False)
