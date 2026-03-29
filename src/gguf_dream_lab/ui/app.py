from __future__ import annotations

from pathlib import Path
from time import time

import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dcc, html

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage, LatentAtlas
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import AppConfig, Basin


def create_dash_app(config: AppConfig) -> Dash:
    runtime = LlamaCppBackend(config.runtime)
    atlas_store = AtlasStorage(config.storage_dir / "atlas")
    atlas = atlas_store.load()
    controller = DreamController(runtime=runtime, atlas=atlas)

    app = Dash(__name__)
    app.title = "GGUF Dream Lab"

    panel_style = {"background": "#1a1f29", "padding": "12px", "borderRadius": "8px", "border": "1px solid #2a3140"}
    control_style = {
        "width": "100%",
        "backgroundColor": "#0f141b",
        "color": "#f2f4f8",
        "border": "1px solid #38445a",
        "borderRadius": "6px",
        "padding": "8px 10px",
    }
    label_style = {"fontSize": "12px", "opacity": 0.85, "marginBottom": "4px", "marginTop": "8px", "display": "block"}
    button_style = {
        "marginTop": "10px",
        "padding": "8px 12px",
        "backgroundColor": "#243041",
        "color": "#f2f4f8",
        "border": "1px solid #3b4a63",
        "borderRadius": "6px",
        "cursor": "pointer",
    }

    app.layout = html.Div(
        style={"backgroundColor": "#101318", "color": "#f2f4f8", "fontFamily": "Inter, Arial", "padding": "16px"},
        children=[
            html.H2("GGUF Dream Lab", style={"marginBottom": "4px"}),
            html.Div("Realtime latent dreaming with coherence-gated commit lane", style={"opacity": 0.8}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginTop": "12px"},
                children=[
                    html.Div(
                        [
                            html.H4("Runtime"),
                            html.Label("Model path", style=label_style),
                            dcc.Input(id="model-path", value=str(config.runtime.model_path or ""), type="text", style=control_style),
                            html.Label("Dream basin", style=label_style),
                            dcc.Dropdown(
                                id="basin",
                                options=[{"label": b.value, "value": b.value} for b in Basin],
                                value=config.dream.basin.value,
                                clearable=False,
                                style={**control_style, "padding": "0"},
                            ),
                            html.Label("Prompt seed", style=label_style),
                            dcc.Input(
                                id="prompt",
                                value=config.dream.prompt,
                                type="text",
                                placeholder="Optional dream seed prompt",
                                style=control_style,
                            ),
                            html.Label("Coherence threshold", style=label_style),
                            dcc.Slider(id="threshold", min=0.3, max=0.95, step=0.01, value=config.dream.coherence_threshold),
                            html.Div(id="status", style={"marginTop": "8px", "fontWeight": "bold"}),
                            html.Div(
                                [
                                    html.Button("Start", id="start-btn", n_clicks=0, style=button_style),
                                    html.Button("Pause", id="pause-btn", n_clicks=0, style={**button_style, "marginLeft": "8px"}),
                                    html.Button("Resume", id="resume-btn", n_clicks=0, style={**button_style, "marginLeft": "8px"}),
                                    html.Button("Stop", id="stop-btn", n_clicks=0, style={**button_style, "marginLeft": "8px"}),
                                ]
                            ),
                        ],
                        style=panel_style,
                    ),
                    html.Div(
                        [
                            html.H4("Provisional Preview"),
                            html.Div(id="preview", style={"minHeight": "180px", "fontSize": "20px", "opacity": 0.65, "fontStyle": "italic"}),
                            html.H4("Committed Transcript"),
                            html.Div(id="committed", style={"minHeight": "180px", "fontSize": "18px"}),
                        ],
                        style=panel_style,
                    ),
                    html.Div(
                        [
                            html.H4("Metrics"),
                            html.Pre(id="metrics", style={"whiteSpace": "pre-wrap", "fontSize": "14px"}),
                            html.H4("Selected State"),
                            html.Div(id="selected-state"),
                        ],
                        style=panel_style,
                    ),
                ],
            ),
            dcc.Graph(id="latent-graph", style={"height": "480px", "marginTop": "12px"}),
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
        prevent_initial_call=True,
    )
    def controls(start, pause, resume, stop, prompt, basin, threshold):
        trigger = ctx.triggered_id
        if trigger is None:
            return {"at": time(), "status": controller.state.status}
        if trigger == "start-btn":
            config.dream.prompt = prompt or ""
            config.dream.basin = Basin(basin)
            config.dream.coherence_threshold = float(threshold)
            controller.start(config.dream)
        elif trigger == "pause-btn":
            controller.pause()
        elif trigger == "resume-btn":
            controller.resume()
        elif trigger == "stop-btn":
            controller.stop()
            atlas_store.save(controller.atlas)
        return {"at": time(), "status": controller.state.status}

    @app.callback(
        Output("status", "children"),
        Input("ticker", "n_intervals"),
    )
    def refresh_status(_):
        detail = f" — {controller.state.status_detail}" if controller.state.status_detail else ""
        status_text = f"Status: {controller.state.status}{detail}"
        return status_text

    @app.callback(
        Output("preview", "children"),
        Output("committed", "children"),
        Output("metrics", "children"),
        Output("latent-graph", "figure"),
        Output("selected-state", "children"),
        Input("ticker", "n_intervals"),
    )
    def refresh_stream(_):
        tick = controller.state.latest_tick
        if tick is None:
            fig = px.scatter(x=[0], y=[0], title="No latent states yet")
            fig.update_layout(template="plotly_dark")
            return "", "", "No ticks yet.", fig, ""

        frame = controller.atlas.to_frame()
        color_mode = "coherence" if "coherence" in frame.columns else None
        fig = px.scatter(
            frame,
            x="x",
            y="y",
            color=color_mode,
            hover_data=["state_id", "run_label", "basin", "step_idx", "preview", "committed", "entropy", "coherence"],
            title="Latent State Trajectory (approximate in baseline mode)",
        )
        fig.update_layout(template="plotly_dark", marker=dict(size=10, line=dict(width=2, color="#ffffff")))
        selected = f"state={tick.state_id} run={tick.run_id} step={tick.step_idx}"
        metrics = (
            f"coherence={tick.coherence:.3f}\n"
            f"entropy={tick.entropy:.3f}\n"
            f"density={tick.local_density:.3f}\n"
            f"token_stability={tick.token_stability:.3f}\n"
            f"smoothness={tick.smoothness:.3f}"
        )
        return tick.preview_text, tick.committed_text, metrics, fig, selected

    return app


def run_ui(config: AppConfig, host: str = "127.0.0.1", port: int = 8050) -> None:
    app = create_dash_app(config)
    app.run(host=host, port=port, debug=False)
