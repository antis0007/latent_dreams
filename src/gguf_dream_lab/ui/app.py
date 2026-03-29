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

    assets_dir = Path(__file__).resolve().parents[3] / "assets"
    app = Dash(__name__, assets_folder=str(assets_dir))
    app.title = "GGUF Dream Lab"

    app.layout = html.Div(
        id="app-root",
        className="app-shell theme-auto",
        children=[
            html.Div(
                className="app-header",
                children=[
                    html.H2("GGUF Dream Lab", className="app-title"),
                    html.Div(id="capability", className="meta-line"),
                    html.Div(id="status", className="status-line"),
                ],
            ),
            html.Div(
                className="layout-grid",
                children=[
                    html.Section(
                        className="panel controls-panel",
                        children=[
                            html.H4("Controls", className="panel-title"),
                            html.Label("Basin", className="control-label"),
                            dcc.Dropdown(
                                id="basin",
                                options=[{"label": b.value, "value": b.value} for b in Basin],
                                value=config.dream.basin.value,
                                className="control-field",
                                clearable=False,
                            ),
                            html.Label("Prompt", className="control-label"),
                            dcc.Input(id="prompt", value=config.dream.prompt, type="text", className="text-input"),
                            html.Label("Tick rate (Hz)", className="control-label"),
                            dcc.Slider(id="tick-hz", min=0.5, max=8, step=0.5, value=config.dream.tick_hz),
                            html.Label("Coherence threshold", className="control-label"),
                            dcc.Slider(id="threshold", min=0.3, max=0.95, step=0.01, value=config.dream.coherence_threshold),
                            html.Label("Noise amplitude", className="control-label"),
                            dcc.Slider(id="noise", min=0.01, max=0.6, step=0.01, value=config.dream.noise_amplitude),
                            html.Label("Branch count", className="control-label"),
                            dcc.Slider(id="branches", min=1, max=5, step=1, value=config.dream.branch_count),
                            html.Div(
                                className="button-row",
                                children=[
                                    html.Button("Start", id="start-btn", className="control-btn"),
                                    html.Button("Pause", id="pause-btn", className="control-btn"),
                                    html.Button("Resume", id="resume-btn", className="control-btn"),
                                    html.Button("Stop", id="stop-btn", className="control-btn stop-btn"),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="panel content-panel",
                        children=[
                            html.H4("Preview (provisional recall)", id="preview-title", className="panel-title"),
                            html.Div(id="preview", className="text-block preview-block"),
                            html.H4("Committed (crystallized recall)", className="panel-title"),
                            html.Div(id="committed", className="text-block"),
                            html.Pre(id="metrics", className="metrics-block"),
                            dcc.Graph(id="latent-graph", className="latent-graph"),
                            html.Div(
                                className="run-controls-row",
                                children=[
                                    html.Div(
                                        className="run-filter-wrap",
                                        children=[
                                            html.Label("Runs to display", className="control-label"),
                                            dcc.Dropdown(
                                                id="run-filter",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="Select one or more runs",
                                                className="control-field",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="run-actions",
                                        children=[
                                            html.Button("Select all", id="select-all-runs-btn", className="control-btn"),
                                            html.Button("Clear selection", id="clear-selection-btn", className="control-btn"),
                                            html.Button("Clear atlas", id="clear-atlas-btn", className="control-btn stop-btn"),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Checklist(
                                id="follow-latest",
                                options=[{"label": "Auto-follow latest step", "value": "follow"}],
                                value=["follow"],
                                className="follow-latest-toggle",
                            ),
                            html.Label("Scrub by step", className="control-label"),
                            dcc.Slider(id="scrub-step", min=0, max=1, step=1, value=0),
                            html.Div(id="step-info", className="meta-line"),
                            html.Div(id="selected-state", className="selected-state"),
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="ticker", interval=int(1000 / max(config.dream.tick_hz, 0.5)), n_intervals=0),
            dcc.Store(id="control-ack"),
            dcc.Store(id="theme-store"),
            dcc.Store(id="run-ui-state", data={"latest_run_id": None}),
            dcc.Interval(id="theme-probe", interval=100, max_intervals=1, n_intervals=0),
        ],
    )

    app.clientside_callback(
        """
        function(_) {
            if (typeof window === 'undefined' || !window.matchMedia) {
                return 'light';
            }
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        """,
        Output("theme-store", "data"),
        Input("theme-probe", "n_intervals"),
    )

    @app.callback(Output("app-root", "className"), Input("theme-store", "data"))
    def apply_theme(theme):
        selected = "dark" if theme == "dark" else "light"
        return f"app-shell theme-{selected}"

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

    @app.callback(
        Output("run-filter", "options"),
        Output("run-filter", "value"),
        Output("run-ui-state", "data"),
        Output("scrub-step", "max"),
        Output("scrub-step", "value"),
        Input("ticker", "n_intervals"),
        Input("start-btn", "n_clicks"),
        Input("select-all-runs-btn", "n_clicks"),
        Input("clear-selection-btn", "n_clicks"),
        Input("clear-atlas-btn", "n_clicks"),
        Input("run-filter", "value"),
        Input("follow-latest", "value"),
        State("run-ui-state", "data"),
        State("scrub-step", "value"),
        prevent_initial_call=False,
    )
    def sync_run_controls(_, __, select_all, clear_selection, clear_atlas, current_filter, follow_latest, ui_state, scrub_step):
        trigger = ctx.triggered_id
        if trigger == "clear-atlas-btn":
            controller.atlas.clear()
            controller.tick_history = []
            controller.state.step_idx = 0
            controller.state.latest_tick = None
            atlas_store.save(controller.atlas)
            return [], [], {"latest_run_id": None}, 1, 0

        frame = controller.atlas.to_frame()
        if frame.empty:
            return [], [], {"latest_run_id": None}, 1, 0

        run_meta = frame[["run_id", "run_label"]].drop_duplicates().tail(40)
        options = [{"label": f"{row.run_label} · {row.run_id[:8]}", "value": row.run_id} for row in run_meta.itertuples(index=False)]
        available = {o["value"] for o in options}
        selected = [rid for rid in (current_filter or []) if rid in available]
        latest_run_id = str(frame["run_id"].iloc[-1])
        previous_latest = (ui_state or {}).get("latest_run_id")

        if trigger == "select-all-runs-btn":
            selected = [o["value"] for o in options]
        elif trigger == "clear-selection-btn":
            selected = []
        elif not selected:
            selected = [o["value"] for o in options]

        if latest_run_id != previous_latest and latest_run_id not in selected:
            selected = selected + [latest_run_id]

        filtered = frame[frame["run_id"].isin(selected)] if selected else frame.iloc[0:0]
        scrub_max = int(filtered["step_idx"].max()) if not filtered.empty else 1
        current_scrub = int(scrub_step or 0)
        auto_follow = "follow" in (follow_latest or [])
        if auto_follow or latest_run_id != previous_latest:
            next_scrub = scrub_max
        else:
            next_scrub = max(0, min(current_scrub, scrub_max))

        return options, selected, {"latest_run_id": latest_run_id}, scrub_max, next_scrub

    @app.callback(Output("status", "children"), Input("ticker", "n_intervals"))
    def refresh_status(_):
        detail = f" ({controller.state.status_detail})" if controller.state.status_detail else ""
        return f"Status: {controller.state.status}{detail}"

    @app.callback(Output("capability", "children"), Input("ticker", "n_intervals"))
    def refresh_capability(_):
        caps = runtime.capabilities()
        return f"Mode: {caps.active_mode.value} | backend: {caps.backend_name} | capture_sites: {caps.capture_sites or ['none']}"

    @app.callback(Output("preview-title", "children"), Input("ticker", "n_intervals"))
    def refresh_preview_title(_):
        caps = runtime.capabilities()
        if caps.supports_instrumented_latents and caps.supports_true_latent_readout:
            return "Preview (true latent readout)"
        return "Preview (prompt-conditioned approximation)"

    @app.callback(
        Output("scrub-step", "value"),
        Input("latent-graph", "clickData"),
        State("scrub-step", "value"),
        prevent_initial_call=True,
    )
    def select_from_graph(click_data, current):
        if not click_data or not click_data.get("points"):
            return current
        point = click_data["points"][0]
        step = point.get("customdata", [None])[0] if point.get("customdata") else point.get("x")
        try:
            return int(step)
        except Exception:
            return current

    @app.callback(
        Output("preview", "children"),
        Output("committed", "children"),
        Output("metrics", "children"),
        Output("latent-graph", "figure"),
        Output("step-info", "children"),
        Output("selected-state", "children"),
        Input("ticker", "n_intervals"),
        Input("scrub-step", "value"),
        Input("run-filter", "value"),
    )
    def refresh_stream(_, scrub_step, run_filter):
        frame = controller.atlas.to_frame()
        if frame.empty:
            fig = px.scatter(x=[0], y=[0], title="No latent states yet")
            fig.update_layout(template="plotly_dark", uirevision="latent-atlas")
            return "", "", "No ticks yet.", fig, "No step selected.", ""

        if run_filter:
            frame = frame[frame["run_id"].isin(run_filter)]
        if frame.empty:
            fig = px.scatter(x=[0], y=[0], title="No runs selected")
            fig.update_layout(template="plotly_dark", uirevision="latent-atlas")
            return "", "", "No ticks for selected runs.", fig, "No step selected.", ""

        max_step = int(frame["step_idx"].max())
        selected_step = max(0, min(int(scrub_step if scrub_step is not None else max_step), max_step))
        selected = frame[frame["step_idx"] == selected_step].tail(1)
        if selected.empty:
            selected = frame[frame["step_idx"] == max_step].tail(1)
            selected_step = max_step
        tick = controller.state.latest_tick

        traj = frame.sort_values("step_idx")
        fig = px.scatter(
            traj,
            x="x",
            y="y",
            color="coherence",
            symbol="run_label",
            custom_data=["step_idx"],
            hover_data=["state_id", "run_label", "run_id", "basin", "step_idx", "preview", "committed", "density", "entropy"],
        )
        for run_id, run_traj in traj.groupby("run_id"):
            fig.add_scatter(
                x=run_traj["x"],
                y=run_traj["y"],
                mode="lines",
                line={"width": 2},
                name=f"trajectory {run_id[:8]}",
                showlegend=False,
            )
        if not selected.empty:
            fig.add_scatter(x=selected["x"], y=selected["y"], mode="markers", marker={"size": 16, "color": "#ff4d6d"}, name="selected")
        fig.update_layout(
            template="plotly_dark",
            title="Latent-state atlas (projection only; control stays high-dimensional)",
            uirevision="latent-atlas",
        )

        selected_txt = ""
        step_info = f"Selected step: {selected_step} / {max_step} | runs: {frame['run_id'].nunique()}"
        preview_text = ""
        committed_text = ""
        metrics = "No ticks yet."
        if not selected.empty:
            row = selected.iloc[0]
            selected_txt = f"selected state={row['state_id']} run={row['run_id'][:8]} phase={row['phase']} basin={row['basin']}"
            preview_text = str(row.get("preview") or "")
            committed_text = str(row.get("committed") or "")
            metrics = (
                f"mode={tick.mode if tick else 'unknown'}\n"
                f"phase={row['phase']}\n"
                f"coherence={float(row['coherence']):.3f}\n"
                f"entropy={float(row['entropy']):.3f}\n"
                f"density={float(row['density']):.3f}"
            )

        if tick is not None and selected_step == max_step:
            metrics = (
                f"mode={tick.mode}\n"
                f"phase={tick.phase}\n"
                f"coherence={tick.coherence:.3f}\n"
                f"entropy={tick.entropy:.3f}\n"
                f"density={tick.local_density:.3f}\n"
                f"stability={tick.token_stability:.3f}"
            )

        return preview_text, committed_text, metrics, fig, step_info, selected_txt
    return app


def run_ui(config: AppConfig, host: str = "127.0.0.1", port: int = 8050) -> None:
    app = create_dash_app(config)
    app.run(host=host, port=port, debug=False)
