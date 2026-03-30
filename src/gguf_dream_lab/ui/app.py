from __future__ import annotations

import ast
import json
from pathlib import Path
from time import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.runtime.capability_contracts import RuntimeBehaviorSnapshot, validate_mode_contract
from gguf_dream_lab.backend.runtime.manager import get_runtime_manager
from gguf_dream_lab.config.models import AppConfig, Basin, BranchSelectionPolicy
from gguf_dream_lab.storage.session_store import SessionStore


def _parse_candidate_scores(candidate_scores: object) -> list[dict]:
    if candidate_scores is None:
        return []
    if isinstance(candidate_scores, list):
        return candidate_scores
    raw = str(candidate_scores).strip()
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, list) else []
    except json.JSONDecodeError:
        try:
            loaded = ast.literal_eval(raw)
            return loaded if isinstance(loaded, list) else []
        except (ValueError, SyntaxError):
            return []


def _compute_branch_divergence(candidate_scores: object) -> float:
    parsed = _parse_candidate_scores(candidate_scores)
    scores = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    if len(scores) < 2:
        return 0.0
    return float(pd.Series(scores, dtype="float64").std(ddof=0))


def _summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["run_id", "mode", "basin"], dropna=False)
        .agg(
            steps=("step_idx", "count"),
            coherence_mean=("coherence", "mean"),
            density_mean=("density", "mean"),
            smoothness_mean=("smoothness", "mean"),
            stability_mean=("stability", "mean"),
            branch_divergence_mean=("branch_divergence", "mean"),
            phase_duration_mean=("phase_duration", "mean"),
            phase_duration_max=("phase_duration", "max"),
        )
        .reset_index()
        .sort_values(["run_id", "mode", "basin"])
    )
    return grouped


def create_dash_app(config: AppConfig) -> Dash:
    runtime = get_runtime_manager().get_runtime(config.runtime)
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
                            html.Label("Branch policy", className="control-label"),
                            dcc.Dropdown(
                                id="branch-policy",
                                options=[{"label": p.value, "value": p.value} for p in BranchSelectionPolicy],
                                value=config.dream.branch_selection_policy.value,
                                className="control-field",
                                clearable=False,
                            ),
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
                            html.Div(
                                className="run-controls-row",
                                children=[
                                    html.Div(
                                        className="run-filter-wrap",
                                        children=[
                                            html.Label("Filter mode", className="control-label"),
                                            dcc.Dropdown(
                                                id="mode-filter",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="All modes",
                                                className="control-field",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="run-filter-wrap",
                                        children=[
                                            html.Label("Filter basin", className="control-label"),
                                            dcc.Dropdown(
                                                id="basin-filter",
                                                options=[],
                                                value=[],
                                                multi=True,
                                                placeholder="All basins",
                                                className="control-field",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(id="latent-graph", className="latent-graph"),
                            dcc.Graph(id="rolling-metrics-graph", className="latent-graph"),
                            html.Div(
                                className="run-controls-row",
                                children=[
                                    html.Button("Export summary CSV", id="export-summary-btn", className="control-btn"),
                                    dcc.Download(id="summary-download"),
                                ],
                            ),
                            html.Pre(id="summary-stats", className="metrics-block"),
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
                            html.H4("Lineage replay", className="panel-title"),
                            html.Label("Lineage path", className="control-label"),
                            dcc.Dropdown(
                                id="lineage-path",
                                options=[],
                                value=None,
                                placeholder="Select a saved lineage path",
                                className="control-field",
                            ),
                            html.Div(
                                className="button-row",
                                children=[
                                    html.Button("Replay", id="replay-start-btn", className="control-btn"),
                                    html.Button("Stop replay", id="replay-stop-btn", className="control-btn"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="ticker", interval=int(1000 / max(config.dream.tick_hz, 0.5)), n_intervals=0),
            dcc.Store(id="control-ack"),
            dcc.Store(id="theme-store"),
            dcc.Store(id="run-ui-state", data={"latest_run_id": None}),
            dcc.Store(id="timeline-cache", data={"run_id": None, "ticks": [], "paths": []}),
            dcc.Interval(id="theme-probe", interval=100, max_intervals=1, n_intervals=0),
            dcc.Interval(id="replay-timer", interval=400, n_intervals=0, disabled=True),
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
        State("branch-policy", "value"),
        prevent_initial_call=True,
    )
    def controls(start, pause, resume, stop, prompt, basin, threshold, tick_hz, noise, branches, branch_policy):
        trigger = ctx.triggered_id
        if trigger == "start-btn":
            config.dream.prompt = prompt or ""
            config.dream.basin = Basin(basin)
            config.dream.coherence_threshold = float(threshold)
            config.dream.tick_hz = float(tick_hz)
            config.dream.noise_amplitude = float(noise)
            config.dream.branch_count = int(branches)
            config.dream.branch_selection_policy = BranchSelectionPolicy(branch_policy)
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
        Input("follow-latest", "value"),
        State("run-filter", "value"),
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

    @app.callback(
        Output("timeline-cache", "data"),
        Output("lineage-path", "options"),
        Output("lineage-path", "value"),
        Input("run-filter", "value"),
        prevent_initial_call=False,
    )
    def load_timeline(run_filter):
        if not run_filter:
            return {"run_id": None, "ticks": [], "paths": []}, [], None
        run_id = run_filter[-1]
        timeline = session_store.load_timeline(run_id)
        options = [{"label": path["label"], "value": path["id"]} for path in timeline.get("paths", [])]
        selected = options[0]["value"] if options else None
        return timeline, options, selected

    @app.callback(
        Output("replay-timer", "disabled"),
        Input("replay-start-btn", "n_clicks"),
        Input("replay-stop-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_replay_mode(_, __):
        return ctx.triggered_id != "replay-start-btn"

    @app.callback(
        Output("scrub-step", "value", allow_duplicate=True),
        Input("replay-timer", "n_intervals"),
        State("replay-timer", "disabled"),
        State("timeline-cache", "data"),
        State("lineage-path", "value"),
        State("scrub-step", "value"),
        prevent_initial_call=True,
    )
    def replay_tick(_, replay_disabled, timeline, lineage_path_id, scrub_step):
        if replay_disabled:
            return scrub_step
        timeline = timeline or {}
        paths = timeline.get("paths", [])
        if not paths:
            return scrub_step
        selected = next((p for p in paths if p.get("id") == lineage_path_id), paths[0])
        state_ids = set(selected.get("state_ids", []))
        path_steps = sorted(
            int(t.get("step_idx", -1))
            for t in timeline.get("ticks", [])
            if str(t.get("state_id")) in state_ids and int(t.get("step_idx", -1)) >= 0
        )
        if not path_steps:
            return scrub_step
        current_step = int(scrub_step or path_steps[0])
        for step in path_steps:
            if step > current_step:
                return step
        return path_steps[0]

    @app.callback(Output("status", "children"), Input("ticker", "n_intervals"))
    def refresh_status(_):
        detail = f" ({controller.state.status_detail})" if controller.state.status_detail else ""
        return f"Status: {controller.state.status}{detail}"

    @app.callback(Output("capability", "children"), Input("ticker", "n_intervals"))
    def refresh_capability(_):
        caps = runtime.capabilities()
        behavior_snapshot = RuntimeBehaviorSnapshot(
            capture=caps.supports_capture,
            reinject=caps.supports_reinject,
            decode_provenance=caps.supports_decode_provenance,
            control_authority=caps.supports_control_authority,
        )
        contract = validate_mode_contract(caps.active_mode, behavior_snapshot)
        suffix = ""
        if contract.downgraded:
            suffix = f" | contract_downgrade_missing={contract.missing_behaviors}"
        verification_suffix = ""
        if caps.instrumentation_downgrade_reasons:
            verification_suffix = (
                f" | verification_source={caps.instrumentation_verification_source}"
                f" | verification_downgrade_reasons={caps.instrumentation_downgrade_reasons}"
            )
        decode_lane = (
            "true_latent_readout"
            if caps.supports_instrumented_latents and caps.supports_true_latent_readout
            else "approximate_prompt_synthesis"
        )
        return (
            f"Mode: {contract.effective_mode.value} | backend: {caps.backend_name} "
            f"| behaviors=capture:{caps.supports_capture},reinject:{caps.supports_reinject},"
            f"decode_provenance:{caps.supports_decode_provenance},control_authority:{caps.supports_control_authority} "
            f"| capture_sites: {caps.capture_sites or ['none']}{suffix}{verification_suffix}"
            f"| decode_lane:{decode_lane} | capture_sites: {caps.capture_sites or ['none']}{suffix}"
        )

    @app.callback(Output("preview-title", "children"), Input("ticker", "n_intervals"))
    def refresh_preview_title(_):
        caps = runtime.capabilities()
        if caps.supports_instrumented_latents and caps.supports_true_latent_readout:
            return "Preview (true latent readout)"
        return "Preview (approximate prompt synthesis)"

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
        Output("rolling-metrics-graph", "figure"),
        Output("mode-filter", "options"),
        Output("mode-filter", "value"),
        Output("basin-filter", "options"),
        Output("basin-filter", "value"),
        Output("summary-stats", "children"),
        Input("ticker", "n_intervals"),
        Input("scrub-step", "value"),
        Input("run-filter", "value"),
        Input("mode-filter", "value"),
        Input("basin-filter", "value"),
        Input("timeline-cache", "data"),
        Input("lineage-path", "value"),
    )
    def refresh_stream(_, scrub_step, run_filter, mode_filter, basin_filter, timeline, lineage_path_id):
        frame = controller.atlas.to_frame()
        if frame.empty:
            fig = px.scatter(x=[0], y=[0], title="No latent states yet")
            fig.update_layout(template="plotly_dark", uirevision="latent-atlas")
            metrics_fig = go.Figure()
            metrics_fig.update_layout(template="plotly_dark", title="Rolling metrics")
            return "", "", "No ticks yet.", fig, "No step selected.", "", metrics_fig, [], [], [], [], "No summary stats yet."

        if run_filter:
            frame = frame[frame["run_id"].isin(run_filter)]
        mode_options = [{"label": str(mode), "value": str(mode)} for mode in sorted(frame.get("mode", pd.Series(dtype=str)).dropna().unique())]
        basin_options = [{"label": str(basin), "value": str(basin)} for basin in sorted(frame["basin"].dropna().unique())]
        available_modes = {opt["value"] for opt in mode_options}
        available_basins = {opt["value"] for opt in basin_options}
        selected_modes = [m for m in (mode_filter or []) if m in available_modes]
        selected_basins = [b for b in (basin_filter or []) if b in available_basins]
        if selected_modes and "mode" in frame.columns:
            frame = frame[frame["mode"].isin(selected_modes)]
        if selected_basins:
            frame = frame[frame["basin"].isin(selected_basins)]
        if frame.empty:
            fig = px.scatter(x=[0], y=[0], title="No runs selected")
            fig.update_layout(template="plotly_dark", uirevision="latent-atlas")
            metrics_fig = go.Figure()
            metrics_fig.update_layout(template="plotly_dark", title="Rolling metrics")
            return (
                "",
                "",
                "No ticks for selected runs.",
                fig,
                "No step selected.",
                "",
                metrics_fig,
                mode_options,
                selected_modes,
                basin_options,
                selected_basins,
                "No summary stats for current filters.",
            )

        metric_frames = []
        for run_id in sorted(frame["run_id"].unique()):
            if run_id == controller.state.run_id and controller.tick_history:
                ticks_df = pd.DataFrame([tick.__dict__ for tick in controller.tick_history])
            else:
                ticks_df = session_store.load_ticks(run_id)
            if ticks_df.empty:
                continue
            ticks_df = ticks_df.copy()
            ticks_df["run_id"] = run_id
            metric_frames.append(ticks_df)

        metrics_frame = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
        if not metrics_frame.empty:
            metrics_frame = metrics_frame.sort_values(["run_id", "step_idx"]).reset_index(drop=True)
            metrics_frame["density"] = metrics_frame.get("local_density", pd.Series(dtype=float))
            metrics_frame["stability"] = metrics_frame.get("token_stability", pd.Series(dtype=float))
            candidate_series = metrics_frame.get("candidate_scores", pd.Series(["[]"] * len(metrics_frame)))
            metrics_frame["branch_divergence"] = candidate_series.apply(_compute_branch_divergence)
            phase_change = metrics_frame.groupby("run_id")["phase"].transform(lambda s: s.ne(s.shift()).astype(int))
            metrics_frame["phase_segment"] = phase_change.groupby(metrics_frame["run_id"]).cumsum()
            metrics_frame["phase_duration"] = metrics_frame.groupby(["run_id", "phase_segment"]).cumcount() + 1
            if selected_modes:
                metrics_frame = metrics_frame[metrics_frame["mode"].isin(selected_modes)]
            if selected_basins:
                run_basin = frame[["run_id", "basin"]].drop_duplicates()
                metrics_frame = metrics_frame.merge(run_basin, on="run_id", how="left")
                metrics_frame = metrics_frame[metrics_frame["basin"].isin(selected_basins)]
            else:
                metrics_frame = metrics_frame.merge(frame[["run_id", "basin"]].drop_duplicates(), on="run_id", how="left")

        max_step = int(frame["step_idx"].max())
        selected_step = max(0, min(int(scrub_step if scrub_step is not None else max_step), max_step))
        selected = frame[frame["step_idx"] == selected_step].tail(1)
        if selected.empty:
            selected = frame[frame["step_idx"] == max_step].tail(1)
            selected_step = max_step
        tick = controller.state.latest_tick
        timeline = timeline or {}
        timeline_ticks = timeline.get("ticks", [])
        lineage_state_ids = set()
        for path in timeline.get("paths", []):
            if path.get("id") == lineage_path_id:
                lineage_state_ids = set(path.get("state_ids", []))
                break

        traj = frame.sort_values("step_idx")
        fig = px.scatter(
            traj,
            x="x",
            y="y",
            color="coherence",
            symbol="run_label",
            custom_data=["step_idx"],
            hover_data=[
                "state_id",
                "run_label",
                "run_id",
                "basin",
                "step_idx",
                "latent_source",
                "preview",
                "committed",
                "density",
                "entropy",
                "basin_sample_count",
                "basin_force_magnitude",
                "basin_prior_spread",
            ],
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
            selected_txt = (
                f"selected state={row['state_id']} run={row['run_id'][:8]} "
                f"phase={row['phase']} basin={row['basin']} latent_source={row.get('latent_source', 'unknown')}"
            )
            preview_text = str(row.get("preview") or "")
            committed_text = str(row.get("committed") or "")
            metrics = (
                f"mode={tick.mode if tick else 'unknown'}\n"
                f"phase={row['phase']}\n"
                f"commit_source={tick.commit_source if tick else 'unknown'}\n"
                f"latent_source={row.get('latent_source', 'unknown')}\n"
                f"coherence={float(row['coherence']):.3f}\n"
                f"entropy={float(row['entropy']):.3f}\n"
                f"density={float(row['density']):.3f}\n"
                f"basin_samples={int(row.get('basin_sample_count', 0))}\n"
                f"basin_force={float(row.get('basin_force_magnitude', 0.0)):.3f}\n"
                f"basin_spread={float(row.get('basin_prior_spread', 0.0)):.3f}"
            )

        if tick is not None and selected_step == max_step:
            metrics = (
                f"mode={tick.mode}\n"
                f"phase={tick.phase}\n"
                f"commit_source={tick.commit_source}\n"
                f"latent_source={tick.latent_source}\n"
                f"coherence={tick.coherence:.3f}\n"
                f"coh_entropy_contrib={tick.coherence_component_entropy:.3f}\n"
                f"coh_density_contrib={tick.coherence_component_density:.3f}\n"
                f"coh_stability_contrib={tick.coherence_component_token_stability:.3f}\n"
                f"coh_smoothness_contrib={tick.coherence_component_smoothness:.3f}\n"
                f"coh_branch_contrib={tick.coherence_component_branch_agreement:.3f}\n"
                f"coh_similarity_contrib={tick.coherence_component_known_state_similarity:.3f}\n"
                f"entropy={tick.entropy:.3f}\n"
                f"density={tick.local_density:.3f}\n"
                f"stability={tick.token_stability:.3f}\n"
                f"basin_samples={tick.basin_sample_count}\n"
                f"basin_mean_coherence={tick.basin_mean_coherence:.3f}\n"
                f"basin_mean_density={tick.basin_mean_density:.3f}\n"
                f"basin_force={tick.basin_force_magnitude:.3f}\n"
                f"basin_spread={tick.basin_prior_spread:.3f}"
            )

        if timeline_ticks:
            replay_row = next((row for row in timeline_ticks if int(row.get("step_idx", -1)) == selected_step), None)
            if replay_row and (not lineage_state_ids or str(replay_row.get("state_id")) in lineage_state_ids):
                preview_text = str(replay_row.get("preview_text") or preview_text)
                committed_text = str(replay_row.get("committed_text") or committed_text)
                metrics = (
                    f"mode={replay_row.get('mode', 'unknown')}\n"
                    f"phase={replay_row.get('phase', 'unknown')}\n"
                    f"coh_entropy_contrib={float(replay_row.get('coherence_component_entropy', 0.0)):.3f}\n"
                    f"coh_density_contrib={float(replay_row.get('coherence_component_density', 0.0)):.3f}\n"
                    f"coh_stability_contrib={float(replay_row.get('coherence_component_token_stability', 0.0)):.3f}\n"
                    f"coh_smoothness_contrib={float(replay_row.get('coherence_component_smoothness', 0.0)):.3f}\n"
                    f"coh_branch_contrib={float(replay_row.get('coherence_component_branch_agreement', 0.0)):.3f}\n"
                    f"coh_similarity_contrib={float(replay_row.get('coherence_component_known_state_similarity', 0.0)):.3f}\n"
                    f"parent_state={replay_row.get('parent_state_id', '')}\n"
                    f"branch_id={replay_row.get('branch_id', '')}\n"
                    f"branch_score={float(replay_row.get('branch_score', 0.0)):.3f}\n"
                    f"basin_samples={int(replay_row.get('basin_sample_count', 0))}\n"
                    f"basin_force={float(replay_row.get('basin_force_magnitude', 0.0)):.3f}\n"
                    f"basin_spread={float(replay_row.get('basin_prior_spread', 0.0)):.3f}\n"
                    f"candidate_scores={replay_row.get('candidate_scores', '[]')}\n"
                    f"rejected_candidates={replay_row.get('rejected_candidates', '[]')}\n"
                    f"selection_trace={replay_row.get('selected_branch_trace', '{}')}\n"
                    f"decode_path_diagnostics={replay_row.get('decode_provenance', 'unknown')}"
                )

        rolling = go.Figure()
        if not metrics_frame.empty:
            rolling_window = 5
            for run_id, run_df in metrics_frame.groupby("run_id"):
                run_df = run_df.sort_values("step_idx")
                smoothed = run_df.copy()
                for col in ["coherence", "density", "smoothness", "stability", "branch_divergence", "phase_duration"]:
                    if col in smoothed.columns:
                        smoothed[col] = smoothed[col].rolling(window=rolling_window, min_periods=1).mean()
                label = str(run_id)[:8]
                for metric_name in ["coherence", "density", "smoothness", "stability", "branch_divergence", "phase_duration"]:
                    if metric_name not in smoothed.columns:
                        continue
                    rolling.add_trace(
                        go.Scatter(
                            x=smoothed["step_idx"],
                            y=smoothed[metric_name],
                            mode="lines",
                            name=f"{metric_name} · {label}",
                            hovertemplate=f"run={label}<br>step=%{{x}}<br>{metric_name}=%{{y:.3f}}<extra></extra>",
                        )
                    )
        rolling.update_layout(template="plotly_dark", title="Rolling metrics (window=5)", uirevision="rolling-metrics")

        summary_table = _summarize_metrics(metrics_frame)
        summary_txt = summary_table.to_string(index=False, float_format=lambda v: f"{v:.4f}") if not summary_table.empty else "No summary stats yet."
        return (
            preview_text,
            committed_text,
            metrics,
            fig,
            step_info,
            selected_txt,
            rolling,
            mode_options,
            selected_modes,
            basin_options,
            selected_basins,
            summary_txt,
        )

    @app.callback(
        Output("summary-download", "data"),
        Input("export-summary-btn", "n_clicks"),
        State("run-filter", "value"),
        State("mode-filter", "value"),
        State("basin-filter", "value"),
        prevent_initial_call=True,
    )
    def export_summary(_, run_filter, mode_filter, basin_filter):
        frame = controller.atlas.to_frame()
        if frame.empty:
            return dcc.send_string("run_id,mode,basin,steps\n", "summary_stats.csv")
        if run_filter:
            frame = frame[frame["run_id"].isin(run_filter)]
        run_basin = frame[["run_id", "basin"]].drop_duplicates()
        metric_frames = []
        for run_id in sorted(frame["run_id"].unique()):
            if run_id == controller.state.run_id and controller.tick_history:
                ticks_df = pd.DataFrame([tick.__dict__ for tick in controller.tick_history])
            else:
                ticks_df = session_store.load_ticks(run_id)
            if ticks_df.empty:
                continue
            ticks_df["run_id"] = run_id
            metric_frames.append(ticks_df)
        metrics_frame = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
        if metrics_frame.empty:
            return dcc.send_string("run_id,mode,basin,steps\n", "summary_stats.csv")
        metrics_frame["density"] = metrics_frame.get("local_density", pd.Series(dtype=float))
        metrics_frame["stability"] = metrics_frame.get("token_stability", pd.Series(dtype=float))
        candidate_series = metrics_frame.get("candidate_scores", pd.Series(["[]"] * len(metrics_frame)))
        metrics_frame["branch_divergence"] = candidate_series.apply(_compute_branch_divergence)
        phase_change = metrics_frame.groupby("run_id")["phase"].transform(lambda s: s.ne(s.shift()).astype(int))
        metrics_frame["phase_segment"] = phase_change.groupby(metrics_frame["run_id"]).cumsum()
        metrics_frame["phase_duration"] = metrics_frame.groupby(["run_id", "phase_segment"]).cumcount() + 1
        metrics_frame = metrics_frame.merge(run_basin, on="run_id", how="left")
        if mode_filter:
            metrics_frame = metrics_frame[metrics_frame["mode"].isin(mode_filter)]
        if basin_filter:
            metrics_frame = metrics_frame[metrics_frame["basin"].isin(basin_filter)]
        summary = _summarize_metrics(metrics_frame)
        return dcc.send_string(summary.to_csv(index=False), "summary_stats.csv")
    return app


def run_ui(config: AppConfig, host: str = "127.0.0.1", port: int = 8050) -> None:
    runtime_manager = get_runtime_manager()
    app = create_dash_app(config)
    try:
        app.run(host=host, port=port, debug=False)
    finally:
        runtime_manager.teardown(config.runtime)
