"""Rich terminal reporting for caliscope's scripting API.

Provides progress bars for long-running operations and colored
quality reports for intrinsic/extrinsic calibration results.

This module imports Rich unconditionally. Core calibration functions
do NOT import this module -- they accept an optional ProgressCallback
protocol parameter instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TaskID
from rich.table import Table
from rich import box

from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput, IntrinsicCalibrationReport
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints


# --- Protocol ---


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for reporting extraction progress.

    Core functions call these methods without knowing whether the
    implementation is Rich, logging, or a no-op.
    """

    def on_video_start(self, cam_id: int, total_frames: int) -> None:
        """Called when extraction begins for a camera's video."""
        ...

    def on_frame(self, cam_id: int, frame_index: int, n_points: int) -> None:
        """Called after each frame is processed."""
        ...

    def on_video_complete(self, cam_id: int) -> None:
        """Called when extraction finishes for a camera's video."""
        ...


# --- Progress ---


class RichProgressBar:
    """Rich progress bar implementing ProgressCallback.

    Usage:
        progress = RichProgressBar()
        points = extract_image_points(videos, tracker, progress=progress)

    Or as context manager for explicit lifecycle:
        with RichProgressBar() as progress:
            points = extract_image_points(videos, tracker, progress=progress)
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._tasks: dict[int, TaskID] = {}
        self._started = False

    def __enter__(self) -> RichProgressBar:
        self._progress.start()
        self._started = True
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.stop()
        self._started = False

    def _ensure_started(self) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

    def on_video_start(self, cam_id: int, total_frames: int) -> None:
        self._ensure_started()
        task_id = self._progress.add_task(f"  cam {cam_id}", total=total_frames)
        self._tasks[cam_id] = task_id

    def on_frame(self, cam_id: int, frame_index: int, n_points: int) -> None:
        if cam_id in self._tasks:
            self._progress.update(self._tasks[cam_id], completed=frame_index + 1)

    def on_video_complete(self, cam_id: int) -> None:
        if cam_id in self._tasks:
            task = self._progress.tasks[self._tasks[cam_id]]
            self._progress.update(self._tasks[cam_id], completed=task.total)


# --- Internal helpers ---


def _quality_badge(value: float, thresholds: list[tuple[float, str, str]]) -> str:
    """Return a Rich-markup colored badge based on value and thresholds.

    Args:
        value: The metric value to evaluate
        thresholds: List of (threshold, label, color) tuples, checked in order.
            The first threshold where value < threshold is used.
            The last entry is used as the fallback (>=).

    Example:
        _quality_badge(0.3, [(0.5, "EXCELLENT", "green"), (1.0, "GOOD", "yellow"), (float("inf"), "POOR", "red")])
        -> "[green]EXCELLENT[/green]"
    """
    for threshold, label, color in thresholds:
        if value < threshold:
            return f"[{color}]{label}[/{color}]"
    # Fallback to last entry
    _, label, color = thresholds[-1]
    return f"[{color}]{label}[/{color}]"


def _sparkline(values: list[float], width: int = 60) -> str:
    """Render a sparkline from a sequence of values.

    Uses 8 Unicode block characters for vertical resolution.
    Values are scaled relative to the maximum in the sequence.
    """
    if not values:
        return ""

    BLOCKS = " ▁▂▃▄▅▆▇█"

    max_val = max(values)
    if max_val == 0:
        return "▁" * min(len(values), width)

    # Truncate if needed
    truncated = len(values) > width
    display_values = values[:width]

    chars = []
    for v in display_values:
        # Scale to 1-8 range (0 maps to lowest visible block)
        idx = int(round(v / max_val * 8))
        idx = max(1, min(8, idx))  # clamp to 1..8 (always show at least ▁)
        chars.append(BLOCKS[idx])

    result = "".join(chars)
    if truncated:
        result += "…"
    return result


# --- Intrinsic Report ---

_RMSE_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.5, "EXCELLENT", "green"),
    (1.0, "GOOD", "yellow"),
    (float("inf"), "POOR", "red"),
]

_COVERAGE_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.60, "LOW", "red"),
    (0.80, "OK", "yellow"),
    (float("inf"), "GOOD", "green"),
]

_EDGE_COVERAGE_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.50, "LOW", "red"),
    (0.75, "OK", "yellow"),
    (float("inf"), "GOOD", "green"),
]

_CORNER_COVERAGE_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.25, "LOW", "red"),
    (0.50, "OK", "yellow"),
    (float("inf"), "GOOD", "green"),
]


def print_intrinsic_report(output: IntrinsicCalibrationOutput, *, console: Console | None = None) -> None:
    """Print intrinsic calibration quality report to terminal."""
    c = console or Console()
    report = output.report
    camera = output.camera

    c.print()
    c.print(f"[bold]Intrinsic Calibration — cam {camera.cam_id}[/bold]")
    c.print("─" * 40)

    # RMSE
    rmse_badge = _quality_badge(report.rmse, _RMSE_THRESHOLDS)
    c.print(f"  Reprojection RMSE:   {report.rmse:.3f} px    {rmse_badge}")
    c.print(f"  Frames used:         {report.frames_used}")
    c.print()

    # Coverage — _quality_badge thresholds check value < threshold.
    # For coverage (higher is better), thresholds are ordered so lower values
    # hit the "bad" buckets first:
    # e.g., coverage=0.88: 0.88 < 0.60? No. 0.88 < 0.80? No. 0.88 < inf? Yes -> GOOD. Correct.
    coverage_badge = _quality_badge(report.coverage_fraction, _COVERAGE_THRESHOLDS)
    edge_badge = _quality_badge(report.edge_coverage_fraction, _EDGE_COVERAGE_THRESHOLDS)
    corner_badge = _quality_badge(report.corner_coverage_fraction, _CORNER_COVERAGE_THRESHOLDS)

    c.print(f"  Coverage:            {report.coverage_fraction:.0%}          {coverage_badge}")
    c.print(f"  Edge coverage:       {report.edge_coverage_fraction:.0%}          {edge_badge}")
    c.print(f"  Corner coverage:     {report.corner_coverage_fraction:.0%}          {corner_badge}")

    if report.orientation_sufficient:
        orient_text = "[green]SUFFICIENT[/green]"
    else:
        orient_text = "[red]INSUFFICIENT[/red]"
    c.print(f"  Orientations:        {report.orientation_count}/8          {orient_text}")
    c.print()

    # Camera matrix
    if camera.matrix is not None:
        fx, fy = camera.matrix[0, 0], camera.matrix[1, 1]
        cx, cy = camera.matrix[0, 2], camera.matrix[1, 2]
        c.print("  Camera Matrix")
        c.print(f"    fx: {fx:.1f}    fy: {fy:.1f}")
        c.print(f"    cx: {cx:.1f}    cy: {cy:.1f}")
        c.print()

    # Distortion
    if camera.distortions is not None:
        coeffs = camera.distortions.ravel()
        if camera.fisheye:
            c.print("  Distortion (fisheye)")
            c.print(f"    k1: {coeffs[0]:.4f}   k2: {coeffs[1]:.4f}   k3: {coeffs[2]:.4f}   k4: {coeffs[3]:.4f}")
        else:
            c.print("  Distortion (standard)")
            c.print(f"    k1: {coeffs[0]:.4f}   k2: {coeffs[1]:.4f}   k3: {coeffs[4]:.4f}")
            c.print(f"    p1: {coeffs[2]:.4f}   p2: {coeffs[3]:.4f}")
    c.print()


# --- Extrinsic Report ---

# Scale thresholds in mm (VolumetricScaleReport properties are already in mm)
_SCALE_THRESHOLDS: list[tuple[float, str, str]] = [
    (3.0, "EXCELLENT", "green"),  # < 3mm
    (5.0, "GOOD", "yellow"),  # 3-5mm
    (float("inf"), "POOR", "red"),  # > 5mm
]

_UNMATCHED_THRESHOLDS: list[tuple[float, str, str]] = [
    (0.05, "", "green"),  # < 5%
    (0.15, "", "yellow"),  # 5-15%
    (float("inf"), "", "red"),  # > 15%
]


def print_extrinsic_report(capture_volume: CaptureVolume, *, console: Console | None = None) -> None:
    """Print extrinsic calibration quality report to terminal."""
    c = console or Console()
    report = capture_volume.reprojection_report
    opt = capture_volume.optimization_status

    c.print()
    c.print("[bold]Extrinsic Calibration Report[/bold]")
    c.print("═" * 50)

    # Optimization section
    c.print()
    c.print("  [bold]Optimization[/bold]")
    if opt is not None:
        status_color = "green" if opt.converged else "red"
        status_text = "CONVERGED" if opt.converged else "NOT CONVERGED"
        c.print(f"    Status:          [{status_color}]{status_text}[/{status_color}] ({opt.termination_reason})")
        c.print(f"    Iterations:      {opt.iterations}")
        c.print(f"    Final cost:      {opt.final_cost:.6f}")
    else:
        c.print("    Status:          [dim]not optimized[/dim]")

    # Reprojection error
    c.print()
    c.print("  [bold]Reprojection Error[/bold]")
    rmse_badge = _quality_badge(report.overall_rmse, _RMSE_THRESHOLDS)
    c.print(f"    Overall RMSE:    {report.overall_rmse:.3f} px     {rmse_badge}")

    unmatched_pct = report.unmatched_rate * 100
    unmatched_color = "red"
    for threshold, _, color in _UNMATCHED_THRESHOLDS:
        if report.unmatched_rate < threshold:
            unmatched_color = color
            break

    c.print(
        f"    Observations:    {report.n_observations_matched:,} matched / "
        f"{report.n_observations_total:,} total "
        f"([{unmatched_color}]{unmatched_pct:.1f}% unmatched[/{unmatched_color}])"
    )
    c.print(f"    3D Points:       {report.n_points:,}")

    # Per-camera breakdown table
    c.print()
    c.print("  [bold]Per-Camera Breakdown[/bold]")
    table = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
    table.add_column("Camera", style="bold")
    table.add_column("Observations", justify="right")
    table.add_column("RMSE (px)", justify="right")

    for cam_id, cam_rmse in sorted(report.by_camera.items()):
        cam_obs = int((report.raw_errors["cam_id"] == cam_id).sum())
        rmse_str = f"[red]{cam_rmse:.3f}[/red]"
        for threshold, _, color in _RMSE_THRESHOLDS:
            if cam_rmse < threshold:
                rmse_str = f"[{color}]{cam_rmse:.3f}[/{color}]"
                break

        table.add_row(f"cam {cam_id}", f"{cam_obs:,}", rmse_str)

    c.print(table)

    # Scale accuracy (only if available)
    scale_report = capture_volume.compute_volumetric_scale_accuracy()
    if scale_report.frame_errors:
        c.print()
        c.print("  [bold]Scale Accuracy (post-alignment)[/bold]")

        # VolumetricScaleReport properties are already in mm
        pooled_rmse_mm = scale_report.pooled_rmse_mm
        median_mm = scale_report.median_rmse_mm
        worst_mm = scale_report.max_rmse_mm
        bias_mm = scale_report.mean_signed_error_mm

        scale_badge = _quality_badge(pooled_rmse_mm, _SCALE_THRESHOLDS)
        c.print(f"    Pooled RMSE:     {pooled_rmse_mm:.2f} mm      {scale_badge}")
        c.print(f"    Median:          {median_mm:.2f} mm")
        c.print(f"    Worst frame:     {worst_mm:.2f} mm")

        bias_sign = "+" if bias_mm >= 0 else ""
        c.print(f"    Bias:           {bias_sign}{bias_mm:.2f} mm")
        c.print(f"    Frames sampled:  {scale_report.n_frames_sampled}")

        # Sparkline of per-frame RMSE (distance_rmse_mm is already in mm)
        frame_rmses = [fe.distance_rmse_mm for fe in scale_report.frame_errors]
        spark = _sparkline(frame_rmses)
        c.print(f"    Sparkline:       {spark}")

    c.print()


# --- Coverage Grid ---


def print_coverage_grid(
    report: IntrinsicCalibrationReport,
    image_points: ImagePoints,
    cam_id: int,
    image_size: tuple[int, int],
    *,
    grid_size: int = 5,
    console: Console | None = None,
) -> None:
    """Print ASCII coverage grid showing corner distribution.

    Requires the original ImagePoints and camera metadata because
    IntrinsicCalibrationReport only stores the fraction, not the
    per-cell counts.
    """
    import numpy as np

    c = console or Console()
    width, height = image_size

    # Filter to selected frames and camera
    df = image_points.df
    mask = (df["cam_id"] == cam_id) & (df["sync_index"].isin(report.selected_frames))
    selected_df = df[mask]

    # Compute grid cell counts
    cell_width = width / grid_size
    cell_height = height / grid_size

    grid = np.zeros((grid_size, grid_size), dtype=int)
    for _, row in selected_df.iterrows():
        col_idx = min(int(row["img_loc_x"] / cell_width), grid_size - 1)
        row_idx = min(int(row["img_loc_y"] / cell_height), grid_size - 1)
        grid[row_idx, col_idx] += 1

    # Print grid as Rich table
    c.print()
    c.print(f"  Corner Distribution ({grid_size}x{grid_size} grid)")
    table = Table(box=box.HEAVY_EDGE, show_header=False, padding=(0, 0))
    for _ in range(grid_size):
        table.add_column(justify="right", width=5)

    for row in range(grid_size):
        cells = []
        for col in range(grid_size):
            count = grid[row, col]
            if count == 0:
                cells.append("[red]    0[/red]")
            else:
                cells.append(f"[green]{count:5d}[/green]")
        table.add_row(*cells)

    c.print(table)
    covered = int(np.count_nonzero(grid))
    total_cells = grid_size * grid_size
    c.print(f"  {covered} of {total_cells} cells covered ({covered / total_cells:.0%})")
    c.print()
