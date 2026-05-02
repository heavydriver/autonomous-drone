"""Generate evaluation graphs and summaries from CSV runtime logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autonomous_drone.models import BoundingBox


@dataclass(frozen=True, slots=True)
class GroundTruthFrame:
    """Single-target ground-truth annotation for one frame."""

    frame_index: int
    present: bool
    bbox: BoundingBox | None
    track_id: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for offline plotting."""

    parser = argparse.ArgumentParser(description="Generate graphs from a follow-run CSV log.")
    parser.add_argument("log_file", help="Path to the per-frame CSV log file.")
    parser.add_argument(
        "--output-dir",
        help="Directory where graphs and summary CSVs will be written.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        help=(
            "Optional annotation CSV with columns "
            "frame_index,present,x1,y1,x2,y2[,track_id]."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for match-based evaluation metrics.",
    )
    parser.add_argument(
        "--center-error-threshold",
        type=float,
        default=0.1,
        help="Normalized center-error threshold used for framing accuracy.",
    )
    return parser.parse_args(argv)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a UTF-8 CSV file into memory."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_related_paths(frame_log_path: Path) -> tuple[Path | None, Path | None, str]:
    """Infer sibling detection and track CSV paths from the frame log name."""

    stem = frame_log_path.stem
    session_name = frame_log_path.parent.name
    if stem == "frames":
        detections_path = frame_log_path.with_name("detections.csv")
        tracks_path = frame_log_path.with_name("tracks.csv")
        return (
            detections_path if detections_path.exists() else None,
            tracks_path if tracks_path.exists() else None,
            session_name,
        )
    if stem.endswith("_frames"):
        session_name = stem[:-7]
        detections_path = frame_log_path.with_name(f"{session_name}_detections.csv")
        tracks_path = frame_log_path.with_name(f"{session_name}_tracks.csv")
        return (
            detections_path if detections_path.exists() else None,
            tracks_path if tracks_path.exists() else None,
            session_name,
        )
    return None, None, session_name


def load_ground_truth_rows(path: Path) -> list[GroundTruthFrame]:
    """Load ground-truth annotations from CSV."""

    frames: list[GroundTruthFrame] = []
    for row in load_csv_rows(path):
        frame_index = _parse_int(row.get("frame_index"))
        if frame_index is None:
            raise ValueError("Ground-truth CSV requires a frame_index column.")
        present = _parse_bool(row.get("present"))
        bbox = _bbox_from_row(row, prefix="")
        if present is None:
            present = bbox is not None
        frames.append(
            GroundTruthFrame(
                frame_index=frame_index,
                present=present,
                bbox=bbox if present else None,
                track_id=row.get("track_id"),
            )
        )
    return frames


def compute_runtime_summary(
    frame_rows: list[dict[str, str]],
    center_error_threshold: float,
) -> dict[str, float]:
    """Compute aggregate runtime metrics from the per-frame log."""

    times = _series(frame_rows, "monotonic_time_s")
    frame_count = len(frame_rows)
    duration_s = max(0.0, times[-1] - times[0]) if len(times) >= 2 else 0.0
    target_present_mask = [_parse_bool(row.get("target_present")) for row in frame_rows]
    follow_allowed_mask = [_parse_bool(row.get("follow_allowed")) for row in frame_rows]
    command_active_mask = [_parse_bool(row.get("command_active")) for row in frame_rows]
    centered_mask = []
    for row, target_present in zip(frame_rows, target_present_mask):
        center_error = _parse_float(row.get("center_error_norm"))
        centered_mask.append(
            bool(target_present) and center_error is not None and center_error <= center_error_threshold
        )

    summary: dict[str, float] = {
        "frame_count": float(frame_count),
        "duration_s": duration_s,
        "mean_fps": _safe_mean(_series(frame_rows, "instant_fps")),
        "p95_total_loop_latency_s": _percentile(
            _series(frame_rows, "total_loop_latency_s"),
            95.0,
        ),
        "mean_detection_latency_s": _safe_mean(
            _series(frame_rows, "detection_latency_s")
        ),
        "mean_tracking_latency_s": _safe_mean(
            _series(frame_rows, "tracking_latency_s")
        ),
        "mean_control_latency_s": _safe_mean(
            _series(frame_rows, "control_latency_s")
        ),
        "mean_mavlink_latency_s": _safe_mean(
            _series(frame_rows, "mavlink_latency_s")
        ),
        "mean_process_cpu_percent": _safe_mean(
            _series(frame_rows, "process_cpu_percent")
        ),
        "peak_process_cpu_percent": _safe_max(
            _series(frame_rows, "process_cpu_percent")
        ),
        "mean_process_rss_mb": _safe_mean(_series(frame_rows, "process_rss_mb")),
        "peak_process_rss_mb": _safe_max(_series(frame_rows, "process_rss_mb")),
        "target_present_ratio": _ratio(target_present_mask),
        "follow_allowed_ratio": _ratio(follow_allowed_mask),
        "command_active_ratio": _ratio(command_active_mask),
        "framing_accuracy_ratio": _ratio(centered_mask),
        "mean_center_error_norm": _safe_mean(
            _series_where(frame_rows, "center_error_norm", target_present_mask)
        ),
        "p95_center_error_norm": _percentile(
            _series_where(frame_rows, "center_error_norm", target_present_mask),
            95.0,
        ),
        "mean_abs_area_ratio_error": _safe_mean(
            [
                abs(value)
                for value in _series_where(frame_rows, "area_ratio_error", target_present_mask)
            ]
        ),
        "max_target_retention_time_s": _safe_max(
            _series(frame_rows, "target_retention_time_s")
        ),
        "track_id_switches": _safe_max(_series(frame_rows, "track_id_switches")),
    }
    return summary


def compute_ground_truth_metrics(
    *,
    frame_rows: list[dict[str, str]],
    detection_rows: list[dict[str, str]],
    ground_truth_rows: list[GroundTruthFrame],
    iou_threshold: float,
) -> dict[str, float]:
    """Compute single-target evaluation metrics against optional labels."""

    detections_by_frame = _group_rows_by_frame(detection_rows)
    frame_by_index = {
        parsed_index: row
        for row in frame_rows
        if (parsed_index := _parse_int(row.get("frame_index"))) is not None
    }
    gt_frames = sorted(ground_truth_rows, key=lambda frame: frame.frame_index)

    detection_tp = 0
    detection_fp = 0
    detection_fn = 0
    detection_correct_presence = 0
    detection_matched_ious: list[float] = []

    tracking_matches = 0
    tracking_fp = 0
    tracking_fn = 0
    id_switches = 0
    previous_matched_track_id: int | None = None

    for gt_frame in gt_frames:
        detections = detections_by_frame.get(gt_frame.frame_index, [])
        predicted_detection_present = len(detections) > 0
        if predicted_detection_present == gt_frame.present:
            detection_correct_presence += 1

        if gt_frame.present and gt_frame.bbox is not None:
            best_detection_iou = _best_iou(detections, gt_frame.bbox)
            if best_detection_iou >= iou_threshold:
                detection_tp += 1
                detection_fp += max(0, len(detections) - 1)
                detection_matched_ious.append(best_detection_iou)
            else:
                detection_fn += 1
                detection_fp += len(detections)
        else:
            detection_fp += len(detections)

        frame_row = frame_by_index.get(gt_frame.frame_index)
        frame_bbox = _bbox_from_row(frame_row, prefix="bbox_") if frame_row is not None else None
        predicted_target_present = bool(
            frame_row is not None and _parse_bool(frame_row.get("target_present"))
        )
        selected_track_id = (
            _parse_int(frame_row.get("selected_track_id")) if frame_row is not None else None
        )
        matched_tracking = (
            gt_frame.present
            and gt_frame.bbox is not None
            and predicted_target_present
            and frame_bbox is not None
            and frame_bbox.intersection_over_union(gt_frame.bbox) >= iou_threshold
        )
        if gt_frame.present:
            if matched_tracking:
                tracking_matches += 1
                if (
                    previous_matched_track_id is not None
                    and selected_track_id is not None
                    and selected_track_id != previous_matched_track_id
                ):
                    id_switches += 1
                if selected_track_id is not None:
                    previous_matched_track_id = selected_track_id
            else:
                tracking_fn += 1
                if predicted_target_present:
                    tracking_fp += 1
                previous_matched_track_id = None
        elif predicted_target_present:
            tracking_fp += 1
            previous_matched_track_id = None

    gt_positive_frames = sum(1 for frame in gt_frames if frame.present)
    total_gt_frames = len(gt_frames)
    summary: dict[str, float] = {
        "detection_precision": _fraction(detection_tp, detection_tp + detection_fp),
        "detection_recall": _fraction(detection_tp, detection_tp + detection_fn),
        "detection_f1": _f1_score(
            _fraction(detection_tp, detection_tp + detection_fp),
            _fraction(detection_tp, detection_tp + detection_fn),
        ),
        "detection_accuracy_frame_presence": _fraction(
            detection_correct_presence,
            total_gt_frames,
        ),
        "detection_mean_iou": _safe_mean(detection_matched_ious),
        "tracking_id_switches": float(id_switches),
        "tracking_mota_single_target": 1.0
        - _fraction(
            tracking_fn + tracking_fp + id_switches,
            gt_positive_frames,
        ),
        "tracking_idf1_single_target": _fraction(
            2 * tracking_matches,
            2 * tracking_matches + tracking_fp + tracking_fn,
        ),
        "tracking_retention_ratio": _fraction(tracking_matches, gt_positive_frames),
    }
    return summary


def write_summary_csv(path: Path, summary: dict[str, float]) -> None:
    """Write a flat metric summary CSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, value))


def generate_plots(
    *,
    frame_rows: list[dict[str, str]],
    output_dir: Path,
    session_name: str,
    ground_truth_rows: list[GroundTruthFrame] | None = None,
    iou_threshold: float = 0.5,
) -> list[Path]:
    """Create PNG graphs for the supplied log rows."""

    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _relative_times(frame_rows)
    output_paths: list[Path] = []

    fps = _series(frame_rows, "instant_fps")
    total_latency = _series(frame_rows, "total_loop_latency_s")
    detection_latency = _series(frame_rows, "detection_latency_s")
    tracking_latency = _series(frame_rows, "tracking_latency_s")
    control_latency = _series(frame_rows, "control_latency_s")
    cpu = _series(frame_rows, "process_cpu_percent")
    rss = _series(frame_rows, "process_rss_mb")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(times, fps, label="FPS", color="tab:blue")
    axes[0].set_ylabel("FPS")
    axes[0].set_title("System Throughput")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, total_latency, label="Total", color="tab:red")
    axes[1].plot(times, detection_latency, label="Detect", color="tab:orange", alpha=0.8)
    axes[1].plot(times, tracking_latency, label="Track", color="tab:green", alpha=0.8)
    axes[1].plot(times, control_latency, label="Control", color="tab:purple", alpha=0.8)
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_title("Pipeline Latency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(times, cpu, label="CPU %", color="tab:brown")
    axes[2].plot(times, rss, label="RSS MB", color="tab:gray")
    axes[2].set_ylabel("Usage")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Process Resource Usage")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    system_path = output_dir / f"{session_name}_system_performance.png"
    fig.savefig(system_path, dpi=160)
    plt.close(fig)
    output_paths.append(system_path)

    target_present = _bool_series(frame_rows, "target_present")
    follow_allowed = _bool_series(frame_rows, "follow_allowed")
    command_active = _bool_series(frame_rows, "command_active")
    detections_count = _series(frame_rows, "detections_count")
    tracks_count = _series(frame_rows, "tracks_count")
    confidence = _series(frame_rows, "selected_confidence")
    retention = _series(frame_rows, "target_retention_time_s")
    switches = _series(frame_rows, "track_id_switches")

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(times, detections_count, label="Detections", color="tab:blue")
    axes[0].plot(times, tracks_count, label="Tracks", color="tab:green")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Perception Counts")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].step(times, target_present, where="post", label="Target Present", color="tab:orange")
    axes[1].step(times, follow_allowed, where="post", label="Follow Allowed", color="tab:red")
    axes[1].step(times, command_active, where="post", label="Command Active", color="tab:purple")
    axes[1].set_ylabel("State")
    axes[1].set_title("Tracking and Gate State")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(times, confidence, color="tab:cyan")
    axes[2].set_ylabel("Confidence")
    axes[2].set_title("Selected Target Confidence")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(times, retention, label="Retention (s)", color="tab:olive")
    axes[3].plot(times, switches, label="ID switches", color="tab:pink")
    axes[3].set_ylabel("Retention / switches")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Target Retention")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    tracking_path = output_dir / f"{session_name}_tracking_quality.png"
    fig.savefig(tracking_path, dpi=160)
    plt.close(fig)
    output_paths.append(tracking_path)

    center_error_x = _series(frame_rows, "center_error_x_norm")
    center_error_y = _series(frame_rows, "center_error_y_norm")
    center_error_norm = _series(frame_rows, "center_error_norm")
    bbox_area_ratio = _series(frame_rows, "bbox_area_ratio")
    desired_area_ratio = _series(frame_rows, "desired_area_ratio")
    command_vx = _series(frame_rows, "command_velocity_forward_m_s")
    command_vy = _series(frame_rows, "command_velocity_right_m_s")
    command_yaw = _series(frame_rows, "command_yaw_rate_rad_s")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(times, center_error_x, label="Center error X", color="tab:blue")
    axes[0].plot(times, center_error_y, label="Center error Y", color="tab:orange")
    axes[0].plot(times, center_error_norm, label="Center error norm", color="tab:green")
    axes[0].set_ylabel("Normalized error")
    axes[0].set_title("Object-In-Frame Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, bbox_area_ratio, label="Observed area ratio", color="tab:red")
    axes[1].plot(times, desired_area_ratio, label="Desired area ratio", color="tab:gray", linestyle="--")
    axes[1].set_ylabel("Area ratio")
    axes[1].set_title("Distance Proxy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(times, command_vx, label="Forward m/s", color="tab:purple")
    axes[2].plot(times, command_vy, label="Right m/s", color="tab:brown")
    axes[2].plot(times, command_yaw, label="Yaw rad/s", color="tab:pink")
    axes[2].set_ylabel("Command")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Control Response")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    control_path = output_dir / f"{session_name}_framing_and_control.png"
    fig.savefig(control_path, dpi=160)
    plt.close(fig)
    output_paths.append(control_path)

    if ground_truth_rows is not None:
        gt_by_index = {frame.frame_index: frame for frame in ground_truth_rows}
        selected_iou: list[float] = []
        detection_iou: list[float] = []
        for row in frame_rows:
            frame_index = _parse_int(row.get("frame_index"))
            gt_frame = gt_by_index.get(frame_index) if frame_index is not None else None
            if gt_frame is None or not gt_frame.present or gt_frame.bbox is None:
                selected_iou.append(float("nan"))
                detection_iou.append(float("nan"))
                continue
            selected_bbox = _bbox_from_row(row, prefix="bbox_")
            selected_iou.append(
                selected_bbox.intersection_over_union(gt_frame.bbox)
                if selected_bbox is not None
                else float("nan")
            )
            detection_iou.append(float("nan"))

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(times, selected_iou, color="tab:blue")
        axes[0].axhline(iou_threshold, color="tab:red", linestyle="--")
        axes[0].set_ylabel("IoU")
        axes[0].set_title("Selected Target IoU vs Ground Truth")
        axes[0].grid(True, alpha=0.3)
        axes[1].step(times, [0.0 if math.isnan(value) else 1.0 for value in selected_iou], where="post")
        axes[1].set_ylabel("Matched")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Ground-Truth Match State")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        gt_path = output_dir / f"{session_name}_ground_truth_alignment.png"
        fig.savefig(gt_path, dpi=160)
        plt.close(fig)
        output_paths.append(gt_path)

    return output_paths


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for graph generation."""

    args = parse_args(argv)
    frame_log_path = Path(args.log_file)
    if not frame_log_path.exists():
        raise FileNotFoundError(f"Log file not found: {frame_log_path}")

    frame_rows = load_csv_rows(frame_log_path)
    if not frame_rows:
        raise ValueError(f"Log file is empty: {frame_log_path}")

    detections_path, _tracks_path, session_name = infer_related_paths(frame_log_path)
    detection_rows = load_csv_rows(detections_path) if detections_path is not None else []
    ground_truth_rows = (
        load_ground_truth_rows(Path(args.ground_truth_csv))
        if args.ground_truth_csv
        else None
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else frame_log_path.parent / f"{session_name}_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_runtime_summary(
        frame_rows,
        center_error_threshold=args.center_error_threshold,
    )
    if ground_truth_rows is not None:
        summary.update(
            compute_ground_truth_metrics(
                frame_rows=frame_rows,
                detection_rows=detection_rows,
                ground_truth_rows=ground_truth_rows,
                iou_threshold=args.iou_threshold,
            )
        )

    summary_path = output_dir / f"{session_name}_summary.csv"
    write_summary_csv(summary_path, summary)
    plot_paths = generate_plots(
        frame_rows=frame_rows,
        output_dir=output_dir,
        session_name=session_name,
        ground_truth_rows=ground_truth_rows,
        iou_threshold=args.iou_threshold,
    )

    print(f"Summary written to {summary_path}")
    for metric_name, metric_value in summary.items():
        print(f"{metric_name}={metric_value:.6f}")
    for plot_path in plot_paths:
        print(f"Saved plot {plot_path}")
    return 0


def _best_iou(rows: Iterable[dict[str, str]], ground_truth_bbox: BoundingBox) -> float:
    """Return the best IoU between logged boxes and the supplied label."""

    best = 0.0
    for row in rows:
        bbox = _bbox_from_row(row, prefix="bbox_")
        if bbox is None:
            continue
        best = max(best, bbox.intersection_over_union(ground_truth_bbox))
    return best


def _relative_times(frame_rows: list[dict[str, str]]) -> list[float]:
    """Return frame timestamps shifted to start at zero."""

    times = _series(frame_rows, "monotonic_time_s")
    if not times:
        return []
    start = times[0]
    return [value - start for value in times]


def _group_rows_by_frame(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    """Group arbitrary CSV rows by frame index."""

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        frame_index = _parse_int(row.get("frame_index"))
        if frame_index is None:
            continue
        grouped.setdefault(frame_index, []).append(row)
    return grouped


def _bbox_from_row(row: dict[str, str] | None, prefix: str) -> BoundingBox | None:
    """Build a bounding box from a CSV row if all coordinates are present."""

    if row is None:
        return None
    x1 = _parse_float(row.get(f"{prefix}x1_px") or row.get(f"{prefix}x1"))
    y1 = _parse_float(row.get(f"{prefix}y1_px") or row.get(f"{prefix}y1"))
    x2 = _parse_float(row.get(f"{prefix}x2_px") or row.get(f"{prefix}x2"))
    y2 = _parse_float(row.get(f"{prefix}y2_px") or row.get(f"{prefix}y2"))
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _series(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    """Return a numeric series for one CSV column, ignoring blanks."""

    values: list[float] = []
    for row in rows:
        value = _parse_float(row.get(column))
        values.append(value if value is not None else float("nan"))
    return values


def _series_where(
    rows: list[dict[str, str]],
    column: str,
    mask: list[bool | None],
) -> list[float]:
    """Return numeric values where a matching boolean mask is true."""

    values: list[float] = []
    for row, enabled in zip(rows, mask):
        if not enabled:
            continue
        value = _parse_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _bool_series(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    """Return a numeric 0/1 series for plotting boolean CSV fields."""

    return [1.0 if _parse_bool(row.get(column)) else 0.0 for row in rows]


def _safe_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean ignoring NaN values."""

    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return float("nan")
    return sum(filtered) / len(filtered)


def _safe_max(values: Iterable[float]) -> float:
    """Return the maximum value ignoring NaN values."""

    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return float("nan")
    return max(filtered)


def _percentile(values: Iterable[float], percentile: float) -> float:
    """Return a simple inclusive percentile ignoring NaN values."""

    filtered = sorted(value for value in values if not math.isnan(value))
    if not filtered:
        return float("nan")
    if len(filtered) == 1:
        return filtered[0]
    rank = (len(filtered) - 1) * (percentile / 100.0)
    low_index = int(math.floor(rank))
    high_index = int(math.ceil(rank))
    if low_index == high_index:
        return filtered[low_index]
    fraction = rank - low_index
    return filtered[low_index] + fraction * (filtered[high_index] - filtered[low_index])


def _ratio(values: Iterable[bool | None]) -> float:
    """Return the fraction of truthy values across a boolean iterable."""

    materialized = [bool(value) for value in values]
    if not materialized:
        return float("nan")
    return sum(materialized) / len(materialized)


def _fraction(numerator: int, denominator: int) -> float:
    """Return a safe floating-point fraction."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""

    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _parse_bool(raw_value: str | None) -> bool | None:
    """Parse common CSV boolean spellings."""

    if raw_value is None or raw_value == "":
        return None
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return None


def _parse_float(raw_value: str | None) -> float | None:
    """Parse a float from a CSV field."""

    if raw_value is None or raw_value == "":
        return None
    return float(raw_value)


def _parse_int(raw_value: str | None) -> int | None:
    """Parse an integer from a CSV field."""

    if raw_value is None or raw_value == "":
        return None
    return int(float(raw_value))


if __name__ == "__main__":
    raise SystemExit(main())
