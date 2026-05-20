from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


def color_for_track(track_id: int) -> tuple[int, int, int]:
    value = int(track_id) * 2654435761 % 255
    return (
        int((value * 3 + 80) % 255),
        int((value * 7 + 120) % 255),
        int((value * 11 + 160) % 255),
    )


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, text_h + 8)
    cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 8, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def load_metadata(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    metadata: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata[int(row["vehicle_id"])] = row
    return metadata


def render_video(source: Path, log_csv: Path, metadata_jsonl: Path | None, output: Path, trail: bool) -> None:
    log = pd.read_csv(log_csv)
    log_by_frame = {int(frame): rows for frame, rows in log.groupby("frame")}
    metadata_by_id = load_metadata(metadata_jsonl)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not create output video: {output}")

    centers_by_track: dict[int, list[tuple[int, int]]] = {}
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        rows = log_by_frame.get(frame_idx)
        if rows is not None:
            for row in rows.itertuples(index=False):
                if pd.isna(row.track_id):
                    continue
                track_id = int(row.track_id)
                color = color_for_track(track_id)
                x1, y1, x2, y2 = map(int, (row.x1, row.y1, row.x2, row.y2))
                center = (int(row.center_x), int(row.center_y))

                centers_by_track.setdefault(track_id, []).append(center)
                centers_by_track[track_id] = centers_by_track[track_id][-32:]

                if trail:
                    points = centers_by_track[track_id]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                confidence = "" if pd.isna(row.confidence) else f" {float(row.confidence):.2f}"
                vehicle_meta = metadata_by_id.get(track_id)
                if vehicle_meta:
                    label = (
                        f"ID {track_id} {row.class_name}{confidence} "
                        f"{vehicle_meta['brand']} {vehicle_meta['color']}"
                    )
                else:
                    label = f"ID {track_id} {row.class_name}{confidence}"
                draw_label(frame, label, x1, y1, color)

        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames - 1}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO tracking CSV logs back onto a video.")
    parser.add_argument("--source", default="Test Video.mp4", help="Original input video.")
    parser.add_argument("--log", default="road_tracking_log.csv", help="Tracking CSV log.")
    parser.add_argument("--metadata", default=None, help="Optional vehicle_metadata.jsonl file.")
    parser.add_argument("--output", default="road_tracking_log_annotated.mp4", help="Annotated output video.")
    parser.add_argument("--trail", action="store_true", help="Draw short center trails for each track.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = Path(args.metadata) if args.metadata else None
    render_video(Path(args.source), Path(args.log), metadata, Path(args.output), args.trail)
    print(f"saved annotated video to {args.output}")


if __name__ == "__main__":
    main()
