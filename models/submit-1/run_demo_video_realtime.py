from __future__ import annotations

import argparse
import sys

import vehicle_metadata_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample road video as a realtime browser-link demo.")
    parser.add_argument("--source", default="Test Video.mp4")
    parser.add_argument("--output", default="demo_vehicle_metadata.jsonl")
    parser.add_argument("--video-output", default="demo_vehicle_metadata_annotated.mp4")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--detect-fps", type=float, default=30.0)
    parser.add_argument("--web-fps", type=float, default=15.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_args = [
        "vehicle_metadata_pipeline.py",
        "--source",
        args.source,
        "--output",
        args.output,
        "--view",
        "--realtime-playback",
        "--detect-fps",
        str(args.detect_fps),
        "--web-host",
        args.web_host,
        "--web-port",
        str(args.web_port),
        "--web-fps",
        str(args.web_fps),
    ]

    if args.save_video:
        pipeline_args.extend(["--save-video", "--video-output", args.video_output])
    if args.print_json:
        pipeline_args.append("--print-json")
    if args.max_frames is not None:
        pipeline_args.extend(["--max-frames", str(args.max_frames)])

    sys.argv = pipeline_args
    vehicle_metadata_pipeline.main()


if __name__ == "__main__":
    main()
