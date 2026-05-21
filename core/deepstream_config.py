import os
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_GIE_CONFIG = Path("/tmp/tensorrt_engine_primary_gie_yolov8n_nano.txt")
POSTPROCESS_OP_TOKENS = {
    b"\x22\x0aBatchedNMS": "BatchedNMS",
    b"\x22\x0eBatchedNMS_TRT": "BatchedNMS_TRT",
    b"\x22\x0cEfficientNMS": "EfficientNMS",
    b"\x22\x10EfficientNMS_TRT": "EfficientNMS_TRT",
    b"\x22\x03Mod": "Mod",
    b"\x22\x11NonMaxSuppression": "NonMaxSuppression",
    b"\x22\x04TopK": "TopK",
}


class DeepStreamConfigError(RuntimeError):
    pass


def deepstream_config() -> Dict[str, object]:
    ds = dict(CONFIG.get("deepstream", {}))
    ds.setdefault("confidence_threshold", CONFIG.get("conf_threshold", 0.25))
    ds.setdefault("iou_threshold", CONFIG.get("iou_threshold", 0.45))
    ds.setdefault("input_width", CONFIG.get("imgsz", 640))
    ds.setdefault("input_height", CONFIG.get("imgsz", 640))
    ds.setdefault("batch_size", 1)
    ds.setdefault("gpu_id", 0)
    ds.setdefault("network_mode", 0)
    ds.setdefault("num_detected_classes", 80)
    ds.setdefault("codec", "h264")
    ds.setdefault("sink_type", "fakesink")
    ds.setdefault("enable_mjpeg_output", False)
    if not ds.get("engine_path"):
        precision = {0: "fp32", 1: "int8", 2: "fp16"}.get(int(ds["network_mode"]), "fp32")
        ds["engine_path"] = (
            f"model_b{int(ds['batch_size'])}_gpu{int(ds['gpu_id'])}_{precision}.engine"
        )
    return ds


def active_camera_ids() -> List[str]:
    configured = CONFIG.get("active_cameras")
    if configured:
        return list(configured)
    return list(CONFIG["cameras"].keys())


def rtsp_source_uris(ds: Dict[str, object], camera_ids: Iterable[str]) -> List[Tuple[str, str]]:
    override_uri = str(ds.get("rtsp_uri") or "").strip()
    cameras = CONFIG.get("cameras", {})
    sources: List[Tuple[str, str]] = []
    for cam_id in camera_ids:
        if override_uri:
            uri = override_uri
        else:
            if cam_id not in cameras:
                raise DeepStreamConfigError(f"No RTSP URI configured for active camera '{cam_id}'.")
            uri = str(cameras[cam_id]).strip()
        sources.append((cam_id, uri))
    return sources


def redact_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if "@" not in parsed.netloc:
        return uri
    host_part = parsed.netloc.rsplit("@", 1)[1]
    return urlunsplit((parsed.scheme, "***:***@" + host_part, parsed.path, parsed.query, parsed.fragment))


def validate_rtsp_sources(ds: Dict[str, object], camera_ids: Iterable[str]) -> None:
    problems: List[str] = []
    for cam_id, uri in rtsp_source_uris(ds, camera_ids):
        try:
            parsed = urlsplit(uri)
            _ = parsed.port
        except ValueError as exc:
            problems.append(f"{cam_id}: invalid RTSP URI {redact_uri(uri)} ({exc})")
            continue

        if parsed.scheme.lower() != "rtsp":
            problems.append(f"{cam_id}: URI must start with rtsp://, got {redact_uri(uri)}")
        if not parsed.hostname:
            problems.append(f"{cam_id}: RTSP URI is missing a hostname: {redact_uri(uri)}")

    if problems:
        raise DeepStreamConfigError("Invalid RTSP source configuration:\n  - " + "\n  - ".join(problems))


def _probe_rtsp_uri(uri: str, timeout_sec: float) -> Tuple[bool, str]:
    parsed = urlsplit(uri)
    host = parsed.hostname
    if not host:
        return False, "missing hostname"
    port = parsed.port or 554

    request = (
        f"OPTIONS {uri} RTSP/1.0\r\n"
        "CSeq: 1\r\n"
        "User-Agent: tensorrt-engine-deepstream-preflight\r\n"
        "\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.settimeout(timeout_sec)
            sock.sendall(request.encode("utf-8"))
            response = sock.recv(4096)
    except socket.timeout:
        return False, "timed out before receiving an RTSP response"
    except OSError as exc:
        return False, str(exc)

    if not response:
        return False, "connection closed without an RTSP response"

    first_line = response.splitlines()[0].decode("latin-1", errors="replace").strip()
    if first_line.startswith("RTSP/"):
        return True, first_line
    return (
        False,
        f"server answered {first_line!r}, not RTSP/1.0. "
        "This usually means the URI points to HTTP, the wrong port, or a non-RTSP mock endpoint.",
    )


def preflight_rtsp_sources(ds: Dict[str, object], camera_ids: Iterable[str]) -> None:
    timeout_sec = float(ds.get("rtsp_preflight_timeout_sec", 3))
    failures: List[str] = []
    for cam_id, uri in rtsp_source_uris(ds, camera_ids):
        ok, detail = _probe_rtsp_uri(uri, timeout_sec)
        if not ok:
            failures.append(f"{cam_id} {redact_uri(uri)}: {detail}")

    if failures:
        raise DeepStreamConfigError(
            "RTSP preflight failed before starting DeepStream:\n"
            "  - "
            + "\n  - ".join(failures)
            + "\nUse a real RTSP URL/port, or set deepstream.rtsp_preflight=False "
            "temporarily if the camera rejects OPTIONS probes."
        )


def resolve_project_path(path_value: Optional[str]) -> str:
    if not path_value:
        return ""

    path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_labels(labels_path: str) -> List[str]:
    resolved = Path(resolve_project_path(labels_path))
    if not resolved.exists():
        raise DeepStreamConfigError(f"DeepStream labels file not found: {resolved}")
    return [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _detect_onnx_postprocess_ops(onnx_path: Path) -> List[str]:
    data = onnx_path.read_bytes()
    return sorted(
        op_name
        for token, op_name in POSTPROCESS_OP_TOKENS.items()
        if token in data
    )


def validate_deepstream_inputs(ds: Dict[str, object]) -> None:
    onnx_path = Path(resolve_project_path(str(ds.get("onnx_model_path") or "")))
    labels_path = Path(resolve_project_path(str(ds.get("labels_path") or "")))
    parser_path = Path(resolve_project_path(str(ds.get("custom_parser_path") or "")))

    missing: List[str] = []
    if not onnx_path.exists():
        missing.append(f"ONNX model: {onnx_path}")
    if not labels_path.exists():
        missing.append(f"labels file: {labels_path}")
    if not parser_path.exists():
        missing.append(f"YOLO custom parser .so: {parser_path}")

    if missing:
        details = "\n  - ".join(missing)
        raise DeepStreamConfigError(
            "DeepStream startup is missing required files:\n"
            f"  - {details}\n"
            "Build/copy these on the Jetson Nano. Do not copy a TensorRT "
            ".engine from Orin, desktop GPU, Colab, or another JetPack stack."
        )

    postprocess_ops = _detect_onnx_postprocess_ops(onnx_path)
    if postprocess_ops:
        ops = ", ".join(postprocess_ops)
        raise DeepStreamConfigError(
            "DeepStream ONNX is not a raw YOLO detector export. "
            f"Found post-processing operators in {onnx_path}: {ops}.\n"
            "Jetson Nano / TensorRT 8 / DeepStream-YOLO expects raw detection "
            "head output and lets the custom parser perform decode/NMS. "
            "Regenerate the ONNX with models/pipeline_1/export_yolo_onnx.py "
            "without --nms, delete the stale .engine file, then start server.py again."
        )


def _config_lines(ds: Dict[str, object]) -> Iterable[str]:
    input_width = int(ds["input_width"])
    input_height = int(ds["input_height"])
    output_blob_names = str(ds.get("output_blob_names") or "").strip()
    engine_create_func = str(
        ds.get("engine_create_func_name") or "NvDsInferYoloCudaEngineGet"
    ).strip()

    yield "# Generated by core.deepstream_config from config.py."
    yield "# Run on Jetson Nano so nvinfer builds this engine on the target stack."
    yield ""
    yield "[property]"
    yield f"gpu-id={int(ds['gpu_id'])}"
    yield f"onnx-file={resolve_project_path(str(ds['onnx_model_path']))}"
    engine_path = str(ds.get("engine_path") or "").strip()
    if engine_path:
        yield f"model-engine-file={resolve_project_path(engine_path)}"
    yield f"labelfile-path={resolve_project_path(str(ds['labels_path']))}"
    yield f"batch-size={int(ds['batch_size'])}"
    yield f"network-mode={int(ds['network_mode'])}"
    yield f"num-detected-classes={int(ds['num_detected_classes'])}"
    yield "process-mode=1"
    yield "network-type=0"
    yield "gie-unique-id=1"
    yield f"interval={int(ds.get('interval', 0))}"
    yield f"workspace-size={int(ds.get('workspace_size', 1024))}"
    yield "cluster-mode=4"
    yield "maintain-aspect-ratio=1"
    yield "symmetric-padding=1"
    yield f"infer-dims=3;{input_height};{input_width}"
    yield "net-scale-factor=0.0039215697906911373"
    yield "model-color-format=0"
    yield "parse-bbox-func-name=NvDsInferParseYolo"
    yield f"custom-lib-path={resolve_project_path(str(ds['custom_parser_path']))}"
    if engine_create_func:
        yield f"engine-create-func-name={engine_create_func}"
    if output_blob_names:
        yield f"output-blob-names={output_blob_names}"
    yield ""
    yield "[class-attrs-all]"
    yield f"pre-cluster-threshold={float(ds['confidence_threshold'])}"
    yield f"nms-iou-threshold={float(ds['iou_threshold'])}"
    yield f"topk={int(ds.get('topk', 300))}"


def write_primary_gie_config(ds: Dict[str, object]) -> str:
    engine_path_value = str(ds.get("engine_path") or "").strip()
    if engine_path_value:
        engine_path = Path(resolve_project_path(engine_path_value))
        engine_path.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_GIE_CONFIG.write_text("\n".join(_config_lines(ds)) + "\n", encoding="utf-8")
    return str(RUNTIME_GIE_CONFIG)
