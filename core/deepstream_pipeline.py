import collections
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from core.deepstream_config import (
    DeepStreamConfigError,
    active_camera_ids,
    deepstream_config,
    load_labels,
    resolve_project_path,
    validate_deepstream_inputs,
    write_primary_gie_config,
)
from core.sse import broadcast_sse
from core.state import MAX_LOG, cam_state, detection_log, detection_log_lock
from routers.api_exporter import push_external_async


log = logging.getLogger("multicam")


class DeepStreamImportError(RuntimeError):
    pass


def _load_deepstream_modules():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib, Gst
    except Exception as exc:
        raise DeepStreamImportError(
            "Could not import GStreamer Python bindings. On Jetson Nano with "
            "DeepStream 6.0/6.0.1, install/use the NVIDIA DeepStream Python "
            "bindings environment and run with system-site-packages."
        ) from exc

    try:
        import pyds
    except Exception as exc:
        raise DeepStreamImportError(
            "Could not import pyds. Build/install DeepStream Python bindings "
            "for the JetPack 4.6.x / DeepStream 6.0.x environment on Nano. "
            "See README_DEEPSTREAM_NANO.md#install-deepstream-python-bindings-pyds."
        ) from exc

    return GLib, Gst, pyds


class DeepStreamPipeline:
    """DeepStream RTSP -> nvinfer pipeline with metadata-driven business logic."""

    def __init__(self):
        self.ds = deepstream_config()
        self.cam_ids = active_camera_ids()
        self.labels = load_labels(str(self.ds["labels_path"]))
        self.gie_config_path = ""
        self.source_id_to_cam_id: Dict[int, str] = {}
        self.fps_times = {
            cid: collections.deque(maxlen=30)
            for cid in self.cam_ids
        }
        self.frame_seen_at: Dict[str, float] = {}
        self.last_mjpeg_at: Dict[str, float] = {}
        self.pipeline = None
        self.mainloop = None
        self.loop_thread: Optional[threading.Thread] = None
        self.stopping = False
        self.restart_pending = False
        self.GLib = None
        self.Gst = None
        self.pyds = None

    def start(self) -> None:
        if not self.cam_ids:
            raise DeepStreamConfigError("No active cameras configured.")
        if int(self.ds.get("batch_size", 1)) != len(self.cam_ids):
            raise DeepStreamConfigError(
                "DeepStream batch_size must match active camera count. "
                f"batch_size={self.ds.get('batch_size')} active={len(self.cam_ids)}. "
                "For Jetson Nano, start with one active camera and batch_size=1."
            )

        validate_deepstream_inputs(self.ds)
        self.gie_config_path = write_primary_gie_config(self.ds)
        self.GLib, self.Gst, self.pyds = _load_deepstream_modules()
        self.Gst.init(None)
        self._build_pipeline()

        self.mainloop = self.GLib.MainLoop()
        self.loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="deepstream-glib-loop",
        )
        self.loop_thread.start()

        ret = self.pipeline.set_state(self.Gst.State.PLAYING)
        if ret == self.Gst.StateChangeReturn.FAILURE:
            raise DeepStreamConfigError("Failed to set DeepStream pipeline to PLAYING.")

        log.info(
            "Started DeepStream pipeline with %d source(s), nvinfer config=%s",
            len(self.cam_ids),
            self.gie_config_path,
        )

    def stop(self) -> None:
        self.stopping = True
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        if self.mainloop is not None:
            self.mainloop.quit()
        if self.loop_thread is not None and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5)

    def _run_loop(self) -> None:
        try:
            self.mainloop.run()
        except Exception:
            log.exception("DeepStream GLib loop stopped unexpectedly")

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise DeepStreamConfigError(
                f"Could not create GStreamer element '{factory}'. "
                "Check DeepStream/GStreamer installation on Jetson Nano."
            )
        return element

    def _build_pipeline(self) -> None:
        Gst = self.Gst
        self.pipeline = Gst.Pipeline.new("jetson-nano-deepstream-yolo")
        if self.pipeline is None:
            raise DeepStreamConfigError("Could not create GStreamer pipeline.")

        streammux = self._make("nvstreammux", "stream-muxer")
        pgie = self._make("nvinfer", "primary-gie")
        nvvidconv = self._make("nvvideoconvert", "pre-osd-convert")
        osd = self._make("nvdsosd", "onscreen-display")

        streammux.set_property("batch-size", int(self.ds["batch_size"]))
        streammux.set_property("width", int(self.ds.get("mux_width", 1280)))
        streammux.set_property("height", int(self.ds.get("mux_height", 720)))
        streammux.set_property("batched-push-timeout", int(self.ds.get("batched_push_timeout_us", 40000)))
        streammux.set_property("live-source", 1)
        streammux.set_property("attach-sys-ts", 1)

        pgie.set_property("config-file-path", self.gie_config_path)

        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(osd)

        for index, cam_id in enumerate(self.cam_ids):
            self._add_rtsp_source(index, cam_id, streammux)

        elements = [streammux, pgie]
        tracker = None
        if bool(self.ds.get("enable_tracker", False)):
            tracker = self._make("nvtracker", "tracker")
            self._configure_tracker(tracker)
            self.pipeline.add(tracker)
            elements.append(tracker)

        elements.extend([nvvidconv, osd])
        sink_elements = self._add_sink_branch(osd)
        elements.extend(sink_elements)

        for current, next_element in zip(elements, elements[1:]):
            if not current.link(next_element):
                raise DeepStreamConfigError(
                    f"Failed to link {current.get_name()} -> {next_element.get_name()}"
                )

        osd_sink_pad = osd.get_static_pad("sink")
        if osd_sink_pad is None:
            raise DeepStreamConfigError("Could not get nvdsosd sink pad for metadata probe.")
        osd_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._metadata_probe, None)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _add_rtsp_source(self, index: int, cam_id: str, streammux) -> None:
        Gst = self.Gst
        codec = str(self.ds.get("codec", "h264")).lower()
        if codec not in ("h264", "h265"):
            raise DeepStreamConfigError("DeepStream codec must be h264 or h265.")

        source = self._make("rtspsrc", f"rtsp-source-{index}")
        depay = self._make(f"rtp{codec}depay", f"rtp-{codec}-depay-{index}")
        parser = self._make(f"{codec}parse", f"{codec}-parse-{index}")
        decoder = self._make("nvv4l2decoder", f"nvv4l2-decoder-{index}")

        uri = self.ds.get("rtsp_uri") or CONFIG["cameras"][cam_id]
        source.set_property("location", str(uri))
        source.set_property("latency", int(self.ds.get("rtsp_latency_ms", 200)))
        source.set_property("drop-on-latency", bool(self.ds.get("drop_on_latency", True)))
        if bool(self.ds.get("rtsp_tcp", True)):
            # 4 is GST_RTSP_LOWER_TRANS_TCP. Avoid importing GstRtsp for DS 6.0.
            source.set_property("protocols", 4)

        for element in (source, depay, parser, decoder):
            self.pipeline.add(element)

        if not depay.link(parser):
            raise DeepStreamConfigError(f"Failed to link depay -> parser for {cam_id}")
        if not parser.link(decoder):
            raise DeepStreamConfigError(f"Failed to link parser -> decoder for {cam_id}")

        source.connect("pad-added", self._on_rtsp_pad_added, depay)

        sinkpad = streammux.get_request_pad("sink_%u")
        if sinkpad is None:
            raise DeepStreamConfigError(f"Could not request streammux sink pad for {cam_id}")
        source_id = self._streammux_source_id(sinkpad, index)
        srcpad = decoder.get_static_pad("src")
        if srcpad is None:
            raise DeepStreamConfigError(f"Could not get decoder src pad for {cam_id}")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise DeepStreamConfigError(f"Failed to link decoder -> streammux for {cam_id}")

        self.source_id_to_cam_id[source_id] = cam_id
        log.info("[%s] DeepStream RTSP source configured: %s (source_id=%d)", cam_id, uri, source_id)

    def _streammux_source_id(self, sinkpad, fallback_index: int) -> int:
        pad_name = sinkpad.get_name() or ""
        if pad_name.startswith("sink_"):
            try:
                return int(pad_name.rsplit("_", 1)[1])
            except ValueError:
                pass
        log.warning("Could not parse streammux pad name '%s'; using source index %d", pad_name, fallback_index)
        return fallback_index

    def _on_rtsp_pad_added(self, source, pad, depay) -> None:
        sinkpad = depay.get_static_pad("sink")
        if sinkpad is None or sinkpad.is_linked():
            return

        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_text = caps.to_string() if caps is not None else ""
        if "application/x-rtp" not in caps_text:
            log.debug("Ignoring non-RTP RTSP pad from %s: %s", source.get_name(), caps_text)
            return

        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            log.error("Failed to link RTSP source pad to depayloader: %s", result)

    def _add_sink_branch(self, osd) -> List[object]:
        sink_type = str(self.ds.get("sink_type", "fakesink")).lower()
        if not bool(self.ds.get("display", False)) and sink_type == "egl":
            sink_type = "fakesink"

        if sink_type == "fakesink":
            sink = self._make("fakesink", "sink")
            self._configure_sink(sink)
            self.pipeline.add(sink)
            return [sink]

        if sink_type == "egl":
            transform = self._make("nvegltransform", "egl-transform")
            sink = self._make("nveglglessink", "egl-sink")
            self._configure_sink(sink)
            self.pipeline.add(transform)
            self.pipeline.add(sink)
            return [transform, sink]

        if sink_type == "appsink":
            if len(self.cam_ids) > 1:
                return self._add_demuxed_appsink_branch()
            return self._add_single_appsink_branch()

        raise DeepStreamConfigError(
            f"Unsupported DeepStream sink_type '{sink_type}'. Use fakesink, egl, or appsink."
        )

    def _add_single_appsink_branch(self) -> List[object]:
        convert, capsfilter, appsink = self._make_appsink_elements("appsink", self.cam_ids[0])
        self.pipeline.add(convert)
        self.pipeline.add(capsfilter)
        self.pipeline.add(appsink)
        return [convert, capsfilter, appsink]

    def _add_demuxed_appsink_branch(self) -> List[object]:
        """Split batched DeepStream output so each active camera has MJPEG frames."""
        demux = self._make("nvstreamdemux", "stream-demuxer")
        self.pipeline.add(demux)

        for index, cam_id in enumerate(self.cam_ids):
            queue = self._make("queue", f"appsink-queue-{index}")
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 1)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)

            convert, capsfilter, appsink = self._make_appsink_elements(f"appsink-{index}", cam_id)

            for element in (queue, convert, capsfilter, appsink):
                self.pipeline.add(element)

            if not queue.link(convert):
                raise DeepStreamConfigError(f"Failed to link appsink queue -> converter for {cam_id}")
            if not convert.link(capsfilter):
                raise DeepStreamConfigError(f"Failed to link appsink converter -> caps for {cam_id}")
            if not capsfilter.link(appsink):
                raise DeepStreamConfigError(f"Failed to link appsink caps -> sink for {cam_id}")

            srcpad = self._request_demux_src_pad(demux, index, cam_id)
            sinkpad = queue.get_static_pad("sink")
            if sinkpad is None:
                raise DeepStreamConfigError(f"Could not get appsink queue sink pad for {cam_id}")
            if srcpad.link(sinkpad) != self.Gst.PadLinkReturn.OK:
                raise DeepStreamConfigError(f"Failed to link demux src_{index} -> appsink branch for {cam_id}")

        return [demux]

    def _make_appsink_elements(self, name_prefix: str, cam_id: str) -> Tuple[object, object, object]:
        convert = self._make("nvvideoconvert", f"{name_prefix}-convert")
        capsfilter = self._make("capsfilter", f"{name_prefix}-caps")
        appsink = self._make("appsink", f"{name_prefix}-sink")
        caps = self.Gst.Caps.from_string("video/x-raw,format=RGBA")
        capsfilter.set_property("caps", caps)
        appsink.set_property("emit-signals", True)
        self._configure_sink(appsink)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        appsink.connect("new-sample", self._on_new_sample, cam_id)
        return convert, capsfilter, appsink

    def _configure_sink(self, sink) -> None:
        sink.set_property("sync", False)
        sink.set_property("async", False)
        sink.set_property("qos", False)

    def _request_demux_src_pad(self, demux, index: int, cam_id: str):
        for pad_name in (f"src_{index}", "src_%u"):
            pad = demux.get_request_pad(pad_name)
            if pad is not None:
                return pad
        raise DeepStreamConfigError(f"Could not request nvstreamdemux src pad for {cam_id}")

    def _configure_tracker(self, tracker) -> None:
        config_path = Path(resolve_project_path(str(self.ds.get("tracker_config_path") or "")))
        if not config_path.exists():
            raise DeepStreamConfigError(f"Tracker config not found: {config_path}")

        int_props = {
            "tracker-width",
            "tracker-height",
            "gpu-id",
            "enable-batch-process",
            "enable-past-frame",
            "display-tracking-id",
        }
        path_props = {"ll-lib-file", "ll-config-file"}

        for raw in config_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if key in int_props:
                tracker.set_property(key, int(value))
            elif key in path_props:
                tracker.set_property(key, resolve_project_path(value))
            else:
                tracker.set_property(key, value)

    def _metadata_probe(self, pad, info, user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return self.Gst.PadProbeReturn.OK

        batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return self.Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = self.pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            cam_id = self.source_id_to_cam_id.get(
                int(frame_meta.source_id),
                f"source{int(frame_meta.source_id)}",
            )
            detections = self._extract_detections(frame_meta)
            self._handle_frame_metadata(cam_id, int(frame_meta.source_id), int(frame_meta.frame_num), detections)

            try:
                frame_list = frame_list.next
            except StopIteration:
                break

        return self.Gst.PadProbeReturn.OK

    def _extract_detections(self, frame_meta) -> List[dict]:
        detections: List[dict] = []
        obj_list = frame_meta.obj_meta_list
        invalid_object_id = getattr(self.pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF)

        while obj_list is not None:
            try:
                obj_meta = self.pyds.NvDsObjectMeta.cast(obj_list.data)
            except StopIteration:
                break

            rect = obj_meta.rect_params
            class_id = int(obj_meta.class_id)
            label = self._object_label(obj_meta, class_id)
            left = max(0.0, float(rect.left))
            top = max(0.0, float(rect.top))
            width = max(0.0, float(rect.width))
            height = max(0.0, float(rect.height))
            confidence = max(0.0, float(obj_meta.confidence))

            detection = {
                "source_id": int(frame_meta.source_id),
                "frame": int(frame_meta.frame_num),
                "class_id": class_id,
                "class_name": label,
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    round(left, 1),
                    round(top, 1),
                    round(left + width, 1),
                    round(top + height, 1),
                ],
            }

            object_id = int(obj_meta.object_id)
            if object_id != int(invalid_object_id):
                detection["track_id"] = object_id

            detections.append(detection)

            try:
                obj_list = obj_list.next
            except StopIteration:
                break

        return detections

    def _object_label(self, obj_meta, class_id: int) -> str:
        raw_label = ""
        try:
            raw_label = self.pyds.get_string(obj_meta.obj_label)
        except Exception:
            raw_label = getattr(obj_meta, "obj_label", "") or ""
        if raw_label:
            return str(raw_label)
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

    def _handle_frame_metadata(self, cam_id: str, source_id: int, frame_idx: int, detections: List[dict]) -> None:
        now = time.time()
        times = self.fps_times.setdefault(cam_id, collections.deque(maxlen=30))
        times.append(now)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) >= 2 else 0.0

        previous_seen_at = self.frame_seen_at.get(cam_id)
        latency_ms = round((now - previous_seen_at) * 1000, 1) if previous_seen_at else 0.0
        self.frame_seen_at[cam_id] = now

        if cam_id in cam_state:
            state = cam_state[cam_id]
            with state["lock"]:
                state["detections"] = detections
                state["fps"] = fps
                state["frame_count"] = frame_idx

        record = {
            "frame": frame_idx,
            "source_id": source_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "camera_id": cam_id,
            "latency_ms": latency_ms,
            "fps": round(fps, 1),
            "num_detections": len(detections),
            "detections": detections,
        }

        with detection_log_lock:
            detection_log.append(record)
            if len(detection_log) > MAX_LOG:
                detection_log.pop(0)

        broadcast_sse(cam_id, record)
        if detections:
            push_external_async(cam_id, record)

    def _on_new_sample(self, sink, cam_id: Optional[str] = None):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        if not bool(self.ds.get("enable_mjpeg_output", False)):
            return self.Gst.FlowReturn.OK
        if cam_id is None:
            if len(self.cam_ids) != 1:
                log.warning("appsink MJPEG sample did not include a camera id.")
                return self.Gst.FlowReturn.OK
            cam_id = self.cam_ids[0]
        if cam_id not in cam_state:
            return self.Gst.FlowReturn.OK

        max_fps = float(CONFIG.get("mjpeg_fps", 0) or 0)
        if max_fps > 0:
            now = time.time()
            previous = self.last_mjpeg_at.get(cam_id, 0.0)
            if now - previous < 1.0 / max_fps:
                return self.Gst.FlowReturn.OK
            self.last_mjpeg_at[cam_id] = now

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return self.Gst.FlowReturn.OK

        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))

        success, map_info = buffer.map(self.Gst.MapFlags.READ)
        if not success:
            return self.Gst.FlowReturn.OK

        try:
            import cv2
            import numpy as np

            frame_rgba = np.ndarray(
                shape=(height, width, 4),
                dtype=np.uint8,
                buffer=map_info.data,
            )
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            quality = int(self.ds.get("jpeg_quality", 70))
            ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                state = cam_state[cam_id]
                with state["lock"]:
                    state["frame_jpg"] = jpg.tobytes()
        finally:
            buffer.unmap(map_info)

        return self.Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message) -> None:
        msg_type = message.type
        if msg_type == self.Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("DeepStream pipeline error: %s | debug=%s", err, debug)
            self._schedule_restart()
        elif msg_type == self.Gst.MessageType.EOS:
            log.warning("DeepStream pipeline reached EOS")
            self._schedule_restart()
        elif msg_type == self.Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            log.warning("DeepStream pipeline warning: %s | debug=%s", warn, debug)

    def _schedule_restart(self) -> None:
        if self.stopping:
            return
        if self.restart_pending:
            return
        self.restart_pending = True
        try:
            self.pipeline.set_state(self.Gst.State.NULL)
        except Exception:
            log.exception("Failed to stop DeepStream pipeline before restart")

        reconnect_sec = int(self.ds.get("reconnect_sec", 5))
        log.info("Scheduling DeepStream pipeline restart in %d seconds", reconnect_sec)
        self.GLib.timeout_add_seconds(reconnect_sec, self._restart_pipeline)

    def _restart_pipeline(self):
        if self.stopping:
            self.restart_pending = False
            return False
        ret = self.pipeline.set_state(self.Gst.State.PLAYING)
        if ret == self.Gst.StateChangeReturn.FAILURE:
            log.error("DeepStream pipeline restart failed; will retry")
            self.GLib.timeout_add_seconds(int(self.ds.get("reconnect_sec", 5)), self._restart_pipeline)
        else:
            self.restart_pending = False
            log.info("DeepStream pipeline restart requested")
        return False
