import logging
from typing import Optional

from core.deepstream_pipeline import DeepStreamPipeline


log = logging.getLogger("multicam")

_deepstream_pipeline: Optional[DeepStreamPipeline] = None


def start_pipelines(model=None) -> DeepStreamPipeline:
    """Start the DeepStream/GStreamer pipeline.

    The old Python path passed a YOLOModel instance into this function. The
    argument is accepted only for compatibility; TensorRT inference is now owned
    by DeepStream nvinfer.
    """
    global _deepstream_pipeline

    if model is not None:
        log.warning("Ignoring legacy model argument; DeepStream nvinfer owns inference.")

    if _deepstream_pipeline is not None:
        return _deepstream_pipeline

    pipeline = DeepStreamPipeline()
    pipeline.start()
    _deepstream_pipeline = pipeline
    return pipeline


def stop_pipelines() -> None:
    global _deepstream_pipeline
    if _deepstream_pipeline is None:
        return
    _deepstream_pipeline.stop()
    _deepstream_pipeline = None
