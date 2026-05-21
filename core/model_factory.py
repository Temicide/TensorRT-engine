from config import CONFIG
from core.tensorrt_engine import YOLOModel
from core.tkdnn_engine import TkDNNDarknetModel, TkDNNDarknetModelPool


def load_detector_model():
    backend = CONFIG.get("inference_backend", "tensorrt_engine")

    if backend == "tensorrt_engine":
        return YOLOModel(CONFIG["model_path"])

    if backend == "tkdnn_darknet":
        tkdnn_config = CONFIG.get("tkdnn", {})
        num_cameras = len(CONFIG.get("active_cameras") or CONFIG.get("cameras", {}))
        num_instances = tkdnn_config.get("num_bridge_instances", num_cameras)
        if num_instances < 1:
            num_instances = 1
        return TkDNNDarknetModelPool(tkdnn_config, num_instances)

    raise RuntimeError(
        "Unsupported inference_backend={!r}. Use 'tensorrt_engine' or "
        "'tkdnn_darknet'.".format(backend)
    )

