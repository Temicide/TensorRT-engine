from config import CONFIG
from core.tensorrt_engine import YOLOModel
from core.tkdnn_engine import TkDNNDarknetModel


def load_detector_model():
    backend = CONFIG.get("inference_backend", "tensorrt_engine")

    if backend == "tensorrt_engine":
        return YOLOModel(CONFIG["model_path"])

    if backend == "tkdnn_darknet":
        return TkDNNDarknetModel(CONFIG.get("tkdnn", {}))

    raise RuntimeError(
        "Unsupported inference_backend={!r}. Use 'tensorrt_engine' or "
        "'tkdnn_darknet'.".format(backend)
    )

