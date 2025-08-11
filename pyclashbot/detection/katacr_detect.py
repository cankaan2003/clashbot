"""Wrapper around KataCR's detection model.

This module provides a thin adapter to use the detection
implementation from `https://github.com/wty-yy/KataCR`.
It expects the `katacr` project to be available in the Python
environment. The heavy lifting is performed by KataCR's
`Infer` class while this wrapper converts screenshots from the
bot into a format suitable for the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from katacr.detection.detect import Infer as KataCRInfer

try:  # pragma: no cover - imported at runtime
    # `Infer` is the high level detection helper used by KataCR's CLI.
    from katacr.detection.detect import Infer as _KataCRInfer
except Exception as exc:  # pragma: no cover - import guard
    _KataCRInfer = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - executed only when import succeeds
    _IMPORT_ERROR = None


class KataCRDetector:
    """Run object detection using KataCR's pretrained YOLO model.

    Parameters
    ----------
    model_name:
        Name of the model directory inside KataCR's ``logs`` folder.
    load_id:
        Checkpoint identifier to load.  Refer to KataCR's documentation
        for available values.
    path_model:
        Optional explicit path to a model checkpoint.  When ``None`` the
        default checkpoints bundled with KataCR are used.
    iou_thre, conf_thre:
        Threshold values forwarded to KataCR's non-maximum suppression
        routine.

    Notes
    -----
    The KataCR project is not distributed on PyPI.  To use this wrapper
    the repository must be available and importable as ``katacr``.  If the
    package cannot be imported an :class:`ImportError` is raised when
    instantiating the detector.
    """

    def __init__(
        self,
        model_name: str = "YOLOv5_v0.5",
        load_id: int = 150,
        path_model: str | None = None,
        iou_thre: float = 0.5,
        conf_thre: float = 0.1,
    ) -> None:
        if _KataCRInfer is None:  # pragma: no cover - runtime check
            raise ImportError(f"KataCR is not installed or failed to import. Original error: {_IMPORT_ERROR}")

        self._infer: KataCRInfer = _KataCRInfer(
            model_name=model_name,
            load_id=load_id,
            path_model=path_model,
            iou_thre=iou_thre,
            conf_thre=conf_thre,
        )

    def __call__(self, image: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """Detect objects within ``image``.

        Parameters
        ----------
        image:
            RGB image as a ``numpy.ndarray`` with shape ``(H, W, 3)`` and
            ``dtype=uint8``.

        Returns
        -------
        list[numpy.ndarray]
            A list of bounding boxes in ``xywh`` format produced by the
            KataCR model.  Each bounding box is a ``numpy.ndarray`` with
            shape ``(N, 7)`` where the columns correspond to
            ``x, y, w, h, conf, state, class``.  The exact format is
            defined by KataCR's ``Infer`` implementation.
        """

        if image.ndim != 3:
            raise ValueError("image must be a HxWx3 RGB array")
        # The `Infer` class expects a numpy array with shape (1, H, W, 3).
        batch: NDArray[np.uint8] = np.ascontiguousarray(image[None, ...])
        boxes: list[NDArray[np.float32]] = self._infer(batch)
        return boxes


__all__ = ["KataCRDetector"]
