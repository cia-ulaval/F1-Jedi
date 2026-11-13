"""Convert hand landmarks into the feature vector used during training and inference."""

def features(landmarks):
    """Preserve camera-relative rotation; remove translation and hand size."""
    import numpy as np

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite x/y/z landmarks")
    centered = points - points[0]
    scale = np.linalg.norm(centered[9])  # wrist to middle-finger MCP
    if scale < 1e-6:
        raise ValueError("Degenerate hand landmarks")
    return (centered / scale).reshape(63)
