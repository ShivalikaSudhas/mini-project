"""
tests/test_pipeline.py — End-to-end sanity tests for the screening pipeline.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import pathlib

# Ensure repo root is importable
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Import smoke tests — catch any NameError / ImportError immediately
# ---------------------------------------------------------------------------

def test_import_visual_model():
    from src.visual_model import (
        VisualForensicModel,
        get_default_visual_model,
        predict_visual,
        extract_visual_features,
        _preprocess_input_image,
    )

def test_import_ocr_model():
    from src.ocr_model import run_ocr, extract_text

def test_import_explainability():
    from src.explainability import visualize_gradcam, generate_and_save_gradcam


# ---------------------------------------------------------------------------
# 2. Unit tests with a dummy image (no checkpoint / dataset needed)
# ---------------------------------------------------------------------------

def _make_dummy_image() -> Image.Image:
    """Create a small synthetic RGB image."""
    return Image.new("RGB", (320, 200), color=(180, 140, 120))


def test_preprocess_pil():
    from src.visual_model import _preprocess_input_image
    img = _make_dummy_image()
    tensor = _preprocess_input_image(img)
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected shape: {tensor.shape}"


def test_preprocess_numpy():
    from src.visual_model import _preprocess_input_image
    arr = np.array(_make_dummy_image())
    tensor = _preprocess_input_image(arr)
    assert tensor.shape == (1, 3, 224, 224)


def test_predict_visual():
    from src.visual_model import predict_visual
    img = _make_dummy_image()
    result = predict_visual(img)
    # Check all expected keys
    for key in ("prediction", "class_id", "probability_genuine",
                "probability_forged", "confidence", "features"):
        assert key in result, f"Missing key: {key}"
    assert result["prediction"] in ("genuine", "forged")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["features"].shape == (1280,)


def test_extract_visual_features():
    from src.visual_model import extract_visual_features
    img = _make_dummy_image()
    feats = extract_visual_features(img)
    assert feats.shape == (1280,)
    # Should be L2-normalised → norm ≈ 1
    assert abs(np.linalg.norm(feats) - 1.0) < 0.01


def test_extract_text_numpy():
    from src.ocr_model import extract_text
    arr = np.array(_make_dummy_image())
    text = extract_text(arr)
    assert isinstance(text, str)


def test_visualize_gradcam():
    from src.visual_model import get_default_visual_model
    from src.explainability import visualize_gradcam
    img = _make_dummy_image()
    model = get_default_visual_model()
    cam = visualize_gradcam(img, model)
    assert isinstance(cam, np.ndarray)
    # Same spatial dimensions as input
    assert cam.shape[:2] == (200, 320)


# ---------------------------------------------------------------------------
# 3. Frontend import test — make sure app.py can be parsed without crashing
# ---------------------------------------------------------------------------

def test_frontend_syntax():
    """Compile frontend/app.py to check for syntax errors (does NOT run Streamlit)."""
    app_path = repo_root / "frontend" / "app.py"
    assert app_path.exists(), f"frontend/app.py not found at {app_path}"
    source = app_path.read_text(encoding="utf-8")
    compile(source, str(app_path), "exec")  # raises SyntaxError on bad code


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
