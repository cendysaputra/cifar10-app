import numpy as np
import pytest
from PIL import Image

from classify import CLASS_NAMES, model_path, preprocess, summarize_prediction


def test_saved_model_outputs_a_probability_distribution():
    pytest.importorskip("tensorflow")
    from tensorflow.keras.models import load_model

    path = model_path()
    if not path.is_file():
        pytest.skip(f"{path.name} tidak ada")

    model = load_model(str(path), compile=False)
    image = Image.new("RGB", (48, 36), (30, 120, 200))
    raw = np.asarray(model(preprocess(image), training=False))
    summary = summarize_prediction(raw)

    assert summary.label in CLASS_NAMES
    total = sum(item.probability for item in summary.ranking)
    assert total == pytest.approx(1.0, abs=1e-4)
    assert all(item.probability >= -1e-5 for item in summary.ranking)
    assert len(summary.ranking) == len(CLASS_NAMES)
