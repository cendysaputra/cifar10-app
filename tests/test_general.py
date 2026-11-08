import numpy as np
import pytest
from PIL import Image

import general
from general import (
    MIN_LABEL_GAP,
    NULL_MARGIN,
    GeneralModelError,
    load_embedding_pack,
    preprocess_general,
    score_embeddings,
)
from general_labels import LABELS, NULL_PROMPTS, SPECIAL_PROMPTS, label_by_id, prompt_texts


def _label(key: str):
    return next(item for item in LABELS if item.key == key)


def test_label_list_covers_everyday_categories():
    assert 200 <= len(LABELS) <= 400
    keys = [item.key for item in LABELS]
    names = [item.name_id for item in LABELS]
    assert len(keys) == len(set(keys))
    assert len(names) == len(set(names))
    for key in ("person", "man", "woman", "house", "building", "rock", "cat", "car", "pizza"):
        assert key in keys
    categories = {item.category for item in LABELS}
    assert categories == {
        "orang",
        "bangunan",
        "alam",
        "hewan",
        "makanan",
        "kendaraan",
        "benda",
        "pakaian",
        "tempat",
    }
    assert all(prompt_texts(item) for item in LABELS)
    assert set(SPECIAL_PROMPTS) <= set(keys)


def test_person_house_and_rock_prompts_are_explicit():
    assert "a photo of a human." in prompt_texts(_label("person"))
    assert prompt_texts(_label("house"))[0] == "a photo of a house."
    rock_prompts = prompt_texts(_label("rock"))
    assert "a photo of a rock." in rock_prompts
    assert "a photo of a stone." in rock_prompts
    assert "a photo of a boulder." in rock_prompts
    assert _label("person").name_id == "Orang"
    assert _label("house").name_id == "Rumah"
    assert _label("rock").name_id == "Batu"
    assert label_by_id("Rumah").key == "house"


def test_default_prompt_uses_an_article_for_vowels():
    assert prompt_texts(_label("apple")) == ("a photo of an apple.",)
    assert prompt_texts(_label("car")) == ("a photo of a car.",)


def test_preprocess_is_clip_batch():
    batch = preprocess_general(Image.new("RGB", (80, 40), (10, 20, 30)))
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == np.float32
    assert batch.flags["C_CONTIGUOUS"]
    assert np.all(np.isfinite(batch))


def test_preprocess_limits_extreme_aspect_ratio():
    batch = preprocess_general(Image.new("RGB", (4000, 8), (255, 0, 0)))
    assert batch.shape == (1, 3, 224, 224)
    assert np.all(np.isfinite(batch))


def test_saved_pack_matches_the_label_list():
    pack = load_embedding_pack()
    expected_rows = sum(len(prompt_texts(item)) for item in LABELS)
    assert pack["embeddings"].shape == (expected_rows, 512)
    assert pack["null_embeddings"].shape == (len(NULL_PROMPTS), 512)
    assert set(np.unique(pack["label_index"])).issuperset(set(range(len(LABELS))))


def test_saved_rock_prompt_predicts_batu():
    pack = load_embedding_pack()
    rock_index = next(index for index, item in enumerate(LABELS) if item.key == "rock")
    row = int(np.flatnonzero(pack["label_index"] == rock_index)[0])
    prediction = score_embeddings(
        pack["embeddings"][row],
        pack["embeddings"],
        pack["label_index"],
        pack["null_embeddings"],
    )
    assert prediction.label == "Batu"
    assert not prediction.low_confidence
    assert prediction.ranking[0].probability >= prediction.ranking[1].probability
    assert len(prediction.ranking) == len(LABELS)


def test_null_prompt_is_not_sure():
    pack = load_embedding_pack()
    prediction = score_embeddings(
        pack["null_embeddings"][0],
        pack["embeddings"],
        pack["label_index"],
        pack["null_embeddings"],
    )
    assert prediction.low_confidence


def test_clear_synthetic_match_is_confident():
    count = len(LABELS)
    width = count + len(NULL_PROMPTS)
    prompts = np.zeros((count, width), dtype=np.float32)
    for index in range(count):
        prompts[index, index] = 1
    nulls = np.zeros((len(NULL_PROMPTS), width), dtype=np.float32)
    for index in range(len(NULL_PROMPTS)):
        nulls[index, count + index] = 1
    prediction = score_embeddings(prompts[0], prompts, np.arange(count), nulls)
    assert prediction.label == LABELS[0].name_id
    assert not prediction.low_confidence
    assert prediction.confidence == pytest.approx(1)


def test_small_gap_is_not_sure():
    count = len(LABELS)
    width = 16
    rng = np.random.default_rng(0)
    prompts = rng.normal(size=(count, width)).astype(np.float32)
    prompts /= np.linalg.norm(prompts, axis=1, keepdims=True)
    prompts[1] = prompts[0] + 0.001 * prompts[2]
    prompts[1] /= np.linalg.norm(prompts[1])
    nulls = rng.normal(size=(len(NULL_PROMPTS), width)).astype(np.float32)
    for index in range(len(nulls)):
        nulls[index] -= prompts[0] * float(np.dot(nulls[index], prompts[0]))
        nulls[index] /= np.linalg.norm(nulls[index])
    prediction = score_embeddings(prompts[0], prompts, np.arange(count), nulls)
    assert prediction.label == LABELS[0].name_id
    assert prediction.low_confidence
    assert NULL_MARGIN > 0
    assert MIN_LABEL_GAP > 0


def test_pack_rejects_mismatched_keys(tmp_path, monkeypatch):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        embeddings=np.zeros((1, 512), dtype=np.float32),
        label_index=np.zeros(1, dtype=np.int32),
        null_embeddings=np.zeros((len(NULL_PROMPTS), 512), dtype=np.float32),
        keys=np.asarray(["bukan-label"]),
    )
    monkeypatch.setattr(general, "embeddings_path", lambda: path)
    with pytest.raises(GeneralModelError):
        load_embedding_pack()
