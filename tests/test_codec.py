import numpy as np
import pytest

from tamis.recognition.codec import decode_embedding, duplicate_key, encode_embedding


def _unit(seed, size=512):
    vector = np.random.default_rng(seed).normal(size=size).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_encoded_embedding_is_a_fixed_width_string():
    # 4-byte float32 scale + 512 int8 values = 516 bytes -> 688 base64 chars.
    # Fixed width regardless of content, unlike the decimal-text encoding this
    # replaced (which averaged ~12,285 characters and varied per value).
    for seed in range(5):
        assert len(encode_embedding(_unit(seed))) == 688


def test_round_trip_stays_within_the_quantization_bound():
    for seed in range(20):
        original = _unit(seed)
        restored = decode_embedding(encode_embedding(original))
        assert restored.dtype == np.float32
        assert restored.shape == original.shape
        assert np.abs(restored - original).max() <= float(np.abs(original).max()) / 254.0


def test_round_trip_preserves_cosine_similarity():
    # The property that actually matters: recognition only ever compares
    # embeddings by cosine similarity.
    for seed in range(20):
        original = _unit(seed)
        restored = decode_embedding(encode_embedding(original))
        cosine = float(np.dot(restored, original) / (np.linalg.norm(restored) * np.linalg.norm(original)))
        assert cosine > 0.9999


@pytest.mark.parametrize("magnitude", [1e-4, 0.18, 1.0, 47.0])
def test_encoding_is_idempotent_at_any_magnitude(magnitude):
    # This is what makes duplicate_key an exact match test between a freshly
    # computed embedding and the copy of itself that went through the sidecar.
    # Dequantizing gives `q * scale`, whose peak is exactly `127 * scale`, so
    # re-quantizing recovers the same scale and the same q.
    original = _unit(0) * magnitude
    once = encode_embedding(original)
    assert encode_embedding(decode_embedding(once)) == once
    assert encode_embedding(decode_embedding(decode_embedding(once).copy())) == once


def test_duplicate_key_matches_across_a_round_trip_but_not_across_faces():
    original = _unit(0)
    assert duplicate_key(decode_embedding(encode_embedding(original))) == duplicate_key(original)
    assert duplicate_key(_unit(1)) != duplicate_key(original)


def test_decode_accepts_the_legacy_plain_float_list():
    # Every version before this encoding stored embeddings as JSON float
    # arrays; those sidecars have to keep loading with no migration step.
    original = _unit(0)
    restored = decode_embedding(original.tolist())
    assert restored.dtype == np.float32
    assert np.allclose(restored, original, atol=1e-6)


def test_all_zero_embedding_round_trips_without_nan():
    zeros = np.zeros(512, dtype=np.float32)
    restored = decode_embedding(encode_embedding(zeros))
    assert not np.isnan(restored).any()
    assert np.all(restored == 0)


def test_decoded_embedding_is_writable():
    # np.frombuffer alone hands back a read-only view onto the decoded bytes,
    # which would make the array unusable anywhere it gets modified in place.
    restored = decode_embedding(encode_embedding(_unit(0)))
    restored[0] = 1.0  # must not raise ValueError: assignment destination is read-only
