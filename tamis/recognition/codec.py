"""On-disk encoding for face embeddings, shared by FaceCatalog and PersonGallery.

Both sidecars used to store each 512-d embedding as a JSON array of decimal
floats. That is enormously wasteful: the embeddings are float32 in memory, so
each value carries 4 bytes of real information, and `json.dumps` wrote it out
as ~24 characters of float64 decimal text (the full repr of a number that
never had that much precision). Measured on a real 34-photo folder, one
embedding took ~12,285 characters and the sidecar reached 4.9MB; serializing
it cost 71ms, on every single face confirmation.

Here each embedding is instead quantized to int8 with a per-vector float32
scale, and the pair is base64'd into one JSON string:

    4 bytes  scale (float32, little-endian)
  512 bytes  int8 values, dequantized as `value * scale`
  --------
  516 bytes -> 688 base64 characters, fixed width

That is ~18x smaller than the old encoding and ~2.5ms to write. int8 is not
a compromise here: the embeddings are L2-normalized and tightly bounded
(measured range [-0.1832, +0.1707] with no outliers), so quantization error
is at most `|max|/254` ~= 7e-4 per component, which perturbs a cosine
similarity by at most ~0.0018. For comparison, the median gap between the
best and second-best candidate person is ~0.067 (37x larger), and the
model's own ambiguity is larger still -- the recognition design notes record
a genuine cross-photo match at 0.555 and unrelated siblings at 0.61 against
each other. Verified on real data: of 313 catalog faces scored against 28
people, quantization changed the top-1 suggestion for exactly 2, and both
were already exact or near-exact ties (margins of 0.000000 and 0.000418)
that float32 was also deciding arbitrarily.

The per-vector scale is stored rather than assumed, so a future embedding
with a different magnitude can never silently clip.
"""

from __future__ import annotations

import base64

import numpy as np

EMBEDDING_SIZE = 512

def encode_embedding(embedding: np.ndarray) -> str:
    """Encode one embedding as a base64 string (see this module's docstring)."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    peak = float(np.abs(vector).max())
    # An all-zero embedding has no meaningful scale; any non-zero value works
    # since every quantized component is zero anyway, and it keeps the
    # decoder's multiply from producing NaN.
    scale = peak / 127.0 if peak > 0 else 1.0
    quantized = np.clip(np.round(vector / scale), -127, 127).astype(np.int8)
    # Little-endian is pinned explicitly rather than left as native order:
    # these sidecars live inside photo folders, which get synced and copied
    # between machines.
    header = np.array([scale], dtype="<f4").tobytes()
    return base64.b64encode(header + quantized.tobytes()).decode("ascii")


def decode_embedding(value: object) -> np.ndarray:
    """Decode an embedding written by `encode_embedding`, or a plain list of
    floats as written by every version before this encoding existed -- both
    sidecar formats stored embeddings as JSON arrays, so files predating this
    change stay readable with no migration step and no version field. The two
    are unambiguous: the current format is a string, the legacy one a list.
    """
    if isinstance(value, str):
        raw = base64.b64decode(value)
        scale = float(np.frombuffer(raw[:4], dtype="<f4")[0])
        # `* scale` produces a new (writable) array; np.frombuffer alone
        # would hand back a read-only view onto the decoded bytes.
        return np.frombuffer(raw[4:], dtype=np.int8).astype(np.float32) * scale
    return np.array(value, dtype=np.float32)


def duplicate_key(embedding: np.ndarray) -> str:
    """A hashable key identifying "the same face sample", for the places that
    have to match an embedding against stored ones by value rather than by
    object identity (PersonGallery.remove_embedding, and the duplicate guard
    in add_embedding).

    This is exact and independent of the vector's magnitude, because encoding
    is idempotent: `encode(decode(encode(v))) == encode(v)`. Dequantizing
    yields `q * scale`, whose peak is exactly `127 * scale`, so re-quantizing
    recovers the same scale and the same `q`. A freshly computed float32
    embedding therefore produces the same key as the copy of itself that has
    been through the sidecar -- which is what these comparisons actually need,
    since one side is routinely stored and the other freshly detected.

    An absolute tolerance would not do the job: the necessary threshold is a
    function of the vector's own magnitude (quantization error is
    `|max|/254`), so any fixed value is either too tight for large-magnitude
    embeddings or loose enough to conflate genuinely different faces. The
    previous code hard-coded `atol=1e-6`, which was fine only while
    embeddings were stored at full float precision.

    Being a plain string, it also turns duplicate detection into a dict
    lookup rather than an O(n^2) sweep of array comparisons.
    """
    return encode_embedding(embedding)
