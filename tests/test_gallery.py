import pytest

np = pytest.importorskip("numpy")  # PersonGallery only needs numpy, but keep it optional like the rest of recognition

import tamis.recognition.gallery as gallery_module  # noqa: E402
from tamis.recognition.gallery import PersonGallery  # noqa: E402


def _empty_gallery(tmp_path):
    return PersonGallery(path=tmp_path / "people.json")


def test_add_person_and_identify_a_close_embedding(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    base = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, base)

    close = base + np.random.default_rng(1).normal(scale=0.01, size=512).astype(np.float32)
    ranked = gallery.identify(close)

    assert len(ranked) == 1
    matched_person, similarity = ranked[0]
    assert matched_person.id == person.id
    assert similarity > 0.9


def test_identify_returns_empty_for_empty_gallery(tmp_path):
    gallery = _empty_gallery(tmp_path)
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert gallery.identify(embedding) == []


def test_identify_ranks_k_nearest_distinct_people_most_similar_first(tmp_path):
    # Alice contributes 2 of the 5 nearest samples (both very close to the
    # query), Bob/Carol/Dave contribute one increasingly-distant sample each
    # -- together exactly the k=5 nearest of 6 total samples. Eve's sample is
    # the least similar of all and should be excluded entirely by k=5, even
    # though her people-count (1 person) would otherwise fit.
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    query = rng.normal(size=512).astype(np.float32)

    def offset_sample(scale):
        return query + rng.normal(scale=scale, size=512).astype(np.float32)

    alice = gallery.add_person("Alice")
    gallery.add_embedding(alice.id, offset_sample(0.01))
    gallery.add_embedding(alice.id, offset_sample(0.02))

    bob = gallery.add_person("Bob")
    gallery.add_embedding(bob.id, offset_sample(0.1))

    carol = gallery.add_person("Carol")
    gallery.add_embedding(carol.id, offset_sample(0.3))

    dave = gallery.add_person("Dave")
    gallery.add_embedding(dave.id, offset_sample(0.6))

    eve = gallery.add_person("Eve")
    gallery.add_embedding(eve.id, offset_sample(3.0))

    ranked = gallery.identify(query, k=5)

    names = [person.name for person, _ in ranked]
    assert names == ["Alice", "Bob", "Carol", "Dave"]  # Alice once, despite 2 samples in the top-5
    assert "Eve" not in names  # her sample wasn't among the k=5 nearest
    similarities = [similarity for _, similarity in ranked]
    assert similarities == sorted(similarities, reverse=True)


def test_identify_k_limits_how_many_samples_are_considered(tmp_path):
    # Alice's sample is the single closest; with k=1, only that one sample is
    # examined, so Bob (a decent but not-nearest match) never gets a chance.
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    query = rng.normal(size=512).astype(np.float32)

    alice = gallery.add_person("Alice")
    gallery.add_embedding(alice.id, query + rng.normal(scale=0.01, size=512).astype(np.float32))

    bob = gallery.add_person("Bob")
    gallery.add_embedding(bob.id, query + rng.normal(scale=0.3, size=512).astype(np.float32))

    ranked = gallery.identify(query, k=1)

    assert [person.name for person, _ in ranked] == ["Alice"]


def test_identify_has_no_similarity_floor(tmp_path):
    # No min_similarity parameter exists anymore -- a genuinely unrelated
    # person still comes back (as long as they fit within k), just with a
    # low score. Callers convey confidence via that score (e.g. color), not
    # by asking identify() to hide poor matches.
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    gallery.add_embedding(person.id, rng.normal(size=512).astype(np.float32))

    unrelated = rng.normal(size=512).astype(np.float32)
    ranked = gallery.identify(unrelated)

    assert len(ranked) == 1
    assert ranked[0][0].id == person.id


def test_add_embedding_to_unknown_person_raises(tmp_path):
    gallery = _empty_gallery(tmp_path)
    with pytest.raises(ValueError):
        gallery.add_embedding("no-such-id", np.zeros(512, dtype=np.float32))


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "people.json"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, embedding)
    gallery.save()

    reloaded = PersonGallery(path=path)

    assert len(reloaded.people) == 1
    assert reloaded.people[0].name == "Alice"
    assert reloaded.people[0].id == person.id
    assert len(reloaded.people[0].embeddings) == 1
    # Embeddings are stored quantized to int8 with a per-vector scale (see
    # tamis.recognition.codec), so a round trip is lossy by design -- bounded
    # by |max|/254 per component. What has to survive is the *similarity*
    # this data exists to compute, not the exact bytes.
    reloaded_embedding = reloaded.people[0].embeddings[0]
    tolerance = float(np.abs(embedding).max()) / 254.0
    assert np.allclose(reloaded_embedding, embedding, atol=tolerance)
    cosine = float(
        np.dot(reloaded_embedding, embedding)
        / (np.linalg.norm(reloaded_embedding) * np.linalg.norm(embedding))
    )
    assert cosine > 0.9999


def test_load_missing_file_starts_empty(tmp_path):
    gallery = PersonGallery(path=tmp_path / "does_not_exist.json")
    assert gallery.people == []
    assert gallery.load_error is None


def test_load_corrupted_existing_file_sets_load_error(tmp_path):
    path = tmp_path / "people.json.gz"
    path.write_bytes(b"not a valid gzip file")

    gallery = PersonGallery(path=path)

    assert gallery.people == []
    assert gallery.load_error is not None
    assert str(path) in gallery.load_error


def test_load_success_leaves_load_error_none(tmp_path):
    path = tmp_path / "people.json"
    gallery = PersonGallery(path=path)
    gallery.add_person("Alice")
    gallery.save()

    reloaded = PersonGallery(path=path)

    assert reloaded.load_error is None


def test_remove_person(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    gallery.remove_person(person.id)
    assert gallery.people == []


def test_rank_all_includes_everyone_regardless_of_similarity_or_k(tmp_path):
    # Unlike identify(), rank_all() has no threshold or k cutoff: even a
    # near-opposite match should still appear, just ranked last.
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    query = rng.normal(size=512).astype(np.float32)

    close = gallery.add_person("Close")
    gallery.add_embedding(close.id, query + rng.normal(scale=0.01, size=512).astype(np.float32))

    far = gallery.add_person("Far")
    gallery.add_embedding(far.id, -query)  # deliberately near-opposite

    ranked = gallery.rank_all(query)

    assert [person.name for person, _ in ranked] == ["Close", "Far"]
    similarities = [similarity for _, similarity in ranked]
    assert similarities == sorted(similarities, reverse=True)


def test_rank_all_excludes_people_with_no_embeddings(tmp_path):
    gallery = _empty_gallery(tmp_path)
    gallery.add_person("NoSamplesYet")
    query = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert gallery.rank_all(query) == []


def test_remove_embedding_removes_the_matching_sample(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    kept = rng.normal(size=512).astype(np.float32)
    removed = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, kept)
    gallery.add_embedding(person.id, removed)

    result = gallery.remove_embedding(person.id, removed)

    assert result is True
    assert len(person.embeddings) == 1
    assert np.allclose(person.embeddings[0], kept)


def test_remove_embedding_deletes_person_left_with_zero_samples(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, embedding)

    result = gallery.remove_embedding(person.id, embedding)

    assert result is True
    assert gallery.find_by_id(person.id) is None
    assert gallery.people == []


def test_remove_embedding_matches_by_value_not_identity(tmp_path):
    # A reloaded-from-JSON array is never the same object as the one that was
    # originally added -- matching must be by value.
    path = tmp_path / "people.json"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, embedding)
    gallery.save()

    reloaded = PersonGallery(path=path)
    reloaded_person = reloaded.people[0]
    same_value_different_object = np.array(embedding, dtype=np.float32, copy=True)

    assert reloaded.remove_embedding(reloaded_person.id, same_value_different_object) is True
    assert reloaded_person.embeddings == []


def test_remove_embedding_returns_false_when_not_found(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    unrelated = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert gallery.remove_embedding(person.id, unrelated) is False
    assert gallery.remove_embedding("no-such-id", unrelated) is False


def test_merge_combines_embeddings_and_removes_the_source(tmp_path):
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    lowercase = gallery.add_person("papa")
    gallery.add_embedding(lowercase.id, rng.normal(size=512).astype(np.float32))
    capitalized = gallery.add_person("Papa")
    gallery.add_embedding(capitalized.id, rng.normal(size=512).astype(np.float32))
    gallery.add_embedding(capitalized.id, rng.normal(size=512).astype(np.float32))

    gallery.merge(keep_id=capitalized.id, remove_id=lowercase.id)

    assert lowercase.id not in {person.id for person in gallery.people}
    kept = gallery.find_by_id(capitalized.id)
    assert kept.name == "Papa"
    assert len(kept.embeddings) == 3
    # The merged-away id doesn't dangle: it redirects to the surviving person,
    # so face records naming it (in any folder, including ones not open at
    # merge time) still resolve. See PersonGallery.merge.
    assert gallery.find_by_id(lowercase.id) is kept
    assert gallery.merged_ids[lowercase.id] == capitalized.id


def test_merge_into_self_raises(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    with pytest.raises(ValueError):
        gallery.merge(keep_id=person.id, remove_id=person.id)


def test_merge_unknown_id_raises(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    with pytest.raises(ValueError):
        gallery.merge(keep_id=person.id, remove_id="no-such-id")


def test_save_writes_gzip_compressed_json(tmp_path):
    import gzip
    import json

    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    gallery.add_embedding(person.id, np.zeros(512, dtype=np.float32))
    gallery.save()

    raw = path.read_bytes()
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        json.loads(raw)  # not readable as plain JSON -- proves it's actually compressed
    decompressed = json.loads(gzip.decompress(raw))
    assert decompressed["people"][0]["name"] == "Alice"


def test_save_writes_atomically(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    gallery.add_person("Alice")
    gallery.save()
    original_bytes = path.read_bytes()

    gallery.add_person("Bob")

    # Writes garbage to whatever file is actually targeted, then fails --
    # simulating a real partial/failed write. A mock that merely raises
    # without writing anything would trivially "pass" even for a
    # non-atomic implementation, since no bytes would ever land anywhere
    # either way -- this is the detail that actually distinguishes them.
    real_write_bytes = Path.write_bytes

    def failing_write_bytes(self, data):
        real_write_bytes(self, b"CORRUPTED-PARTIAL-WRITE")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

    with pytest.raises(OSError):
        gallery.save()

    # The original must be untouched -- save() writes to a temp file first
    # and only replaces the original once the write fully succeeds.
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".tamis_write_*"))


def test_load_migrates_from_legacy_uncompressed_file_without_deleting_it(tmp_path):
    import json

    legacy_path = tmp_path / "people.json"
    legacy_path.write_text(json.dumps({"people": [{"id": "abc123", "name": "Alice", "embeddings": [[0.0] * 512]}]}))

    gz_path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=gz_path)

    assert [p.name for p in gallery.people] == ["Alice"]
    assert not gz_path.exists()  # nothing written yet, this was a read-only migration check

    gallery.save()

    assert gz_path.exists()
    assert legacy_path.exists()  # left in place, not deleted

    reloaded = PersonGallery(path=gz_path)
    assert [p.name for p in reloaded.people] == ["Alice"]


def test_load_migrates_from_the_pre_rename_picsel_directory(tmp_path, monkeypatch):
    import gzip
    import json

    # Only the real DEFAULT_GALLERY_PATH falls back to the pre-rename
    # (picSel -> Tamis) ~/.picsel/ location -- a caller-supplied custom path
    # (every other test in this file) must never consult it. Monkeypatch
    # both module-level path constants to isolated tmp_path locations so
    # this test can exercise that exact branch without touching the real
    # home directory.
    new_path = tmp_path / "tamis" / "people.json.gz"
    legacy_path = tmp_path / "picsel" / "people.json.gz"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(
        gzip.compress(
            json.dumps({"people": [{"id": "abc123", "name": "Alice", "embeddings": [[0.0] * 512]}]}).encode()
        )
    )
    monkeypatch.setattr(gallery_module, "DEFAULT_GALLERY_PATH", new_path)
    monkeypatch.setattr(gallery_module, "_LEGACY_GALLERY_PATH", legacy_path)

    gallery = PersonGallery(path=new_path)

    assert [p.name for p in gallery.people] == ["Alice"]
    assert not new_path.exists()  # read-only migration check, nothing written yet
    assert legacy_path.exists()  # left in place, not deleted

    gallery.save()

    assert new_path.exists()


def test_load_does_not_consult_the_legacy_picsel_path_for_a_custom_path(tmp_path, monkeypatch):
    # The pre-rename fallback is gated to DEFAULT_GALLERY_PATH specifically --
    # a custom path (what every test in this file normally uses, and what
    # the real app uses for a test/throwaway PersonGallery) must never reach
    # for it, even if it happens to exist.
    import gzip
    import json

    real_legacy_path = tmp_path / "picsel_home" / "people.json.gz"
    real_legacy_path.parent.mkdir(parents=True)
    real_legacy_path.write_bytes(
        gzip.compress(json.dumps({"people": [{"id": "x", "name": "ShouldNotAppear", "embeddings": []}]}).encode())
    )
    monkeypatch.setattr(gallery_module, "_LEGACY_GALLERY_PATH", real_legacy_path)

    gallery = PersonGallery(path=tmp_path / "custom" / "people.json.gz")

    assert gallery.people == []


def test_export_then_import_into_a_fresh_gallery(tmp_path):
    source = _empty_gallery(tmp_path)
    person = source.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    source.add_embedding(person.id, embedding)

    export_path = tmp_path / "exported.json.gz"
    source.export_to(export_path)

    destination = PersonGallery(path=tmp_path / "other" / "people.json.gz")
    added = destination.import_from(export_path)

    assert added == 1
    assert len(destination.people) == 1
    imported_person = destination.people[0]
    assert imported_person.name == "Alice"
    assert imported_person.id != person.id  # never reuses the source's id
    # Lossy by design -- embeddings are stored quantized, see the round-trip
    # test above.
    assert np.allclose(imported_person.embeddings[0], embedding, atol=float(np.abs(embedding).max()) / 254.0)


def test_import_merges_into_existing_person_with_the_same_name(tmp_path):
    source = _empty_gallery(tmp_path)
    source_person = source.add_person("Alice")
    source.add_embedding(source_person.id, np.random.default_rng(0).normal(size=512).astype(np.float32))
    export_path = tmp_path / "exported.json.gz"
    source.export_to(export_path)

    destination = _empty_gallery(tmp_path)
    existing = destination.add_person("Alice")
    destination.add_embedding(existing.id, np.random.default_rng(1).normal(size=512).astype(np.float32))

    added = destination.import_from(export_path)

    assert added == 0  # merged into the existing "Alice", not added as new
    assert len(destination.people) == 1
    assert len(destination.people[0].embeddings) == 2


def test_import_is_atomic_on_a_malformed_entry(tmp_path):
    import gzip
    import json

    malformed_path = tmp_path / "malformed_export.json.gz"
    malformed_path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "people": [
                        {"id": "a", "name": "Alice", "embeddings": [[0.0] * 512]},
                        {"id": "b", "embeddings": [[0.0] * 512]},  # missing "name"
                    ]
                }
            ).encode("utf-8")
        )
    )

    destination = _empty_gallery(tmp_path)
    with pytest.raises(KeyError):
        destination.import_from(malformed_path)

    # Alice (parsed fine, listed before the malformed entry) must not have
    # been merged in -- the whole import failed, so nothing should change.
    assert destination.people == []


def test_import_accepts_plain_uncompressed_json(tmp_path):
    import json

    plain_path = tmp_path / "plain_export.json"
    plain_path.write_text(json.dumps({"people": [{"id": "x", "name": "Bob", "embeddings": [[0.0] * 512]}]}))

    destination = _empty_gallery(tmp_path)
    added = destination.import_from(plain_path)

    assert added == 1
    assert destination.people[0].name == "Bob"


def test_similarity_to_matches_the_closest_sample(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    rng = np.random.default_rng(0)
    base = rng.normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, rng.normal(size=512).astype(np.float32))  # unrelated, far sample
    gallery.add_embedding(person.id, base)  # the close one

    close = base + rng.normal(scale=0.01, size=512).astype(np.float32)
    assert gallery.similarity_to(person.id, close) > 0.9


def test_similarity_to_unknown_person_is_zero(tmp_path):
    gallery = _empty_gallery(tmp_path)
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert gallery.similarity_to("no-such-id", embedding) == 0.0


def test_similarity_to_person_with_no_samples_is_zero(tmp_path):
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("NoSamplesYet")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert gallery.similarity_to(person.id, embedding) == 0.0


def test_add_embedding_ignores_a_sample_the_person_already_has(tmp_path):
    # Samples carry no identity, so nothing downstream can notice or undo a
    # double-add; duplicates only slow every save down (a real gallery was
    # measured 34% redundant) without improving matching, since a person is
    # scored by their single closest sample.
    gallery = _empty_gallery(tmp_path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)

    gallery.add_embedding(person.id, embedding)
    gallery.add_embedding(person.id, embedding)
    gallery.add_embedding(person.id, embedding.copy())  # equal by value, different object

    assert len(person.embeddings) == 1


def test_add_embedding_recognises_a_sample_that_went_through_disk(tmp_path):
    # The duplicate guard has to survive quantization: the stored copy comes
    # back slightly different from the freshly-computed float32 it was made
    # from, which is exactly why matching is on the encoded form rather than
    # a float tolerance.
    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, embedding)
    gallery.save()

    reloaded = PersonGallery(path=path)
    reloaded.add_embedding(reloaded.people[0].id, embedding)

    assert len(reloaded.people[0].embeddings) == 1


def test_remove_embedding_matches_a_sample_that_went_through_disk(tmp_path):
    # Same reasoning as above, for the other direction: relabeling a face has
    # to be able to find and drop the sample added under its previous label,
    # or that sample is orphaned in the wrong person's reference data.
    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    gallery.add_embedding(person.id, embedding)
    gallery.save()

    reloaded = PersonGallery(path=path)
    assert reloaded.remove_embedding(reloaded.people[0].id, embedding) is True


def test_merge_does_not_duplicate_samples_both_people_already_had(tmp_path):
    # The same face is routinely labeled in more than one folder (the face
    # cache is per-folder, the gallery is global), so the two entries being
    # merged very often overlap.
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    shared = rng.normal(size=512).astype(np.float32)
    lowercase = gallery.add_person("papa")
    gallery.add_embedding(lowercase.id, shared)
    capitalized = gallery.add_person("Papa")
    gallery.add_embedding(capitalized.id, shared.copy())
    gallery.add_embedding(capitalized.id, rng.normal(size=512).astype(np.float32))

    gallery.merge(keep_id=capitalized.id, remove_id=lowercase.id)

    assert len(gallery.find_by_id(capitalized.id).embeddings) == 2


def test_merge_redirects_survive_a_save_and_reload(tmp_path):
    # The redirect is what makes a merge reach folders that were closed when
    # it happened, so it has to outlive the session that did the merging.
    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    rng = np.random.default_rng(0)
    lowercase = gallery.add_person("papa")
    gallery.add_embedding(lowercase.id, rng.normal(size=512).astype(np.float32))
    capitalized = gallery.add_person("Papa")
    gallery.add_embedding(capitalized.id, rng.normal(size=512).astype(np.float32))
    gallery.merge(keep_id=capitalized.id, remove_id=lowercase.id)
    gallery.save()

    reloaded = PersonGallery(path=path)

    assert reloaded.find_by_id(lowercase.id) is not None
    assert reloaded.find_by_id(lowercase.id).id == capitalized.id


def test_merge_chains_resolve_to_the_final_survivor(tmp_path):
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    a = gallery.add_person("a")
    gallery.add_embedding(a.id, rng.normal(size=512).astype(np.float32))
    b = gallery.add_person("b")
    gallery.add_embedding(b.id, rng.normal(size=512).astype(np.float32))
    c = gallery.add_person("c")
    gallery.add_embedding(c.id, rng.normal(size=512).astype(np.float32))

    gallery.merge(keep_id=b.id, remove_id=a.id)
    gallery.merge(keep_id=c.id, remove_id=b.id)

    assert gallery.find_by_id(a.id).id == c.id
    assert gallery.find_by_id(b.id).id == c.id


def test_resolve_id_terminates_on_a_cyclic_redirect_map(tmp_path):
    # A corrupted or hand-edited file must not be able to hang the app.
    gallery = _empty_gallery(tmp_path)
    gallery.merged_ids = {"a": "b", "b": "a"}
    assert gallery.resolve_id("a") in {"a", "b"}


def test_load_drops_duplicate_samples(tmp_path):
    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    person = gallery.add_person("Alice")
    embedding = np.random.default_rng(0).normal(size=512).astype(np.float32)
    # Appended directly, bypassing add_embedding's guard, to stand in for a
    # gallery written before that guard existed.
    person.embeddings.extend([embedding, embedding.copy(), embedding.copy()])
    gallery.save()

    reloaded = PersonGallery(path=path)

    assert len(reloaded.people[0].embeddings) == 1


def test_load_drops_a_sample_claimed_by_two_people(tmp_path):
    # Such a sample makes one face vote for both people, producing an exact
    # tie that identify() then breaks arbitrarily. Nothing records which label
    # came last, so it is dropped from both rather than guessed.
    path = tmp_path / "people.json.gz"
    gallery = PersonGallery(path=path)
    rng = np.random.default_rng(0)
    contested = rng.normal(size=512).astype(np.float32)
    alice = gallery.add_person("Alice")
    alice.embeddings.extend([contested, rng.normal(size=512).astype(np.float32)])
    bob = gallery.add_person("Bob")
    bob.embeddings.extend([contested.copy(), rng.normal(size=512).astype(np.float32)])
    gallery.save()

    reloaded = PersonGallery(path=path)

    for person in reloaded.people:
        assert len(person.embeddings) == 1
        assert not np.allclose(person.embeddings[0], contested, atol=1e-2)


def test_remove_embedding_via_a_merged_away_id_still_deletes_the_emptied_person(tmp_path):
    # find_by_id resolves merge redirects, so the person found may not be the
    # id that was passed in -- removal has to act on the resolved person.
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    shared = rng.normal(size=512).astype(np.float32)
    old = gallery.add_person("papa")
    gallery.add_embedding(old.id, shared)
    new = gallery.add_person("Papa")
    gallery.add_embedding(new.id, rng.normal(size=512).astype(np.float32))
    gallery.merge(keep_id=new.id, remove_id=old.id)

    for sample in list(gallery.find_by_id(new.id).embeddings):
        assert gallery.remove_embedding(old.id, sample) is True  # reached via the stale id

    assert gallery.people == []


def test_merging_two_ids_that_resolve_to_the_same_person_raises(tmp_path):
    gallery = _empty_gallery(tmp_path)
    rng = np.random.default_rng(0)
    a = gallery.add_person("a")
    gallery.add_embedding(a.id, rng.normal(size=512).astype(np.float32))
    b = gallery.add_person("b")
    gallery.add_embedding(b.id, rng.normal(size=512).astype(np.float32))
    gallery.merge(keep_id=b.id, remove_id=a.id)

    with pytest.raises(ValueError):
        gallery.merge(keep_id=b.id, remove_id=a.id)  # a now redirects to b
