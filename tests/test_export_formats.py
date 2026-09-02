"""17lands exports do not all have the same shape, and the differences bite.

Three real cases, all found by trying to ingest sets beyond the first one
and all reproduced here synthetically:

  - AFR.PremierDraft is a gzipped **tar** containing one CSV, under the
    same `.csv.gz` name as every other set. Read as plain gzip, its first
    "column" is a tar header block and the vocabulary is silently garbage.
  - AFR also has no `rank`, `user_n_games_bucket` or
    `user_game_win_rate_bucket` column, so requiring the union of every
    set's metadata columns rejected a perfectly good export.
  - SIR.PremierDraft has exactly one row in 1.6M whose recorded pick is not
    in its recorded pack. Treating that as fatal costs the whole set.
"""

from __future__ import annotations

import csv
import gzip
import io
import tarfile

import numpy as np
import pytest

from src.data.dataset import PickData
from src.data.ingest import ingest
from src.data.vocab import build_vocabulary, is_gzipped_tar, open_text, read_header

from .synthetic import CARDS, make_drafts, write_export


def _retar(csv_gz_path, tar_gz_path, member_name="draft_data_public.TST.csv"):
    """Repackages a gzipped CSV as a gzipped tar holding that one CSV."""
    with gzip.open(csv_gz_path, "rb") as handle:
        payload = handle.read()
    with tarfile.open(tar_gz_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


# --------------------------------------------------------------------------
# A .csv.gz that is really a .tar.gz
# --------------------------------------------------------------------------

@pytest.fixture
def tarred_export(tmp_path):
    plain = tmp_path / "plain.csv.gz"
    write_export(plain, make_drafts(np.random.default_rng(0), 6))
    tarred = tmp_path / "draft_data_public.AFRLIKE.PremierDraft.csv.gz"
    _retar(plain, tarred)
    return plain, tarred


def test_a_gzipped_tar_is_detected(tarred_export):
    plain, tarred = tarred_export
    assert is_gzipped_tar(tarred)
    assert not is_gzipped_tar(plain)


def test_reading_a_tarred_export_as_plain_gzip_would_have_been_wrong(tarred_export):
    """The failure this guards against is silent, not loud: the tar header
    parses as CSV perfectly well and yields a nonsense first column.
    """
    _, tarred = tarred_export
    with gzip.open(tarred, "rt", encoding="utf-8", errors="replace", newline="") as h:
        naive_header = next(csv.reader(h))
    real_header = read_header(tarred)

    # The 512-byte tar header runs straight into the CSV's first line, so
    # the first column comes back as archive metadata glued to a real name.
    # Nothing raises; the file just parses into something subtly wrong.
    assert naive_header != real_header
    assert naive_header[0] != real_header[0] == "expansion"
    assert "ustar" in naive_header[0]


def test_vocabulary_and_ingest_see_through_the_tar(tarred_export):
    plain, tarred = tarred_export

    assert read_header(tarred) == read_header(plain)
    vocab = build_vocabulary(tarred)
    assert vocab.size == len(CARDS)
    assert "Zidane, Tantalus Thief" in vocab.card_to_id

    stats = ingest(tarred, tarred.parent / "processed", verbose=False)
    assert stats.rows == 6 * 3 * 14
    assert stats.geometry.picks_per_pack == 14


def test_open_text_closes_the_archive_with_the_stream(tarred_export):
    _, tarred = tarred_export
    handle = open_text(tarred)
    assert handle.readline()
    handle.close()
    assert handle.closed


# --------------------------------------------------------------------------
# Metadata columns that only some sets carry
# --------------------------------------------------------------------------

OPTIONAL_IN_AFR = ("rank", "user_n_games_bucket", "user_game_win_rate_bucket")


@pytest.fixture
def sparse_meta_export(tmp_path):
    path = tmp_path / "draft_data_public.SPARSE.PremierDraft.csv.gz"
    write_export(
        path, make_drafts(np.random.default_rng(1), 6), drop_columns=OPTIONAL_IN_AFR
    )
    return path


def test_an_export_without_rank_still_builds_a_vocabulary(sparse_meta_export):
    header = read_header(sparse_meta_export)
    for column in OPTIONAL_IN_AFR:
        assert column not in header
    vocab = build_vocabulary(sparse_meta_export)
    assert vocab.size == len(CARDS)


def test_an_export_without_rank_still_ingests(sparse_meta_export, tmp_path):
    out = tmp_path / "processed"
    stats = ingest(sparse_meta_export, out, verbose=False)
    assert stats.rows == 6 * 3 * 14
    assert set(stats.missing_meta_columns) == {"rank", "user_game_win_rate_bucket"}
    assert "no ['rank'" in stats.summary()

    data = PickData.load(out)
    assert data.dropped_drafts == 0
    # The absent columns get neutral defaults rather than fabricated values.
    assert list(data.rank_names) == ["unknown"]
    assert (data.rank_code == 0).all()
    assert np.isnan(data.win_rate_bucket).all()


def test_a_genuinely_unusable_export_is_still_rejected(tmp_path):
    """Dropping the requirement on optional columns must not drop it on the
    four columns ingest cannot work without.
    """
    path = tmp_path / "broken.csv.gz"
    write_export(
        path, make_drafts(np.random.default_rng(2), 2), drop_columns=("pick_number",)
    )
    with pytest.raises(ValueError, match="cannot do without"):
        build_vocabulary(path)


# --------------------------------------------------------------------------
# A pick that is not in its pack
# --------------------------------------------------------------------------

def _corrupt_one_pick(src, dest, n_rows_to_break=1):
    """Rewrites an export with some rows' `pick` set to a card not in the pack."""
    with gzip.open(src, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    index = {c: i for i, c in enumerate(header)}

    broken = 0
    for row in body:
        if broken >= n_rows_to_break:
            break
        absent = [
            c for c in CARDS if row[index[f"pack_card_{c}"]] == "0"
        ]
        if absent:
            row[index["pick"]] = absent[0]
            broken += 1
    assert broken == n_rows_to_break

    with gzip.open(dest, "wt", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([header] + body)


def test_a_single_impossible_row_does_not_cost_the_whole_set(tmp_path):
    """SIR really does contain one such row in 1.6M. Raising on it threw
    away a 1.6M-row corpus over a single bad record.
    """
    good = tmp_path / "good.csv.gz"
    write_export(good, make_drafts(np.random.default_rng(3), 40))
    broken = tmp_path / "broken.csv.gz"
    _corrupt_one_pick(good, broken)

    stats = ingest(broken, tmp_path / "processed", verbose=False)
    assert stats.dropped_rows == 1
    assert stats.rows == 40 * 42 - 1
    assert "1 malformed rows dropped" in stats.summary()

    # The bad row's draft is short by one, so PickData removes the rest of
    # it -- the corrupt row can never leak into a neighbouring pool.
    data = PickData.load(tmp_path / "processed")
    assert data.dropped_drafts == 1
    assert data.size == 39 * 42


def test_too_many_impossible_rows_still_raises(tmp_path):
    """A handful is corpus noise; a lot means the header is misaligned, and
    silently dropping those would turn a broken parse into a smaller corpus
    that looks fine.
    """
    good = tmp_path / "good.csv.gz"
    write_export(good, make_drafts(np.random.default_rng(4), 4))
    broken = tmp_path / "broken.csv.gz"
    _corrupt_one_pick(good, broken, n_rows_to_break=20)

    with pytest.raises(ValueError, match="max_bad_row_fraction"):
        ingest(broken, tmp_path / "processed", verbose=False)
