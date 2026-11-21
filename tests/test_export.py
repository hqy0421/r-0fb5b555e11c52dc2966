#!/usr/bin/env python
import csv
import io
import logging
import os
import unittest
from pathlib import Path

import pytest

import pdfplumber
from pdfplumber.export import _chunk_name, _read_chunk, _write_chunk

logging.disable(logging.ERROR)

HERE = os.path.abspath(os.path.dirname(__file__))
DEMO = os.path.join(HERE, "pdfs/pdffill-demo.pdf")
CHELSEA = os.path.join(HERE, "pdfs/chelsea_pdta.pdf")


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return f.read()


class Interrupted(Exception):
    pass


class ObjectParseCounter:
    """
    Temporarily count Page.objects parses, and/or fail on the Nth one.

    Counting ``Page.objects`` accesses is a proxy for "did the export actually
    re-extract this page?": a resumed run should only touch pages it cannot
    reuse from disk.
    """

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.page_numbers = []
        self._original = None

    def __enter__(self):
        self._original = pdfplumber.page.Page.objects
        original = self._original
        counter = self

        def objects(self):
            counter.page_numbers.append(self.page_number)
            if (
                counter.fail_on is not None
                and len(counter.page_numbers) >= counter.fail_on
            ):
                raise Interrupted()
            return original.fget(self)

        pdfplumber.page.Page.objects = property(objects)
        return self

    def __exit__(self, *exc):
        assert self._original is not None
        pdfplumber.page.Page.objects = self._original


class Test(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(
            os.path.join(
                os.path.dirname(__file__), "pdfs", ".tmp-export-" + str(id(self))
            )
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _target(self, name="out.csv"):
        self.tmp.mkdir(parents=True, exist_ok=True)
        return self.tmp / name

    def test_matches_to_csv(self):
        cases = [
            {},
            {"precision": 3},
            {"include_attrs": ["page_number", "text"]},
            {"exclude_attrs": ["matrix", "mcid", "ncs"]},
            {"object_types": ["char", "rect"]},
        ]
        for kw in cases:
            with self.subTest(kw=kw):
                with pdfplumber.open(DEMO) as pdf:
                    expected = pdf.to_csv(**kw)
                    target = self._target()
                    pdf.to_csv_file(target, **kw)
                with open(target, newline="", encoding="utf-8") as f:
                    assert f.read() == expected

    def test_matches_to_csv_subset_of_pages(self):
        with pdfplumber.open(DEMO, pages=[1, 3, 5]) as pdf:
            expected = pdf.to_csv(precision=2)
            target = self._target()
            pdf.to_csv_file(target, precision=2)
        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected

    def test_all_object_types_ordering(self):
        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv(object_types=None)
            target = self._target()
            pdf.to_csv_file(target, object_types=None)
        with open(target, newline="", encoding="utf-8") as f:
            got = f.read()
        assert got == expected
        # Object types appear in first-appearance order; "annot" follows the
        # other types on each page, exactly as with .to_csv(...).
        old_limit = csv.field_size_limit()
        csv.field_size_limit(64 * 1024 * 1024)
        try:
            rows = list(csv.DictReader(io.StringIO(got, newline="")))
        finally:
            csv.field_size_limit(old_limit)
        seen = []
        for row in rows:
            if row["object_type"] not in seen:
                seen.append(row["object_type"])
        assert seen == ["line", "char", "rect", "image", "annot", "curve"]

    def test_header_includes_late_attributes(self):
        """
        The header must describe *everything* in the output, even when the
        attribute first shows up on a later page or a later object.
        """
        with pdfplumber.open(CHELSEA) as pdf:
            target = self._target()
            pdf.to_csv_file(target)
            expected = pdf.to_csv()
        with open(target, newline="", encoding="utf-8") as f:
            got = f.read()

        header = next(csv.reader(io.StringIO(got, newline="")))
        old_limit = csv.field_size_limit()
        csv.field_size_limit(64 * 1024 * 1024)
        try:
            reader = csv.DictReader(io.StringIO(got, newline=""))
            keys = set()
            for row in reader:
                keys.update(k for k, v in row.items() if v is not None and k)
        finally:
            csv.field_size_limit(old_limit)

        assert not (keys - set(header))
        assert got == expected

    def test_resume_skips_completed_pages(self):
        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv()
        target = self._target()

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=4):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target)

        parts = Path(str(target) + ".parts")
        assert parts.is_dir()
        assert not target.exists()

        with ObjectParseCounter() as counter:
            with pdfplumber.open(DEMO) as pdf:
                pdf.to_csv_file(target)

        # Pages 1-3 were completed, so only pages 4+ need re-extraction.
        assert min(counter.page_numbers) == 4
        assert not parts.exists()
        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected

    def test_resume_result_matches_uninterrupted(self):
        uninterrupted = self._target("uninterrupted.csv")
        resumed = self._target("resumed.csv")

        with pdfplumber.open(DEMO) as pdf:
            pdf.to_csv_file(uninterrupted, precision=3)

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=5):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(resumed, precision=3)

        with pdfplumber.open(DEMO) as pdf:
            pdf.to_csv_file(resumed, precision=3)

        assert resumed.read_bytes() == uninterrupted.read_bytes()

    def test_corrupt_and_truncated_chunks_are_reextracted(self):
        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv()
        target = self._target()

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=5):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target)

        parts = Path(str(target) + ".parts")
        # Case 1: invalid JSON.
        (parts / _chunk_name(2)).write_text('{"page_number": 2, "groups": [')
        # Case 2: valid JSON, but truncated/incomplete content.
        chunk3 = parts / _chunk_name(3)
        data = chunk3.read_bytes()
        chunk3.write_bytes(data[: len(data) // 2])
        # Case 3: well-formed but wrong page number.
        groups = _read_chunk(parts / _chunk_name(4), 4)
        assert groups is not None
        _write_chunk(parts / _chunk_name(4), 999, groups)

        with ObjectParseCounter() as counter:
            with pdfplumber.open(DEMO) as pdf:
                pdf.to_csv_file(target)

        assert counter.page_numbers[:3] == [2, 3, 4]
        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected

    def test_config_change_invalidates_checkpoints(self):
        target = self._target()
        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=4):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target, precision=3)

        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv(precision=1)
            pdf.to_csv_file(target, precision=1)

        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected

    def test_different_pdf_at_same_path_invalidates(self):
        import shutil

        self.tmp.mkdir(parents=True, exist_ok=True)
        source = self.tmp / "doc.pdf"
        shutil.copy(DEMO, source)
        target = self.tmp / "out.csv"

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=4):
                with pdfplumber.open(source) as pdf:
                    pdf.to_csv_file(target)

        shutil.copy(os.path.join(HERE, "pdfs/hello_structure.pdf"), source)
        with pdfplumber.open(source) as pdf:
            expected = pdf.to_csv()
            pdf.to_csv_file(target)

        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected

    def test_resume_false_disables_checkpoint_reuse(self):
        target = self._target()
        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=4):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target)

        with ObjectParseCounter() as counter:
            with pdfplumber.open(DEMO) as pdf:
                pdf.to_csv_file(target, resume=False)

        assert counter.page_numbers == [1, 2, 3, 4, 5, 6, 7]

    def test_target_untouched_until_complete(self):
        target = self._target()
        target.write_text("PREVIOUS CONTENT")

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=3):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target)

        assert target.read_text() == "PREVIOUS CONTENT"

        # No half-written target or temp files left behind.
        leftovers = sorted(
            p.name for p in self.tmp.iterdir() if p.name != "out.csv.parts"
        )
        assert leftovers == ["out.csv"]

    def test_custom_checkpoint_dir(self):
        target = self._target()
        checkpoint = self.tmp / "checkpoints"

        with self.assertRaises(Interrupted):
            with ObjectParseCounter(fail_on=4):
                with pdfplumber.open(DEMO) as pdf:
                    pdf.to_csv_file(target, checkpoint_dir=checkpoint)

        assert checkpoint.is_dir()
        assert not Path(str(target) + ".parts").exists()

        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv()
            pdf.to_csv_file(target, checkpoint_dir=checkpoint)

        with open(target, newline="", encoding="utf-8") as f:
            assert f.read() == expected
        assert not checkpoint.exists()

    def test_write_to_stdout(self):
        import contextlib

        with pdfplumber.open(DEMO) as pdf:
            expected = pdf.to_csv(precision=3)
        buf = io.StringIO()
        # "-" delegates to the in-memory exporter (no checkpointing).
        with contextlib.redirect_stdout(buf):
            with pdfplumber.open(DEMO) as pdf:
                pdf.to_csv_file("-", precision=3)
        assert buf.getvalue() == expected
        assert not (self.tmp / "out.csv.parts").exists()


def test_cli_csv_output_file(tmp_path):
    from pdfplumber.cli import main

    out = tmp_path / "out.csv"
    main([DEMO, "--format", "csv", "--precision", "3", "--output", str(out)])
    with pdfplumber.open(DEMO) as pdf:
        assert _read(out) == pdf.to_csv(precision=3)


def test_cli_csv_resume(tmp_path):
    out = tmp_path / "out.csv"
    with pdfplumber.open(DEMO) as pdf:
        expected = pdf.to_csv()

    with pytest.raises(Interrupted):
        with ObjectParseCounter(fail_on=4):
            from pdfplumber.cli import main

            main([DEMO, "--format", "csv", "--output", str(out)])

    assert not out.exists()
    with ObjectParseCounter() as counter:
        from pdfplumber.cli import main

        main([DEMO, "--format", "csv", "--output", str(out)])

    assert min(counter.page_numbers) == 4
    assert _read(out) == expected


def test_cli_no_resume(tmp_path):
    out = tmp_path / "out.csv"
    with pytest.raises(Interrupted):
        with ObjectParseCounter(fail_on=4):
            from pdfplumber.cli import main

            main([DEMO, "--format", "csv", "--output", str(out)])

    with ObjectParseCounter() as counter:
        from pdfplumber.cli import main

        main([DEMO, "--format", "csv", "--output", str(out), "--no-resume"])

    assert counter.page_numbers == [1, 2, 3, 4, 5, 6, 7]
    with pdfplumber.open(DEMO) as pdf:
        assert _read(out) == pdf.to_csv()


def test_iter_pages_matches_pages():
    with pdfplumber.open(DEMO) as pdf:
        assert [p.page_number for p in pdf.iter_pages()] == [
            p.page_number for p in pdf.pages
        ]


def test_memory_stays_bounded():
    """
    The file exporter streams rows to disk instead of accumulating them, so
    peak traced memory should be far below the in-memory exporter's.
    """
    import gc
    import tracemalloc

    target = Path(os.path.join(HERE, "pdfs", ".tmp-mem-export.csv"))
    parts = Path(str(target) + ".parts")
    try:
        gc.collect()
        tracemalloc.start()
        with pdfplumber.open(CHELSEA) as pdf:
            pdf.to_csv_file(target)
        _cur, peak_file = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()
        tracemalloc.start()
        with pdfplumber.open(CHELSEA) as pdf:
            pdf.to_csv()
        _cur, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak_file < peak_mem / 4
    finally:
        target.unlink(missing_ok=True)
        import shutil

        shutil.rmtree(parts, ignore_errors=True)
