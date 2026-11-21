"""Resumable, page-at-a-time file export."""

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

from ._version import __version__
from .convert import CSV_COLS_REQUIRED, CSV_COLS_TO_PREPEND, Serializer

try:  # pragma: nocover
    import pdfminer as _pdfminer

    PDFMINER_VERSION: Optional[str] = _pdfminer.__version__
except Exception:  # pragma: nocover
    PDFMINER_VERSION = None

PathLike = Union[str, "os.PathLike[str]"]

# Bump whenever the on-disk checkpoint layout below changes.
PARTS_VERSION = 1

MANIFEST_NAME = "manifest.json"
CHUNK_PREFIX = "page-"
CHUNK_SUFFIX = ".json"
TMP_PREFIX = ".tmp"

T_cell = Optional[str]
T_row = Dict[str, T_cell]
T_group = Tuple[str, List[T_row]]


def _cell(value: Any) -> T_cell:
    """
    Convert a serialized value into the exact text that ``csv.DictWriter``
    would write for it. Checkpointing that text (rather than JSON-typed values)
    keeps the final CSV byte-identical whether or not the export was resumed.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _object_cells(raw: Any, serialized: Any) -> T_row:
    """
    Mirror ``Container.to_csv``'s field handling.

    A field whose *raw* value is a dict (e.g. ``annot["data"]``) has never been
    eligible for a CSV column, so it is omitted here; everything else is kept.
    """
    if not isinstance(raw, dict) or not isinstance(serialized, dict):
        return {}
    return {
        key: _cell(value)
        for key, value in serialized.items()
        if key in raw and type(raw[key]) is not dict
    }


def _serialize_objs(serializer: Serializer, objs: Sequence[Any]) -> List[T_row]:
    return [_object_cells(obj, serializer.serialize(obj)) for obj in objs]


def _groups_for_page(
    page: Any,
    object_types: Optional[List[str]],
    serializer: Serializer,
    known_types: List[str],
) -> List[T_group]:
    """
    Return this page's rows, grouped by object type, in exactly the order
    ``Container.to_csv`` would emit them.

    With ``object_types=None`` the type order is the order in which types first
    appear across the selected pages. Growing ``known_types`` incrementally
    reproduces that order without parsing later pages up front.
    """
    groups: List[T_group] = []

    if object_types is None:
        objects = page.objects
        for kind in objects.keys():
            if kind not in known_types:
                known_types.append(kind)

        for kind in known_types:
            objs = objects.get(kind, [])
            if len(objs):
                groups.append((kind, _serialize_objs(serializer, objs)))

        annots = page.annots
        if len(annots):
            groups.append(("annot", _serialize_objs(serializer, annots)))
    else:
        for kind in object_types:
            objs = getattr(page, kind + "s")
            if len(objs):
                groups.append((kind, _serialize_objs(serializer, objs)))

    return groups


def _digest_stream(stream: Any) -> Optional[str]:
    """SHA-256 of the PDF's bytes, restoring the stream position afterwards."""
    h = hashlib.sha256()
    try:
        if not stream.seekable():
            return None
        pos: Optional[int] = stream.tell()
    except Exception:
        return None
    try:
        stream.seek(0)
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    finally:
        try:
            stream.seek(pos)
        except Exception:  # pragma: nocover
            pass
    return h.hexdigest()


def _laparams_dict(laparams: Any) -> Optional[Dict[str, Any]]:
    if laparams is None:
        return None
    return {key: getattr(laparams, key) for key in vars(laparams)}


def _container_descriptor(container: Any) -> Dict[str, Any]:
    """
    Describe derived containers (cropped, filtered, deduped pages). Their
    output can differ even when the underlying PDF and page number match, so
    those details participate in the fingerprint.

    Callables are fingerprinted via ``repr``: stable within a process, not
    across processes. That makes cross-process reuse conservative for derived
    containers, which is the safe direction.
    """
    desc: Dict[str, Any] = {"container_type": type(container).__name__}

    bbox = getattr(container, "bbox", None)
    if bbox is not None:
        try:
            desc["bbox"] = [repr(x) for x in bbox]
        except Exception:  # pragma: nocover
            desc["bbox"] = repr(bbox)

    for attr in ("_crop_fn", "filter_fn"):
        fn = getattr(container, attr, None)
        if fn is not None:
            desc[attr] = repr(fn)

    parent = getattr(container, "parent_page", None)
    if parent is not None:
        desc["parent_page"] = getattr(parent, "page_number", None)

    return desc


def _fingerprint(
    container: Any,
    object_types: Optional[List[str]],
    precision: Optional[int],
    include_attrs: Optional[List[str]],
    exclude_attrs: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Everything that can affect the produced CSV. If any of it differs, no
    checkpointed page may be reused. The PDF's bytes identify the document, so
    resuming stays safe even if the file was re-written in place.
    """
    pdf = getattr(container, "pdf", container)
    stream = getattr(pdf, "stream", None)
    path = getattr(pdf, "path", None)
    pages_to_parse = getattr(pdf, "pages_to_parse", None)

    source: Dict[str, Any] = {}
    digest = _digest_stream(stream) if stream is not None else None
    if digest is not None:
        source["pdf_sha256"] = digest
    else:  # pragma: nocover
        source["pdf_path"] = None if path is None else str(path)
        try:
            source["pdf_size"] = os.fstat(stream.fileno()).st_size  # type: ignore[union-attr]  # noqa: E501
        except Exception:
            source["pdf_size"] = None

    return {
        "parts_version": PARTS_VERSION,
        "pdfplumber_version": __version__,
        "pdfminer_version": PDFMINER_VERSION,
        **source,
        "page_number": getattr(container, "page_number", None),
        "pages": None if pages_to_parse is None else list(pages_to_parse),
        "object_types": None if object_types is None else list(object_types),
        "include_attrs": None if include_attrs is None else list(include_attrs),
        "exclude_attrs": None if exclude_attrs is None else list(exclude_attrs),
        "precision": precision,
        "laparams": _laparams_dict(getattr(pdf, "laparams", None)),
        "unicode_norm": getattr(pdf, "unicode_norm", None),
        "raise_unicode_errors": getattr(pdf, "raise_unicode_errors", None),
        "container": _container_descriptor(container),
    }


def _chunk_name(page_number: int) -> str:
    return f"{CHUNK_PREFIX}{page_number:012d}{CHUNK_SUFFIX}"


def _chunk_number(name: str) -> Optional[int]:
    if not (name.startswith(CHUNK_PREFIX) and name.endswith(CHUNK_SUFFIX)):
        return None
    core = name[len(CHUNK_PREFIX) : -len(CHUNK_SUFFIX)]
    if not core.isdigit():
        return None
    return int(core)


def _atomic_write(path: Path, write_fn: Callable[[TextIO], None]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=TMP_PREFIX + "-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: nocover
            pass
        raise


def _write_chunk(path: Path, page_number: int, groups: List[T_group]) -> None:
    """
    Write one page's rows. The JSON is emitted incrementally, so only the
    page's rows (already in memory) are needed -- no full-document string.

    The file is written under a temporary name and renamed only when complete,
    so a crash mid-write can never look like a finished page.
    """

    def write(f: TextIO) -> None:
        f.write('{"page_number":')
        f.write(str(page_number))
        f.write(',"groups":[')
        for i, (kind, rows) in enumerate(groups):
            if i:
                f.write(",")
            f.write("[")
            f.write(json.dumps(kind, ensure_ascii=True))
            f.write(",[")
            for j, row in enumerate(rows):
                if j:
                    f.write(",")
                f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            f.write("]]")
        f.write("]}")

    _atomic_write(path, write)


def _read_chunk(path: Path, page_number: int) -> Optional[List[T_group]]:
    """
    Return the cached groups, or None if the chunk is missing, truncated,
    corrupt, or otherwise not a complete checkpoint for the expected page.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict) or data.get("page_number") != page_number:
        return None

    raw_groups = data.get("groups")
    if not isinstance(raw_groups, list):
        return None

    groups: List[T_group] = []
    for item in raw_groups:
        if not isinstance(item, list) or len(item) != 2:
            return None
        kind, rows = item
        if not isinstance(kind, str) or not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                return None
            for key, value in row.items():
                if not isinstance(key, str):
                    return None
                if value is not None and not isinstance(value, str):
                    return None
        groups.append((kind, rows))
    return groups


def _remove_stale_temps(parts_dir: Path) -> None:
    for entry in parts_dir.iterdir():
        if entry.name.startswith(TMP_PREFIX):
            try:
                entry.unlink()
            except OSError:  # pragma: nocover
                pass


def _clear_parts_dir(parts_dir: Path) -> None:
    """
    Delete this exporter's files from ``parts_dir`` and remove the directory
    if it is now empty.

    Only known checkpoint artefacts are removed: with a user-supplied
    ``checkpoint_dir`` the directory may contain unrelated files, which must
    survive.
    """
    try:
        entries = list(parts_dir.iterdir())
    except OSError:
        return

    for entry in entries:
        name = entry.name
        if not (
            name == MANIFEST_NAME
            or name.startswith(TMP_PREFIX)
            or _chunk_number(name) is not None
        ):
            continue
        try:
            entry.unlink()
        except OSError:  # pragma: nocover
            pass

    try:
        parts_dir.rmdir()
    except OSError:
        pass


def _prepare_parts_dir(
    parts_dir: Path, fingerprint: Dict[str, Any], resume: bool
) -> bool:
    """
    Validate an existing checkpoint directory, or (re)initialise it.
    Returns True only when existing per-page checkpoints may be reused.
    """
    manifest = {"fingerprint": fingerprint}

    if parts_dir.is_dir():
        try:
            with open(parts_dir / MANIFEST_NAME, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None
        if resume and existing == manifest:
            _remove_stale_temps(parts_dir)
            return True
        _clear_parts_dir(parts_dir)
    elif parts_dir.exists():  # pragma: nocover
        # A non-directory is in the way; the path was asked for explicitly.
        parts_dir.unlink()

    parts_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(f: TextIO) -> None:
        f.write(json.dumps(manifest, ensure_ascii=True, sort_keys=True))

    _atomic_write(parts_dir / MANIFEST_NAME, write_manifest)
    return False


def _column_names(fields: Set[str], serializer: Serializer) -> List[str]:
    non_required = CSV_COLS_TO_PREPEND + list(
        sorted(fields - set(CSV_COLS_REQUIRED + CSV_COLS_TO_PREPEND))
    )
    return CSV_COLS_REQUIRED + list(filter(serializer.attr_filter, non_required))


def _iter_page_groups(
    container: Any,
    object_types: Optional[List[str]],
    serializer: Serializer,
    checkpoint_page: Callable[[int, List[T_group]], None],
    reuse_page: Optional[Callable[[int], Optional[List[T_group]]]],
    fields: Set[str],
) -> Iterator[Tuple[int, List[T_group]]]:
    """
    Walk the container page by page, yielding each page's rows after they are
    fully in hand and checkpointed.

    Only one page's rows and the discovered column/type metadata are ever held
    in memory. Pages created solely for this iteration are closed once
    consumed, releasing their layout/object caches promptly; pages the caller
    already materialized are left alone.

    ``fields`` accumulates every attribute actually present in the output, so
    the header stays complete even when a later page (or a later object within
    a page) introduces an attribute earlier pages lacked.
    """
    known_types: List[str] = []

    for page in container.iter_pages():
        try:
            page_number = page.page_number
            cached = reuse_page(page_number) if reuse_page is not None else None
            if cached is not None:
                for kind, _rows in cached:
                    if kind != "annot" and kind not in known_types:
                        known_types.append(kind)
                groups = cached
            else:
                groups = _groups_for_page(page, object_types, serializer, known_types)
                checkpoint_page(page_number, groups)
            for _kind, rows in groups:
                for row in rows:
                    fields.update(row.keys())
            yield page_number, groups
        finally:
            if getattr(container, "_iter_pages_are_transient", False):
                page.close()


def _chunk_paths(parts_dir: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for entry in parts_dir.iterdir():
        number = _chunk_number(entry.name)
        if number is not None:
            found.append((number, entry))
    return sorted(found)


def _write_rows(stream: TextIO, parts_dir: Path, cols: List[str]) -> None:
    writer = csv.DictWriter(
        stream,
        fieldnames=cols,
        extrasaction="ignore",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )
    writer.writeheader()
    for _number, chunk_path in _chunk_paths(parts_dir):
        with open(chunk_path, encoding="utf-8") as f:
            data = json.load(f)
        for _kind, rows in data["groups"]:
            writer.writerows(rows)


def _assemble(
    parts_dir: Path, target: str, serializer: Serializer, fields: Set[str]
) -> None:
    """
    Stream every checkpoint into a temporary CSV, then atomically replace the
    target. The target is never truncated or partially written.
    """
    cols = _column_names(fields, serializer)

    target_path = Path(target)
    parent = target_path.parent if str(target_path.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        dir=str(parent), prefix=TMP_PREFIX + "-" + target_path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            _write_rows(stream, parts_dir, cols)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: nocover
            pass
        raise


def to_csv_file(
    container: Any,
    path: PathLike,
    object_types: Optional[List[str]] = None,
    precision: Optional[int] = None,
    include_attrs: Optional[List[str]] = None,
    exclude_attrs: Optional[List[str]] = None,
    resume: bool = True,
    checkpoint_dir: Optional[PathLike] = None,
) -> None:
    """
    Write ``container``'s objects to ``path`` as CSV, one page at a time.

    Unlike ``Container.to_csv(stream)``, this never holds all pages' rows in
    memory, and it checkpoints each fully-written page to disk. If the export
    is interrupted, calling it again with the same PDF and configuration
    reuses those pages; corrupt or incomplete pages are re-extracted. The
    target file is only replaced, atomically, once every page is done.

    ``path == "-"`` writes to stdout without checkpointing.
    """
    target = os.fspath(path)

    if target == "-":
        result = container.to_csv(
            None,
            object_types,
            precision=precision,
            include_attrs=include_attrs,
            exclude_attrs=exclude_attrs,
        )
        if result:
            sys.stdout.write(result)
            sys.stdout.flush()
        return

    serializer = Serializer(
        precision=precision,
        include_attrs=include_attrs,
        exclude_attrs=exclude_attrs,
    )

    fingerprint = _fingerprint(
        container, object_types, precision, include_attrs, exclude_attrs
    )
    parts_dir = (
        Path(os.fspath(checkpoint_dir))
        if checkpoint_dir is not None
        else Path(target + ".parts")
    )

    reusable = _prepare_parts_dir(parts_dir, fingerprint, resume=resume)

    # Column metadata: every attribute actually present in the output, kept at
    # header granularity only. Unlike Container.to_csv, this is a union across
    # *all* objects (not just each type's first object on each page), so a
    # later page that introduces a new attribute cannot lose its column --
    # matching what resumed runs (which re-aggregate from checkpoints) produce.
    fields: Set[str] = set()

    def reuse_page(page_number: int) -> Optional[List[T_group]]:
        chunk = parts_dir / _chunk_name(page_number)
        groups = _read_chunk(chunk, page_number)
        if groups is None and chunk.exists():
            try:
                chunk.unlink()
            except OSError:  # pragma: nocover
                pass
        return groups

    def checkpoint_page(page_number: int, groups: List[T_group]) -> None:
        _write_chunk(parts_dir / _chunk_name(page_number), page_number, groups)

    for _page_number, _groups in _iter_page_groups(
        container,
        object_types,
        serializer,
        checkpoint_page=checkpoint_page,
        reuse_page=reuse_page if reusable else None,
        fields=fields,
    ):
        pass

    _assemble(parts_dir, target, serializer, fields)
    _clear_parts_dir(parts_dir)


def write_text_file(path: PathLike, text: str) -> None:
    """Atomically write ``text`` to ``path`` (``"-"`` means stdout)."""
    write_text_chunks(path, [text])


def write_text_chunks(path: PathLike, chunks: Iterable[str]) -> None:
    """Atomically write an iterable of strings to ``path``, streaming them."""
    target = os.fspath(path)
    if target == "-":
        for chunk in chunks:
            sys.stdout.write(chunk)
        sys.stdout.flush()
        return

    target_path = Path(target)
    parent = target_path.parent if str(target_path.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(parent), prefix=TMP_PREFIX + "-" + target_path.name + "-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: nocover
            pass
        raise
