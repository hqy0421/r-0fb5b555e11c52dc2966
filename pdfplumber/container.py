import csv
import json
from io import StringIO
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Set, TextIO

from . import utils
from ._typing import T_obj, T_obj_list
from .convert import CSV_COLS_REQUIRED, CSV_COLS_TO_PREPEND, Serializer
from .export import PathLike, to_csv_file


class Container(object):
    cached_properties = ["_rect_edges", "_curve_edges", "_edges", "_objects"]

    # True for containers whose iter_pages() creates throwaway Page objects
    # (rather than yielding long-lived, caller-owned ones).
    _iter_pages_are_transient = False

    @property
    def pages(self) -> Optional[List[Any]]:  # pragma: nocover
        raise NotImplementedError

    def iter_pages(self) -> Iterator[Any]:
        """
        Yield this container's pages one at a time.

        Unlike ``.pages``, implementations are encouraged not to materialize
        the full list, so that per-page data can be released as soon as it has
        been consumed.
        """
        pages = self.pages
        if pages is None:
            yield self
        else:
            yield from pages

    @property
    def objects(self) -> Dict[str, T_obj_list]:  # pragma: nocover
        raise NotImplementedError

    def to_dict(
        self, object_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:  # pragma: nocover
        raise NotImplementedError

    def flush_cache(self, properties: Optional[List[str]] = None) -> None:
        props = self.cached_properties if properties is None else properties
        for p in props:
            if hasattr(self, p):
                delattr(self, p)

    @property
    def rects(self) -> T_obj_list:
        return self.objects.get("rect", [])

    @property
    def lines(self) -> T_obj_list:
        return self.objects.get("line", [])

    @property
    def curves(self) -> T_obj_list:
        return self.objects.get("curve", [])

    @property
    def images(self) -> T_obj_list:
        return self.objects.get("image", [])

    @property
    def chars(self) -> T_obj_list:
        return self.objects.get("char", [])

    @property
    def textboxverticals(self) -> T_obj_list:
        return self.objects.get("textboxvertical", [])

    @property
    def textboxhorizontals(self) -> T_obj_list:
        return self.objects.get("textboxhorizontal", [])

    @property
    def textlineverticals(self) -> T_obj_list:
        return self.objects.get("textlinevertical", [])

    @property
    def textlinehorizontals(self) -> T_obj_list:
        return self.objects.get("textlinehorizontal", [])

    @property
    def rect_edges(self) -> T_obj_list:
        if hasattr(self, "_rect_edges"):
            return self._rect_edges
        rect_edges_gen = (utils.rect_to_edges(r) for r in self.rects)
        self._rect_edges: T_obj_list = list(chain(*rect_edges_gen))
        return self._rect_edges

    @property
    def curve_edges(self) -> T_obj_list:
        if hasattr(self, "_curve_edges"):
            return self._curve_edges
        curve_edges_gen = (utils.curve_to_edges(r) for r in self.curves)
        self._curve_edges: T_obj_list = list(chain(*curve_edges_gen))
        return self._curve_edges

    @property
    def edges(self) -> T_obj_list:
        if hasattr(self, "_edges"):
            return self._edges
        line_edges = list(map(utils.line_to_edge, self.lines))
        self._edges: T_obj_list = line_edges + self.rect_edges + self.curve_edges
        return self._edges

    @property
    def horizontal_edges(self) -> T_obj_list:
        def test(x: T_obj) -> bool:
            return bool(x["orientation"] == "h")

        return list(filter(test, self.edges))

    @property
    def vertical_edges(self) -> T_obj_list:
        def test(x: T_obj) -> bool:
            return bool(x["orientation"] == "v")

        return list(filter(test, self.edges))

    def to_json(
        self,
        stream: Optional[TextIO] = None,
        object_types: Optional[List[str]] = None,
        include_attrs: Optional[List[str]] = None,
        exclude_attrs: Optional[List[str]] = None,
        precision: Optional[int] = None,
        indent: Optional[int] = None,
    ) -> Optional[str]:

        data = self.to_dict(object_types)

        serialized = Serializer(
            precision=precision,
            include_attrs=include_attrs,
            exclude_attrs=exclude_attrs,
        ).serialize(data)

        if stream is None:
            return json.dumps(serialized, indent=indent)
        else:
            json.dump(serialized, stream, indent=indent)
            return None

    def to_csv(
        self,
        stream: Optional[TextIO] = None,
        object_types: Optional[List[str]] = None,
        precision: Optional[int] = None,
        include_attrs: Optional[List[str]] = None,
        exclude_attrs: Optional[List[str]] = None,
    ) -> Optional[str]:
        if stream is None:
            stream = StringIO()
            to_string = True
        else:
            to_string = False

        if object_types is None:
            object_types = list(self.objects.keys()) + ["annot"]

        serialized = []
        fields: Set[str] = set()

        serializer = Serializer(
            precision=precision,
            include_attrs=include_attrs,
            exclude_attrs=exclude_attrs,
        )
        for page in self.iter_pages():
            for t in object_types:
                objs = getattr(page, t + "s")
                if len(objs):
                    serialized += serializer.serialize(objs)
                    new_keys = [k for k, v in objs[0].items() if type(v) is not dict]
                    fields = fields.union(set(new_keys))

        non_req_cols = CSV_COLS_TO_PREPEND + list(
            sorted(set(fields) - set(CSV_COLS_REQUIRED + CSV_COLS_TO_PREPEND))
        )

        cols = CSV_COLS_REQUIRED + list(filter(serializer.attr_filter, non_req_cols))

        w = csv.DictWriter(
            stream,
            fieldnames=cols,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        w.writeheader()
        w.writerows(serialized)

        if to_string:
            stream.seek(0)
            return stream.read()
        else:
            return None

    def to_csv_file(
        self,
        path: PathLike,
        object_types: Optional[List[str]] = None,
        precision: Optional[int] = None,
        include_attrs: Optional[List[str]] = None,
        exclude_attrs: Optional[List[str]] = None,
        resume: bool = True,
        checkpoint_dir: Optional[PathLike] = None,
    ) -> None:
        """
        Write this container's objects to ``path`` as CSV, one page at a time.

        The output is identical to ``.to_csv(...)``, but rows are streamed to
        disk rather than accumulated in memory. Each fully-written page is
        checkpointed; if the process is interrupted, calling this method again
        with the same PDF and configuration skips those pages. Corrupt or
        incomplete checkpoints are re-extracted. ``path`` is replaced
        atomically, and only after every page has been written.

        Pass ``resume=False`` to ignore (and discard) any existing checkpoints.
        Pass ``path="-"`` to write to stdout without checkpointing.
        """
        to_csv_file(
            self,
            path,
            object_types=object_types,
            precision=precision,
            include_attrs=include_attrs,
            exclude_attrs=exclude_attrs,
            resume=resume,
            checkpoint_dir=checkpoint_dir,
        )
