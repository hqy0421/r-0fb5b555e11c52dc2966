#!/usr/bin/env python
import argparse
import json
import sys
from collections import defaultdict, deque
from itertools import chain
from typing import Any, DefaultDict, Dict, List, Optional

from .export import write_text_chunks, write_text_file
from .pdf import PDF

if len(sys.argv) == 1:
    sys.argv.append("--help")


def parse_page_spec(p_str: str) -> List[int]:
    if "-" in p_str:
        start, end = map(int, p_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(p_str)]


def parse_args(args_raw: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("pdfplumber")

    parser.add_argument("infile", nargs="?", type=argparse.FileType("rb"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--structure",
        help="Write the structure tree as JSON.  "
        "All other arguments except --pages, --laparams, and --indent will be ignored",
        action="store_true",
    )
    group.add_argument(
        "--structure-text",
        help="Write the structure tree as JSON including text contents.  "
        "All other arguments except --pages, --laparams, and --indent will be ignored",
        action="store_true",
    )

    parser.add_argument("--format", choices=["csv", "json", "text"], default="csv")

    parser.add_argument("--types", nargs="+")

    parser.add_argument(
        "--include-attrs",
        nargs="+",
        help="Include *only* these object attributes in output.",
    )

    parser.add_argument(
        "--exclude-attrs",
        nargs="+",
        help="Exclude these object attributes from output.",
    )

    parser.add_argument("--laparams", type=json.loads)

    parser.add_argument("--precision", type=int)

    parser.add_argument("--pages", nargs="+", type=parse_page_spec)

    parser.add_argument(
        "--indent", type=int, help="Indent level for JSON pretty-printing."
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write to PATH instead of stdout. For CSV, extraction proceeds "
        "page by page and resumes automatically after an interruption. "
        'Use "-" for stdout.',
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing CSV checkpoints when writing to --output.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        metavar="PATH",
        help="Directory for CSV checkpoints. Defaults to <output>.parts.",
    )

    args = parser.parse_args(args_raw)
    if args.pages is not None:
        args.pages = list(chain(*args.pages))
    return args


def add_text_to_mcids(pdf: PDF, data: List[Dict[str, Any]]) -> None:
    page_contents: DefaultDict[int, Any] = defaultdict(lambda: defaultdict(str))
    for page in pdf.pages:
        text_contents = page_contents[page.page_number]
        for c in page.chars:
            mcid = c.get("mcid")
            if mcid is None:
                continue
            text_contents[mcid] += c["text"]
    d = deque(data)
    while d:
        el = d.popleft()
        if "children" in el:
            d.extend(el["children"])
        pageno = el.get("page_number")
        if pageno is None:
            continue
        text_contents = page_contents[pageno]
        if "mcids" in el:
            el["text"] = [text_contents[mcid] for mcid in el["mcids"]]


def main(args_raw: List[str] = sys.argv[1:]) -> None:
    args = parse_args(args_raw)
    output: Optional[str] = args.output

    with PDF.open(args.infile, pages=args.pages, laparams=args.laparams) as pdf:
        if args.structure:
            text = json.dumps(pdf.structure_tree, indent=args.indent) + "\n"
            _emit(text, output)
        elif args.structure_text:
            tree = pdf.structure_tree
            add_text_to_mcids(pdf, tree)
            text = json.dumps(tree, indent=args.indent, ensure_ascii=False) + "\n"
            _emit(text, output)
        elif args.format == "csv":
            if output is None or output == "-":
                pdf.to_csv(
                    sys.stdout,
                    args.types,
                    precision=args.precision,
                    include_attrs=args.include_attrs,
                    exclude_attrs=args.exclude_attrs,
                )
            else:
                pdf.to_csv_file(
                    output,
                    args.types,
                    precision=args.precision,
                    include_attrs=args.include_attrs,
                    exclude_attrs=args.exclude_attrs,
                    resume=not args.no_resume,
                    checkpoint_dir=args.checkpoint_dir,
                )
        elif args.format == "text":
            if output is None or output == "-":
                for page in pdf.pages:
                    print(page.extract_text(layout=True))
            else:
                write_text_chunks(
                    output,
                    (
                        page.extract_text(layout=True) + "\n"
                        for page in pdf.iter_pages()
                    ),
                )
        else:
            if output is None or output == "-":
                pdf.to_json(
                    sys.stdout,
                    args.types,
                    precision=args.precision,
                    include_attrs=args.include_attrs,
                    exclude_attrs=args.exclude_attrs,
                    indent=args.indent,
                )
            else:
                json_text = pdf.to_json(
                    None,
                    args.types,
                    precision=args.precision,
                    include_attrs=args.include_attrs,
                    exclude_attrs=args.exclude_attrs,
                    indent=args.indent,
                )
                write_text_file(output, json_text or "")


def _emit(text: str, output: Optional[str]) -> None:
    if output is None or output == "-":
        sys.stdout.write(text)
    else:
        write_text_file(output, text)


if __name__ == "__main__":
    main()
