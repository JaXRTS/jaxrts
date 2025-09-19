"""
Submodule handeling the citation of references
"""

from collections.abc import Iterable
from pathlib import Path

import pybtex
import pybtex.database
from pybtex.style.labels import BaseLabelStyle


class _NoLabelStyle(BaseLabelStyle):
    """
    This is a helper label style, only returning an empty string. Hence, the
    label with be ``[] ``, for all entries. Since this length is well defined,
    we can easily strip this value.
    """

    def format_labels(self, sorted_entries):
        for number, entry in enumerate(sorted_entries):
            yield ""


literature_file = Path(__file__).parent / "literature.bib"


def get_formatted_ref_string(
    bibkeys: str | Iterable[str], comment: str | None = None
) -> str:
    """
    Return biblography entries as a human readable string.
    """
    if isinstance(bibkeys, str):
        bibkeys = [bibkeys]

    # The [3:] part is for stripping the label -- which is empty, anyways
    try:
        out_list = [
            s[3:]
            for s in pybtex.format_from_file(
                literature_file,
                "plain",
                bibkeys,
                output_backend="text",
                label_style=_NoLabelStyle,
            ).split("\n")
        ]
    except KeyError as e:
        key = e.args[0]
        raise KeyError(
            f"Could not find biblography entry {key} in {literature_file}"
        )
    if comment is not None:
        out_list = [comment, *out_list]
    # Filter out all empty strings:
    out_list = [string for string in out_list if string.strip() != ""]
    return "\n".join(out_list)


def get_bibtex_ref_string(
    bibkeys: str | Iterable[str], comment: str | None = None
) -> str:
    """
    Return biblography entries as a bibtex style string.
    """
    if isinstance(bibkeys, str):
        bibkeys = [bibkeys]

    try:
        out_list = [
            pybtex.database.parse_file(literature_file)
            .entries[key]
            .to_string("bibtex")
            for key in bibkeys
        ]
    except KeyError as e:
        key = e.args[0]
        raise KeyError(
            f"Could not find biblography entry {key} in {literature_file}"
        )
    if comment is not None:
        out_list = [comment, *out_list]
    # Filter out all empty strings:
    out_list = [string for string in out_list if string.strip() != ""]
    return "\n".join(out_list)
