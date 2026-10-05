from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting import BaseSortingStyle
from pybtex.style.formatting.unsrt import Style as UnsrtStyle

LATEX_TO_UNICODE = {
    '\\"{a}': "ä",
    '\\"{o}': "ö",
    '\\"{u}': "ü",
    '\\"{A}': "Ä",
    '\\"{O}': "Ö",
    '\\"{U}': "Ü",
    "\\'{a}": "á",
    "\\'{e}": "é",
    "\\'{i}": "í",
    "\\'{o}": "ó",
    "\\'{u}'": "ú",
    "\\'{A}": "Á",
    "\\'{E}": "É",
    "\\'{I}": "Í",
    "\\'{O}": "Ó",
    "\\'{U}": "Ú",
    '\\"a': "ä",
    '\\"o': "ö",
    '\\"u': "ü",
    '\\"A': "Ä",
    '\\"O': "Ö",
    '\\"U': "Ü",
    "\\'a": "á",
    "\\'e": "é",
    "\\'i": "í",
    "\\'o": "ó",
    "\\'u'": "ú",
    "\\'A": "Á",
    "\\'E": "É",
    "\\'I": "Í",
    "\\'O": "Ó",
    "\\'U": "Ú",
    "\\ss{}": "ß",
    "\\ss": "ß",
}


def fix_latex(name: str) -> str:
    """Replace LaTeX escape sequences with proper unicode characters."""
    for latex, uni in LATEX_TO_UNICODE.items():
        name = name.replace(latex, uni)
    # Strip any remaining braces
    name = name.replace("{", "").replace("}", "")
    return name


def get_last(person) -> str:
    last_parts = person.last_names
    last = str(last_parts[0]) if last_parts else "Anon"
    return fix_latex(last)


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        counts = {}  # track raw labels for disambiguation

        # First pass: generate raw labels
        raw_labels = []
        for entry in sorted_entries:
            persons = entry.persons.get("author", [])
            year = entry.fields.get("year", "")

            if len(persons) == 0:
                name_part = "Anon."
            elif len(persons) == 1:
                name_part = get_last(persons[0])
            elif len(persons) == 2:
                name_part = (
                    f"{get_last(persons[0])} and {get_last(persons[1])}"
                )
            else:
                name_part = f"{get_last(persons[0])} et al."

            label = f"{name_part}, {year}"
            raw_labels.append(label)
            counts[label] = counts.get(label, 0) + 1

        # Second pass: disambiguate duplicate labels with a, b, c, ...
        seen = {}
        final_labels = []
        for label in raw_labels:
            if counts[label] > 1:
                idx = seen.get(label, 0)
                seen[label] = idx + 1
                final_labels.append(f"{label}{chr(ord('a') + idx)}")
            else:
                final_labels.append(label)

        yield from final_labels


class AuthorYearSortingStyle(BaseSortingStyle):
    def sort(self, entries):
        def sort_key(entry):
            persons = entry.persons.get("author", [])
            year = entry.fields.get("year", "")
            last = get_last(persons[0]) if persons else "Anon"
            return (last.lower(), year)

        return sorted(entries, key=sort_key)


class AuthorYearStyle(UnsrtStyle):
    default_label_style = AuthorYearLabelStyle
    default_sorting_style = AuthorYearSortingStyle
