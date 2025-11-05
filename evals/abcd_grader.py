import re
import sys


_PATTERNS = [

    re.compile(
        r'''(?ix)
        (?:\*{1,2}|_{1,2})
        Answer[s]?
        \s*[:\-–]?
        (?:\*{1,2}|_{1,2})
        \s*
        ([ABCD])\b
        ''',
        re.X
    ),


    re.compile(r'''(?ix)
        ^\s*
        (?:\*{1,2}|_{1,2})?
        Answer:?
        (?:\*{1,2}|_{1,2})?
        \s*:?\s*
        (?:\*{1,2}|_{1,2})?
        ([ABCD])
        (?:\*{1,2}|_{1,2})?
        \s*
    ''', re.MULTILINE),


    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)'),


    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b'),


    re.compile(r'(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b'),



    re.compile(r'(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}', re.MULTILINE),


    re.compile(r'(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),


    re.compile(r'(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),


    re.compile(r'(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])'),


    re.compile(r'(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])'),


    re.compile(r'(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}'),


    re.compile(r'''(?x)
        (?<![A-Za-z0-9])
        (?:\*{1,2}|_{1,2})
        \s*([ABCD])\)
        [^*_\n]+?
        (?:\*{1,2}|_{1,2})
        (?![A-Za-z0-9])
    '''),


    re.compile(r'''(?x)^\s*
        (?:\*{1,2}|_{1,2})?
        ([ABCD])
        (?:\*{1,2}|_{1,2})?
        \s*[\.\)\-–:]?
        \s*.*$
    ''', re.MULTILINE),
]


def extract_abcd(text: str) -> str | None:
    """
    Сканировать текст (с сохранением обёрток Markdown/LaTeX) и вернуть `'A'`, `'B'`, `'C'` или `'D'`, если обнаружено указание правильного ответа.
    В противном случае вернуть `None`.
    """
    matches = []
    for prio, pat in enumerate(_PATTERNS):
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in 'ABCD':
                matches.append((prio, m, letter))

    matches.sort(key=lambda triple: (
        triple[0],
        len(triple[1].group(0))
    ))
    for _, match, letter in matches:
        return letter
    return text.removeprefix('**')[:1]


def main():
    if len(sys.argv) > 1:

        for fn in sys.argv[1:]:
            with open(fn, encoding='utf8') as fp:
                text = fp.read()
            ans = extract_abcd(text)
            print(f"{fn} ➜ {ans!r}")
    else:

        for line in sys.stdin:
            ans = extract_abcd(line)
            print(f"{line} ➜ {ans!r}")


if __name__ == "__main__":
    main()

