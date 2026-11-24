"""Fold straight-line kernel variants into one kernel with constexpr switches.

The row generators emit each implementation of a kernel (CP4 / CP8-reload /
running-product backward, fused or split prefix, ...) as its own straight-line
``@triton.jit`` function.  As FLA's kernels do with ``USE_G`` / ``IS_VARLEN``,
``merge_kernels`` turns them into a single function: lines every variant
shares are written once, and each run of lines only some variants have sits
under ``if <flags>:``.  The flags are ``tl.constexpr``, so Triton compiles only
the taken branch, and a constexpr ``if`` does not open a scope: each variant's
specialization visits exactly its original statements in their original order.
``merge_kernels`` checks that for every variant before returning.

Parameters merge the same way.  A parameter only some variants take is passed
as ``None`` by the others, which Triton specializes away, so each variant also
keeps its original signature.
"""
from __future__ import annotations

import difflib
import itertools


def _split(text: str) -> tuple[list[str], list[str], list[str]]:
    """(decorators, parameter lines, body lines) of one generated kernel."""
    lines = text.rstrip("\n").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("def "))
    close = lines.index("):", start)
    assert lines[start].endswith("("), lines[start]
    return lines[:start], lines[start + 1:close], lines[close + 1:]


def _merge(sequences: dict[str, list[str]]) -> list[tuple[str, frozenset]]:
    """Order-preserving union of line sequences, each line tagged with its variants."""
    merged: list[tuple[str, frozenset]] = []
    for key, lines in sequences.items():
        if not merged:
            merged = [(line, frozenset((key,))) for line in lines]
            continue
        matcher = difflib.SequenceMatcher(None, [line for line, _ in merged], lines, autojunk=False)
        out = []
        for op, a0, a1, b0, b1 in matcher.get_opcodes():
            if op == "equal":
                out += [(line, tags | {key}) for line, tags in merged[a0:a1]]
            else:
                out += merged[a0:a1]
                out += [(line, frozenset((key,))) for line in lines[b0:b1]]
        merged = out
    for key, lines in sequences.items():
        assert [line for line, tags in merged if key in tags] == lines, f"variant {key} not preserved"
    return merged


def _condition(tags: frozenset, flags: dict[str, dict[str, bool]]) -> str:
    """The simplest flag expression true for exactly the variants in ``tags``."""
    names = sorted({name for values in flags.values() for name in values})
    literals = [(name, lambda f, n=name: f[n]) for name in names]
    literals += [(f"not {name}", lambda f, n=name: not f[n]) for name in names]

    def exact(test):
        return all(bool(test(values)) == (key in tags) for key, values in flags.items())

    for text, test in literals:
        if exact(test):
            return text
    for (ta, fa), (tb, fb) in itertools.combinations(literals, 2):
        if exact(lambda f: fa(f) and fb(f)):
            return f"{ta} and {tb}"
        if exact(lambda f: fa(f) or fb(f)):
            return f"{ta} or {tb}"
    raise ValueError(f"no two-literal condition selects {sorted(tags)}")


def _is_statement(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def merge_kernels(name: str, variants: dict[str, str], flags: dict[str, dict[str, bool]],
                  comment: str = "") -> str:
    """One ``@triton.jit`` kernel ``name`` from straight-line ``variants``.

    ``flags[key]`` gives the constexpr values that select variant ``key``;
    every variant must set the same flag names.
    """
    assert variants.keys() == flags.keys()
    names = sorted({n for values in flags.values() for n in values})
    assert all(sorted(values) == names for values in flags.values())
    parts = {key: _split(text) for key, text in variants.items()}
    decorators = {tuple(d) for d, _, _ in parts.values()}
    assert len(decorators) == 1, "variants must share their decorators"
    everyone = frozenset(variants)

    out = [f"# {line}" if line else "#" for line in comment.split("\n")] if comment else []
    out += list(next(iter(decorators)))
    out.append(f"def {name}(")
    for line, tags in _merge({key: params for key, (_, params, _) in parts.items()}):
        note = "" if tags == everyone else f"  # None unless {_condition(tags, flags)}"
        out.append(line + note)
    out.append("    " + " ".join(f"{flag}: tl.constexpr," for flag in names))
    out.append("):")
    merged = _merge({key: body for key, (_, _, body) in parts.items()})
    for tags, group in itertools.groupby(merged, key=lambda item: item[1]):
        lines = [line for line, _ in group]
        if tags == everyone or not any(_is_statement(line) for line in lines):
            out += lines
            continue
        out.append(f"    if {_condition(tags, flags)}:")
        out += [("    " + line) if line else line for line in lines]
    return "\n".join(out) + "\n"


__all__ = ["merge_kernels"]
