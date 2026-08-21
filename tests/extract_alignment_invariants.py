#!/usr/bin/env python3
"""Extract the ml_alignment_autograd log fields that must stay identical.

The diagnostics refactor deliberately changes some log content (RMS tables,
before statistics, the ``(n0=, n1=)`` labels, the removed ``W:`` fields), so a
whole-file diff of two runs is expected to differ.  This tool projects a log
onto only the quantities the refactor must NOT change, so that
``diff <(extract baseline) <(extract candidate)`` is a meaningful gate.

Usage:
    extract_alignment_invariants.py <log> --mode {surface,free} --layers 16:6,23:4

Fail-closed: exits non-zero with a message on stderr whenever an expected
section, field, or record count is missing.  A silently empty extraction would
make the diff pass for two equally broken logs.
"""

from __future__ import annotations

import argparse
import re
import sys

MEAN_TABLE_HEADER = "=== Per-layer residual mean"
RMS_TABLE_HEADER = "=== Per-layer residual RMS"
TRUTH_SECTION_HEADER = "=== ML vs TRUE misalignment"

DOF_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")

# Per-layer chi2 tokens look like ``v16l6=1.23``.  The lookbehind keeps the
# weight-diagnostic tokens ``Wv16l6: raw=...`` from being captured: those are
# followed by a colon rather than ``=``, and are preceded by a letter.
LAYER_CHI2_RE = re.compile(r"(?<![A-Za-z])v(\d+)l(\d+)=([-\d.eE+]+)")

# ``  vol=16 lay=6 (n=93)   0.1234mm   0.0012mm   0.5678mm   0.0034mm``
# Group 3 is X_before and group 4 is X_after; only X_after is invariant.
TABLE_ROW_RE = re.compile(
    r"\s*vol=(\d+) lay=(\d+).*?([-\d.]+)mm\s+([-\d.]+)mm"
)
TRUTH_LAYER_RE = re.compile(r"\s*vol=(\d+) lay=(\d+) \(n=\d+\)\s*$")
TRUTH_DOF_RE = re.compile(
    r"\s*(\w+): corr=([-+\d.naN]+)\s+recovery=([-+\d.naN]+)"
)


def fail(message):
    print(f"extract_alignment_invariants: {message}", file=sys.stderr)
    sys.exit(1)


def parse_layers(spec):
    layers = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        volume, _, layer = item.partition(":")
        layers.append((int(volume), int(layer)))
    if not layers:
        fail(f"--layers {spec!r} did not yield any volume:layer pair")
    return layers


def extract_surface_iterations(lines):
    """Surface mode logs ``Epoch [NNN/M] | LR: | Chi2: | L2: | Total: | ...``."""
    records = []
    for line in lines:
        if not line.startswith("Epoch ["):
            continue
        index = re.search(r"Epoch \[(\d+)/(\d+)\]", line)
        if index is None:
            fail(f"Epoch line without a parsable index: {line!r}")
        fields = []
        for key in ("LR", "Chi2", "L2", "Total"):
            match = re.search(rf"\b{key}: ([-\d.eE+]+)", line)
            if match is None:
                fail(f"Epoch line missing {key!r} field: {line!r}")
            fields.append(f"{key}={match.group(1)}")
        chi2_by_layer = LAYER_CHI2_RE.findall(line)
        layer_tokens = [f"v{v}l{l}={c}" for v, l, c in chi2_by_layer]
        records.append((
            int(index.group(1)),
            int(index.group(2)),
            "ITER {:04d} {} {}".format(
                int(index.group(1)), " ".join(fields), " ".join(layer_tokens)
            ).rstrip(),
        ))
    return records


def extract_free_iterations(lines):
    """Free mode logs ``Joint [NNNN/M] chi2/component=X invalid=Y%``."""
    records = []
    for line in lines:
        if not line.startswith("Joint ["):
            continue
        index = re.search(r"Joint \[(\d+)/(\d+)\]", line)
        if index is None:
            fail(f"Joint line without a parsable index: {line!r}")
        chi2 = re.search(r"chi2/component=([-\d.eE+]+)", line)
        invalid = re.search(r"invalid=([-\d.]+)%", line)
        if chi2 is None:
            fail(f"Joint line missing chi2/component: {line!r}")
        if invalid is None:
            fail(f"Joint line missing invalid fraction: {line!r}")
        records.append((
            int(index.group(1)),
            int(index.group(2)),
            "ITER {:04d} chi2={} invalid={}".format(
                int(index.group(1)), chi2.group(1), invalid.group(1)
            ),
        ))
    return records


def expected_surface_epochs(total):
    """Mirror ``epoch % 10 == 0 or epoch == 1`` over ``range(1, total + 1)``."""
    return {e for e in range(1, total + 1) if e % 10 == 0 or e == 1}


def expected_free_iterations(total):
    """Mirror ``iteration % 100 == 0 or iteration == total - 1``."""
    return {i for i in range(total) if i % 100 == 0 or i == total - 1}


def check_iteration_records(records, mode, lines):
    if not records:
        fail(f"no iteration records found for --mode {mode}")
    totals = {total for _, total, _ in records}
    if len(totals) != 1:
        fail(f"inconsistent iteration totals across records: {sorted(totals)}")
    total = totals.pop()
    seen = {index for index, _, _ in records}

    early_stop = next(
        (m for m in (re.search(r"early stop at iteration (\d+)", l) for l in lines) if m),
        None,
    )
    if early_stop is not None:
        limit = int(early_stop.group(1))
        expected = {i for i in expected_free_iterations(total) if i <= limit}
    elif mode == "surface":
        expected = expected_surface_epochs(total)
    else:
        expected = expected_free_iterations(total)

    if seen != expected:
        fail(
            f"--mode {mode}: iteration records {sorted(seen)} "
            f"do not match the expected {sorted(expected)} for total={total}"
        )


def extract_mean_x_after(lines, layers):
    """X_after column of the mean table only; the RMS table is allowed to change."""
    begin = next((i for i, l in enumerate(lines) if MEAN_TABLE_HEADER in l), None)
    if begin is None:
        fail(f"missing section header {MEAN_TABLE_HEADER!r}")
    end = next(
        (i for i, l in enumerate(lines[begin:], begin) if RMS_TABLE_HEADER in l), None
    )
    if end is None:
        fail(
            f"missing section header {RMS_TABLE_HEADER!r}; the mean table cannot be "
            "delimited and both tables share the same row format"
        )

    found = {}
    for line in lines[begin:end]:
        match = TABLE_ROW_RE.match(line)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        found[key] = match.group(4)

    missing = [f"{v}:{l}" for v, l in layers if (v, l) not in found]
    if missing:
        fail(f"mean table has no row for target layer(s) {', '.join(missing)}")
    return [f"MEAN_X_AFTER vol={v} lay={l} {found[(v, l)]}" for v, l in layers]


def extract_truth(lines, layers):
    begin = next((i for i, l in enumerate(lines) if TRUTH_SECTION_HEADER in l), None)
    if begin is None:
        fail(f"missing section header {TRUTH_SECTION_HEADER!r}")

    records = {}
    current = None
    for line in lines[begin + 1:]:
        layer = TRUTH_LAYER_RE.match(line)
        if layer is not None:
            current = (int(layer.group(1)), int(layer.group(2)))
            records.setdefault(current, {})
            continue
        dof = TRUTH_DOF_RE.match(line)
        if dof is not None and current is not None and dof.group(1) in DOF_NAMES:
            records[current][dof.group(1)] = (dof.group(2), dof.group(3))
        elif line.startswith("===") and current is not None:
            break

    out = []
    for volume, layer in layers:
        if (volume, layer) not in records:
            fail(f"truth section has no entry for target layer {volume}:{layer}")
        entries = records[(volume, layer)]
        for name in DOF_NAMES:
            if name not in entries:
                fail(f"truth entry for {volume}:{layer} is missing DOF {name!r}")
            corr, recovery = entries[name]
            out.append(
                f"TRUTH vol={volume} lay={layer} {name} corr={corr} recovery={recovery}"
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--mode", choices=("surface", "free"), required=True)
    parser.add_argument("--layers", required=True, help="e.g. 16:6,23:4")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    try:
        lines = open(args.log, encoding="utf-8").read().splitlines()
    except OSError as exc:
        fail(f"cannot read {args.log}: {exc}")

    if args.mode == "surface":
        records = extract_surface_iterations(lines)
    else:
        records = extract_free_iterations(lines)
    check_iteration_records(records, args.mode, lines)

    out = [text for _, _, text in records]
    out += extract_truth(lines, layers)
    out += extract_mean_x_after(lines, layers)
    print("\n".join(out))


if __name__ == "__main__":
    main()
