"""Recover artefacts from the base64 blocks colab_bootstrap.py frames into stdout.

    python3 scripts/decode_artifacts.py <captured-output> <destination-dir>

Downloading artefacts off a Colab VM after the training script exits does not
work: the session becomes unreachable at teardown, and every download fails by
any name. stdout is the channel that survives, because `colab exec` has
already streamed it. The bootstrap frames each file into it; this takes them
back out.

Two properties matter more than speed here.

**A block is only written if its SHA-256 matches.** The stream can be
truncated by a lost session, or interleaved with training output, and a
corrupt `params.msgpack` that looks like a real one is worse than no file at
all. Mismatches are reported on stderr and the file is left alone.

**An existing good file is never replaced by a worse one.** A segment can be
decoded more than once -- from the live capture and again from the log -- and
re-running must be safe.

Prints one line per artefact to stdout so the driver can log what happened,
and exits non-zero only if a block was found and could not be recovered.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import sys
from pathlib import Path

BEGIN = "===MTG-ARTEFACT-BEGIN"
END = "===MTG-ARTEFACT-END"

# Deliberately tolerant about what sits in front of the marker: `colab exec`
# and `tee` both prepend, and a block is still perfectly recoverable when
# something has decorated its lines.
BEGIN_RE = re.compile(
    rf"{re.escape(BEGIN)}\s+(?P<name>\S+)\s+(?P<size>\d+)\s+(?P<sha>[0-9a-f]{{64}})"
)
END_RE = re.compile(rf"{re.escape(END)}\s+(?P<name>\S+)")
B64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")


def decode(text: str, dest: Path) -> tuple[list[str], list[str]]:
    """Writes every intact block into `dest`. Returns (written, failed)."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    failed: list[str] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        begin = BEGIN_RE.search(lines[i])
        if not begin:
            i += 1
            continue

        name = begin.group("name")
        size = int(begin.group("size"))
        want = begin.group("sha")

        chunks: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            end = END_RE.search(lines[i])
            if end and end.group("name") == name:
                closed = True
                i += 1
                break
            if BEGIN_RE.search(lines[i]):
                break  # a new block started; this one never closed
            stripped = lines[i].strip()
            if stripped and B64_RE.match(stripped):
                chunks.append(stripped)
            i += 1

        if not closed:
            failed.append(f"{name}: block never closed (stream truncated?)")
            continue

        try:
            raw = base64.b64decode("".join(chunks), validate=True)
        except (binascii.Error, ValueError) as error:
            failed.append(f"{name}: base64 did not decode ({error})")
            continue

        if len(raw) != size:
            failed.append(f"{name}: {len(raw)}B decoded, header says {size}B")
            continue
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            failed.append(f"{name}: sha256 {got[:12]} != {want[:12]}")
            continue

        target = dest / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == want:
            written.append(f"{name} (already present, identical)")
            continue
        target.write_bytes(raw)
        written.append(f"{name} ({len(raw):,}B)")

    return written, failed


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    source, dest = Path(argv[1]), Path(argv[2])
    if not source.is_file():
        print(f"no such capture: {source}", file=sys.stderr)
        return 2

    written, failed = decode(
        source.read_text(encoding="utf-8", errors="replace"), dest
    )
    for line in written:
        print(f"decoded {line}")
    for line in failed:
        print(f"FAILED  {line}", file=sys.stderr)
    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
