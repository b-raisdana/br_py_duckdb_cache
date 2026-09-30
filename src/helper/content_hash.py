"""Deterministic 7-character content hashes embedded in every dump filename.

Dump names are content-addressed, which gives two properties the debugging
workflow needs: the same logical output lands on the same file, so repeated
runs reuse one dump instead of littering duplicates, and any change to that
output lands on a new file, so the previous version stays readable.

The hash runs over the exact bytes of the file being written (Parquet payload
for a frame, normalized JSON text for the descriptor), so it is computed before
the final name is assembled. CRC-32 is used on purpose: a debug artefact is no
security boundary, and CRC-32 is the fastest checksum in the standard library
(no extra dependency, no cryptographic cost on large frames).
"""

from __future__ import annotations

import re
import zlib

HASH_LENGTH = 7
CONTENT_HASH_PATTERN = re.compile(rf"[0-9a-f]{{{HASH_LENGTH}}}")


def content_hash(payload: bytes) -> str:
    """Return exactly :data:`HASH_LENGTH` lowercase hex characters for ``payload``."""
    return f"{zlib.crc32(payload):08x}"[:HASH_LENGTH]


def is_content_hash(text: str) -> bool:
    """True when ``text`` is a well-formed content hash."""
    return CONTENT_HASH_PATTERN.fullmatch(text) is not None
