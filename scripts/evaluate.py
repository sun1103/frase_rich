#!/usr/bin/env python
"""CLI wrapper: leave-one-complex-out retrieval benchmark."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
