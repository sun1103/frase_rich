#!/usr/bin/env python
"""CLI wrapper: build the rich weighted FRASE index."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frase_rich.build import main  # noqa: E402

if __name__ == "__main__":
    main()
