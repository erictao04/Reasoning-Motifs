from __future__ import annotations

import sys

from analysis.cli import main


if __name__ == "__main__":
    main(["skipgrams", *sys.argv[1:]])
