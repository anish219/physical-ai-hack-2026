"""Entry point replacing /home/pah/pah/armserver.py; use the Dream environment."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / "physical-ai-hacks" / "src"))

from dream_hackathon.armserver import main

if __name__ == "__main__":
    main()
