"""
Main Entrypoint — Accident Detection System (Person 1 + Person 2)
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import main

if __name__ == "__main__":
    main()
