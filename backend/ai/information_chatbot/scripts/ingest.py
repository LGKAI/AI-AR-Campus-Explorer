"""
This file has been moved to ingest.py (project root).

To run the ingest pipeline:
    python ingest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
exec(open(os.path.join(os.path.dirname(__file__), "..", "ingest.py")).read())
