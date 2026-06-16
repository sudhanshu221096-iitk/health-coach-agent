import sys
import os

# Add the project root to Python path
sys.path.insert(0, "/opt/render/project/src")

# Patch sqlite3 before chromadb loads
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from api.main import app
