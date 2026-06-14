"""
rag/_compat.py

Patches sqlite3 to the pysqlite3-binary version before ChromaDB imports.
Must be imported before any chromadb import in the process.
"""
import sys

try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # System sqlite is new enough
