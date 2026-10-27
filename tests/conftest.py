"""
Shared test fixtures and configuration.
Ensures src/ modules are importable as a package (with relative imports).
"""

import sys
from pathlib import Path

# Insert the PROJECT ROOT (trading_scanner/) so that `from src.xxx import ...` works.
# This is needed because src/ modules use relative imports (from .exceptions import ...).
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Also make `import src.xxx` resolve to the package
import importlib
if 'src' not in sys.modules:
    src_path = _project_root / 'src'
    spec = importlib.util.spec_from_file_location('src', src_path / '__init__.py', submodule_search_locations=[str(src_path)])
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules['src'] = mod
        spec.loader.exec_module(mod)
