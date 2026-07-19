"""
ULAL — Unified Lifelong Action Learning.

Import through the package root:

    import src.config.config as cfg
    from src.data.cache import load_feature_cache
    from src.cl import run_cl_stream

Or load everything into a notebook namespace:

    %run /content/ulal/src/imports.py

This file also makes `src` an explicit regular package. Without it `src` resolves as a
PEP 420 namespace package, which works but breaks in confusing ways if another `src`
directory ever appears earlier on sys.path.
"""

__version__ = "0.2.0"
