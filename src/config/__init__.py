# Import config through the package path:
#     import src.config.config as cfg
#
# This file used to re-export names from src.config.config, which recursed when
# the same file was also reachable as top-level `config` (two sys.path roots).
# imports.py now uses a single root, so no re-export is needed here.
