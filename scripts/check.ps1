$ErrorActionPreference = "Stop"

python -m pytest
python -m lightning_decoding.cli --help

