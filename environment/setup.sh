#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).
#
# The gym needs only numpy (exact scoring, models) and regex (GPT-2
# pre-tokenizer). The previously listed torch/transformers served an unscored
# GPT-2 runner that has been removed.

set -e

uv pip install --system regex==2024.11.6 numpy==1.26.4
