#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python 01_collect_snapshot.py
python 02_build_measures.py
python 03_llm_coding.py
python 04_analysis.py
python 05_package_results.py
