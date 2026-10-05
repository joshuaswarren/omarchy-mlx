#!/usr/bin/env bash
# Remove the FamGlmDsa M2 scratch dir after evidence archiving.
set -eu
if [ -d ~/dsa-fam-correctness ]; then rm -r ~/dsa-fam-correctness; fi
echo "scratch removed; disk now:"
df -h /home | tail -1
