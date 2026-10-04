#!/usr/bin/env bash
# Launch the tests build detached (jw16).
set -uo pipefail
nohup nice -n 19 bash /var/tmp/dfuse/build_tests.sh > /var/tmp/dfuse/build_tests-status.log 2>&1 &
echo "TESTS-BUILD-STARTED pid=$!"
