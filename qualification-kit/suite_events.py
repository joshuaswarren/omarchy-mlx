"""pytest plugin: capture negative outcomes independently of JUnit/subtest formatting."""
import json
import os
import re
from collections import Counter
from pathlib import Path

failures = {}
skips = []
seen = Counter()
invalid = 0


def failure_class(report):
    message = str(getattr(getattr(report.longrepr, 'reprcrash', None), 'message', ''))
    if ('[omarchy]' in message and ('is not implemented' in message
                                    or 'maxStorageBufferRange' in message)):
        return 'vulkan_refusal'
    if message.startswith('UnboundLocalError:') and "'custom_kernel'" in message:
        return 'upstream_harness'
    if re.fullmatch(r'contains \d+ failed subtests?', str(report.longrepr)):
        return 'upstream_subtest_summary'
    if message.startswith(('AssertionError', 'assert ')):
        return 'assertion'
    return 'other'


def pytest_runtest_logreport(report):
    global invalid
    if report.when != 'call' and report.passed:
        return
    node = report.nodeid.rsplit('/', 1)[-1]
    context = getattr(report, 'context', None)
    if context:
        fields = ','.join(f'{k}={v}' for k, v in sorted(context.kwargs.items()))
        node += f'::subtest({context.msg or ""};{fields})'
        seen[node] += 1
        if seen[node] > 1:
            node += f'#{seen[node]}'
    if report.passed:
        return
    if not re.fullmatch(r"test_\w+\.py::[\w.:(),'\[\]<>+=#; -]+", node, re.ASCII):
        invalid += 1
        return
    if report.failed:
        failures[node if report.when == 'call' else f'{node}::{report.when}'] = failure_class(report)
    elif report.skipped:
        skips.append(node)


def pytest_sessionfinish(session, exitstatus):
    Path(os.environ['QUALIFY_SUITE_EVENTS']).write_text(json.dumps({
        'failures': failures, 'skips': skips, 'invalid': invalid,
    }))
