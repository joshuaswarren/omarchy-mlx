# Harbor offline sub-steps (Q-104: issues #62/#63/#64)

Three named offline sub-steps, one per harbor gap - each completable by a
pane without multi-node hardware, with the artifact in this branch.

## #62 - concurrent inference receipt harness

`tools/harbor/concurrency_harness.py`: the 30x4 concurrent OpenAI-compatible
request runner the two-node receipt needs. Aggregate tokens/s, p50/p95,
HTTP 200 rate and empty-completion accounting are built in; `--endpoint
http://127.0.0.1:0` stub mode validates the harness's own accounting
offline (validated: 12/12 stub requests, 100% ok, p50 0.05 s). When the
two-node serve exists the real run is one command with `--out receipt.json`.

Offline-provable artifact: the harness itself + the stub-mode summary.

## #63 - native catalog/chat app failure contract

`tests/harbor/test_catalog_app_contract.py`: the failure-path rows written
before the app exists (offline hub keeps the bundled catalog browsable, a
failed pull discards partial bytes and registers nothing, an unreachable
gateway preserves the typed message and sends nothing partial). All eight
contract rows assert unconditionally today; they go live when
`omarchy_cluster.catalog_app` lands, then the app's real tests inherit the
same file.

Offline-provable artifact: the contract skeleton (8 rows asserted green
offline in this branch).

## #64 - the mDNS advertisement discloses no cluster token

`tests/test_advertise_no_token.py`: four tests proving `advertise_start`
publishes only the service name, type and port - no token material in the
argv on either the dns-sd (macOS) or avahi-publish (Linux) path, with a
long-name case pushing the argv length. The accept side (a rogue copying a
joined name, an unjoined advertiser) is already covered in
tests/test_discover_trust.py.

Offline-provable artifact: four passing tests, no hardware.
