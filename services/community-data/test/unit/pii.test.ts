import { describe, expect, test } from "bun:test";
import { parseHostAliases, scanPii } from "../../src/pii";
import fixture from "./fixtures/payload-v1.json";

const CLEAN_VALUES = [
  "[redacted]",
  "[redacted-mac]",
  "[redacted-ip4]",
  "[redacted-ip6]",
  "[redacted-uuid]",
  "[home]",
  "[host]",
  "[user]",
];

describe("server-side PII scan", () => {
  test("clean collector payload passes", () => {
    expect(scanPii(JSON.stringify(fixture))).toBeNull();
  });

  test("collector placeholder text passes", () => {
    for (const value of CLEAN_VALUES) {
      expect(scanPii(`"model":"Machine ${value} build"`)).toBeNull();
    }
  });

  test.each([
    ["mac", '"model":"eth0 00:1A:2B:3C:4D:5E"'],
    ["ipv4", '"model":"gateway at 192.168.1.42"'],
    ["ipv4", '"model":"subnet 10.0.0.0/8"'],
    ["ipv6", '"model":"link fe80::1"'],
    ["ipv6", '"model":"ula fd00:1234:5678::1"'],
    ["uuid", '"kernel":"uuid 123e4567-e89b-12d3-a456-426614174000"'],
    ["home_path", '"model":"lives in /home/joshua/src"'],
    ["home_path", '"model":"C:\\\\Users\\\\joshua\\\\docs"'],
    ["credential", '"model":"token ghp_0123456789abcdefghijklmnop"'],
    ["credential", '"model":"AKIA0123456789ABCDEF"'],
    ["credential", '"model":"slack xoxb-0123456789-abcdef"'],
    ["credential", '"model":"token sk-0123456789abcdefghijklmnop"'],
    ["credential", '"auth":"Bearer abcdef123456"'],
    ["credential", '"note":"password = \\"hunter2hunter2\\""'],
    ["hostname", '"model":"joshuas-macbook.local"'],
  ])("refuses %s hit", (kind, snippet) => {
    const kinds = scanPii(snippet);
    expect(kinds).not.toBeNull();
    expect(Object.keys(kinds as object)).toContain(kind);
  });

  test("serial numbers in JSON shape are refused", () => {
    const kinds = scanPii('"note":"serial_number": "C02XJ1ABCDEF"');
    expect(kinds).not.toBeNull();
    expect(Object.keys(kinds as object)).toContain("serial");
  });

  test("redaction count keys do not false-positive", () => {
    const summary = {
      redaction_summary: { mac: 2, ipv4: 1, uuid: 3, credential: 1 },
      model: "Mac mini",
      files: [{ path: "quick.json", bytes: 10, sha256: "a".repeat(64) }],
    };
    expect(scanPii(JSON.stringify(summary))).toBeNull();
  });

  test("kernel version strings do not false-positive as IPv4", () => {
    expect(scanPii('"kernel":"6.9.1-asahi"')).toBeNull();
  });

  test("iBoot firmware versions in both deep summary fields pass", () => {
    expect(scanPii(JSON.stringify({
      boot_chain: "iboot2=iBoot-10151.140.19.700.2",
      ane_port_detail: { devicetree: { boot: { chosen: {
        "asahi,iboot2-version": "iBoot-10151.140.19.700.2",
        "asahi,system-fw-version": "iBoot-20712.1.2.0.0",
      } } } },
    }))).toBeNull();
  });

  test("firmware exception does not hide addresses or CIDR suffixes", () => {
    for (const value of [
      "iBoot-10151.140.19.700.2 gateway 198.51.100.7",
      "iBoot-198.51.100.7",
      "iBoot-20712.1.2.0.0/8",
      "driver version: 198.51.100.7",
      "iBoot-1.2.3.4.198.51.100.7",
      "iBoot-999.1.2.198.51.100.7",
      "iBoot-1.198.51.100.7",
      "iBoot-999.1.2.3.4.198.51.100.7",
      "prefixiBoot-10151.140.19.700.2",
      "iBoot-10151.192.168.1.5",
      "iBoot-20712.198.51.100.7",
    ]) {
      expect(scanPii(JSON.stringify({ boot_chain: value }))).toHaveProperty("ipv4");
    }
  });

  test("firmware recognition accepts consistent case variants", () => {
    for (const prefix of ["iBoot", "IBOOT", "Iboot", "iboot"]) {
      expect(scanPii(JSON.stringify({ boot_chain:
        `${prefix}-10151.140.19.700.2` }))).toBeNull();
    }
  });

  test("distant firmware tokens do not exempt later addresses", () => {
    const value = "iBoot-10151.140.19.700.2 " + "x".repeat(1024) +
      " 198.51.100.7";
    expect(scanPii(JSON.stringify({ boot_chain: value }))).toEqual({ ipv4: 1 });
    expect(scanPii("198.51.100.7 ".repeat(20000))).toEqual({ ipv4: 50 });
  });

  test("host alias inside a systemd unit path is refused", () => {
    const line =
      '"dmesg":["Oct  1 21:58:41 [host] systemd[1]: ' +
      '/etc/systemd/system/box1-ane.service:9: Ignoring"]';
    const kinds = scanPii(line, ["box1", "box1-max"]);
    expect(kinds).not.toBeNull();
    expect(Object.keys(kinds as object)).toContain("hostname_alias");
    // The same line passes when no aliases ride the request.
    expect(scanPii(line)).toBeNull();
  });

  test("hex runs and longer words do not match an alias", () => {
    expect(scanPii('"iomem":"0xdeadbeef beefcake"', ["deadbeef"])).toBeNull();
  });

  test("alias parsing filters junk and caps the list", () => {
    expect(parseHostAliases("box1, x, ab, ok-name99, BOX1")).toEqual([
      "box1",
      "ok-name99",
    ]);
    expect(parseHostAliases(null)).toEqual([]);
    expect(parseHostAliases("a,".repeat(40) + "last1").length).toBeLessThanOrEqual(16);
  });
});
