import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "decode_pair", HERE / "scripts" / "decode_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DecodePairGate(unittest.TestCase):
    def setUp(self):
        self.gate = load_gate()

    def test_pass_within_five_percent(self):
        v = self.gate.verdict(100.0, 104.9)
        self.assertEqual(v['verdict'], 'pass')

    def test_refuse_over_five_percent_high(self):
        v = self.gate.verdict(106.0, 100.0)
        self.assertEqual(v['verdict'], 'refuse')

    def test_refuse_over_five_percent_low(self):
        v = self.gate.verdict(100.0, 106.0)
        self.assertEqual(v['verdict'], 'refuse')

    def test_exact_boundary_passes(self):
        v = self.gate.verdict(100.0, 105.0)
        self.assertEqual(v['verdict'], 'pass')

    def test_ids_mismatch_refuses(self):
        v = self.gate.verdict(100.0, 100.5, kit_ids='aaa',
                              packaged_ids='bbb')
        self.assertEqual(v['verdict'], 'refuse')
        self.assertEqual(v['reason'], 'ids_divergence')
        v = self.gate.verdict(100.0, 100.5, kit_ids='aaa',
                              packaged_ids='aaa')
        self.assertEqual(v['verdict'], 'pass')

    def test_missing_rate_refuses(self):
        for kit, pkg in ((0.0, 100.0), (100.0, 0.0), (None, 100.0)):
            v = self.gate.verdict(kit, pkg)
            self.assertEqual(v['verdict'], 'refuse', (kit, pkg))

    def test_non_finite_rate_refuses(self):
        nan, inf = math.nan, math.inf
        for kit, pkg in ((nan, 10.0), (10.0, nan), (inf, 10.0),
                         (10.0, inf), (-inf, 10.0), (inf, inf),
                         ('nan', '10'), (True, True), (10**400, 10.0)):
            v = self.gate.verdict(kit, pkg)
            self.assertEqual(v['verdict'], 'refuse', (kit, pkg))
            self.assertEqual(v['reason'], 'unreadable_rate', (kit, pkg))

    def test_invalid_max_delta_refuses(self):
        for max_delta in (math.nan, math.inf, -math.inf, -0.01, True,
                          10**400):
            v = self.gate.verdict(1.0, 1000.0, max_delta)
            self.assertEqual(v['verdict'], 'refuse', max_delta)
            self.assertEqual(v['reason'], 'invalid_max_delta', max_delta)

    def test_cli_non_finite_refuses_with_strict_json(self):
        cases = (
            (['--kit-tps', 'nan', '--packaged-tps', '10'],
             'unreadable_rate'),
            (['--kit-tps', 'inf', '--packaged-tps', 'inf'],
             'unreadable_rate'),
            (['--kit-tps=-inf', '--packaged-tps', '10'],
             'unreadable_rate'),
            (['--kit-tps', '10', '--packaged-tps', 'inf'],
             'unreadable_rate'),
            (['--kit-tps', '1', '--packaged-tps', '1000',
              '--max-delta', 'nan'], 'invalid_max_delta'),
            (['--kit-tps', '1', '--packaged-tps', '1000',
              '--max-delta', 'inf'], 'invalid_max_delta'),
            (['--kit-tps', '1', '--packaged-tps', '1000',
              '--max-delta=-0.01'], 'invalid_max_delta'),
        )

        def reject(constant):
            raise ValueError(f'non-JSON constant {constant}')

        for args, reason in cases:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / 'pair.json'
                rc = self.gate.main(args + [
                    '--model', 'model-x', '--tokens', '64',
                    '--tag', 'qualification-r8', '--out', str(out)])
                self.assertEqual(rc, 2, args)
                rec = json.loads(out.read_text(), parse_constant=reject)
                self.assertEqual(rec['verdict'], 'refuse', args)
                self.assertEqual(rec['reason'], reason, args)

    def test_receipt_carries_the_pair(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'pair.json'
            rc = self.gate.main([
                '--kit-tps', '62.85', '--packaged-tps', '63.10',
                '--kit-ids', 'aaa', '--packaged-ids', 'aaa',
                '--model', 'model-x', '--tokens', '64',
                '--tag', 'qualification-r8', '--out', str(out)])
            self.assertEqual(rc, 0)
            rec = json.loads(out.read_text())
            self.assertEqual(rec['kit_decode_tps'], 62.85)
            self.assertEqual(rec['packaged_decode_tps'], 63.10)
            self.assertEqual(rec['verdict'], 'pass')
            self.assertEqual(rec['tag'], 'qualification-r8')
            self.assertIn('delta_frac', rec)

    def test_cli_refuse_exit(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'pair.json'
            rc = self.gate.main([
                '--kit-tps', '50.0', '--packaged-tps', '63.10',
                '--model', 'model-x', '--tokens', '64',
                '--tag', 'qualification-r8', '--out', str(out)])
            self.assertEqual(rc, 2)
            rec = json.loads(out.read_text())
            self.assertEqual(rec['verdict'], 'refuse')


if __name__ == '__main__':
    unittest.main()
