import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from mlx_omarchy_assistant.server import AssistantServer, voice_ready


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = AssistantServer(('127.0.0.1', 0), Path(self.temp.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port
        self.origin = f'http://127.0.0.1:{self.port}'
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        self.addCleanup(conn.close)
        data = json.dumps(body) if body is not None else None
        conn.request(method, path, data, headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def login(self):
        status, headers, data = self.request('POST', '/api/session', {'token': self.server.bootstrap},
                                            {'Origin': self.origin, 'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        return {'Cookie': headers['Set-Cookie'].split(';')[0], 'Origin': self.origin,
                'X-Assistant-CSRF': json.loads(data)['csrf'], 'Content-Type': 'application/json'}

    def test_untrusted_origin_host_and_missing_auth_are_refused(self):
        self.assertEqual(self.request('GET', '/api/status')[0], 401)
        self.assertEqual(self.request('GET', '/', headers={'Host': 'attacker.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/session', {'token': self.server.bootstrap},
            {'Origin': 'https://attacker.example', 'Content-Type': 'application/json'})[0], 403)

    def test_encoded_api_paths_require_authentication(self):
        self.assertEqual(self.request('GET', '/%61pi/conversations')[0], 401)
        self.assertEqual(self.request('POST', '/%61pi/conversations', {'save': True})[0], 401)

    def test_bootstrap_is_one_use_and_mutations_require_csrf(self):
        token = self.server.bootstrap
        headers = self.login()
        self.assertEqual(self.request('POST', '/api/session', {'token': token},
            {'Origin': self.origin, 'Content-Type': 'application/json'})[0], 401)
        missing = dict(headers)
        missing.pop('X-Assistant-CSRF')
        self.assertEqual(self.request('POST', '/api/conversations', {'save': False}, missing)[0], 403)
        self.assertEqual(self.request('POST', '/api/conversations', {'save': False}, headers)[0], 201)

    def test_history_export_and_delete_use_real_http(self):
        headers = self.login()
        status, _, data = self.request('POST', '/api/conversations', {'save': True}, headers)
        self.assertEqual(status, 201)
        cid = json.loads(data)['id']
        self.assertEqual(self.request('GET', f'/api/conversations/{cid}/export', headers=headers)[0], 200)
        self.assertEqual(self.request('DELETE', f'/api/conversations/{cid}', headers=headers)[0], 200)
        self.assertEqual(self.request('GET', f'/api/conversations/{cid}', headers=headers)[0], 404)
    def test_resume_without_a_saved_pair_is_absent(self):
        class Gpu:
            def acquire(self, blocking=False):
                return True
            def release(self):
                return None
        class Coordinator:
            gpu = Gpu()
            def close(self):
                return None
        class Manager:
            def adopt_saved(self):
                raise RuntimeError("no saved pair lock to adopt")
            def start(self):
                raise AssertionError("start should not run when no lock exists")
            def cancel(self):
                return None
            def stop(self):
                return {"stopped": True}
        self.server._manager = Manager()
        self.server._coordinator = Coordinator()
        headers = self.login()
        status, _, _ = self.request("POST", "/api/resume", {}, headers)
        self.assertEqual(status, 202)
        self.server.setup_thread.join(2)
        self.assertEqual(self.server.setup_state["state"], "absent")

    def test_context_selection_preserves_constraints_and_rejects_unknown_fields(self):
        headers = self.login()
        _, _, raw = self.request("POST", "/api/conversations", {}, headers)
        cid = json.loads(raw)["id"]
        path = f"/api/conversations/{cid}/context"
        body = {"selected_turn_ids": [], "pinned_constraints": "Use SI units."}
        status, _, raw = self.request("POST", path, body, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["context"], body)
        self.assertEqual(self.request("POST", path, dict(body, silent_trim=True), headers)[0], 400)
        _, _, raw = self.request("GET", f"/api/conversations/{cid}", headers=headers)
        self.assertEqual(json.loads(raw)["context"], body)

    def test_headers_block_remote_resources_and_caching(self):
        headers = self.login()
        status, response_headers, _ = self.request('GET', '/api/conversations', headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(response_headers['Cache-Control'], 'no-store')
        self.assertIn("connect-src 'self'", response_headers['Content-Security-Policy'])
        self.assertEqual(response_headers['X-Content-Type-Options'], 'nosniff')

    def test_speech_requires_an_admitted_voice_reservation(self):
        headers = self.login()
        status, _, raw = self.request('POST', '/api/speak', {}, headers)
        self.assertEqual(status, 400)
        self.assertIn('Enable voice in model setup', json.loads(raw)['error'])

    def test_speech_stream_carries_turn_identity_without_retained_audio(self):
        from types import SimpleNamespace

        class PcmSource:
            def synthesize_chunks(self, text, cancel):
                yield SimpleNamespace(sample_rate=24000, data=b'\x01\x00\x02\x00')

            def close(self):
                pass

        from unittest.mock import Mock
        self.server._manager = Mock()
        self.server._manager.status.return_value = {'voice': {'requested': True}}
        self.server._manager.stop.return_value = {'stopped': True}
        headers = self.login()
        cid = self.server.store.create()['id']
        turn = self.server.store.begin(cid, 'Read this')
        self.server.store.emit(cid, turn, 'text', {'text': 'A visible sentence.'})
        self.server.store.finish(cid, turn)
        self.server._synthesis = PcmSource()
        status, response_headers, raw = self.request('POST', '/api/speak',
            {'conversation_id': cid, 'turn_id': turn, 'text': 'A visible sentence.', 'sentence_sequence': 2}, headers)
        self.assertEqual(status, 200)
        self.assertEqual(response_headers['Content-Type'], 'text/event-stream')
        events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: ')]
        self.assertEqual(events[0]['turn_id'], turn)
        self.assertEqual(events[0]['sentence_sequence'], 2)
        self.assertEqual(events[0]['data'], 'AQACAA==')
        self.assertFalse(events[-1]['stopped'])
        self.assertFalse(self.server.coordinator.gpu.locked())
    def test_transfer_rejects_process_overrides_and_reports_real_file_errors(self):
        headers = self.login()
        self.assertEqual(self.request('POST', '/api/transfer',
            {'action': 'inspect', 'python': '/bin/sh'}, headers)[0], 400)
        bad = Path(self.temp.name) / 'broken.zip'
        bad.write_bytes(b'not a zip archive')
        self.assertEqual(self.request('POST', '/api/transfer',
            {'action': 'inspect', 'bundle': str(bad)}, headers)[0], 202)
        self.server.transfer_thread.join(3)
        status, _, raw = self.request('GET', '/api/transfer', headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['state'], 'error')
        self.assertIn('zip', json.loads(raw)['error'].lower())
        self.assertFalse(self.server.coordinator.gpu.locked())

if __name__ == '__main__':
    unittest.main()


class VoiceReadyTests(unittest.TestCase):
    def test_stored_receipt_without_a_live_runtime_is_not_ready(self):
        self.assertFalse(voice_ready({"qualified": True, "ready": False}, "ready"))
        self.assertFalse(voice_ready({"ready": True}, "unqualified"))
        self.assertFalse(voice_ready({"ready": True}, "missing"))
        self.assertTrue(voice_ready({"ready": True, "qualified": True}, "ready"))
