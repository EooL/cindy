"""Regression tests for the public bridge entry points.

Protects the reviewed bugs: unknown/paid prices becoming zero, incompatible SQLite
queries, missing/wrong usage buckets, and false-success errors. No existing Cindy
suite executes this external Python bridge. Unit fixtures use no real home/config.
Run: python -m unittest discover -s docs/examples/opencode-bridge -p test_bridge.py
"""
import importlib.util
import io
import json
import pathlib
import sqlite3
from contextlib import closing
import tempfile
import threading
import unittest
from unittest.mock import patch

import opencode_cindy_bridge as bridge
from opencode_transport import BridgeError, Conversation, OpenCodeTransport, usage_from_messages


class Handler:
    def __init__(self):
        self.wfile = io.BytesIO()
        self.status = None
        self.json = None
        self.headers = {}
    def send_response(self, status): self.status = status
    def send_header(self, key, value): self.headers[key] = value
    def end_headers(self): pass
    def _send_json(self, status, obj): self.status, self.json = status, obj


class BridgeTests(unittest.TestCase):
    def test_cost_boundaries_preserve_zero_without_manufacturing_prices(self):
        self.assertEqual(bridge.model_cost({'input': 12, 'output': 30, 'cache': {'read': 0, 'write': 2}}),
                         {'input': 12, 'output': 30, 'cacheRead': 0, 'cacheWrite': 2})
        for value in [None, [], {}, {'input': True, 'output': '2', 'cacheRead': -1, 'cacheWrite': float('inf')}]:
            self.assertEqual(bridge.model_cost(value), {})

    def test_configured_prices_and_namespaced_duplicates(self):
        config = {'provider': {'one': {'models': {'same': {'cost': {'input': 12, 'output': 30}}}},
                               'two': {'models': {'same': {}}}}}
        with patch.object(bridge, 'opencode_command', return_value=['fixture']), \
             patch.object(bridge, 'load_opencode_config', return_value=config), \
             patch.object(bridge.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ''
            models = bridge.discover_dynamic_models()
        by_id = {m['id']: m for m in models}
        self.assertEqual(by_id['one/same']['cost'], {'input': 12, 'output': 30})
        self.assertEqual(by_id['two/same']['cost'], {})

    def test_modern_sqlite_readonly_and_honest_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'fixture.db'
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('CREATE TABLE session(cost REAL, tokens_input INT, tokens_output INT, '
                             'tokens_cache_read INT, tokens_cache_write INT, tokens_reasoning INT, time_updated INT)')
                conn.execute('INSERT INTO session VALUES (1.5, 100, 20, 30, 40, 5, 50000)')
                conn.commit()
            before = path.read_bytes()
            stats = bridge.get_opencode_stats(path, now_ms=50000)
            self.assertTrue(stats['available'])
            self.assertEqual(stats['tokens_input'], 100)
            self.assertEqual(stats['tokens_cache_write'], 40)
            self.assertEqual(stats['total_cost'], 1.5)
            self.assertEqual(stats['recently_updated_sessions'], {'5h': 1, '7d': 1})
            self.assertEqual(path.read_bytes(), before)
            with patch.object(bridge, 'get_opencode_stats', return_value=stats):
                quota = bridge.build_quota_payload()
            self.assertEqual(quota['windows'], [])
            self.assertFalse(quota['quota_available'])

    def test_unsupported_schema_is_unavailable_not_measured_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'fixture.db'
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('CREATE TABLE session(id TEXT)')
            self.assertEqual(bridge.get_opencode_stats(path)['reason'], 'unsupported_schema')

    def test_usage_sums_steps_cache_and_reasoning(self):
        messages = [{'info': {'role': 'assistant', 'time': {'completed': 1}, 'tokens': {
            'input': 100, 'output': 20, 'reasoning': 5, 'cache': {'read': 30, 'write': 40}}}},
            {'info': {'role': 'assistant', 'finish': 'stop', 'tokens': {'input': 1, 'output': 2}}},
            {'info': {'role': 'user', 'tokens': {'input': 999}}}]
        result = usage_from_messages(messages)
        self.assertEqual(result['prompt_tokens'], 171)
        self.assertEqual(result['completion_tokens'], 27)
        self.assertEqual(result['total_tokens'], 198)
        self.assertEqual(result['prompt_tokens_details']['cached_tokens'], 30)
        self.assertIsNone(usage_from_messages([{'info': {'role': 'assistant', 'tokens': {'input': 0}}}]))
        self.assertIsNone(usage_from_messages([{'info': {'role': 'assistant', 'finish': 'stop', 'tokens': {}}}]))

    def test_completed_stream_contains_measured_usage(self):
        transport = OpenCodeTransport('http://unused', request_json=lambda *args: None)
        conversation = Conversation('fixture', '/fixture', 'fixture', {}, 'auto')
        conversation.events.put(('delta', {'reasoning_content': 'Think'}))
        conversation.events.put(('delta', {'content': 'Hel'}))
        usage = {'prompt_tokens': 123, 'completion_tokens': 45, 'total_tokens': 168}
        conversation.events.put(('done', ({'info': {'finish': 'stop'}, 'parts': [
            {'type': 'text', 'text': 'Hello'}, {'type': 'reasoning', 'text': 'Think'}]}, usage)))
        handler = Handler()
        with patch.object(transport, '_new', return_value=conversation):
            transport.chat({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}, handler, [])
        chunks = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertEqual([c['usage'] for c in chunks if 'usage' in c], [usage])
        self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices']), 'Hello')
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'stop')

    def test_errors_before_and_after_stream_start_are_not_success(self):
        for after_start in (False, True):
            with self.subTest(after_start=after_start):
                transport = OpenCodeTransport('http://unused', request_json=lambda *args: None)
                conversation = Conversation('fixture', '/fixture', 'fixture', {}, 'auto')
                if after_start:
                    conversation.events.put(('delta', {'content': 'Partial'}))
                conversation.events.put(('error', BridgeError(502, 'upstream_failure', 'Failed')))
                handler = Handler()
                with patch.object(transport, '_new', return_value=conversation):
                    if after_start:
                        transport.chat({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}, handler, [])
                        self.assertIn('"error"', handler.wfile.getvalue().decode())
                        self.assertNotIn('"finish_reason": "stop"', handler.wfile.getvalue().decode())
                    else:
                        with self.assertRaises(BridgeError) as raised:
                            transport.chat({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Hi'}]}, handler, [])
                        self.assertEqual(raised.exception.status, 502)
                        self.assertIsNone(handler.json)

    def test_completed_tool_history_does_not_prevent_new_user_turn(self):
        transport = OpenCodeTransport('http://unused')
        self.assertIsNone(transport._resume({'model': 'fixture', 'messages': [
            {'role': 'tool', 'tool_call_id': 'call_bridge_old', 'content': 'Result'},
            {'role': 'assistant', 'content': 'Done'}, {'role': 'user', 'content': 'Next'}]}))
        with self.assertRaises(BridgeError):
            transport._resume({'model': 'fixture', 'messages': [
                {'role': 'tool', 'tool_call_id': 'call_bridge_old', 'content': 'Result'}]})

    def test_native_mcp_result_returns_only_to_its_own_conversation(self):
        transport = OpenCodeTransport('http://unused', request_json=lambda *args: None)
        conversations = [Conversation(token, '/fixture', 'fixture', {
            'echo': {'name': 'echo', 'inputSchema': {'type': 'object'}}}, 'auto')
            for token in ('one', 'two')]
        replies = {}
        workers = []
        calls = []
        for conversation in conversations:
            transport.conversations[conversation.token] = conversation
            worker = threading.Thread(target=lambda c=conversation: replies.update({c.token: transport.mcp(c.token,
                {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': 'echo', 'arguments': {'value': c.token}}})}))
            worker.start()
            workers.append(worker)
            kind, call = conversation.events.get(timeout=1)
            self.assertEqual(kind, 'tool')
            calls.append(call)
        self.assertNotEqual(calls[0].id, calls[1].id)
        try:
            with self.assertRaises(BridgeError):
                transport._resume({'model': 'fixture', 'messages': [
                    {'role': 'tool', 'tool_call_id': call.id, 'content': 'mixed'} for call in calls]})
            self.assertFalse(any(call.resolved for call in calls))
            for conversation, call in zip(conversations, calls):
                conversation.events.put(('done', ({'info': {'finish': 'stop'}, 'parts': [
                    {'type': 'text', 'text': conversation.token}]}, None)))
                handler = Handler()
                transport.chat({'model': 'fixture', 'messages': [
                    {'role': 'tool', 'tool_call_id': call.id, 'content': conversation.token}]}, handler, [])
                self.assertEqual(handler.json['choices'][0]['message']['content'], conversation.token)
        finally:
            for conversation in conversations:
                transport._close(conversation)
            for worker in workers:
                worker.join(timeout=1)
        self.assertEqual(replies['one']['result']['content'][0]['text'], 'one')
        self.assertEqual(replies['two']['result']['content'][0]['text'], 'two')

    def test_text_json_never_becomes_an_executed_tool_call(self):
        transport = OpenCodeTransport('http://unused', request_json=lambda *args: None)
        conversation = Conversation('fixture', '/fixture', 'fixture', {}, 'required')
        conversation.events.put(('done', ({'info': {'finish': 'stop'}, 'parts': [
            {'type': 'text', 'text': '{"name":"echo","arguments":{"value":"fake"}}'}]}, None)))
        with patch.object(transport, '_new', return_value=conversation):
            with self.assertRaises(BridgeError) as raised:
                transport.chat({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Use echo'}]}, Handler(), [])
        self.assertEqual(raised.exception.code, 'tool_choice_not_honored')
        self.assertEqual(conversation.pending, {})

    def test_capacity_rejection_leaves_no_temp_directories(self):
        transport = OpenCodeTransport('http://unused', request_json=lambda *args: None)
        for i in range(32):
            transport.conversations[f'slot_{i}'] = Conversation(f'slot_{i}', f'/tmp/{i}', 'model', {}, 'auto')
        before = set(pathlib.Path(tempfile.gettempdir()).glob('cindy-opencode-*'))
        with self.assertRaises(BridgeError) as raised:
            transport._new({'model': 'fixture', 'messages': [{'role': 'user', 'content': 'hi'}]}, [])
        self.assertEqual(raised.exception.status, 429)
        self.assertEqual(raised.exception.code, 'bridge_busy')
        after = set(pathlib.Path(tempfile.gettempdir()).glob('cindy-opencode-*'))
        self.assertEqual(after - before, set())

    def test_usage_fetch_failure_still_returns_completed_reply(self):
        call_count = 0
        def mock_request(path, conversation, body=None):
            nonlocal call_count
            call_count += 1
            if body is not None:
                return {'info': {'finish': 'stop'}, 'parts': [{'type': 'text', 'text': 'Completed successfully'}]}
            raise BridgeError(502, 'opencode_http_error', 'Transient network glitch during usage fetch')

        transport = OpenCodeTransport('http://unused', request_json=mock_request)
        conversation = Conversation('fixture', '/fixture', 'fixture', {}, 'auto', session_id='ses_123')
        transport._run(conversation, {'model': {'providerID': 'opencode', 'modelID': 'fixture'}, 'parts': []})
        kind, (result, usage) = conversation.events.get_nowait()
        self.assertEqual(kind, 'done')
        self.assertEqual(result['parts'][0]['text'], 'Completed successfully')
        self.assertIsNone(usage)


if __name__ == '__main__':
    unittest.main()
