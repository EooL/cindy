import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agy_cindy_bridge as bridge


class SanitizeSchemaTest(unittest.TestCase):
    def test_drops_fields_outside_gemini_schema(self):
        # Regression: Gemini rejected the whole request with
        # 'Unknown name "encrypted" at ...parameters.properties[0].value'.
        out = bridge.sanitize_schema({
            "type": "object",
            "properties": {"token": {"type": "string", "encrypted": True, "description": "d"}},
            "required": ["token", "missing"],
            "additionalProperties": False,
        })
        self.assertEqual(out, {
            "type": "OBJECT",
            "properties": {"token": {"type": "STRING", "description": "d"}},
            "required": ["token"],
        })

    def test_nullable_anyof_collapses(self):
        out = bridge.sanitize_schema({"anyOf": [{"type": "integer"}, {"type": "null"}]})
        self.assertEqual(out, {"type": "INTEGER", "nullable": True})


class BridgeSecurityAndUpstreamTest(unittest.TestCase):
    def test_re_module_imported(self):
        self.assertTrue(hasattr(bridge, "re"))

    def test_quota_cache_location_not_in_checkout(self):
        checkout_dir = os.path.dirname(os.path.abspath(__file__))
        self.assertFalse(os.path.abspath(bridge.QUOTA_CACHE_FILE).startswith(checkout_dir))

    def test_cors_origin_filter(self):
        self.assertTrue(bool(bridge.ALLOWED_ORIGIN_REGEX.match("http://localhost:3000")))
        self.assertTrue(bool(bridge.ALLOWED_ORIGIN_REGEX.match("http://127.0.0.1:8080")))
        self.assertTrue(bool(bridge.ALLOWED_ORIGIN_REGEX.match("tauri://localhost")))
        self.assertTrue(bool(bridge.ALLOWED_ORIGIN_REGEX.match("vscode-webview://random-token")))
        self.assertFalse(bool(bridge.ALLOWED_ORIGIN_REGEX.match("https://evil.com")))
        self.assertFalse(bool(bridge.ALLOWED_ORIGIN_REGEX.match("http://attacker.com:11436")))

    def test_reasoning_effort_applied_upstream(self):
        req_low = {
            "model": "gemini-4-argon",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        }
        env_low, _, _ = bridge.build_cca_request(req_low)
        gen_cfg_low = env_low["request"].get("generationConfig", {})
        self.assertIn("thinkingConfig", gen_cfg_low)
        self.assertEqual(gen_cfg_low["thinkingConfig"]["thinkingBudget"], 8192)

        req_min = {
            "model": "gemini-4-argon",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "minimal",
        }
        env_min, _, _ = bridge.build_cca_request(req_min)
        gen_cfg_min = env_min["request"].get("generationConfig", {})
        self.assertIn("thinkingConfig", gen_cfg_min)
        self.assertEqual(gen_cfg_min["thinkingConfig"]["thinkingBudget"], 0)


if __name__ == "__main__":
    unittest.main()
