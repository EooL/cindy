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

    def test_schema_defs_and_ref_dereferencing(self):
        schema = {
            "$defs": {
                "UserRole": {
                    "type": "string",
                    "enum": ["admin", "editor", "viewer"],
                },
                "Profile": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"$ref": "#/$defs/UserRole"},
                    },
                    "required": ["name", "role"],
                },
            },
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/Profile"},
            },
            "required": ["user"],
        }
        sanitized = bridge.sanitize_schema(schema)
        self.assertEqual(sanitized, {
            "type": "OBJECT",
            "properties": {
                "user": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "role": {"type": "STRING", "enum": ["admin", "editor", "viewer"]},
                    },
                    "required": ["name", "role"],
                },
            },
            "required": ["user"],
        })

    def test_schema_circular_ref_safety(self):
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/Node"},
                    },
                },
            },
            "type": "object",
            "properties": {
                "head": {"$ref": "#/$defs/Node"},
            },
        }
        sanitized = bridge.sanitize_schema(schema)
        self.assertEqual(sanitized["type"], "OBJECT")
        self.assertEqual(sanitized["properties"]["head"]["type"], "OBJECT")
        self.assertEqual(sanitized["properties"]["head"]["properties"]["next"], {})

    def test_strict_cca_allowlist_strips_disallowed_keys(self):
        schema = {
            "title": "SearchQuery",
            "description": "A search query with validation",
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 100,
                    "description": "search query text",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
            },
            "required": ["q"],
        }
        out = bridge.sanitize_schema(schema)
        self.assertEqual(out, {
            "type": "OBJECT",
            "description": "A search query with validation",
            "properties": {
                "q": {"type": "STRING", "description": "search query text"},
                "limit": {"type": "INTEGER"},
                "tags": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": ["q"],
        })
        self.assertNotIn("title", out)
        self.assertNotIn("minLength", out["properties"]["q"])
        self.assertNotIn("minimum", out["properties"]["limit"])
        self.assertNotIn("minItems", out["properties"]["tags"])


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

    def test_supports_fast_mode_is_false_in_catalog(self):
        for model in bridge.MODEL_CATALOG:
            self.assertFalse(model.get("supports_fast", False), f"{model['id']} should not advertise supports_fast")

    def test_preserve_actual_backend_model(self):
        req_opus = {
            "model": "claude-opus-5-5",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "high",
        }
        env_opus, _, _ = bridge.build_cca_request(req_opus)
        self.assertEqual(env_opus["model"], "claude-opus-5-5-high")

        req_argon = {
            "model": "gemini-4-argon",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "high",
        }
        env_argon, _, _ = bridge.build_cca_request(req_argon)
        self.assertEqual(env_argon["model"], "gemini-3.8-flash-tiered")


if __name__ == "__main__":
    unittest.main()
