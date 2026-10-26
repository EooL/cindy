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


if __name__ == "__main__":
    unittest.main()
