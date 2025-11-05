"""
Unit tests for AI Engine JSON parsing and error handling
"""

import unittest
import sys
import os
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.ai_engine import QualityEnhancedAI


class MockQualityEnhancedAI:
    """Mock class to test _parse_response without requiring LLM"""

    def __init__(self):
        self.logger = MockLogger()

    def _parse_response(self, content: str):
        """Use the actual method from QualityEnhancedAI"""
        # Import the actual implementation
        from modules.ai_engine import QualityEnhancedAI as RealAI
        # Create a temporary instance just to borrow the method
        temp = type('temp', (), {
            'logger': self.logger,
            '_parse_response': RealAI._parse_response.__get__(self, MockQualityEnhancedAI)
        })()
        return temp._parse_response(content)


class MockLogger:
    """Mock logger for testing"""

    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)


class TestAIEngineJSONParsing(unittest.TestCase):
    """Test cases for AI Engine JSON parsing"""

    def setUp(self):
        """Set up test fixtures"""
        self.ai = MockQualityEnhancedAI()

    def test_parse_response_with_json_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        content = '''```json
{
  "analysis": "Test analysis",
  "changes": [],
  "next_steps": []
}
```'''
        result = self.ai._parse_response(content)
        self.assertEqual(result["analysis"], "Test analysis")
        self.assertEqual(result["changes"], [])

    def test_parse_response_with_regular_markdown(self):
        """Test parsing JSON wrapped in regular code blocks"""
        content = '''```
{
  "analysis": "Test analysis",
  "changes": [],
  "next_steps": []
}
```'''
        result = self.ai._parse_response(content)
        self.assertEqual(result["analysis"], "Test analysis")

    def test_parse_response_with_braces_only(self):
        """Test parsing JSON with just braces"""
        content = '''Some text before {"analysis": "Test", "changes": [], "next_steps": []} some text after'''
        result = self.ai._parse_response(content)
        self.assertEqual(result["analysis"], "Test")

    def test_parse_response_malformed_json(self):
        """Test handling of malformed JSON"""
        content = '''```json
{
  "analysis": "Missing closing brace",
  "changes": []
```'''
        result = self.ai._parse_response(content)
        # Should return fallback structure
        self.assertIn("unable", result["analysis"].lower())
        self.assertEqual(result["changes"], [])
        # Should have logged an error
        self.assertGreater(len(self.ai.logger.errors), 0)

    def test_parse_response_invalid_markdown_structure(self):
        """Test handling of invalid markdown structure"""
        content = "```json\nNo closing backticks"
        result = self.ai._parse_response(content)
        # Should handle gracefully and return fallback
        self.assertIsInstance(result, dict)
        self.assertIn("analysis", result)

    def test_parse_response_adds_missing_fields(self):
        """Test that missing fields are added with defaults"""
        content = '{"analysis": "Test only"}'
        result = self.ai._parse_response(content)
        self.assertIn("changes", result)
        self.assertIn("next_steps", result)
        self.assertEqual(result["analysis"], "Test only")

    def test_parse_response_empty_content(self):
        """Test handling of empty content"""
        content = ""
        result = self.ai._parse_response(content)
        # Should return fallback structure
        self.assertIsInstance(result, dict)
        self.assertIn("analysis", result)


if __name__ == '__main__':
    unittest.main()
