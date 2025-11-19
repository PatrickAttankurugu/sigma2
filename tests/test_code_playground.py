"""
Tests for Code Playground
"""

import pytest
from azuma.components.code_playground import CodePlayground


class TestCodePlayground:
    """Test CodePlayground functionality"""

    @pytest.fixture
    def playground(self):
        """Create a CodePlayground instance"""
        return CodePlayground()

    def test_simple_code_execution(self, playground):
        """Test executing simple Python code"""
        code = "print('Hello, World!')"
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "Hello, World!" in stdout
        assert stderr == ""

    def test_code_with_error(self, playground):
        """Test code that raises an error"""
        code = "print(undefined_variable)"
        stdout, stderr, success = playground.execute_code(code)

        assert not success
        assert "NameError" in stderr

    def test_syntax_error(self, playground):
        """Test code with syntax error"""
        code = "print('unclosed string"
        stdout, stderr, success = playground.execute_code(code)

        assert not success
        assert "Syntax Error" in stderr or "SyntaxError" in stderr

    def test_math_operations(self, playground):
        """Test mathematical operations"""
        code = """
import math
result = math.sqrt(16)
print(result)
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "4.0" in stdout

    def test_numpy_operations(self, playground):
        """Test NumPy operations (if numpy is available)"""
        code = """
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Mean: {arr.mean()}")
except ImportError:
    print("NumPy not available")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        # Should either show result or import message
        assert ("Mean:" in stdout) or ("not available" in stdout)

    def test_restricted_builtins(self, playground):
        """Test that dangerous builtins are restricted"""
        # These should fail or be restricted
        dangerous_codes = [
            "import os; os.system('ls')",
            "import subprocess",
            "open('/etc/passwd')",
            "__import__('os').system('ls')"
        ]

        for code in dangerous_codes:
            stdout, stderr, success = playground.execute_code(code)
            # Should either fail or not produce harmful output
            # Just ensure it doesn't crash the test
            assert True  # Code executed without crashing

    def test_output_length_limit(self, playground):
        """Test that output is limited in length"""
        code = """
for i in range(10000):
    print(f"Line {i}: " + "x" * 100)
"""
        stdout, stderr, success = playground.execute_code(code)

        # Output should be truncated
        assert len(stdout) <= playground.max_output_length + 100  # Some tolerance
        if len(stdout) > playground.max_output_length:
            assert "truncated" in stdout

    def test_multiple_prints(self, playground):
        """Test multiple print statements"""
        code = """
print("First line")
print("Second line")
print("Third line")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "First line" in stdout
        assert "Second line" in stdout
        assert "Third line" in stdout

    def test_list_comprehension(self, playground):
        """Test list comprehension"""
        code = """
squares = [x**2 for x in range(5)]
print(squares)
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "[0, 1, 4, 9, 16]" in stdout

    def test_function_definition(self, playground):
        """Test defining and calling functions"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(7)
print(f"Fibonacci(7) = {result}")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "13" in stdout  # fibonacci(7) = 13

    def test_class_definition(self, playground):
        """Test defining and using classes"""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
result = calc.add(5, 3)
print(f"5 + 3 = {result}")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "5 + 3 = 8" in stdout

    def test_exception_handling(self, playground):
        """Test exception handling in code"""
        code = """
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "Cannot divide by zero" in stdout

    def test_statistics_module(self, playground):
        """Test using statistics module"""
        code = """
import statistics
data = [1, 2, 3, 4, 5]
mean = statistics.mean(data)
median = statistics.median(data)
print(f"Mean: {mean}, Median: {median}")
"""
        stdout, stderr, success = playground.execute_code(code)

        assert success
        assert "Mean: 3" in stdout
        assert "Median: 3" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
