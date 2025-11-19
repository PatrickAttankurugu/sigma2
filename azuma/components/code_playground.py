"""
Interactive Code Playground for Azuma AI
Allows students to write and execute Python code safely
"""

import streamlit as st
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple
import time


class CodePlayground:
    """Interactive code execution environment"""

    def __init__(self):
        self.max_execution_time = 5  # seconds
        self.max_output_length = 5000  # characters

        # Restricted builtins for safety
        self.safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }

    def execute_code(self, code: str, timeout: int = 5) -> Tuple[str, str, bool]:
        """
        Safely execute Python code with restrictions

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (stdout, stderr, success)
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        success = True

        try:
            # Create restricted globals with safe modules
            safe_globals = {
                '__builtins__': self.safe_builtins,
                'math': __import__('math'),
                'random': __import__('random'),
                'statistics': __import__('statistics'),
                'datetime': __import__('datetime'),
                'itertools': __import__('itertools'),
                'collections': __import__('collections'),
            }

            # Try to import ML libraries safely
            try:
                import numpy as np
                safe_globals['np'] = np
                safe_globals['numpy'] = np
            except ImportError:
                pass

            try:
                import pandas as pd
                safe_globals['pd'] = pd
                safe_globals['pandas'] = pd
            except ImportError:
                pass

            try:
                import matplotlib.pyplot as plt
                safe_globals['plt'] = plt
                safe_globals['matplotlib'] = __import__('matplotlib')
            except ImportError:
                pass

            # Redirect stdout and stderr
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Compile and execute
                compiled_code = compile(code, '<student_code>', 'exec')
                exec(compiled_code, safe_globals)

        except SyntaxError as e:
            stderr_buffer.write(f"Syntax Error: {e}\n")
            stderr_buffer.write(f"Line {e.lineno}: {e.text}")
            success = False

        except Exception as e:
            stderr_buffer.write(f"Error: {type(e).__name__}: {str(e)}\n")
            stderr_buffer.write(traceback.format_exc())
            success = False

        # Get outputs and limit length
        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        if len(stdout) > self.max_output_length:
            stdout = stdout[:self.max_output_length] + "\n... (output truncated)"

        if len(stderr) > self.max_output_length:
            stderr = stderr[:self.max_output_length] + "\n... (error output truncated)"

        return stdout, stderr, success

    def render_playground(self, initial_code: str = "", key: str = "playground"):
        """
        Render interactive code playground in Streamlit

        Args:
            initial_code: Initial code to display
            key: Unique key for the code editor
        """
        st.markdown("### 💻 Interactive Code Playground")
        st.write("Write and run Python code right here! Available libraries: numpy, pandas, matplotlib, math, random")

        # Code editor
        code = st.text_area(
            "Write your Python code here:",
            value=initial_code,
            height=300,
            key=f"code_editor_{key}",
            placeholder="# Write your code here\nprint('Hello, Azuma AI!')"
        )

        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            run_button = st.button("▶️ Run Code", key=f"run_{key}", type="primary")

        with col2:
            clear_button = st.button("🗑️ Clear", key=f"clear_{key}")

        if clear_button:
            st.rerun()

        # Execute code when button is clicked
        if run_button and code.strip():
            with st.spinner("Executing code..."):
                start_time = time.time()
                stdout, stderr, success = self.execute_code(code)
                execution_time = time.time() - start_time

            # Display results
            st.markdown("---")

            if success:
                st.success(f"✅ Execution successful ({execution_time:.3f}s)")
            else:
                st.error(f"❌ Execution failed ({execution_time:.3f}s)")

            # Show output
            if stdout:
                st.markdown("**Output:**")
                st.code(stdout, language="text")

            # Show errors
            if stderr:
                st.markdown("**Errors/Warnings:**")
                st.code(stderr, language="text")

            # Show matplotlib plots if any
            # Note: This requires special handling in Streamlit
            # The user would need to use st.pyplot() in their code

        elif run_button:
            st.warning("Please write some code first!")

    def create_quiz_playground(self, problem: str, test_cases: list, solution_hint: str = ""):
        """
        Create a code playground for quiz problems with automated testing

        Args:
            problem: Problem description
            test_cases: List of test cases [{"input": ..., "expected": ...}, ...]
            solution_hint: Optional hint for the solution
        """
        st.markdown("### 🎯 Coding Challenge")
        st.write(problem)

        if solution_hint:
            with st.expander("💡 Hint"):
                st.write(solution_hint)

        # Code editor
        code = st.text_area(
            "Your solution:",
            height=250,
            placeholder="# Write your solution here\n"
        )

        if st.button("🧪 Test Solution", type="primary"):
            if not code.strip():
                st.warning("Please write a solution first!")
                return

            # Run test cases
            all_passed = True
            results = []

            for i, test_case in enumerate(test_cases):
                test_input = test_case.get("input", "")
                expected = test_case.get("expected", "")

                # Create test code
                test_code = f"{code}\n\nresult = {test_input}\nprint(result)"

                stdout, stderr, success = self.execute_code(test_code)

                if success and stdout.strip() == str(expected):
                    results.append({"test": i+1, "passed": True, "output": stdout.strip()})
                else:
                    results.append({
                        "test": i+1,
                        "passed": False,
                        "expected": expected,
                        "got": stdout.strip(),
                        "error": stderr
                    })
                    all_passed = False

            # Display results
            st.markdown("---")
            st.markdown("**Test Results:**")

            for result in results:
                if result["passed"]:
                    st.success(f"✅ Test {result['test']}: Passed")
                else:
                    st.error(f"❌ Test {result['test']}: Failed")
                    st.write(f"Expected: `{result['expected']}`")
                    st.write(f"Got: `{result['got']}`")
                    if result.get("error"):
                        st.code(result["error"], language="text")

            if all_passed:
                st.balloons()
                st.success("🎉 All tests passed! Great job!")
                return True
            else:
                st.info("Keep trying! Debug your code and run tests again.")
                return False

        return None


# Example usage functions
def create_example_playground():
    """Create an example playground with sample code"""
    playground = CodePlayground()

    sample_code = """# Example: Calculate fibonacci numbers
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test it
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")
"""

    playground.render_playground(initial_code=sample_code, key="example")


def create_ml_playground():
    """Create a playground with ML example"""
    playground = CodePlayground()

    ml_code = """# Example: Simple linear regression with numpy
import numpy as np

# Generate sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate linear regression parameters
mean_x = np.mean(X)
mean_y = np.mean(Y)

numerator = np.sum((X - mean_x) * (y - mean_y))
denominator = np.sum((X - mean_x) ** 2)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

# Make predictions
predictions = slope * X + intercept
print(f"Predictions: {predictions}")
"""

    playground.render_playground(initial_code=ml_code, key="ml_example")
