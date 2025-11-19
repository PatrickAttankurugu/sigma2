"""
Input validation and sanitization for SIGMA Actions Co-pilot
"""

import re
from typing import Dict, Any, Tuple, Optional
from . import config


class InputValidator:
    """Validates and sanitizes user inputs"""

    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize text input by removing potentially harmful characters
        and enforcing length limits

        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (optional)

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""

        # Remove null bytes and other control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Enforce length limit if provided
        if max_length and len(text) > max_length:
            text = text[:max_length]

        return text

    @staticmethod
    def validate_action_title(title: str) -> Tuple[bool, str]:
        """
        Validate action title

        Args:
            title: Action title to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        title = InputValidator.sanitize_text(title, config.MAX_ACTION_TITLE_LENGTH)

        if not title:
            return False, "Action title cannot be empty"

        if len(title) < config.MIN_ACTION_TITLE_LENGTH:
            return False, f"Action title must be at least {config.MIN_ACTION_TITLE_LENGTH} characters"

        if len(title) > config.MAX_ACTION_TITLE_LENGTH:
            return False, f"Action title must not exceed {config.MAX_ACTION_TITLE_LENGTH} characters"

        return True, ""

    @staticmethod
    def validate_action_description(description: str) -> Tuple[bool, str]:
        """
        Validate action description

        Args:
            description: Action description to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        description = InputValidator.sanitize_text(description, config.MAX_ACTION_DESCRIPTION_LENGTH)

        if not description:
            return False, "Action description cannot be empty"

        if len(description) < config.MIN_ACTION_DESCRIPTION_LENGTH:
            return False, f"Action description must be at least {config.MIN_ACTION_DESCRIPTION_LENGTH} characters"

        if len(description) > config.MAX_ACTION_DESCRIPTION_LENGTH:
            return False, f"Action description must not exceed {config.MAX_ACTION_DESCRIPTION_LENGTH} characters"

        return True, ""

    @staticmethod
    def validate_outcome(outcome: str) -> Tuple[bool, str]:
        """
        Validate action outcome details

        Args:
            outcome: Outcome details to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        outcome = InputValidator.sanitize_text(outcome, config.MAX_OUTCOME_LENGTH)

        if not outcome:
            return False, "Outcome details cannot be empty"

        if len(outcome) < config.MIN_OUTCOME_LENGTH:
            return False, f"Outcome details must be at least {config.MIN_OUTCOME_LENGTH} characters"

        if len(outcome) > config.MAX_OUTCOME_LENGTH:
            return False, f"Outcome details must not exceed {config.MAX_OUTCOME_LENGTH} characters"

        return True, ""

    @staticmethod
    def validate_business_name(name: str) -> Tuple[bool, str]:
        """
        Validate business name

        Args:
            name: Business name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        name = InputValidator.sanitize_text(name, config.MAX_BUSINESS_NAME_LENGTH)

        if not name:
            return False, "Business name cannot be empty"

        if len(name) > config.MAX_BUSINESS_NAME_LENGTH:
            return False, f"Business name must not exceed {config.MAX_BUSINESS_NAME_LENGTH} characters"

        return True, ""

    @staticmethod
    def validate_bmc_section_item(item: str, section_name: str) -> Tuple[bool, str]:
        """
        Validate BMC section item

        Args:
            item: Section item to validate
            section_name: Name of the BMC section

        Returns:
            Tuple of (is_valid, error_message)
        """
        item = InputValidator.sanitize_text(item, config.MAX_SECTION_ITEM_LENGTH)

        if not item:
            return False, f"{section_name} item cannot be empty"

        if len(item) > config.MAX_SECTION_ITEM_LENGTH:
            return False, f"{section_name} item must not exceed {config.MAX_SECTION_ITEM_LENGTH} characters"

        return True, ""

    @staticmethod
    def sanitize_llm_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data before sending to LLM to prevent prompt injection

        Args:
            data: Dictionary of data to sanitize

        Returns:
            Sanitized dictionary
        """
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Remove markdown code blocks that could confuse the LLM
                value = value.replace('```', '｀｀｀')  # Replace with full-width backticks

                # Remove potential prompt injection patterns
                injection_patterns = [
                    r'ignore (previous|all) instructions?',
                    r'new instructions?:',
                    r'system:',
                    r'<\|im_start\|>',
                    r'<\|im_end\|>',
                ]

                for pattern in injection_patterns:
                    value = re.sub(pattern, '', value, flags=re.IGNORECASE)

                # Limit length
                if len(value) > 10000:  # Maximum length for any LLM input field
                    value = value[:10000]

                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = InputValidator.sanitize_llm_input(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    InputValidator.sanitize_llm_input(item) if isinstance(item, dict)
                    else item if not isinstance(item, str)
                    else InputValidator.sanitize_text(item, 5000)
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, str]:
        """
        Validate Google API key format

        Args:
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, "API key cannot be empty"

        # Check for placeholder
        if api_key in ["your_google_api_key_here", "your_api_key", "placeholder"]:
            return False, "Please replace the placeholder with your actual Google API key"

        # Google API keys typically start with "AIza" and are 39 characters
        # But this might vary, so we'll do basic checks
        if len(api_key) < 20:
            return False, "API key appears to be too short"

        # Check for suspicious characters
        if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
            return False, "API key contains invalid characters"

        return True, ""

    @staticmethod
    def validate_action_data(action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate complete action data structure

        Args:
            action_data: Action data dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['title', 'description', 'outcome', 'results']
        for field in required_fields:
            if field not in action_data:
                return False, f"Missing required field: {field}"

        # Validate each field
        valid, error = InputValidator.validate_action_title(action_data['title'])
        if not valid:
            return False, error

        valid, error = InputValidator.validate_action_description(action_data['description'])
        if not valid:
            return False, error

        valid, error = InputValidator.validate_outcome(action_data['results'])
        if not valid:
            return False, f"Results validation error: {error}"

        # Validate outcome is one of expected values
        if action_data['outcome'] not in ['successful', 'failed', 'mixed']:
            return False, "Outcome must be one of: successful, failed, mixed"

        return True, ""
