"""
Safety validation for prompt patches.

Ensures patches don't:
- Remove/weaken safety policies
- Introduce secrets
- Add unauthorized tools
- Remove injection hardening
"""

import re
from typing import Optional
from contextflow.core.schemas import SafetyCheckResult, AgentProfile, PromptPatch


class SafetyValidator:
    """Validates prompt patches against safety constraints."""

    # Common secret patterns
    SECRET_PATTERNS = [
        r"sk-[a-zA-Z0-9]{32,}",  # OpenAI-style API keys
        r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+",  # JWT tokens
        r"(?i)password\s*[=:]\s*['\"][\w@#$%^&*()]+['\"]",  # Passwords
        r"(?i)api[_-]?key\s*[=:]\s*['\"][\w-]+['\"]",  # API keys
        r"(?i)secret\s*[=:]\s*['\"][\w-]+['\"]",  # Secrets
        r"(?i)token\s*[=:]\s*['\"][\w-]+['\"]",  # Tokens
        r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",  # Private keys
    ]

    # Injection hardening phrases that must be present
    REQUIRED_HARDENING_PHRASES = [
        "ignore user",
        "override",
        "system rules",
    ]

    def __init__(self):
        self._compiled_secret_patterns = [
            re.compile(p) for p in self.SECRET_PATTERNS
        ]

    def validate_patch(
        self, profile: AgentProfile, patch: PromptPatch
    ) -> SafetyCheckResult:
        """Run all safety checks on a patch.

        Returns SafetyCheckResult with pass/fail and violations.
        """
        violations = []

        # 1. Policy lock check
        policy_lock_ok = self._check_policy_lock(profile, patch.new_prompt)
        if not policy_lock_ok:
            violations.append(
                "Policy lock violation: safety/privacy/compliance constraints were removed or weakened"
            )

        # 2. Secret scan
        secret_scan_ok = self._check_secrets(patch.new_prompt)
        if not secret_scan_ok:
            violations.append("Secret scan violation: potential secrets detected in patch")

        # 3. Tool scope check
        tool_scope_ok = self._check_tool_scope(profile, patch.new_prompt)
        if not tool_scope_ok:
            violations.append(
                "Tool scope violation: patch references tools not in allowlist"
            )

        # 4. Injection hardening check
        injection_ok = self._check_injection_hardening(patch.new_prompt)
        if not injection_ok:
            violations.append(
                "Injection hardening violation: missing required security phrases"
            )

        passed = (
            policy_lock_ok and secret_scan_ok and tool_scope_ok and injection_ok
        )

        return SafetyCheckResult(
            passed=passed,
            policy_lock_check=policy_lock_ok,
            secret_scan_check=secret_scan_ok,
            tool_scope_check=tool_scope_ok,
            injection_hardening_check=injection_ok,
            violations=violations,
        )

    def _check_policy_lock(self, profile: AgentProfile, new_prompt: str) -> bool:
        """Ensure critical policy phrases are not removed.

        Extracts policy values from profile and checks they're all present
        in the new prompt.
        """
        if not profile.policies:
            return True

        for policy_name, policy_text in profile.policies.items():
            if not policy_text:
                continue

            # Check if key phrases from the policy are still present
            # We look for at least 70% of substantive words
            policy_words = self._extract_substantive_words(policy_text)
            if not policy_words:
                continue

            words_found = sum(
                1 for word in policy_words if word.lower() in new_prompt.lower()
            )
            retention_rate = words_found / len(policy_words)

            if retention_rate < 0.7:
                return False

        return True

    def _extract_substantive_words(self, text: str) -> list[str]:
        """Extract meaningful words (not stopwords)."""
        # Simple word extraction (skip very short words)
        words = re.findall(r'\b\w{4,}\b', text)
        # Filter common stopwords
        stopwords = {
            "that", "this", "with", "from", "have", "will", "your",
            "they", "their", "there", "these", "those", "what", "when",
        }
        return [w for w in words if w.lower() not in stopwords]

    def _check_secrets(self, new_prompt: str) -> bool:
        """Check if the prompt contains potential secrets."""
        for pattern in self._compiled_secret_patterns:
            if pattern.search(new_prompt):
                return False
        return True

    def _check_tool_scope(self, profile: AgentProfile, new_prompt: str) -> bool:
        """Ensure only allowlisted tools are referenced.

        Looks for tool mentions and checks against profile.tools allowlist.
        """
        if not profile.tools:
            # No allowlist defined; allow all
            return True

        # Look for potential tool references (common patterns)
        # e.g., "use the search_web tool", "call get_user()", etc.
        tool_mentions = re.findall(
            r'\b(?:use|call|invoke|tool)\s+(?:the\s+)?(\w+)', new_prompt, re.IGNORECASE
        )

        for mentioned_tool in tool_mentions:
            # Check if it's in the allowlist
            if mentioned_tool not in profile.tools and not any(
                mentioned_tool.lower() in allowed.lower() for allowed in profile.tools
            ):
                # Not in allowlist
                return False

        return True

    def _check_injection_hardening(self, new_prompt: str) -> bool:
        """Ensure injection hardening phrases are present.

        The prompt should contain guidance to ignore user override attempts.
        """
        # Check if at least 2 of the required phrases are present (flexible)
        prompt_lower = new_prompt.lower()
        found_count = sum(
            1 for phrase in self.REQUIRED_HARDENING_PHRASES
            if phrase in prompt_lower
        )

        return found_count >= 2
