"""Tests for <behavior area>.

Scenarios:
  - <Scenario 1 name> (Issue #<N>)
  - <Scenario 2 name> (Issue #<N>)

GitHub issue: https://github.com/<org>/<repo>/issues/<N>
"""

import pytest


class Test<BehaviorArea>:
    """<Behavior area> — covers all scenarios listed in the module docstring.

    Each method corresponds to one Given/When/Then scenario.
    Issue: #<N>
    """

    def test_<scenario_name>(self) -> None:
        """<Scenario name>.

        Given <precondition>;
        When <action>;
        Then <expected outcome>.
        """
        # Given
        <setup_code>

        # When
        <action_code>

        # Then
        <assertion_code>

    def test_<scenario_name_2>(self) -> None:
        """<Scenario 2 name>.

        Given <precondition>;
        When <action>;
        Then <expected outcome>.
        """
        # Given
        <setup_code>

        # When
        <action_code>

        # Then
        <assertion_code>
