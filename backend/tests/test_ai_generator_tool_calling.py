import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Focused tests for AI generator tool calling functionality to diagnose failures"""

    def test_generate_response_with_tool_failure_returns_generic_error(self):
        """Test that tool failures result in generic error responses that look like 'Query failed'"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock AI requesting tool use
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].id = "tool_123"
            tool_response.content[0].input = {"query": "test"}

            # Mock final response when tool fails
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = (
                "I'm unable to find information about that topic."
            )

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Create mock tool manager that fails
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_manager.execute_tool.return_value = (
                "No relevant content found."  # Tool returns empty result
            )

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "What is covered in lesson 1?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # This simulates the "Query failed" scenario - AI can't find content
            assert "unable to find" in response.lower() or "no" in response.lower()

    def test_generate_response_with_max_results_zero_causes_empty_tool_results(self):
        """Test how AI responds when tools return empty results due to MAX_RESULTS=0"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock AI requesting search tool
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].id = "tool_123"
            tool_response.content[0].input = {"query": "lesson content"}

            # Mock AI's response to empty tool results
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = (
                "I couldn't find any relevant information in the course materials."
            )

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Mock tool manager returning empty results (simulating MAX_RESULTS=0 bug)
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_manager.execute_tool.return_value = "No relevant content found."

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "What topics are covered in the course?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Verify the AI received empty tool results and responded appropriately
            assert "couldn't find" in response.lower() or "no" in response.lower()
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="lesson content"
            )

    def test_generate_response_tool_parameter_validation_failure(self):
        """Test AI behavior when tool parameters are invalid"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock AI requesting tool with invalid parameters
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].id = "tool_123"
            tool_response.content[0].input = {}  # Missing required 'query' parameter

            # Mock final response after tool parameter error
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = "I encountered an error while searching."

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Mock tool manager that raises exception for invalid parameters
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {
                    "name": "search_course_content",
                    "input_schema": {"required": ["query"]},
                }
            ]
            mock_tool_manager.execute_tool.side_effect = Exception(
                "Missing required parameter: query"
            )

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "Search for information",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Should handle tool parameter errors gracefully
            assert "error" in response.lower()

    def test_generate_response_with_course_name_resolution_failure(self):
        """Test AI behavior when course name cannot be resolved"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock AI requesting search with specific course
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].id = "tool_123"
            tool_response.content[0].input = {
                "query": "lesson 1 content",
                "course_name": "Nonexistent Course",
            }

            # Mock AI's response to course not found error
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = "I couldn't find a course with that name."

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Mock tool returning course not found error
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_manager.execute_tool.return_value = (
                "No course found matching 'Nonexistent Course'"
            )

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "What's in lesson 1 of Nonexistent Course?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Should indicate course not found
            assert "couldn't find" in response.lower() or "course" in response.lower()

    def test_generate_response_anthropic_api_failure(self):
        """Test behavior when Anthropic API calls fail"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock API failure
            mock_client.messages.create.side_effect = Exception(
                "API Error: Invalid API key"
            )

            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]

            generator = AIGenerator("invalid-key", "claude-sonnet-4-20250514")

            # Should raise exception on API failure
            with pytest.raises(Exception) as exc_info:
                generator.generate_response(
                    "What is AI?",
                    tools=mock_tool_manager.get_tool_definitions(),
                    tool_manager=mock_tool_manager,
                )

            assert "API Error" in str(exc_info.value)

    def test_generate_response_with_tool_timeout_simulation(self):
        """Test AI behavior when tool execution times out"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock AI requesting tool
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].id = "tool_123"
            tool_response.content[0].input = {"query": "complex query"}

            # Mock final response after timeout
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = (
                "The search took too long to complete. Please try a simpler query."
            )

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Mock tool manager that simulates timeout
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_manager.execute_tool.return_value = (
                "Search error: Request timeout after 30 seconds"
            )

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "Analyze all course content in detail",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Should handle timeout gracefully
            assert "timeout" in response.lower() or "too long" in response.lower()

    def test_generate_response_tool_choice_logic_for_different_query_types(self):
        """Test that AI chooses appropriate tools based on query type"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Test different query types and expected tool usage
            query_scenarios = [
                {
                    "query": "What is covered in lesson 1?",
                    "expected_tool": "search_course_content",
                    "expected_params": {"query": "lesson 1 content"},
                },
                {
                    "query": "Show me the course outline",
                    "expected_tool": "get_course_outline",
                    "expected_params": {"course_name": "course"},
                },
                {
                    "query": "What is machine learning?",  # General question
                    "expected_tool": None,  # Should not use tools for general knowledge
                    "expected_params": None,
                },
            ]

            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"},
                {"name": "get_course_outline"},
            ]

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            for scenario in query_scenarios:
                # Reset mocks for each scenario
                mock_client.messages.create.reset_mock()
                mock_tool_manager.execute_tool.reset_mock()

                if scenario["expected_tool"]:
                    # Mock tool use response
                    tool_response = Mock()
                    tool_response.stop_reason = "tool_use"
                    tool_response.content = [Mock()]
                    tool_response.content[0].type = "tool_use"
                    tool_response.content[0].name = scenario["expected_tool"]
                    tool_response.content[0].id = "tool_123"
                    tool_response.content[0].input = scenario["expected_params"]

                    final_response = Mock()
                    final_response.content = [Mock()]
                    final_response.content[0].text = "Tool-based response"

                    mock_client.messages.create.side_effect = [
                        tool_response,
                        final_response,
                    ]
                    mock_tool_manager.execute_tool.return_value = "Tool result"
                else:
                    # Mock direct response without tool use
                    direct_response = Mock()
                    direct_response.stop_reason = "stop"
                    direct_response.content = [Mock()]
                    direct_response.content[0].text = "Direct knowledge response"

                    mock_client.messages.create.return_value = direct_response

                response = generator.generate_response(
                    scenario["query"],
                    tools=mock_tool_manager.get_tool_definitions(),
                    tool_manager=mock_tool_manager,
                )

                if scenario["expected_tool"]:
                    # Verify tool was called correctly
                    mock_tool_manager.execute_tool.assert_called_once()
                    assert "Tool-based response" in response
                else:
                    # Verify no tool was called
                    mock_tool_manager.execute_tool.assert_not_called()
                    assert "Direct knowledge response" in response

    def test_generate_response_sequential_tool_calling_failure_recovery(self):
        """Test recovery when sequential tool calls fail at different stages"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock Round 1: AI calls outline tool successfully
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "get_course_outline"
            round1_response.content[0].id = "tool_1"
            round1_response.content[0].input = {"course_name": "Test Course"}

            # Mock Round 2: AI calls search tool but it fails
            round2_response = Mock()
            round2_response.stop_reason = "tool_use"
            round2_response.content = [Mock()]
            round2_response.content[0].type = "tool_use"
            round2_response.content[0].name = "search_course_content"
            round2_response.content[0].id = "tool_2"
            round2_response.content[0].input = {"query": "specific topic"}

            # Mock final response after tool failure
            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = (
                "I found the course outline but couldn't retrieve specific lesson content."
            )

            mock_client.messages.create.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            # Mock tool manager with mixed success/failure
            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "get_course_outline"},
                {"name": "search_course_content"},
            ]
            mock_tool_manager.execute_tool.side_effect = [
                "Course Outline: Lesson 1, Lesson 2, Lesson 3",  # First tool succeeds
                "No relevant content found.",  # Second tool fails
            ]

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            response = generator.generate_response(
                "Give me details about Test Course lessons",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Should indicate partial success
            assert "outline" in response.lower() and "couldn't" in response.lower()
            assert mock_tool_manager.execute_tool.call_count == 2

    def test_generate_response_system_prompt_effectiveness(self):
        """Test that the system prompt properly guides tool usage"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock a response without tool usage for general question
            general_response = Mock()
            general_response.stop_reason = "stop"
            general_response.content = [Mock()]
            general_response.content[0].text = (
                "General knowledge response without tools"
            )

            mock_client.messages.create.return_value = general_response

            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            # Test general knowledge question that shouldn't use tools
            response = generator.generate_response(
                "What is the capital of France?",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Verify system prompt was used and tools weren't called for general question
            call_args = mock_client.messages.create.call_args[1]
            system_content = call_args["system"]

            # Check key parts of system prompt are present
            assert "course materials and educational content" in system_content
            assert "Content Search Tool" in system_content
            assert "General knowledge questions" in system_content

            # Verify no tools were called for general knowledge
            mock_tool_manager.execute_tool.assert_not_called()

    def test_generate_response_with_conversation_history_affecting_tool_usage(self):
        """Test how conversation history affects tool calling decisions"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock response that considers conversation history
            context_response = Mock()
            context_response.stop_reason = "tool_use"
            context_response.content = [Mock()]
            context_response.content[0].type = "tool_use"
            context_response.content[0].name = "search_course_content"
            context_response.content[0].id = "tool_123"
            context_response.content[0].input = {
                "query": "lesson 2 content",
                "course_name": "Course A",  # Should extract from context
            }

            final_response = Mock()
            final_response.content = [Mock()]
            final_response.content[0].text = "Based on Course A lesson 2..."

            mock_client.messages.create.side_effect = [context_response, final_response]

            mock_tool_manager = Mock()
            mock_tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"}
            ]
            mock_tool_manager.execute_tool.return_value = (
                "Lesson 2 covers advanced topics"
            )

            generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

            # Provide conversation history that mentions a specific course
            conversation_history = "User: Tell me about Course A\nAssistant: Course A covers programming basics"

            response = generator.generate_response(
                "What's in lesson 2?",  # Ambiguous without context
                conversation_history=conversation_history,
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
            )

            # Verify conversation history was included in system prompt
            call_args = mock_client.messages.create.call_args_list[0][1]
            assert conversation_history in call_args["system"]

            # Should have called tool with course context from history
            assert "Course A" in response
