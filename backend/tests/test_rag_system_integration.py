import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemIntegration:
    """End-to-end integration tests for RAGSystem to identify failure points"""

    def test_query_end_to_end_with_max_results_zero_failure(self, broken_config):
        """Test complete query flow when MAX_RESULTS=0 causes failure"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session,
        ):
            # Setup broken vector store (MAX_RESULTS=0)
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Simulate MAX_RESULTS=0 causing empty search results
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            # Mock AI response when tools return nothing
            mock_ai_gen.return_value.generate_response.return_value = "I couldn't find relevant information."
            mock_session.return_value.get_conversation_history.return_value = None

            rag_system = RAGSystem(broken_config)

            # Execute query that should work but fails due to config
            response, sources, source_links = rag_system.query("What is covered in lesson 1?")

            # Assert this produces a "Query failed" type response
            assert "couldn't find" in response.lower() or "no" in response.lower()
            assert sources == []
            assert source_links == []

            # Verify the vector store was initialized with MAX_RESULTS=0
            mock_vector_store.assert_called_once_with(
                broken_config.CHROMA_PATH,
                broken_config.EMBEDDING_MODEL,
                0  # This is the problem
            )

    def test_query_end_to_end_with_proper_config_success(self, test_config):
        """Test complete query flow with proper configuration"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session,
        ):
            # Setup proper vector store
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Mock successful search results
            mock_vector_store_instance.search.return_value = SearchResults(
                documents=["Lesson 1 covers introduction to AI and basic concepts."],
                metadata=[{"course_title": "AI Fundamentals", "lesson_number": 1, "chunk_index": 0}],
                distances=[0.1]
            )
            
            # Mock AI successful response
            mock_ai_gen.return_value.generate_response.return_value = "Based on the course materials, lesson 1 covers introduction to AI."
            mock_session.return_value.get_conversation_history.return_value = None

            rag_system = RAGSystem(test_config)

            # Mock tool manager responses
            rag_system.tool_manager.get_last_sources = Mock(return_value=["AI Fundamentals - Lesson 1"])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=["https://example.com/lesson1"])

            # Execute same query that failed with broken config
            response, sources, source_links = rag_system.query("What is covered in lesson 1?")

            # Assert this produces a successful response
            assert "lesson 1" in response.lower()
            assert "AI" in response
            assert sources == ["AI Fundamentals - Lesson 1"]
            assert source_links == ["https://example.com/lesson1"]

            # Verify proper config was used
            mock_vector_store.assert_called_once_with(
                test_config.CHROMA_PATH,
                test_config.EMBEDDING_MODEL,
                5  # Proper MAX_RESULTS value
            )

    def test_query_with_api_key_missing_or_invalid(self, test_config):
        """Test query behavior when Anthropic API key is missing or invalid"""
        # Create config with invalid API key
        invalid_config = test_config
        invalid_config.ANTHROPIC_API_KEY = ""

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock AI generator to raise authentication error
            mock_ai_gen.return_value.generate_response.side_effect = Exception("API Error: Invalid API key")

            rag_system = RAGSystem(invalid_config)

            # Query should fail with API error
            with pytest.raises(Exception) as exc_info:
                rag_system.query("What is machine learning?")
            
            assert "API" in str(exc_info.value) and "key" in str(exc_info.value)

    def test_query_with_empty_vector_store(self, test_config):
        """Test query behavior when vector store has no data"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup empty vector store
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            # Mock AI response to empty search results
            mock_ai_gen.return_value.generate_response.return_value = "I don't have information about that topic in the course materials."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            response, sources, source_links = rag_system.query("What is covered in the course?")

            # Should indicate no information available
            assert "don't have information" in response.lower() or "no" in response.lower()
            assert sources == []
            assert source_links == []

    def test_query_with_corrupted_vector_store_data(self, test_config):
        """Test query behavior when vector store returns corrupted data"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup vector store returning corrupted data
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Return corrupted search results
            mock_vector_store_instance.search.return_value = SearchResults(
                documents=[""],  # Empty document
                metadata=[{}],   # Empty metadata
                distances=[None] # Invalid distance
            )
            
            mock_ai_gen.return_value.generate_response.return_value = "I found some information but it appears to be corrupted."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=["Unknown Source"])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[None])

            response, sources, source_links = rag_system.query("What is AI?")

            # Should handle corrupted data gracefully
            assert isinstance(response, str)
            assert len(response) > 0

    def test_query_with_network_timeout_simulation(self, test_config):
        """Test query behavior when vector store operations timeout"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup vector store that times out
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.search.return_value = SearchResults.empty("Search error: Request timeout")
            
            mock_ai_gen.return_value.generate_response.return_value = "The search operation timed out. Please try again later."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            response, sources, source_links = rag_system.query("Analyze all course content")

            # Should handle timeout gracefully
            assert "timeout" in response.lower() or "try again" in response.lower()

    def test_query_flow_with_session_management_errors(self, test_config):
        """Test query flow when session management fails"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session,
        ):
            # Setup session manager that fails
            mock_session.return_value.get_conversation_history.side_effect = Exception("Session database error")
            mock_ai_gen.return_value.generate_response.return_value = "Response without session context"

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Should still work without session context
            response, sources, source_links = rag_system.query("What is AI?", session_id="broken-session")

            assert "Response without session context" in response

    def test_query_with_tool_registration_failure(self, test_config):
        """Test query behavior when tools fail to register properly"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            rag_system = RAGSystem(test_config)
            
            # Simulate tool registration failure
            rag_system.tool_manager.get_tool_definitions = Mock(return_value=[])
            
            # Mock AI response without tools
            mock_ai_gen.return_value.generate_response.return_value = "I don't have access to course search tools."

            response, sources, source_links = rag_system.query("What is in lesson 1?")

            # Should indicate tools are unavailable
            assert "don't have access" in response.lower() or "tools" in response.lower()

    def test_query_with_document_processing_never_completed(self, test_config):
        """Test query behavior when no documents were ever processed"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup vector store with no course data
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.get_existing_course_titles.return_value = []
            mock_vector_store_instance.get_course_count.return_value = 0
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            mock_ai_gen.return_value.generate_response.return_value = "No course materials are currently loaded in the system."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            response, sources, source_links = rag_system.query("What courses are available?")

            # Should indicate no courses loaded
            assert "no course" in response.lower() or "not loaded" in response.lower()
            
            # Verify analytics show empty state
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 0
            assert analytics["course_titles"] == []

    def test_query_prompt_format_affecting_ai_behavior(self, test_config):
        """Test how the RAG system's prompt formatting affects AI responses"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock AI generator to capture the prompt format
            mock_ai_instance = Mock()
            mock_ai_gen.return_value = mock_ai_instance
            mock_ai_instance.generate_response.return_value = "Test response"

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Execute query
            response, _, _ = rag_system.query("What is machine learning?")

            # Verify the prompt format used by RAG system
            call_args = mock_ai_instance.generate_response.call_args[1]
            prompt_query = call_args["query"]
            
            # Should contain the RAG system's prompt format
            assert "Answer this question about course materials:" in prompt_query
            assert "What is machine learning?" in prompt_query

    def test_query_source_tracking_integration(self, test_config):
        """Test that source tracking works correctly through the complete flow"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            mock_ai_gen.return_value.generate_response.return_value = "Response with sources"

            rag_system = RAGSystem(test_config)
            
            # Mock multiple sources from different lessons
            expected_sources = ["Course A - Lesson 1", "Course A - Lesson 2", "Course B - Lesson 1"]
            expected_links = ["https://a1.com", "https://a2.com", "https://b1.com"]
            
            rag_system.tool_manager.get_last_sources = Mock(return_value=expected_sources)
            rag_system.tool_manager.get_last_source_links = Mock(return_value=expected_links)
            rag_system.tool_manager.reset_sources = Mock()

            response, sources, source_links = rag_system.query("Compare lesson content across courses")

            # Verify sources are properly tracked and returned
            assert sources == expected_sources
            assert source_links == expected_links
            
            # Verify sources are reset after retrieval
            rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_error_propagation_from_all_layers(self, test_config):
        """Test how errors from different layers propagate through the system"""
        error_scenarios = [
            {
                "layer": "vector_store",
                "error": "Database connection failed",
                "expected_behavior": "should_handle_gracefully"
            },
            {
                "layer": "ai_generator", 
                "error": "API rate limit exceeded",
                "expected_behavior": "should_raise_exception"
            },
            {
                "layer": "tool_manager",
                "error": "Tool execution failed",
                "expected_behavior": "should_handle_gracefully"
            }
        ]
        
        for scenario in error_scenarios:
            with (
                patch("rag_system.DocumentProcessor"),
                patch("rag_system.VectorStore") as mock_vector_store,
                patch("rag_system.AIGenerator") as mock_ai_gen,
                patch("rag_system.SessionManager"),
            ):
                rag_system = RAGSystem(test_config)
                
                if scenario["layer"] == "vector_store":
                    mock_vector_store.return_value.search.return_value = SearchResults.empty(f"Search error: {scenario['error']}")
                    mock_ai_gen.return_value.generate_response.return_value = f"I encountered an error: {scenario['error']}"
                elif scenario["layer"] == "ai_generator":
                    mock_ai_gen.return_value.generate_response.side_effect = Exception(scenario['error'])
                elif scenario["layer"] == "tool_manager":
                    rag_system.tool_manager.get_last_sources = Mock(return_value=[])
                    rag_system.tool_manager.get_last_source_links = Mock(return_value=[])
                    mock_ai_gen.return_value.generate_response.return_value = f"Tool error: {scenario['error']}"

                if scenario["expected_behavior"] == "should_raise_exception":
                    with pytest.raises(Exception) as exc_info:
                        rag_system.query("Test query")
                    assert scenario['error'] in str(exc_info.value)
                else:
                    response, sources, source_links = rag_system.query("Test query")
                    assert scenario['error'] in response

    def test_query_performance_with_realistic_load(self, test_config):
        """Test query performance characteristics with realistic loads"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup realistic search results
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.search.return_value = SearchResults(
                documents=[f"Course content document {i}" for i in range(5)],
                metadata=[{"course_title": f"Course {i}", "lesson_number": 1} for i in range(5)],
                distances=[0.1 * i for i in range(5)]
            )
            
            mock_ai_gen.return_value.generate_response.return_value = "Response based on multiple course sources"

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[f"Course {i} - Lesson 1" for i in range(5)])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[f"https://course{i}.com" for i in range(5)])

            # Execute multiple queries in sequence
            for i in range(10):
                response, sources, source_links = rag_system.query(f"Query {i} about course content")
                
                # Each query should complete successfully
                assert isinstance(response, str)
                assert len(response) > 0
                assert len(sources) == 5
                assert len(source_links) == 5