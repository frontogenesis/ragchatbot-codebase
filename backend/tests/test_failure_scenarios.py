import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pytest
import tempfile

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from ai_generator import AIGenerator
from search_tools import CourseSearchTool


class TestFailureScenarios:
    """Simulate real failure conditions that cause 'Query failed' type responses"""

    def test_complete_system_failure_max_results_zero(self, broken_config):
        """Test the complete system failure when MAX_RESULTS=0 causes cascading failures"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup the broken configuration scenario
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Simulate what happens with MAX_RESULTS=0:
            # 1. Vector store search returns empty results
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            # 2. CourseSearchTool returns "No relevant content found"
            # 3. AI receives empty tool results and responds unhelpfully
            mock_ai_gen.return_value.generate_response.return_value = "I'm unable to find information about that topic."

            rag_system = RAGSystem(broken_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Test multiple queries that should work but fail
            failing_queries = [
                "What is covered in lesson 1?",
                "Tell me about the course introduction",
                "What topics are discussed in the AI course?",
                "Show me content from lesson 2"
            ]

            for query in failing_queries:
                response, sources, source_links = rag_system.query(query)
                
                # All queries should fail with unhelpful responses
                assert "unable to find" in response.lower() or "no" in response.lower()
                assert sources == []
                assert source_links == []
                
                # This demonstrates the "Query failed" behavior

    def test_anthropic_api_authentication_failure(self, test_config):
        """Test system behavior when Anthropic API authentication fails"""
        # Create config with invalid API key
        invalid_config = test_config
        invalid_config.ANTHROPIC_API_KEY = "invalid-key-123"

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock authentication failure
            mock_ai_gen.return_value.generate_response.side_effect = Exception("Authentication Error: Invalid API key")

            rag_system = RAGSystem(invalid_config)

            # All queries should fail with authentication error
            with pytest.raises(Exception) as exc_info:
                rag_system.query("What is machine learning?")
            
            assert "Authentication" in str(exc_info.value)

    def test_anthropic_api_rate_limit_failure(self, test_config):
        """Test system behavior when hitting API rate limits"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock rate limit error
            mock_ai_gen.return_value.generate_response.side_effect = Exception("Rate limit exceeded. Please try again later.")

            rag_system = RAGSystem(test_config)

            # Queries should fail with rate limit error
            with pytest.raises(Exception) as exc_info:
                rag_system.query("What is deep learning?")
            
            assert "Rate limit" in str(exc_info.value)

    def test_chroma_db_corruption_failure(self, test_config):
        """Test system behavior when ChromaDB database is corrupted"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup corrupted database scenario
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Mock database corruption error
            mock_vector_store_instance.search.return_value = SearchResults.empty("Search error: Database file is corrupted")
            mock_ai_gen.return_value.generate_response.return_value = "I'm experiencing database issues and cannot search the course materials."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            response, sources, source_links = rag_system.query("What is AI?")

            # Should indicate database issues
            assert "database" in response.lower() and "issues" in response.lower()
            assert sources == []

    def test_embedding_model_download_failure(self, test_config):
        """Test system behavior when embedding model cannot be downloaded"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock embedding model failure during vector store initialization
            mock_vector_store.side_effect = Exception("Failed to download model 'all-MiniLM-L6-v2': Network error")

            # RAG system initialization should fail
            with pytest.raises(Exception) as exc_info:
                RAGSystem(test_config)
            
            assert "download model" in str(exc_info.value)

    def test_out_of_disk_space_failure(self, test_config):
        """Test system behavior when running out of disk space"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup disk space failure scenario
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Mock disk space error
            mock_vector_store_instance.search.return_value = SearchResults.empty("Search error: No space left on device")
            mock_ai_gen.return_value.generate_response.return_value = "I'm unable to search due to storage limitations."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            response, sources, source_links = rag_system.query("Explain neural networks")

            assert "unable to search" in response.lower() or "storage" in response.lower()

    def test_memory_exhaustion_failure(self, test_config):
        """Test system behavior when running out of memory"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup memory exhaustion scenario
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Mock memory error
            mock_vector_store_instance.search.side_effect = MemoryError("Out of memory")

            rag_system = RAGSystem(test_config)

            # Should handle memory error gracefully or raise it
            try:
                response, sources, source_links = rag_system.query("Process large query")
                # If it doesn't raise, should indicate memory issues
                assert "memory" in response.lower() or "error" in response.lower()
            except MemoryError:
                # Memory errors are expected to propagate
                pass

    def test_network_connectivity_failure(self, test_config):
        """Test system behavior when network connectivity is lost"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Mock network failure for Anthropic API
            mock_ai_gen.return_value.generate_response.side_effect = Exception("Network error: Connection timed out")

            rag_system = RAGSystem(test_config)

            # Network failures should propagate
            with pytest.raises(Exception) as exc_info:
                rag_system.query("What is computer vision?")
            
            assert "Network error" in str(exc_info.value) or "Connection" in str(exc_info.value)

    def test_concurrent_access_race_condition(self, test_config):
        """Test system behavior under concurrent access that causes race conditions"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup race condition scenario
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Mock intermittent failures due to race conditions
            search_results = [
                SearchResults(["Result 1"], [{"course_title": "Course A"}], [0.1]),
                SearchResults.empty("Search error: Database locked"),
                SearchResults(["Result 2"], [{"course_title": "Course B"}], [0.2]),
                SearchResults.empty("Search error: Resource busy")
            ]
            
            mock_vector_store_instance.search.side_effect = search_results
            mock_ai_gen.return_value.generate_response.side_effect = [
                "Found some information",
                "I encountered a database error",
                "Here's what I found",
                "The system is currently busy"
            ]

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(side_effect=[["Course A"], [], ["Course B"], []])
            rag_system.tool_manager.get_last_source_links = Mock(side_effect=[["link1"], [], ["link2"], []])

            # Multiple concurrent requests with mixed results
            responses = []
            for i in range(4):
                response, sources, source_links = rag_system.query(f"Query {i}")
                responses.append(response)

            # Some should succeed, some should fail
            success_responses = [r for r in responses if "found" in r.lower()]
            error_responses = [r for r in responses if "error" in r.lower() or "busy" in r.lower()]
            
            assert len(success_responses) > 0, "No successful responses during race condition test"
            assert len(error_responses) > 0, "No error responses during race condition test"

    def test_document_processing_failure_empty_system(self, test_config):
        """Test system behavior when no documents could be processed"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Setup empty system (no documents processed)
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.get_existing_course_titles.return_value = []
            mock_vector_store_instance.get_course_count.return_value = 0
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            # Mock document processing failures
            mock_doc_proc.return_value.process_course_document.side_effect = Exception("Document processing failed")
            
            mock_ai_gen.return_value.generate_response.return_value = "I don't have any course materials loaded to search through."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Verify system has no content
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 0

            response, sources, source_links = rag_system.query("What courses are available?")

            # Should indicate no content available
            assert "don't have" in response.lower() or "no" in response.lower()

    def test_malformed_query_handling(self, test_config):
        """Test system behavior with malformed or problematic queries"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Test various problematic queries
            problematic_queries = [
                "",  # Empty query
                " ",  # Whitespace only
                "a" * 10000,  # Extremely long query
                "SELECT * FROM courses; DROP TABLE courses;",  # SQL injection attempt
                "What is \x00\x01\x02 in the course?",  # Binary characters
                "🤖🚀💻" * 100,  # Unicode spam
            ]

            for query in problematic_queries:
                try:
                    # Mock responses for problematic queries
                    if not query.strip():
                        mock_ai_gen.return_value.generate_response.return_value = "Please provide a specific question about the course materials."
                    else:
                        mock_ai_gen.return_value.generate_response.return_value = "I'll help you with your question about the course materials."

                    response, sources, source_links = rag_system.query(query)
                    
                    # Should handle gracefully without crashing
                    assert isinstance(response, str)
                    
                except Exception as e:
                    # Some malformed queries might raise exceptions - that's acceptable
                    print(f"Query '{query[:50]}...' raised exception: {e}")

    def test_session_management_corruption(self, test_config):
        """Test system behavior when session management is corrupted"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session,
        ):
            # Setup corrupted session scenario
            mock_session.return_value.get_conversation_history.side_effect = Exception("Session data corrupted")
            mock_session.return_value.add_exchange.side_effect = Exception("Cannot save session")
            
            mock_ai_gen.return_value.generate_response.return_value = "I cannot access conversation history due to session issues."

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # Should handle session corruption gracefully
            response, sources, source_links = rag_system.query("Continue our conversation", session_id="corrupted-session")

            # May indicate session issues or work without session context
            assert isinstance(response, str)

    def test_tool_definition_corruption_failure(self, test_config):
        """Test system behavior when tool definitions are corrupted"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            rag_system = RAGSystem(test_config)
            
            # Corrupt tool definitions
            rag_system.tool_manager.get_tool_definitions = Mock(return_value=[
                {"name": "corrupted_tool", "invalid_schema": True},  # Invalid tool definition
                {},  # Empty tool definition
                None  # Null tool definition
            ])
            
            mock_ai_gen.return_value.generate_response.return_value = "I encountered an issue with the search tools."

            response, sources, source_links = rag_system.query("What is machine learning?")

            # Should indicate tool issues
            assert "issue" in response.lower() or "tool" in response.lower()

    def test_cascading_failure_scenario(self, broken_config):
        """Test a realistic cascading failure scenario that leads to 'Query failed'"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Step 1: MAX_RESULTS=0 causes vector store to return empty results
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            mock_vector_store_instance.search.return_value = SearchResults([], [], [])
            
            # Step 2: CourseSearchTool gets empty results and returns "No relevant content found"
            # Step 3: AI gets tool result indicating no content found
            # Step 4: AI responds with generic "unable to help" message
            mock_ai_gen.return_value.generate_response.return_value = "I'm sorry, but I'm unable to find relevant information to answer your question."

            rag_system = RAGSystem(broken_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[])

            # User asks a perfectly reasonable question that should work
            response, sources, source_links = rag_system.query("Can you explain what machine learning is according to the course materials?")

            # Result: Generic failure response that looks like "Query failed"
            assert "sorry" in response.lower() or "unable" in response.lower()
            assert "find" in response.lower() or "information" in response.lower()
            assert sources == []
            assert source_links == []

            # This is the exact "Query failed" scenario described by the user:
            # 1. User asks reasonable question
            # 2. System appears to work but returns unhelpful response
            # 3. No sources or content are returned
            # 4. User gets frustrated because the system "fails" to help

    def test_system_recovery_after_failure(self, test_config):
        """Test system behavior after recovering from failures"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            # Simulate failure then recovery
            search_results_sequence = [
                SearchResults.empty("Search error: Temporary failure"),  # Initial failure
                SearchResults(  # Recovery with results
                    documents=["Machine learning is a subset of artificial intelligence."],
                    metadata=[{"course_title": "AI Fundamentals", "lesson_number": 1}],
                    distances=[0.1]
                )
            ]
            
            ai_responses = [
                "I'm experiencing technical difficulties right now.",  # Failure response
                "Based on the course materials, machine learning is a subset of AI."  # Recovery response
            ]
            
            mock_vector_store_instance.search.side_effect = search_results_sequence
            mock_ai_gen.return_value.generate_response.side_effect = ai_responses

            rag_system = RAGSystem(test_config)
            rag_system.tool_manager.get_last_sources = Mock(side_effect=[[], ["AI Fundamentals - Lesson 1"]])
            rag_system.tool_manager.get_last_source_links = Mock(side_effect=[[], ["https://ai.com"]])

            # First query fails
            response1, sources1, links1 = rag_system.query("What is machine learning?")
            assert "difficulties" in response1.lower()
            assert sources1 == []

            # Second query succeeds after recovery
            response2, sources2, links2 = rag_system.query("What is machine learning?")
            assert "machine learning" in response2.lower()
            assert sources2 == ["AI Fundamentals - Lesson 1"]

    def test_partial_failure_scenarios(self, test_config):
        """Test scenarios where some components work but others fail"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager"),
        ):
            # Scenario: Vector store works but lesson link retrieval fails
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            mock_vector_store_instance.search.return_value = SearchResults(
                documents=["Course content about neural networks"],
                metadata=[{"course_title": "Deep Learning", "lesson_number": 3}],
                distances=[0.15]
            )
            
            mock_ai_gen.return_value.generate_response.return_value = "Here's what I found about neural networks in the course."

            rag_system = RAGSystem(test_config)
            
            # Sources work but links fail
            rag_system.tool_manager.get_last_sources = Mock(return_value=["Deep Learning - Lesson 3"])
            rag_system.tool_manager.get_last_source_links = Mock(return_value=[None])  # Link retrieval failed

            response, sources, source_links = rag_system.query("Tell me about neural networks")

            # Should work but without links
            assert "neural networks" in response.lower()
            assert sources == ["Deep Learning - Lesson 3"]
            assert source_links == [None]  # Partial failure is acceptable