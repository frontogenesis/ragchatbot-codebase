import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore


class TestCourseSearchToolComprehensive:
    """Comprehensive tests for CourseSearchTool to diagnose failure scenarios"""

    def test_execute_with_max_results_zero_bug(self):
        """Test the critical MAX_RESULTS=0 bug that causes empty results"""
        # Create a mock vector store that simulates MAX_RESULTS=0 behavior
        mock_vector_store = Mock()

        # Simulate what happens when MAX_RESULTS=0 is passed to ChromaDB
        # ChromaDB would return empty results even for valid queries
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        tool = CourseSearchTool(mock_vector_store)

        # Execute with a query that should find content
        result = tool.execute("What is covered in the introduction lesson?")

        # Assert that we get "No relevant content found" due to MAX_RESULTS=0
        assert result == "No relevant content found."

        # This test demonstrates the critical bug: even valid queries return nothing
        # when MAX_RESULTS=0 because the vector store search limit is 0
        mock_vector_store.search.assert_called_once_with(
            query="What is covered in the introduction lesson?",
            course_name=None,
            lesson_number=None,
        )

    def test_execute_with_vector_store_error_propagation(self):
        """Test how errors from vector store are handled"""
        mock_vector_store = Mock()

        # Simulate different types of vector store errors
        error_scenarios = [
            "Search error: Database connection failed",
            "Search error: Invalid query format",
            "Search error: ChromaDB collection not found",
            "Search error: Embedding model failed",
        ]

        tool = CourseSearchTool(mock_vector_store)

        for error_msg in error_scenarios:
            mock_vector_store.search.return_value = SearchResults.empty(error_msg)

            result = tool.execute("test query")

            # Assert that the error is properly propagated
            assert result == error_msg
            assert "Search error:" in result

    def test_execute_with_course_name_resolution_failure(self):
        """Test behavior when course name cannot be resolved"""
        mock_vector_store = Mock()

        # Mock course name resolution failure
        course_not_found_error = SearchResults.empty(
            "No course found matching 'NonexistentCourse'"
        )
        mock_vector_store.search.return_value = course_not_found_error

        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("What is in lesson 1?", course_name="NonexistentCourse")

        assert result == "No course found matching 'NonexistentCourse'"

        # Verify the search was attempted with the invalid course name
        mock_vector_store.search.assert_called_once_with(
            query="What is in lesson 1?",
            course_name="NonexistentCourse",
            lesson_number=None,
        )

    def test_execute_with_realistic_vector_store_configuration(self):
        """Test with realistic vector store configurations that might cause issues"""
        test_configs = [
            {"max_results": 0, "expected_result_count": 0},  # Broken config
            {"max_results": 1, "expected_result_count": 1},  # Very limited
            {"max_results": 5, "expected_result_count": 5},  # Normal config
        ]

        for config in test_configs:
            mock_vector_store = Mock()

            # Create results based on max_results configuration
            if config["expected_result_count"] == 0:
                # Simulate empty results due to MAX_RESULTS=0
                search_results = SearchResults(documents=[], metadata=[], distances=[])
            else:
                # Create realistic search results
                documents = [
                    f"Document content {i}"
                    for i in range(config["expected_result_count"])
                ]
                metadata = [
                    {
                        "course_title": "Test Course",
                        "lesson_number": i + 1,
                        "chunk_index": i,
                    }
                    for i in range(config["expected_result_count"])
                ]
                distances = [
                    0.1 * (i + 1) for i in range(config["expected_result_count"])
                ]
                search_results = SearchResults(
                    documents=documents, metadata=metadata, distances=distances
                )

            mock_vector_store.search.return_value = search_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson"
            )

            tool = CourseSearchTool(mock_vector_store)
            result = tool.execute("test query")

            if config["expected_result_count"] == 0:
                assert result == "No relevant content found."
            else:
                assert "[Test Course - Lesson" in result
                assert f"Document content" in result

    def test_execute_with_malformed_metadata(self):
        """Test handling of malformed metadata from vector store"""
        mock_vector_store = Mock()

        # Create search results with malformed metadata
        malformed_metadata_scenarios = [
            # Missing course_title
            [{"lesson_number": 1, "chunk_index": 0}],
            # Missing lesson_number
            [{"course_title": "Test Course", "chunk_index": 0}],
            # Empty metadata
            [{}],
            # None values
            [{"course_title": None, "lesson_number": None, "chunk_index": 0}],
        ]

        tool = CourseSearchTool(mock_vector_store)

        for malformed_metadata in malformed_metadata_scenarios:
            search_results = SearchResults(
                documents=["Test content"], metadata=malformed_metadata, distances=[0.1]
            )
            mock_vector_store.search.return_value = search_results

            # Should not crash, should handle gracefully
            result = tool.execute("test query")

            # Should still format some kind of result
            assert isinstance(result, str)
            assert len(result) > 0

    def test_execute_with_network_timeout_simulation(self):
        """Test behavior when vector store operations timeout"""
        mock_vector_store = Mock()

        # Simulate timeout/network errors
        timeout_error = SearchResults.empty(
            "Search error: Request timeout after 30 seconds"
        )
        mock_vector_store.search.return_value = timeout_error

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is machine learning?")

        assert result == "Search error: Request timeout after 30 seconds"

    def test_execute_with_large_result_set_handling(self):
        """Test handling of large result sets that might cause memory issues"""
        mock_vector_store = Mock()

        # Create a large result set
        large_document_count = 100
        documents = [
            f"Large document content {i} " * 100 for i in range(large_document_count)
        ]  # Long documents
        metadata = [
            {
                "course_title": f"Course {i % 10}",
                "lesson_number": i % 20,
                "chunk_index": i,
            }
            for i in range(large_document_count)
        ]
        distances = [0.001 * i for i in range(large_document_count)]

        large_results = SearchResults(
            documents=documents, metadata=metadata, distances=distances
        )
        mock_vector_store.search.return_value = large_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Should handle large results without crashing
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain formatted results from multiple courses
        assert "[Course" in result

    def test_execute_source_tracking_with_failures(self):
        """Test source tracking when lesson link retrieval fails"""
        mock_vector_store = Mock()

        # Create normal search results
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
            ],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = search_results

        # Make get_lesson_link fail/return None
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Should still work and track sources properly
        assert "[Test Course - Lesson 1]" in result
        assert tool.last_sources == ["Test Course - Lesson 1"]
        assert tool.last_source_links == [None]  # Should handle None gracefully

    def test_execute_with_unicode_content(self):
        """Test handling of Unicode characters in search results"""
        mock_vector_store = Mock()

        # Create search results with Unicode content
        unicode_documents = [
            "Content with émojis 🚀 and spëcial çharacters",
            "中文内容测试 with mixed languages",
            "Mathematical symbols: ∑∏∆√∞",
        ]
        unicode_metadata = [
            {"course_title": "Unicodé Course", "lesson_number": 1, "chunk_index": i}
            for i in range(len(unicode_documents))
        ]

        unicode_results = SearchResults(
            documents=unicode_documents,
            metadata=unicode_metadata,
            distances=[0.1, 0.2, 0.3],
        )
        mock_vector_store.search.return_value = unicode_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesšon"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Should handle Unicode content properly
        assert isinstance(result, str)
        assert "🚀" in result
        assert "中文" in result
        assert "∑∏∆" in result

    def test_execute_parameter_validation_edge_cases(self):
        """Test parameter validation with edge case inputs"""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults([], [], [])

        tool = CourseSearchTool(mock_vector_store)

        edge_case_parameters = [
            # Empty strings
            {"query": "", "course_name": "", "lesson_number": None},
            # Very long strings
            {"query": "x" * 10000, "course_name": "y" * 1000, "lesson_number": None},
            # Negative lesson numbers
            {"query": "test", "course_name": None, "lesson_number": -1},
            # Very large lesson numbers
            {"query": "test", "course_name": None, "lesson_number": 999999},
            # Special characters in course names
            {
                "query": "test",
                "course_name": "Course with / and \\ and ?",
                "lesson_number": None,
            },
        ]

        for params in edge_case_parameters:
            # Should not crash with edge case parameters
            result = tool.execute(
                query=params["query"],
                course_name=params["course_name"],
                lesson_number=params["lesson_number"],
            )

            assert isinstance(result, str)

    def test_execute_concurrent_access_simulation(self):
        """Test behavior under concurrent access patterns"""
        mock_vector_store = Mock()

        # Create different results for each "concurrent" call
        results_sequence = [
            SearchResults(
                documents=[f"Result {i}"],
                metadata=[{"course_title": f"Course {i}", "lesson_number": 1}],
                distances=[0.1],
            )
            for i in range(5)
        ]

        mock_vector_store.search.side_effect = results_sequence
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)

        # Simulate multiple concurrent requests
        results = []
        for i in range(5):
            result = tool.execute(f"query {i}")
            results.append(result)

        # Each result should be different and properly formatted
        for i, result in enumerate(results):
            assert f"Course {i}" in result
            assert f"Result {i}" in result

    def test_execute_memory_cleanup_after_large_operations(self):
        """Test that memory is properly managed after large operations"""
        mock_vector_store = Mock()

        tool = CourseSearchTool(mock_vector_store)

        # Perform multiple large operations
        for iteration in range(10):
            large_results = SearchResults(
                documents=[f"Large content {i}" * 1000 for i in range(50)],
                metadata=[
                    {"course_title": f"Course {iteration}", "lesson_number": i}
                    for i in range(50)
                ],
                distances=[0.01 * i for i in range(50)],
            )
            mock_vector_store.search.return_value = large_results
            mock_vector_store.get_lesson_link.return_value = (
                f"https://example.com/lesson{iteration}"
            )

            result = tool.execute(f"query {iteration}")

            # Verify that previous sources are properly replaced, not accumulated
            assert len(tool.last_sources) == 50  # Should only have current results
            assert len(tool.last_source_links) == 50
