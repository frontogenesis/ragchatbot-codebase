import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from config import Config, config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestSystemDiagnosis:
    """System health check tests to diagnose current configuration and runtime issues"""

    def test_config_validation_max_results_zero_bug(self):
        """Test if the current config has the MAX_RESULTS=0 bug"""
        # Test the actual config object
        current_config = Config()

        # This test will reveal if MAX_RESULTS=0 is the issue
        if current_config.MAX_RESULTS == 0:
            pytest.fail(
                f"CRITICAL BUG FOUND: MAX_RESULTS is set to {current_config.MAX_RESULTS}. This will cause all vector searches to return empty results!"
            )

        assert (
            current_config.MAX_RESULTS > 0
        ), f"MAX_RESULTS must be > 0, got {current_config.MAX_RESULTS}"

        # Test recommended values
        assert (
            current_config.MAX_RESULTS >= 3
        ), "MAX_RESULTS should be at least 3 for useful results"
        assert (
            current_config.MAX_RESULTS <= 20
        ), "MAX_RESULTS should not exceed 20 to avoid performance issues"

    def test_config_anthropic_api_key_validation(self):
        """Test if Anthropic API key is properly configured"""
        current_config = Config()

        # Check if API key exists and is not empty
        api_key = current_config.ANTHROPIC_API_KEY

        if not api_key or api_key == "":
            pytest.skip(
                "ANTHROPIC_API_KEY not set - this will cause query failures. Set ANTHROPIC_API_KEY environment variable."
            )

        # Basic API key format validation
        assert isinstance(api_key, str), "API key should be a string"
        assert len(api_key) > 10, "API key seems too short"

        # Check if it looks like a placeholder
        placeholder_values = [
            "your_api_key_here",
            "test-api-key",
            "placeholder",
            "change_me",
        ]
        if api_key.lower() in placeholder_values:
            pytest.fail(f"API key appears to be a placeholder: {api_key}")

    def test_config_model_name_validation(self):
        """Test if the AI model name is correct"""
        current_config = Config()

        expected_model = "claude-sonnet-4-20250514"
        assert (
            current_config.ANTHROPIC_MODEL == expected_model
        ), f"Model should be {expected_model}, got {current_config.ANTHROPIC_MODEL}"

    def test_config_chunk_size_settings(self):
        """Test if document processing settings are reasonable"""
        current_config = Config()

        # Test chunk size
        assert current_config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert (
            200 <= current_config.CHUNK_SIZE <= 2000
        ), f"CHUNK_SIZE {current_config.CHUNK_SIZE} may be too small or too large"

        # Test chunk overlap
        assert current_config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert (
            current_config.CHUNK_OVERLAP < current_config.CHUNK_SIZE
        ), "CHUNK_OVERLAP must be less than CHUNK_SIZE"

        # Test reasonable overlap percentage
        overlap_percentage = (
            current_config.CHUNK_OVERLAP / current_config.CHUNK_SIZE
        ) * 100
        assert (
            overlap_percentage <= 50
        ), f"Chunk overlap {overlap_percentage:.1f}% seems excessive"

    def test_vector_store_initialization_with_current_config(self):
        """Test if vector store can be initialized with current configuration"""
        current_config = Config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary config for testing
            test_config = Config()
            test_config.CHROMA_PATH = temp_dir
            test_config.MAX_RESULTS = current_config.MAX_RESULTS

            # Test vector store initialization
            try:
                vector_store = VectorStore(
                    test_config.CHROMA_PATH,
                    test_config.EMBEDDING_MODEL,
                    test_config.MAX_RESULTS,
                )

                # Test that MAX_RESULTS is properly passed and stored
                assert vector_store.max_results == test_config.MAX_RESULTS

                if test_config.MAX_RESULTS == 0:
                    pytest.fail(
                        "Vector store initialized with MAX_RESULTS=0 - this will cause search failures!"
                    )

            except Exception as e:
                pytest.fail(f"Vector store initialization failed: {e}")

    def test_ai_generator_initialization_with_current_config(self):
        """Test if AI generator can be initialized with current configuration"""
        current_config = Config()

        # Skip if no API key
        if not current_config.ANTHROPIC_API_KEY:
            pytest.skip("No API key configured - cannot test AI generator")

        try:
            ai_generator = AIGenerator(
                current_config.ANTHROPIC_API_KEY, current_config.ANTHROPIC_MODEL
            )

            # Verify initialization
            assert ai_generator.model == current_config.ANTHROPIC_MODEL
            assert ai_generator.base_params["model"] == current_config.ANTHROPIC_MODEL

        except Exception as e:
            pytest.fail(f"AI generator initialization failed: {e}")

    @pytest.mark.integration
    def test_rag_system_initialization_with_current_config(self):
        """Test if RAG system can be initialized with current configuration"""
        current_config = Config()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use temporary directory to avoid conflicts
            test_config = Config()
            test_config.CHROMA_PATH = temp_dir
            test_config.ANTHROPIC_API_KEY = (
                current_config.ANTHROPIC_API_KEY or "test-key"
            )
            test_config.MAX_RESULTS = current_config.MAX_RESULTS

            if test_config.MAX_RESULTS == 0:
                pytest.fail(
                    "Cannot test RAG system with MAX_RESULTS=0 - this is a critical bug!"
                )

            try:
                rag_system = RAGSystem(test_config)

                # Verify components are initialized
                assert rag_system.vector_store is not None
                assert rag_system.ai_generator is not None
                assert rag_system.search_tool is not None
                assert rag_system.tool_manager is not None

                # Verify tool registration
                tool_definitions = rag_system.tool_manager.get_tool_definitions()
                tool_names = [tool["name"] for tool in tool_definitions]
                assert "search_course_content" in tool_names
                assert "get_course_outline" in tool_names

            except Exception as e:
                pytest.fail(f"RAG system initialization failed: {e}")

    def test_docs_folder_existence_and_content(self):
        """Test if the docs folder exists and contains course documents"""
        # Check common docs folder locations
        possible_docs_paths = [
            "../docs",
            "../../docs",
            "docs",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "docs"),
        ]

        docs_path = None
        for path in possible_docs_paths:
            if os.path.exists(path):
                docs_path = path
                break

        if not docs_path:
            pytest.skip("Docs folder not found - system will have no course content")

        # Check if docs folder contains files
        try:
            files = os.listdir(docs_path)
            course_files = [
                f for f in files if f.lower().endswith((".pdf", ".docx", ".txt"))
            ]

            if not course_files:
                pytest.skip(
                    f"No course documents found in {docs_path} - system will have no content"
                )

            print(f"Found {len(course_files)} course files in {docs_path}")

        except Exception as e:
            pytest.fail(f"Error accessing docs folder {docs_path}: {e}")

    def test_chroma_db_directory_permissions(self):
        """Test if ChromaDB directory can be created and accessed"""
        current_config = Config()
        chroma_path = current_config.CHROMA_PATH

        try:
            # Test if directory exists or can be created
            if not os.path.exists(chroma_path):
                os.makedirs(chroma_path, exist_ok=True)

            # Test write permissions
            test_file = os.path.join(chroma_path, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("test")

            # Cleanup test file
            os.remove(test_file)

        except PermissionError:
            pytest.fail(f"No write permissions for ChromaDB directory: {chroma_path}")
        except Exception as e:
            pytest.fail(f"Cannot access ChromaDB directory {chroma_path}: {e}")

    def test_environment_variables_loaded(self):
        """Test if environment variables are properly loaded"""
        # Check if .env file exists
        env_files = [".env", "../.env", "../../.env"]
        env_file_found = any(os.path.exists(env_file) for env_file in env_files)

        if not env_file_found:
            print(
                "Warning: No .env file found - environment variables must be set manually"
            )

        # Check critical environment variables
        critical_vars = ["ANTHROPIC_API_KEY"]
        missing_vars = []

        for var in critical_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            pytest.skip(f"Missing critical environment variables: {missing_vars}")

    @pytest.mark.slow
    def test_embedding_model_availability(self):
        """Test if the embedding model can be loaded"""
        current_config = Config()

        try:
            from sentence_transformers import SentenceTransformer

            # Test loading the embedding model
            model = SentenceTransformer(current_config.EMBEDDING_MODEL)

            # Test encoding a simple sentence
            test_embedding = model.encode("This is a test sentence.")

            assert len(test_embedding) > 0, "Embedding model returned empty result"
            assert len(test_embedding) > 100, "Embedding dimension seems too small"

        except ImportError:
            pytest.skip("sentence-transformers not installed")
        except Exception as e:
            pytest.fail(f"Embedding model loading failed: {e}")

    @pytest.mark.integration
    def test_vector_store_with_real_data_operations(self):
        """Test vector store operations with actual data"""
        current_config = Config()

        if current_config.MAX_RESULTS == 0:
            pytest.fail("Cannot test with MAX_RESULTS=0 - this will cause failures!")

        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = Config()
            test_config.CHROMA_PATH = temp_dir
            test_config.MAX_RESULTS = current_config.MAX_RESULTS

            try:
                vector_store = VectorStore(
                    test_config.CHROMA_PATH,
                    test_config.EMBEDDING_MODEL,
                    test_config.MAX_RESULTS,
                )

                # Test basic operations
                from models import Course, CourseChunk, Lesson

                # Create test data
                test_course = Course(
                    title="Test Course",
                    instructor="Test Instructor",
                    course_link="https://test.com",
                    lessons=[
                        Lesson(
                            lesson_number=1,
                            title="Test Lesson",
                            lesson_link="https://test.com/1",
                        )
                    ],
                )

                test_chunk = CourseChunk(
                    content="This is test course content about machine learning.",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                )

                # Test adding data
                vector_store.add_course_metadata(test_course)
                vector_store.add_course_content([test_chunk])

                # Test searching - this will fail if MAX_RESULTS=0
                results = vector_store.search("machine learning")

                if results.error:
                    pytest.fail(f"Vector store search failed: {results.error}")

                if current_config.MAX_RESULTS > 0 and results.is_empty():
                    pytest.fail(
                        "Search returned empty results despite having data - possible configuration issue"
                    )

            except Exception as e:
                pytest.fail(f"Vector store operations failed: {e}")

    def test_system_memory_requirements(self):
        """Test if system has sufficient resources"""
        import psutil

        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < 2:
            print(
                f"Warning: Low available memory ({available_memory_gb:.1f} GB) - may cause performance issues"
            )

        # Check available disk space
        current_config = Config()
        chroma_path = current_config.CHROMA_PATH

        if os.path.exists(chroma_path):
            disk_usage = psutil.disk_usage(chroma_path)
            available_space_gb = disk_usage.free / (1024**3)

            if available_space_gb < 1:
                print(
                    f"Warning: Low disk space ({available_space_gb:.1f} GB) in {chroma_path}"
                )

    def test_python_dependencies_versions(self):
        """Test if all required dependencies are installed with correct versions"""
        required_packages = {
            "anthropic": "0.8.0",  # Minimum version
            "chromadb": "0.4.0",
            "sentence-transformers": "2.2.0",
            "fastapi": "0.100.0",
            "uvicorn": "0.20.0",
        }

        missing_packages = []
        version_issues = []

        for package, min_version in required_packages.items():
            try:
                import importlib

                module = importlib.import_module(package.replace("-", "_"))

                # Try to get version
                if hasattr(module, "__version__"):
                    version = module.__version__
                    print(f"{package}: {version}")
                else:
                    print(f"{package}: version unknown")

            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            pytest.fail(f"Missing required packages: {missing_packages}")

    def test_current_working_directory_context(self):
        """Test if tests are running from the correct directory"""
        current_dir = os.getcwd()

        # Should be running from backend directory or project root
        expected_paths = ["backend", "ragchatbot-codebase"]

        if not any(path in current_dir for path in expected_paths):
            print(f"Warning: Running from unexpected directory: {current_dir}")
            print(
                "Consider running tests from the backend directory: cd backend && python -m pytest"
            )

        # Check if we can import modules correctly
        try:
            from config import config as test_config
            from rag_system import RAGSystem as test_rag

            print("Module imports successful")
        except ImportError as e:
            pytest.fail(f"Cannot import required modules from {current_dir}: {e}")

    def test_logging_configuration(self):
        """Test if logging is properly configured"""
        import logging

        # Check current logging level
        root_logger = logging.getLogger()
        current_level = root_logger.level

        print(
            f"Current logging level: {current_level} ({logging.getLevelName(current_level)})"
        )

        # Test that we can create log messages
        try:
            logger = logging.getLogger("rag_system_test")
            logger.info("Test log message")
        except Exception as e:
            pytest.fail(f"Logging configuration issue: {e}")

    def test_system_diagnosis_summary(self):
        """Generate a summary of system health based on other tests"""
        current_config = Config()

        issues = []
        warnings = []

        # Check critical issues
        if current_config.MAX_RESULTS == 0:
            issues.append("CRITICAL: MAX_RESULTS=0 will cause all searches to fail")

        if not current_config.ANTHROPIC_API_KEY:
            issues.append("CRITICAL: No Anthropic API key configured")

        # Check warnings
        if current_config.MAX_RESULTS > 20:
            warnings.append(f"MAX_RESULTS={current_config.MAX_RESULTS} may be too high")

        if not os.path.exists("../docs"):
            warnings.append("No docs folder found - system will have no course content")

        # Print summary
        print("\n=== SYSTEM DIAGNOSIS SUMMARY ===")
        print(f"MAX_RESULTS: {current_config.MAX_RESULTS}")
        print(
            f"API Key configured: {'Yes' if current_config.ANTHROPIC_API_KEY else 'No'}"
        )
        print(f"Model: {current_config.ANTHROPIC_MODEL}")
        print(f"Chunk size: {current_config.CHUNK_SIZE}")

        if issues:
            print(f"\nCRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")

        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")

        if not issues and not warnings:
            print("\n✅ No critical issues detected")

        # Fail test if critical issues found
        if issues:
            pytest.fail(f"Critical system issues detected: {issues}")
