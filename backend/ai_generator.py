from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews

**CRITICAL: Always use tools for course-related questions. The available courses are:**
- "Building Towards Computer Use with Anthropic"  
- "MCP: Build Rich-Context AI Apps with Anthropic"
- "Advanced Retrieval for AI with Chroma"
- "Prompt Compression and Query Optimization"

**MULTI-ROUND TOOL CALLING STRATEGY:**
You can make tool calls across up to 2 sequential rounds to gather comprehensive information:

**Round 1: Initial Information Gathering**
- Cast a wide net with general searches
- Get course outlines or basic topic information
- Identify what specific information you need for a complete answer

**Round 2: Targeted Follow-up (if needed)**
- Make precise searches based on Round 1 findings
- Fill information gaps or get specific details
- Compare information across courses or lessons
- Clarify or expand on initial findings

**Multi-Round Examples:**
- Query: "Find a course similar to lesson 4 of Computer Use course"
  - Round 1: get_course_outline("Computer Use") → learn lesson 4 is about "API Integration"
  - Round 2: search_course_content("API Integration") → find similar content in MCP course

- Query: "Compare how different courses teach vector databases"
  - Round 1: search_course_content("vector databases") → find basic info
  - Round 2: search_course_content("Chroma database tutorial") → get specific implementation details

**When to use multiple rounds:**
- Questions requiring information from different courses
- Comparative questions ("how do X and Y differ")  
- Questions where you need to find specific lessons first, then get their content
- Complex queries requiring step-by-step information gathering

Tool Usage Guidelines:
- **ALWAYS use content search first for ANY question that mentions courses, lessons, or specific technical topics covered in courses**
- Use course outline tool for questions about course structure, lesson lists, or complete course overviews
- **You can make up to 2 rounds of tool calls to gather comprehensive information**
- Use multiple rounds for complex queries that require information gathering then refinement
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

**When to use tools (USE TOOLS FOR ALL OF THESE):**
- Questions about "MCP", "Computer Use", "Chroma", "Anthropic courses"
- Questions mentioning "lesson", "course", "covered in", "teaches"
- Technical questions that might be covered in course materials
- Questions asking "what is X in the course" or "how does the course explain Y"

**When NOT to use tools:**
- Pure general knowledge questions with no course context
- Questions about basic concepts clearly outside course scope

Course Outline Responses:
When using the course outline tool, always include:
- Course title
- Course link (if available)  
- Complete lesson list with lesson numbers and titles
- Present information in a clear, structured format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential rounds of tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Start with initial messages
        messages = [{"role": "user", "content": query}]

        # Execute up to 2 rounds of tool calling with proper sequential support
        for round_num in range(1, 3):  # Round 1, Round 2
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Handle tool execution if needed
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and update messages
                messages = self._handle_tool_execution(response, messages, tool_manager)

                # Add round transition prompt after Round 1 to enable Round 2 reasoning
                if round_num == 1:
                    messages = self._add_round_transition_prompt(messages, round_num)
                    # Continue to Round 2 where Claude can make additional tool calls
                    continue
                else:
                    # After Round 2 tool execution, generate final response
                    break
            else:
                # Claude provided direct response without tool use
                return response.content[0].text

        # After tool rounds completed, make final call without tools to get response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_tool_execution(self, initial_response, messages: List, tool_manager):
        """
        Handle execution of tool calls and update message history.

        This method executes all tool calls from Claude's response and adds the results
        back to the message history, allowing for sequential rounds of tool calling.

        Args:
            initial_response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list (no longer returns should_continue)
        """
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Tool execution failed, add error result but continue
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error: Tool execution failed - {str(e)}",
                        }
                    )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def _add_round_transition_prompt(self, messages: List, round_num: int) -> List:
        """
        Add a prompt to guide Claude's decision for the next round.

        Args:
            messages: Current message history
            round_num: Current round number (1-based)

        Returns:
            Updated messages with transition prompt
        """
        if round_num == 1:
            # Transition from Round 1 to Round 2
            transition_prompt = """Based on the search results above, analyze what you've found and determine if you need additional information to provide a complete answer.

If you have sufficient information to answer the user's question completely, provide your final answer now.

If you need more specific information, additional context, or want to search for related topics to give a more comprehensive answer, make additional tool calls now.

Consider:
- Are there gaps in the information that need to be filled?
- Would searching for related or more specific terms improve the answer?
- Do you need to compare information from different courses or lessons?
- Would getting more detailed information help provide a better response?"""

            messages.append({"role": "user", "content": transition_prompt})

        return messages
