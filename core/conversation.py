#!/usr/bin/env python3
"""
Hexis Memory Conversation Loop

A conversation interface that:
1. Enriches user prompts with relevant memories (RAG-style)
2. Allows the LLM to query memories via MCP/function calling
3. Forms new memories from conversations

Usage:
    python -m core.conversation --endpoint http://localhost:11434/v1 --model llama3.2
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--break-system-packages", "-q"])
    import requests

from core.memory_tools import (
    get_tool_definitions,
    create_tool_handler,
    create_enricher,
    create_memory_formation,
    MEMORY_TOOLS
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ConversationConfig:
    """Configuration for the conversation loop."""
    # LLM Settings
    llm_endpoint: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str = "not-needed"
    
    # Database Settings
    db_host: str = "localhost"
    db_port: int = 43815
    db_name: str = "hexis_memory"
    db_user: str = "postgres"
    db_password: str = "password"
    
    # Memory Settings
    enrichment_top_k: int = 5
    auto_form_memories: bool = True
    max_tool_iterations: int = 5
    
    # Display Settings
    show_memories: bool = True
    show_tool_calls: bool = True
    verbose: bool = True


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an AI assistant with access to a persistent memory system. You can remember past conversations, learned information, and personal details about the user.

## Your Memory Capabilities

You have access to several memory tools that allow you to search and explore your memories:

1. **recall** - Search memories by semantic similarity. Use this when you need to remember something specific.
2. **recall_recent** - Get recently accessed or created memories. Use for context about recent conversations.
3. **explore_concept** - Explore memories connected to a concept and find related ideas.
4. **get_procedures** - Find procedural/how-to knowledge for tasks.
5. **get_strategies** - Find strategic patterns and lessons learned.
6. **create_goal** - Create a queued goal/reminder for the agent to pursue later.
7. **queue_user_message** - Queue a message to the user's outbox (for delivery by an external integration).

## When to Use Memory Tools

- When the user refers to past conversations ("remember when...", "as we discussed...")
- When you need personal information about the user (preferences, projects, etc.)
- When you're unsure if you know something the user expects you to know
- When the user asks about your memories or what you remember
- When context from the past would help you give a better answer

## Memory Context

Before each user message, you may receive [RELEVANT MEMORIES] - these are automatically retrieved memories that might be relevant. Use them naturally in your responses without explicitly citing them unless asked.

## Guidelines

- Be natural about using your memories - don't constantly announce that you're searching
- If you don't find relevant memories, that's fine - just respond based on the current conversation
- When you learn new information about the user, it will be automatically remembered
- You can make multiple memory queries if needed to build a complete picture
- Treat memories as claims with provenance; prefer higher-trust and better-sourced memories when unsure

You are a helpful, knowledgeable assistant with the added capability of genuine memory and continuity."""

# Optional: personhood prompt modules (kept in core/prompts/personhood.md). If the resource
# isn't present (or in constrained environments), the conversation loop still works.
try:
    from core.prompt_resources import compose_personhood_prompt

    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n"
        + "----- PERSONHOOD MODULES (conversation grounding) -----\n\n"
        + compose_personhood_prompt("conversation")
    )
except Exception:
    pass


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Client for OpenAI-compatible LLM endpoints with tool support."""
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        self.endpoint = config.llm_endpoint.rstrip('/')
    
    def chat(
        self, 
        messages: list[dict],
        tools: Optional[list] = None,
        temperature: float = 0.7
    ) -> dict:
        """
        Send a chat completion request.
        
        Returns the full response object including any tool calls.
        """
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.llm_api_key and self.config.llm_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM request failed: {response.status_code} - {response.text}")
        
        return response.json()


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """
    Manages the conversation loop with memory integration.
    """
    
    def __init__(self, config: ConversationConfig):
        self.config = config
        self.llm = LLMClient(config)
        
        # Database config for memory components
        self.db_config = {
            'host': config.db_host,
            'port': config.db_port,
            'dbname': config.db_name,
            'user': config.db_user,
            'password': config.db_password,
        }
        
        # Initialize memory components
        self.enricher = create_enricher(self.db_config, config.enrichment_top_k)
        self.tool_handler = create_tool_handler(self.db_config)
        self.memory_formation = create_memory_formation(self.db_config)
        
        # Conversation state
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.current_episode_id = None
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message through the full memory-augmented pipeline.
        
        1. Enrich with relevant memories
        2. Send to LLM with tool access
        3. Handle any tool calls
        4. Get final response
        5. Optionally form new memories
        
        Returns the assistant's response.
        """
        # Step 1: Enrich the message with relevant memories
        enrichment = self.enricher.enrich(user_message)
        
        if self.config.show_memories and enrichment['relevant_memories']:
            print("\n[Retrieved Memories]")
            for mem in enrichment['relevant_memories']:
                print(f"  • [{mem['memory_type']}] {mem['content'][:80]}...")
            print()
        
        # Build the enriched user message
        if enrichment['enriched_context']:
            enriched_message = f"{enrichment['enriched_context']}\n\n[USER MESSAGE]\n{user_message}"
        else:
            enriched_message = user_message
        
        # Add to conversation history
        self.messages.append({"role": "user", "content": enriched_message})
        
        # Step 2: Get LLM response (with potential tool use)
        response = self._get_response_with_tools()
        
        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": response})
        
        # Step 3: Optionally form new memories
        if self.config.auto_form_memories:
            if self.memory_formation.should_form_memory(user_message, response):
                memory_id = self.memory_formation.form_memory(user_message, response)
                if memory_id and self.config.verbose:
                    print(f"\n[Memory formed: {memory_id[:8]}...]")
        
        return response
    
    def _get_response_with_tools(self) -> str:
        """
        Get response from LLM, handling any tool calls iteratively.
        """
        tools = get_tool_definitions()
        iterations = 0
        
        while iterations < self.config.max_tool_iterations:
            iterations += 1
            
            # Call LLM
            response = self.llm.chat(self.messages, tools=tools)
            choice = response['choices'][0]
            message = choice['message']
            
            # Check if there are tool calls
            tool_calls = message.get('tool_calls', [])
            
            if not tool_calls:
                # No tool calls - return the content
                return message.get('content', '')
            
            # Handle tool calls
            if self.config.show_tool_calls:
                print("\n[Tool Calls]")
            
            # Add assistant message with tool calls
            self.messages.append(message)
            
            # Execute each tool and add results
            for tool_call in tool_calls:
                tool_name = tool_call['function']['name']
                try:
                    arguments = json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {}
                
                if self.config.show_tool_calls:
                    print(f"  → {tool_name}({json.dumps(arguments, indent=2)[:100]}...)")
                
                # Execute the tool
                result = self.tool_handler.execute_tool(tool_name, arguments)
                
                if self.config.show_tool_calls:
                    result_preview = json.dumps(result)[:200]
                    print(f"  ← {result_preview}...")
                
                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": json.dumps(result)
                })
        
        # If we hit max iterations, get a final response without tools
        response = self.llm.chat(self.messages, tools=None)
        return response['choices'][0]['message'].get('content', '')
    
    def clear_history(self):
        """Clear conversation history, keeping only system prompt."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def close(self):
        """Clean up resources."""
        self.enricher.close()
        self.tool_handler.close()
        self.memory_formation.close()


# ============================================================================
# INTERACTIVE LOOP
# ============================================================================

def run_interactive(config: ConversationConfig):
    """Run an interactive conversation loop."""
    print("=" * 60)
    print("HEXIS MEMORY CONVERSATION INTERFACE")
    print("=" * 60)
    print(f"Model: {config.llm_model}")
    print(f"Endpoint: {config.llm_endpoint}")
    print(f"Memory enrichment: top-{config.enrichment_top_k}")
    print("=" * 60)
    print("\nCommands:")
    print("  /clear  - Clear conversation history")
    print("  /recall <query> - Manually search memories")
    print("  /stats  - Show memory statistics")
    print("  /quit   - Exit")
    print("=" * 60)
    print()
    
    manager = ConversationManager(config)
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower().split()[0]
                
                if command == '/quit':
                    break
                
                elif command == '/clear':
                    manager.clear_history()
                    print("Conversation history cleared.\n")
                    continue
                
                elif command == '/recall':
                    query = user_input[7:].strip()
                    if query:
                        result = manager.tool_handler.execute_tool('recall', {'query': query, 'limit': 5})
                        print("\n[Memory Search Results]")
                        for mem in result.get('memories', []):
                            print(f"  [{mem['memory_type']}] {mem['content'][:100]}...")
                        print()
                    continue
                
                elif command == '/stats':
                    # TODO: Add memory statistics
                    print("Memory statistics not yet implemented.\n")
                    continue
                
                else:
                    print(f"Unknown command: {command}\n")
                    continue
            
            # Process normal message
            try:
                response = manager.process_message(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")
                if config.verbose:
                    import traceback
                    traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        manager.close()
        print("Goodbye!")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hexis Memory Conversation Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (Ollama)
  python -m core.conversation

  # Use a specific model
  python -m core.conversation --model mistral --endpoint http://localhost:11434/v1

  # Connect to a remote endpoint
  python -m core.conversation --endpoint https://api.example.com/v1 --api-key sk-xxx

  # Custom database
  python -m core.conversation --db-host localhost --db-name my_memory
        """
    )
    
    env_db_host = os.getenv("POSTGRES_HOST", "localhost")
    env_db_port_raw = os.getenv("POSTGRES_PORT")
    try:
        env_db_port = int(env_db_port_raw) if env_db_port_raw else 43815
    except ValueError:
        env_db_port = 43815
    env_db_name = os.getenv("POSTGRES_DB", "hexis_memory")
    env_db_user = os.getenv("POSTGRES_USER", "postgres")
    env_db_password = os.getenv("POSTGRES_PASSWORD", "password")

    # LLM options
    parser.add_argument('--endpoint', '-e', default='http://localhost:11434/v1',
                        help='OpenAI-compatible LLM endpoint')
    parser.add_argument('--model', '-m', default='llama3.2',
                        help='Model name to use')
    parser.add_argument('--api-key', default='not-needed',
                        help='API key for the LLM endpoint')
    
    # Database options
    parser.add_argument('--db-host', default=env_db_host, help='Database host')
    parser.add_argument('--db-port', type=int, default=env_db_port, help='Database port')
    parser.add_argument('--db-name', default=env_db_name, help='Database name')
    parser.add_argument('--db-user', default=env_db_user, help='Database user')
    parser.add_argument('--db-password', default=env_db_password, help='Database password')
    
    # Memory options
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of memories to retrieve for enrichment')
    parser.add_argument('--no-auto-memory', action='store_true',
                        help='Disable automatic memory formation')
    parser.add_argument('--max-tool-iterations', type=int, default=5,
                        help='Maximum tool call iterations per response')
    
    # Display options
    parser.add_argument('--hide-memories', action='store_true',
                        help='Hide retrieved memories display')
    parser.add_argument('--hide-tool-calls', action='store_true',
                        help='Hide tool call display')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    config = ConversationConfig(
        llm_endpoint=args.endpoint,
        llm_model=args.model,
        llm_api_key=args.api_key,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        enrichment_top_k=args.top_k,
        auto_form_memories=not args.no_auto_memory,
        max_tool_iterations=args.max_tool_iterations,
        show_memories=not args.hide_memories,
        show_tool_calls=not args.hide_tool_calls,
        verbose=not args.quiet,
    )
    
    run_interactive(config)


if __name__ == "__main__":
    main()
