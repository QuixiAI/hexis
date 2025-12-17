#!/usr/bin/env python3
"""
AGI Memory Ingestion Pipeline

Ingests documents (markdown, PDF, code, text) and converts them into
structured memories using an LLM for analysis and classification.

Usage:
    python ingest.py --input ./documents --endpoint http://localhost:11434/v1
    python ingest.py --file document.pdf --endpoint http://localhost:8000/v1
"""

import argparse
import json
import hashlib
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Generator
from enum import Enum
import mimetypes

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--break-system-packages", "-q"])
    import requests

try:
    from openrouter import OpenRouter
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False

from datetime import datetime, timezone

from cognitive_memory_api import CognitiveMemorySync, MemoryInput as ApiMemoryInput, MemoryType as ApiMemoryType


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Pipeline configuration."""
    # LLM Settings
    llm_endpoint: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str = "not-needed"  # For local endpoints
    llm_provider: str = "openai"
    
    # Database Settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "agi_memory"
    db_user: str = "postgres"
    db_password: str = "password"
    
    # Chunking Settings
    chunk_size: int = 2000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    
    # Processing Settings
    batch_size: int = 5  # Chunks to process before committing
    verbose: bool = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    STRATEGIC = "strategic"


@dataclass
class ExtractedMemory:
    """A memory extracted from a document chunk."""
    memory_type: MemoryType
    content: str
    importance: float = 0.5
    
    # Type-specific fields
    # Episodic
    emotional_valence: Optional[float] = None
    context: Optional[dict] = None
    
    # Semantic
    confidence: Optional[float] = None
    category: Optional[list] = None
    related_concepts: Optional[list] = None
    
    # Procedural
    steps: Optional[list] = None
    prerequisites: Optional[list] = None
    
    # Strategic
    pattern_description: Optional[str] = None
    context_applicability: Optional[dict] = None
    
    # Metadata
    source_file: str = ""
    source_chunk: int = 0
    concepts: list = field(default_factory=list)
    relationships: list = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A chunk of a document for processing."""
    content: str
    source_file: str
    chunk_index: int
    total_chunks: int
    file_type: str
    metadata: dict = field(default_factory=dict)


# ============================================================================
# DOCUMENT READERS
# ============================================================================

class DocumentReader:
    """Base class for document readers."""
    
    @staticmethod
    def read(file_path: Path) -> str:
        raise NotImplementedError


class MarkdownReader(DocumentReader):
    """Reads markdown files."""
    
    @staticmethod
    def read(file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


class TextReader(DocumentReader):
    """Reads plain text files."""
    
    @staticmethod
    def read(file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


class CodeReader(DocumentReader):
    """Reads code files with language detection."""
    
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript-react',
        '.tsx': 'typescript-react',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c-header',
        '.hpp': 'cpp-header',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.ps1': 'powershell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
    }
    
    @classmethod
    def read(cls, file_path: Path) -> str:
        language = cls.LANGUAGE_MAP.get(file_path.suffix.lower(), 'unknown')
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return f"[Language: {language}]\n[File: {file_path.name}]\n\n{content}"


class PDFReader(DocumentReader):
    """Reads PDF files."""
    
    @staticmethod
    def read(file_path: Path) -> str:
        try:
            import pdfplumber
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "--break-system-packages", "-q"])
            import pdfplumber
        
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {i + 1}]\n{page_text}")
                
                # Also extract tables
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table:
                        table_str = "\n".join([" | ".join(str(cell) if cell else "" for cell in row) for row in table])
                        text_parts.append(f"[Table {j + 1} on Page {i + 1}]\n{table_str}")
        
        return "\n\n".join(text_parts)


def get_reader(file_path: Path) -> DocumentReader:
    """Get the appropriate reader for a file type."""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return PDFReader()
    elif suffix in ['.md', '.markdown']:
        return MarkdownReader()
    elif suffix in CodeReader.LANGUAGE_MAP:
        return CodeReader()
    else:
        return TextReader()


# ============================================================================
# CHUNKING
# ============================================================================

class SmartChunker:
    """
    Intelligently chunks documents while respecting structure.
    """
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: str, file_path: Path) -> Generator[DocumentChunk, None, None]:
        """Chunk a document based on its type."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.md', '.markdown']:
            chunks = self._chunk_markdown(content)
        elif suffix in CodeReader.LANGUAGE_MAP:
            chunks = self._chunk_code(content)
        else:
            chunks = self._chunk_text(content)
        
        total = len(chunks)
        for i, chunk_content in enumerate(chunks):
            yield DocumentChunk(
                content=chunk_content,
                source_file=str(file_path),
                chunk_index=i,
                total_chunks=total,
                file_type=suffix,
                metadata={
                    'filename': file_path.name,
                    'size_bytes': len(content.encode('utf-8')),
                }
            )
    
    def _chunk_markdown(self, content: str) -> list[str]:
        """Chunk markdown by headers and sections."""
        # Split by headers
        header_pattern = r'^(#{1,6}\s+.+)$'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        for i, section in enumerate(sections):
            if re.match(header_pattern, section):
                current_header = section
                continue
            
            section_with_header = f"{current_header}\n{section}" if current_header else section
            
            if len(current_chunk) + len(section_with_header) <= self.chunk_size:
                current_chunk += section_with_header
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If section itself is too large, split it
                if len(section_with_header) > self.chunk_size:
                    sub_chunks = self._chunk_text(section_with_header)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = section_with_header
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
    def _chunk_code(self, content: str) -> list[str]:
        """Chunk code by functions/classes."""
        # Try to split on function/class definitions
        # This is a simplified approach - could be enhanced with AST parsing
        
        patterns = [
            r'^(def\s+\w+.*?(?=\ndef\s+|\nclass\s+|\Z))',  # Python functions
            r'^(class\s+\w+.*?(?=\nclass\s+|\Z))',  # Python classes
            r'^(function\s+\w+.*?(?=\nfunction\s+|\Z))',  # JS functions
            r'^(const\s+\w+\s*=\s*(?:async\s*)?\(.*?\)\s*=>.*?(?=\nconst\s+|\Z))',  # Arrow functions
        ]
        
        # If we can't find good split points, fall back to text chunking
        chunks = self._chunk_text(content)
        return chunks
    
    def _chunk_text(self, content: str) -> list[str]:
        """Simple text chunking with overlap."""
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        
        # Try to split on paragraph boundaries
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle paragraphs larger than chunk_size
                if len(para) > self.chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Add overlap
        if self.overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk
                    prev_overlap = chunks[i-1][-self.overlap:]
                    chunk = f"...{prev_overlap}\n\n{chunk}"
                overlapped_chunks.append(chunk)
            return overlapped_chunks
        
        return chunks


# ============================================================================
# LLM INTERFACE
# ============================================================================

class LLMClient:
    """Client for OpenAI-compatible LLM endpoints."""
    
    def __init__(self, config: Config):
        self.config = config
        self.endpoint = config.llm_endpoint.rstrip('/')
        self.openrouter_client = None
        if self.config.llm_provider == "openrouter":
            if HAS_OPENROUTER:
                 self.openrouter_client = OpenRouter(api_key=self.config.llm_api_key)
            else:
                 print("Warning: OpenRouter provider selected but openrouter package not installed. Falling back to generic HTTP.")

    def complete(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Send a chat completion request."""
        if self.config.llm_provider == "openrouter" and self.openrouter_client:
            response = self.openrouter_client.chat.send(
                model=self.config.llm_model,
                messages=messages,
                # temperature=temperature # SDK might not support this arg directly in send, need to check
            )
            # Handle response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            elif isinstance(response, dict):
                 if 'choices' in response and len(response['choices']) > 0:
                     return response['choices'][0]['message']['content']
                 elif 'message' in response:
                     return response['message']['content']
            return str(response)

        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        
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
        
        return response.json()["choices"][0]["message"]["content"]


# ============================================================================
# MEMORY EXTRACTION PROMPTS
# ============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing documents and extracting structured memories for an AI memory system.

Your task is to analyze a document chunk and extract memories that would be valuable for an AI to recall later. You must output valid JSON only, no other text.

The memory types are:
1. EPISODIC - Events, experiences, narratives, things that happened
2. SEMANTIC - Facts, knowledge, definitions, information with a confidence level
3. PROCEDURAL - How-to knowledge, steps, processes, instructions
4. STRATEGIC - Patterns, meta-knowledge, lessons learned, heuristics

For each memory, assess:
- importance: 0.0-1.0 (how valuable is this to remember?)
- concepts: key concepts/entities this memory relates to
- relationships: how this memory might connect to other concepts

Output format (JSON array):
```json
{
  "memories": [
    {
      "type": "semantic|episodic|procedural|strategic",
      "content": "Clear, standalone memory content that makes sense without the original document",
      "importance": 0.0-1.0,
      "concepts": ["concept1", "concept2"],
      
      // For SEMANTIC memories:
      "confidence": 0.0-1.0,
      "category": ["category1", "category2"],
      "related_concepts": ["related1", "related2"],
      
      // For EPISODIC memories:
      "emotional_valence": -1.0 to 1.0,
      "context": {"when": "...", "where": "...", "who": "..."},
      
      // For PROCEDURAL memories:
      "steps": ["step1", "step2", "step3"],
      "prerequisites": ["prereq1", "prereq2"],
      
      // For STRATEGIC memories:
      "pattern_description": "Description of the pattern or heuristic",
      "context_applicability": {"domains": [...], "conditions": [...]}
    }
  ],
  "cross_references": [
    {"from_concept": "X", "to_concept": "Y", "relationship": "CAUSES|SUPPORTS|CONTRADICTS|ASSOCIATED"}
  ]
}
```

Be selective - not every sentence needs to become a memory. Focus on:
- Key facts and knowledge
- Important procedures and how-to information
- Significant events or experiences
- Valuable patterns and insights

Make each memory self-contained - it should make sense when recalled without the original document context."""


def create_analysis_prompt(chunk: DocumentChunk) -> str:
    """Create the analysis prompt for a document chunk."""
    return f"""Analyze this document chunk and extract structured memories.

SOURCE: {chunk.source_file}
TYPE: {chunk.file_type}
CHUNK: {chunk.chunk_index + 1} of {chunk.total_chunks}

CONTENT:
{chunk.content}

Extract all valuable memories from this content. Output valid JSON only."""


# ============================================================================
# MEMORY EXTRACTION
# ============================================================================

class MemoryExtractor:
    """Extracts memories from document chunks using an LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
    
    def extract(self, chunk: DocumentChunk) -> list[ExtractedMemory]:
        """Extract memories from a document chunk."""
        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": create_analysis_prompt(chunk)}
        ]
        
        try:
            response = self.llm.complete(messages)
            memories = self._parse_response(response, chunk)
            return memories
        except Exception as e:
            if self.config.verbose:
                print(f"  Warning: Failed to extract memories from chunk {chunk.chunk_index}: {e}")
            return []
    
    def _parse_response(self, response: str, chunk: DocumentChunk) -> list[ExtractedMemory]:
        """Parse the LLM response into ExtractedMemory objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        # Clean up common issues
        json_str = json_str.strip()
        if not json_str.startswith('{'):
            # Try to find the JSON object
            start = json_str.find('{')
            if start != -1:
                json_str = json_str[start:]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.config.verbose:
                print(f"  Warning: Failed to parse JSON: {e}")
            return []
        
        memories = []
        for mem_data in data.get("memories", []):
            try:
                memory_type = MemoryType(mem_data.get("type", "semantic").lower())
                
                memory = ExtractedMemory(
                    memory_type=memory_type,
                    content=mem_data.get("content", ""),
                    importance=float(mem_data.get("importance", 0.5)),
                    source_file=chunk.source_file,
                    source_chunk=chunk.chunk_index,
                    concepts=mem_data.get("concepts", []),
                )
                
                # Type-specific fields
                if memory_type == MemoryType.SEMANTIC:
                    memory.confidence = float(mem_data.get("confidence", 0.8))
                    memory.category = mem_data.get("category", [])
                    memory.related_concepts = mem_data.get("related_concepts", [])
                
                elif memory_type == MemoryType.EPISODIC:
                    memory.emotional_valence = float(mem_data.get("emotional_valence", 0.0))
                    memory.context = mem_data.get("context", {})
                
                elif memory_type == MemoryType.PROCEDURAL:
                    memory.steps = mem_data.get("steps", [])
                    memory.prerequisites = mem_data.get("prerequisites", [])
                
                elif memory_type == MemoryType.STRATEGIC:
                    memory.pattern_description = mem_data.get("pattern_description", "")
                    memory.context_applicability = mem_data.get("context_applicability", {})
                
                # Store cross-references
                memory.relationships = data.get("cross_references", [])
                
                if memory.content:  # Only add if there's actual content
                    memories.append(memory)
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"  Warning: Failed to parse memory: {e}")
                continue
        
        return memories


# ============================================================================
# DATABASE STORAGE
# ============================================================================

class MemoryStore:
    """Stores extracted memories in Postgres via the core CognitiveMemory API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: CognitiveMemorySync | None = None
    
    def connect(self):
        """Connect to the database (sync wrapper around asyncpg pool)."""
        if self.client is not None:
            return
        dsn = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
        self.client = CognitiveMemorySync.connect(dsn, min_size=1, max_size=5)
    
    def close(self):
        """Close the database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
    
    def store_memory(self, memory: ExtractedMemory) -> Optional[str]:
        ids = self.store_memories([memory])
        return str(ids[0]) if ids else None

    def store_memories(self, memories: list[ExtractedMemory]) -> list[str]:
        """Store a batch of extracted memories and return their IDs."""
        if not memories:
            return []
        if self.client is None:
            self.connect()
        assert self.client is not None

        now = datetime.now(timezone.utc).isoformat()
        items: list[tuple[ExtractedMemory, str]] = []
        for m in memories:
            h = hashlib.sha256(f"{m.memory_type.value}\n{m.content}".encode("utf-8")).hexdigest()
            items.append((m, h))

        # Idempotency: skip already-ingested (source_file + content_hash).
        existing: dict[tuple[str, str], str] = {}
        by_source: dict[str, list[str]] = {}
        for m, h in items:
            by_source.setdefault(m.source_file, []).append(h)
        for src, hashes in by_source.items():
            try:
                receipts = self.client.get_ingestion_receipts(src, hashes)
            except Exception:
                receipts = {}
            for hh, mid in receipts.items():
                existing[(src, hh)] = str(mid)

        inputs: list[ApiMemoryInput] = []
        receipt_rows: list[dict] = []
        for m, h in items:
            if (m.source_file, h) in existing:
                continue
            api_type = ApiMemoryType(m.memory_type.value)
            source_ref = {
                "kind": "document",
                "ref": m.source_file,
                "label": f"{Path(m.source_file).name}#chunk{m.source_chunk}",
                "observed_at": now,
                "trust": 0.7,
                "content_hash": h,
            }

            context: Optional[dict] = None
            if api_type == ApiMemoryType.EPISODIC:
                context = {
                    "type": "ingest",
                    "source_file": m.source_file,
                    "source_chunk": m.source_chunk,
                    "extracted": m.context or {},
                }
            elif api_type == ApiMemoryType.PROCEDURAL:
                context = {"steps": m.steps or []}
            elif api_type == ApiMemoryType.STRATEGIC:
                context = {"pattern_description": m.pattern_description, "context_applicability": m.context_applicability}

            inputs.append(
                ApiMemoryInput(
                    content=m.content,
                    type=api_type,
                    importance=m.importance,
                    emotional_valence=float(m.emotional_valence or 0.0),
                    context=context,
                    concepts=[str(c).strip().lower() for c in (m.concepts or []) if str(c).strip()],
                    source_attribution=source_ref,
                    source_references=[source_ref] if api_type == ApiMemoryType.SEMANTIC else None,
                )
            )
            receipt_rows.append({"source_file": m.source_file, "chunk_index": int(m.source_chunk), "content_hash": h})

        if not inputs:
            return []

        ids = self.client.remember_batch(inputs)
        created = [str(i) for i in ids]

        # Record receipts best-effort; failures should not fail ingestion after commit.
        try:
            for row, mid in zip(receipt_rows, created):
                row["memory_id"] = mid
            self.client.record_ingestion_receipts(receipt_rows)
        except Exception:
            pass

        return created
    
    def commit(self):
        """Compatibility no-op (writes are committed per statement in asyncpg)."""
    
    def rollback(self):
        """Compatibility no-op (writes are committed per statement in asyncpg)."""


# ============================================================================
# INGESTION PIPELINE
# ============================================================================

class IngestionPipeline:
    """Main ingestion pipeline orchestrator."""
    
    SUPPORTED_EXTENSIONS = {
        '.md', '.markdown',  # Markdown
        '.txt', '.text',  # Plain text
        '.pdf',  # PDF
        '.py', '.js', '.ts', '.jsx', '.tsx',  # Code
        '.java', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php', '.swift',
        '.kt', '.scala', '.r', '.sql',
        '.sh', '.bash', '.zsh', '.ps1',
        '.yaml', '.yml', '.json', '.xml',
        '.html', '.css', '.scss', '.less',
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.chunker = SmartChunker(config.chunk_size, config.chunk_overlap)
        self.extractor = MemoryExtractor(config)
        self.store = MemoryStore(config)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_processed': 0,
            'memories_created': 0,
            'errors': 0,
        }
    
    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file. Returns number of memories created."""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 0
        
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {file_path.suffix}")
            return 0
        
        if self.config.verbose:
            print(f"\nProcessing: {file_path}")
        
        # Read the document
        reader = get_reader(file_path)
        try:
            content = reader.read(file_path)
        except Exception as e:
            print(f"  Error reading file: {e}")
            self.stats['errors'] += 1
            return 0
        
        if self.config.verbose:
            print(f"  Read {len(content)} characters")
        
        # Chunk the document
        chunks = list(self.chunker.chunk(content, file_path))
        if self.config.verbose:
            print(f"  Split into {len(chunks)} chunks")
        
        # Process each chunk
        memories_created = 0
        for i, chunk in enumerate(chunks):
            if self.config.verbose:
                print(f"  Processing chunk {i + 1}/{len(chunks)}...", end=" ")
            
            # Extract memories
            memories = self.extractor.extract(chunk)
            
            if self.config.verbose:
                print(f"extracted {len(memories)} memories")
            
            # Store memories (batched for fewer DB round-trips)
            try:
                created_ids = self.store.store_memories(memories)
            except Exception as e:
                self.stats["errors"] += 1
                if self.config.verbose:
                    print(f"    Error storing batch: {e}")
                created_ids = []

            memories_created += len(created_ids)
            if self.config.verbose:
                for memory, memory_id in zip(memories, created_ids):
                    print(f"    + [{memory.memory_type.value}] {memory.content[:60]}... ({memory_id[:8]}...)")
            
            self.stats['chunks_processed'] += 1
            
            # Commit periodically
            if (i + 1) % self.config.batch_size == 0:
                self.store.commit()
        
        # Final commit for this file
        self.store.commit()
        
        self.stats['files_processed'] += 1
        self.stats['memories_created'] += memories_created
        
        if self.config.verbose:
            print(f"  Created {memories_created} memories from {file_path.name}")
        
        return memories_created
    
    def ingest_directory(self, dir_path: Path, recursive: bool = True) -> int:
        """Ingest all supported files in a directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Directory not found: {dir_path}")
            return 0
        
        pattern = '**/*' if recursive else '*'
        files = [f for f in dir_path.glob(pattern) 
                 if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        
        if self.config.verbose:
            print(f"Found {len(files)} files to process")
        
        total_memories = 0
        for file_path in files:
            total_memories += self.ingest_file(file_path)
        
        return total_memories
    
    def print_stats(self):
        """Print ingestion statistics."""
        print("\n" + "=" * 50)
        print("INGESTION COMPLETE")
        print("=" * 50)
        print(f"Files processed:   {self.stats['files_processed']}")
        print(f"Chunks processed:  {self.stats['chunks_processed']}")
        print(f"Memories created:  {self.stats['memories_created']}")
        print(f"Errors:            {self.stats['errors']}")
        print("=" * 50)
    
    def close(self):
        """Clean up resources."""
        self.store.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AGI Memory Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single file
  python ingest.py --file document.pdf
  
  # Ingest a directory
  python ingest.py --input ./documents
  
  # Use a specific LLM endpoint
  python ingest.py --input ./docs --endpoint http://localhost:8000/v1 --model mistral
  
  # Custom database connection
  python ingest.py --file doc.md --db-host localhost --db-name my_memory
        """
    )
    
    env_db_host = os.getenv("POSTGRES_HOST", "localhost")
    env_db_port_raw = os.getenv("POSTGRES_PORT")
    try:
        env_db_port = int(env_db_port_raw) if env_db_port_raw else 5432
    except ValueError:
        env_db_port = 5432
    env_db_name = os.getenv("POSTGRES_DB", "agi_memory")
    env_db_user = os.getenv("POSTGRES_USER", "postgres")
    env_db_password = os.getenv("POSTGRES_PASSWORD", "password")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', type=Path, help='Single file to ingest')
    input_group.add_argument('--input', '-i', type=Path, help='Directory to ingest')
    
    # LLM options
    parser.add_argument('--endpoint', '-e', default='http://localhost:11434/v1',
                        help='OpenAI-compatible LLM endpoint (default: http://localhost:11434/v1)')
    parser.add_argument('--model', '-m', default='llama3.2',
                        help='Model name to use (default: llama3.2)')
    parser.add_argument('--api-key', default='not-needed',
                        help='API key for the LLM endpoint')
    parser.add_argument('--provider', default='openai',
                        help='LLM provider: openai, ollama, openrouter')
    
    # Database options
    parser.add_argument('--db-host', default=env_db_host, help='Database host')
    parser.add_argument('--db-port', type=int, default=env_db_port, help='Database port')
    parser.add_argument('--db-name', default=env_db_name, help='Database name')
    parser.add_argument('--db-user', default=env_db_user, help='Database user')
    parser.add_argument('--db-password', default=env_db_password, help='Database password')
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Target chunk size in characters (default: 2000)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not recursively process directories')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Build config
    config = Config(
        llm_endpoint=args.endpoint,
        llm_model=args.model,
        llm_api_key=args.api_key,
        llm_provider=args.provider,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )
    
    # Run pipeline
    pipeline = IngestionPipeline(config)
    
    try:
        if args.file:
            pipeline.ingest_file(args.file)
        else:
            pipeline.ingest_directory(args.input, recursive=not args.no_recursive)
        
        pipeline.print_stats()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        pipeline.store.rollback()
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
