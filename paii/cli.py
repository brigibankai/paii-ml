"""
CLI entry point for PAII Systems.
"""

import argparse
import logging
import sys
from typing import Optional

from paii.app import PAIISystem
import paii.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="PAII Systems: Local-first vector search"
    )
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add text to the index")
    add_parser.add_argument(
        "text",
        type=str,
        help="Text to add"
    )
    add_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source metadata"
    )
    
    # PDF command
    pdf_parser = subparsers.add_parser("pdf", help="Ingest a PDF file")
    pdf_parser.add_argument(
        "path",
        type=str,
        help="Path to PDF file"
    )
    pdf_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source name (default: filename)"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument(
        "query",
        type=str,
        help="Query text"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)"
    )
    
    # Info command
    subparsers.add_parser("info", help="Show system information")
    
    # Global options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Embedding provider (default: local)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (local: all-MiniLM-L6-v2, openai: text-embedding-3-small)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create system
    embedding_kwargs = {}
    if args.model:
        if args.provider == "local":
            embedding_kwargs["model_name"] = args.model
        else:
            embedding_kwargs["model"] = args.model
    
    if args.api_key:
        embedding_kwargs["api_key"] = args.api_key
    
    system = PAIISystem(embedding_provider=args.provider, **embedding_kwargs)
    
    # Handle commands
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "add":
            entry_id = system.add(args.text, metadata={"source": args.source or "cli"})
            print(f"✅ Added entry {entry_id}")
        
        elif args.command == "pdf":
            num_chunks = system.ingest_pdf(args.path, source_name=args.source)
            print(f"✅ Ingested {num_chunks} chunks from PDF")
        
        elif args.command == "search":
            results = system.search(args.query, top_k=args.top_k)
            if not results:
                print("❌ No results found")
            else:
                print(f"\nFound {len(results)} results:\n")
                for i, result in enumerate(results, 1):
                    print(f"{i}. [{result.score:.3f}] {result.text[:100]}...")
                    if result.metadata:
                        print(f"   Metadata: {result.metadata}")
        
        elif args.command == "info":
            info = system.info()
            print("\nPAII System Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
