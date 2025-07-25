#!/usr/bin/env python3
"""
Enhanced RAG System - Main CLI Interface
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.pipeline import RAGPipeline, create_pipeline, quick_setup
from src.utils.config import ConfigManager, get_config
from src.utils.logging import setup_logging, get_logger
from src.utils.errors import RAGError, format_error_response


def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG System - Process documents and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a document
  python main.py process https://arxiv.org/pdf/2101.00001.pdf
  
  # Start interactive query mode
  python main.py query --interactive
  
  # Query with specific document
  python main.py query "What is the main contribution?" --load-index ./data/indices/auto_save
  
  # Start API server
  python main.py serve --port 8000
  
  # Show system statistics
  python main.py stats
        """
    )
    
    # Global options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output (errors only)')
    parser.add_argument('--output-format', choices=['text', 'json'], default='text',
                       help='Output format')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a document')
    process_parser.add_argument('source', help='Document source (URL or file path)')
    process_parser.add_argument('--document-id', help='Custom document ID')
    process_parser.add_argument('--split-strategy', choices=['fixed_size', 'semantic', 'section_aware', 'adaptive'],
                               help='Document splitting strategy')
    process_parser.add_argument('--save-index', type=str, help='Save index to specified path')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', nargs='?', help='Question to ask')
    query_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    query_parser.add_argument('--load-index', type=str, help='Load index from specified path')
    query_parser.add_argument('--k', type=int, help='Number of documents to retrieve')
    query_parser.add_argument('--strategy', choices=['basic', 'enhanced'], default='enhanced',
                             help='Query strategy')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, help='Server host')
    serve_parser.add_argument('--port', type=int, help='Server port')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    stats_parser.add_argument('--load-index', type=str, help='Load index from specified path')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    config_subparsers.add_parser('show', help='Show current configuration')
    config_subparsers.add_parser('validate', help='Validate configuration')
    
    config_init_parser = config_subparsers.add_parser('init', help='Initialize default configuration')
    config_init_parser.add_argument('--force', action='store_true', help='Overwrite existing config')
    
    return parser


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Setup logging based on CLI arguments"""
    log_level = "INFO"
    
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    
    setup_logging({
        'level': log_level,
        'format': 'simple' if args.output_format == 'text' else 'structured',
        'output': 'console'
    })


def print_output(data: Any, format_type: str = 'text') -> None:
    """Print output in specified format"""
    if format_type == 'json':
        print(json.dumps(data, indent=2, default=str))
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print(data)


def handle_process_command(args: argparse.Namespace) -> int:
    """Handle document processing command"""
    try:
        logger = get_logger("cli")
        logger.info("Starting document processing", source=args.source)
        
        # Create pipeline
        pipeline = create_pipeline(args.config)
        
        # Process document
        from src.core.splitter import SplitStrategy
        split_strategy = None
        if args.split_strategy:
            split_strategy = SplitStrategy(args.split_strategy)
        
        result = pipeline.process_document(
            source=args.source,
            document_id=args.document_id,
            split_strategy=split_strategy
        )
        
        if result.success:
            output_data = {
                "status": "success",
                "document_id": result.document_id,
                "chunks": result.chunk_count,
                "processing_time_ms": result.processing_time_ms,
                "title": result.metadata.title if result.metadata else "Unknown",
                "authors": result.metadata.authors if result.metadata else []
            }
            
            # Save index if requested
            if args.save_index:
                success = pipeline.save_index(args.save_index)
                output_data["index_saved"] = success
                output_data["index_path"] = args.save_index
            
            print_output(output_data, args.output_format)
            return 0
        else:
            error_data = {
                "status": "error",
                "error": result.error_message,
                "processing_time_ms": result.processing_time_ms
            }
            print_output(error_data, args.output_format)
            return 1
            
    except RAGError as e:
        print_output(format_error_response(e), args.output_format)
        return 1
    except Exception as e:
        print_output({"status": "error", "error": str(e)}, args.output_format)
        return 1


def handle_query_command(args: argparse.Namespace) -> int:
    """Handle query command"""
    try:
        logger = get_logger("cli")
        
        # Create pipeline
        pipeline = create_pipeline(args.config)
        
        # Load index if specified
        if args.load_index:
            success = pipeline.load_index(args.load_index)
            if not success:
                print_output({"status": "error", "error": "Failed to load index"}, args.output_format)
                return 1
        
        if args.interactive:
            return run_interactive_query(pipeline, args)
        else:
            if not args.question:
                print("Error: Question required for non-interactive mode")
                return 1
            
            return run_single_query(pipeline, args.question, args)
            
    except RAGError as e:
        print_output(format_error_response(e), args.output_format)
        return 1
    except Exception as e:
        print_output({"status": "error", "error": str(e)}, args.output_format)
        return 1


def run_single_query(pipeline: RAGPipeline, question: str, args: argparse.Namespace) -> int:
    """Run a single query"""
    try:
        result = pipeline.query(
            question=question,
            k=args.k,
            strategy=args.strategy
        )
        
        if args.output_format == 'json':
            output_data = {
                "question": question,
                "answer": result.answer,
                "sources": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} 
                           for doc in result.sources],
                "metadata": result.metadata,
                "processing_time_ms": result.processing_time_ms,
                "tokens_used": result.tokens_used,
                "cost_usd": result.cost_usd
            }
            print_output(output_data, args.output_format)
        else:
            print(f"\nQuestion: {question}")
            print(f"Answer: {result.answer}")
            print(f"\nSources: {len(result.sources)} documents")
            print(f"Processing time: {result.processing_time_ms}ms")
            print(f"Tokens used: {result.tokens_used}")
            print(f"Cost: ${result.cost_usd:.4f}")
        
        return 0
        
    except Exception as e:
        print_output({"status": "error", "error": str(e)}, args.output_format)
        return 1


def run_interactive_query(pipeline: RAGPipeline, args: argparse.Namespace) -> int:
    """Run interactive query mode"""
    print("Enhanced RAG System - Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help")
                print("  stats - Show system statistics")
                print("  clear - Clear screen")
                print("  quit/exit - Exit interactive mode")
                print("  Or just ask any question!")
                continue
            
            if question.lower() == 'stats':
                stats = pipeline.get_stats()
                print_output(stats, 'text')
                continue
            
            if question.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            if not question:
                continue
            
            # Process query
            print("Processing...")
            result = pipeline.query(
                question=question,
                k=args.k,
                strategy=args.strategy
            )
            
            print(f"\nAnswer: {result.answer}")
            print(f"\nSources: {len(result.sources)} documents")
            print(f"Time: {result.processing_time_ms}ms | Tokens: {result.tokens_used} | Cost: ${result.cost_usd:.4f}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return 0


def handle_serve_command(args: argparse.Namespace) -> int:
    """Handle API server command"""
    try:
        # Import here to avoid dependency issues if FastAPI not installed
        from src.api.server import create_app
        import uvicorn
        
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
        
        # Override with CLI args
        host = args.host or config.api.host
        port = args.port or config.api.port
        
        # Create FastAPI app
        app = create_app(config)
        
        print(f"Starting RAG API server at http://{host}:{port}")
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=args.reload,
            log_level="info"
        )
        
        return 0
        
    except ImportError:
        print("Error: FastAPI and uvicorn required for API server")
        print("Install with: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        return 1


def handle_stats_command(args: argparse.Namespace) -> int:
    """Handle stats command"""
    try:
        pipeline = create_pipeline(args.config)
        
        # Load index if specified
        if args.load_index:
            pipeline.load_index(args.load_index)
        
        stats = pipeline.get_stats()
        print_output(stats, args.output_format)
        
        return 0
        
    except Exception as e:
        print_output({"status": "error", "error": str(e)}, args.output_format)
        return 1


def handle_config_command(args: argparse.Namespace) -> int:
    """Handle configuration commands"""
    try:
        config_manager = ConfigManager(args.config)
        
        if args.config_action == 'show':
            config = config_manager.get_config()
            config_dict = config_manager._config_to_dict(config)
            print_output(config_dict, args.output_format)
        
        elif args.config_action == 'validate':
            config = config_manager.load_config()  # This validates
            print_output({"status": "valid", "message": "Configuration is valid"}, args.output_format)
        
        elif args.config_action == 'init':
            config_path = Path(args.config or './config/config.yaml')
            
            if config_path.exists() and not args.force:
                print(f"Configuration file already exists: {config_path}")
                print("Use --force to overwrite")
                return 1
            
            config_manager._create_default_config(config_path)
            print(f"Default configuration created: {config_path}")
        
        return 0
        
    except Exception as e:
        print_output({"status": "error", "error": str(e)}, args.output_format)
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging_from_args(args)
    
    # Handle commands
    if args.command == 'process':
        return handle_process_command(args)
    elif args.command == 'query':
        return handle_query_command(args)
    elif args.command == 'serve':
        return handle_serve_command(args)
    elif args.command == 'stats':
        return handle_stats_command(args)
    elif args.command == 'config':
        return handle_config_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())