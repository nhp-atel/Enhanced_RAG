"""
Command-line interface for the RAG system with rich progress indicators.
"""
import click
import asyncio
from pathlib import Path
from typing import Optional, List
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
import json

from config_example import RAGConfig, get_config
from enhanced_document_processor import process_research_paper
from langchain_openai import ChatOpenAI

console = Console()
logger = logging.getLogger(__name__)


class RAGSystemCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            model=self.config.chat_model,
            temperature=self.config.temperature
        )
        self.rag_system = None  # Initialize lazily
    
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
🚀 Enhanced RAG System CLI
Universal Research Paper Analysis with Dynamic Processing
        """
        console.print(Panel(welcome_text, style="bold blue"))
    
    def process_paper_interactive(
        self, 
        source: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """Interactive paper processing with progress indicators."""
        if not source:
            source = Prompt.ask(
                "Enter PDF source (URL or local path)",
                default="https://arxiv.org/pdf/2101.00001"
            )
        
        if not output_dir:
            output_dir = Path(Prompt.ask(
                "Output directory", 
                default="./processed_papers"
            ))
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Main processing task
            main_task = progress.add_task("Processing paper...", total=100)
            
            try:
                # Step 1: Download/Load PDF
                progress.update(main_task, description="📥 Loading PDF...", advance=10)
                # Your PDF loading logic here
                
                # Step 2: Extract metadata
                progress.update(main_task, description="🔍 Extracting metadata...", advance=20)
                # Your metadata extraction logic here
                
                # Step 3: Generate summary
                progress.update(main_task, description="📝 Generating summary...", advance=25)
                # Your summary generation logic here
                
                # Step 4: Extract concepts
                progress.update(main_task, description="🧠 Extracting concepts...", advance=20)
                # Your concept extraction logic here
                
                # Step 5: Create embeddings
                progress.update(main_task, description="🔮 Creating embeddings...", advance=15)
                # Your embedding creation logic here
                
                # Step 6: Build vector store
                progress.update(main_task, description="🗃️ Building vector store...", advance=10)
                # Your vector store creation logic here
                
                progress.update(main_task, description="✅ Processing complete!", advance=0)
                
                # Display results
                self.display_processing_results(source, output_dir)
                
            except Exception as e:
                progress.update(main_task, description=f"❌ Error: {str(e)}")
                console.print(f"[red]Processing failed: {e}[/red]")
                raise
    
    def display_processing_results(self, source: str, output_dir: Path):
        """Display processing results in a formatted table."""
        table = Table(title="Processing Results")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        # Mock results - replace with actual data
        table.add_row("PDF Source", "✅ Loaded", source)
        table.add_row("Metadata", "✅ Extracted", "Title, Authors, Date identified")
        table.add_row("Summary", "✅ Generated", "1,247 characters")
        table.add_row("Concepts", "✅ Extracted", "23 technical terms, 15 key concepts")
        table.add_row("Embeddings", "✅ Created", "156 document chunks + 38 concept embeddings")
        table.add_row("Vector Store", "✅ Built", "FAISS index with 194 total vectors")
        
        console.print(table)
    
    def interactive_query_session(self):
        """Start interactive query session."""
        if not self.rag_system:
            console.print("[red]No papers processed yet. Please process a paper first.[/red]")
            return
        
        console.print("\n[bold green]🤖 Interactive Query Session Started[/bold green]")
        console.print("Type 'quit', 'exit', or 'q' to end the session.\n")
        
        while True:
            try:
                question = Prompt.ask("[bold blue]Your question")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                # Process query with spinner
                with Progress(
                    SpinnerColumn(),
                    TextColumn("🔍 Searching for relevant information..."),
                    console=console
                ) as progress:
                    search_task = progress.add_task("", total=None)
                    
                    # Your query processing logic here
                    # answer = self.rag_system.query(question)
                    answer = f"Mock answer for: {question}"  # Replace with actual logic
                
                # Display answer
                answer_panel = Panel(
                    answer,
                    title="[bold green]📝 Answer[/bold green]",
                    border_style="green"
                )
                console.print(answer_panel)
                console.print()  # Add spacing
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error processing query: {e}[/red]")
    
    def show_system_status(self):
        """Display system status and configuration."""
        status_table = Table(title="🔧 System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="yellow")
        
        # Configuration status
        status_table.add_row("Configuration", "✅ Loaded", f"Model: {self.config.chat_model}")
        status_table.add_row("OpenAI API", "✅ Connected", "API key configured")
        status_table.add_row("Vector Store", "📁 Ready", f"Path: {self.config.vector_store_path}")
        status_table.add_row("Cache", 
                           "✅ Enabled" if self.config.enable_caching else "❌ Disabled",
                           f"TTL: {self.config.cache_ttl_seconds}s")
        
        console.print(status_table)
        
        # Display current configuration
        config_panel = Panel(
            self._format_config_display(),
            title="[bold blue]📋 Current Configuration[/bold blue]",
            border_style="blue"
        )
        console.print(config_panel)
    
    def _format_config_display(self) -> str:
        """Format configuration for display."""
        config_dict = {
            "Model Settings": {
                "Chat Model": self.config.chat_model,
                "Embedding Model": self.config.embedding_model,
                "Temperature": self.config.temperature
            },
            "Processing": {
                "Chunk Size": self.config.chunk_size,
                "Chunk Overlap": self.config.chunk_overlap,
                "Batch Size": self.config.batch_size
            },
            "Performance": {
                "Caching": self.config.enable_caching,
                "Max Concurrent": self.config.max_concurrent_requests
            }
        }
        
        formatted = ""
        for section, settings in config_dict.items():
            formatted += f"[bold]{section}:[/bold]\n"
            for key, value in settings.items():
                formatted += f"  {key}: {value}\n"
            formatted += "\n"
        
        return formatted
    
    def batch_process_papers(self, paper_list_file: Path):
        """Process multiple papers from a file."""
        if not paper_list_file.exists():
            console.print(f"[red]File not found: {paper_list_file}[/red]")
            return
        
        # Read paper URLs/paths
        with open(paper_list_file, 'r') as f:
            papers = [line.strip() for line in f if line.strip()]
        
        console.print(f"[blue]Processing {len(papers)} papers...[/blue]")
        
        # Process with progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            batch_task = progress.add_task("Processing papers...", total=len(papers))
            
            results = []
            for i, paper in enumerate(papers):
                progress.update(
                    batch_task, 
                    description=f"Processing paper {i+1}/{len(papers)}: {paper[:50]}...",
                    advance=1
                )
                
                try:
                    # Your processing logic here
                    result = {"paper": paper, "status": "success"}
                    results.append(result)
                except Exception as e:
                    result = {"paper": paper, "status": "failed", "error": str(e)}
                    results.append(result)
        
        # Display batch results
        self._display_batch_results(results)
    
    def _display_batch_results(self, results: List[dict]):
        """Display batch processing results."""
        results_table = Table(title="📊 Batch Processing Results")
        results_table.add_column("Paper", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details", style="yellow")
        
        for result in results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            status = f"{status_emoji} {result['status'].title()}"
            details = result.get("error", "Successfully processed")
            
            results_table.add_row(
                result["paper"][:50] + "..." if len(result["paper"]) > 50 else result["paper"],
                status,
                details
            )
        
        console.print(results_table)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        console.print(f"\n[green]✅ {successful}/{len(results)} papers processed successfully[/green]")


# Click CLI commands
@click.group()
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, config_file):
    """Enhanced RAG System CLI - Universal Research Paper Analysis"""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file
    
    # Initialize CLI app
    app = RAGSystemCLI()
    app.display_welcome()
    ctx.obj['app'] = app


@cli.command()
@click.argument('source', required=False)
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--interactive/--no-interactive', default=True, help='Interactive mode')
@click.pass_context
def process(ctx, source, output_dir, interactive):
    """Process a research paper (URL or local file)"""
    app = ctx.obj['app']
    
    if interactive:
        app.process_paper_interactive(source, Path(output_dir) if output_dir else None)
    else:
        # Non-interactive processing
        console.print(f"[blue]Processing: {source}[/blue]")
        # Your non-interactive processing logic here


@cli.command()
@click.pass_context
def query(ctx):
    """Start interactive query session"""
    app = ctx.obj['app']
    app.interactive_query_session()


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and configuration"""
    app = ctx.obj['app']
    app.show_system_status()


@cli.command()
@click.argument('paper_list_file', type=click.Path(exists=True))
@click.pass_context
def batch(ctx, paper_list_file):
    """Process multiple papers from a file"""
    app = ctx.obj['app']
    app.batch_process_papers(Path(paper_list_file))


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['json', 'table']), default='table')
@click.pass_context
def list_papers(ctx, output_format):
    """List processed papers"""
    # Your logic to list processed papers
    console.print("[blue]Listing processed papers...[/blue]")


if __name__ == "__main__":
    cli()