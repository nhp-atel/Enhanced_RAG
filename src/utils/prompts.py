"""
Prompt management system for external prompt templates.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

from ..interfaces import PromptTemplate
from ..utils.errors import ConfigurationError, ValidationError


class SimplePromptTemplate(PromptTemplate):
    """Simple implementation of PromptTemplate"""
    
    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the prompt with given variables"""
        # Validate required variables
        missing_vars = self.validate_variables(**kwargs)
        if missing_vars:
            raise ValidationError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Format templates
        try:
            formatted_system = self.system_prompt.format(**kwargs) if self.system_prompt else ""
            formatted_human = self.human_prompt.format(**kwargs) if self.human_prompt else ""
            
            messages = []
            if formatted_system:
                messages.append({"role": "system", "content": formatted_system})
            if formatted_human:
                messages.append({"role": "user", "content": formatted_human})
            
            return messages
            
        except KeyError as e:
            raise ValidationError(f"Missing template variable: {e}")
        except Exception as e:
            raise ValidationError(f"Template formatting error: {e}")


class PromptManager:
    """Manages external prompt templates from YAML/JSON files"""
    
    def __init__(self, prompts_file: Optional[str] = None):
        self.prompts_file = prompts_file or "./config/prompts.yaml"
        self.prompts: Dict[str, PromptTemplate] = {}
        self.domains: Dict[str, Dict[str, Any]] = {}
        
        # Load prompts on initialization
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from configuration file"""
        prompts_path = Path(self.prompts_file)
        
        if not prompts_path.exists():
            raise ConfigurationError(f"Prompts file not found: {self.prompts_file}")
        
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                if prompts_path.suffix.lower() == '.yaml' or prompts_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Load main prompts
            if 'prompts' in data:
                for name, template_data in data['prompts'].items():
                    self.prompts[name] = SimplePromptTemplate(name, template_data)
            
            # Load domain-specific configurations if present
            if 'domains' in data:
                self.domains = data['domains']
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load prompts from {self.prompts_file}: {e}")
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """Get prompt template by name"""
        if name not in self.prompts:
            raise ValidationError(f"Prompt template '{name}' not found")
        
        return self.prompts[name]
    
    def list_prompts(self) -> List[str]:
        """List available prompt names"""
        return list(self.prompts.keys())
    
    def add_prompt(self, name: str, template_data: Dict[str, Any]) -> None:
        """Add a new prompt template"""
        self.prompts[name] = SimplePromptTemplate(name, template_data)
    
    def reload_prompts(self) -> None:
        """Reload prompts from file"""
        self.prompts.clear()
        self.domains.clear()
        self.load_prompts()
    
    def get_domain_prompt(self, domain: str, prompt_type: str) -> Optional[PromptTemplate]:
        """Get domain-specific prompt if available"""
        if domain in self.domains:
            domain_config = self.domains[domain]
            if prompt_type in domain_config:
                template_data = domain_config[prompt_type]
                return SimplePromptTemplate(f"{domain}_{prompt_type}", template_data)
        
        return None
    
    def format_prompt(self, name: str, **kwargs) -> List[Dict[str, str]]:
        """Format prompt with variables - convenience method"""
        template = self.get_prompt(name)
        return template.format(**kwargs)
    
    def validate_prompt(self, name: str, **kwargs) -> List[str]:
        """Validate prompt variables - convenience method"""
        template = self.get_prompt(name)
        return template.validate_variables(**kwargs)
    
    def get_few_shot_examples(self, domain: str) -> List[Dict[str, str]]:
        """Get few-shot examples for a domain"""
        if domain in self.domains:
            return self.domains[domain].get('few_shot_examples', [])
        return []
    
    def create_few_shot_prompt(
        self, 
        base_prompt_name: str, 
        domain: str, 
        max_examples: int = 3,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Create a prompt with few-shot examples"""
        # Get base prompt
        base_template = self.get_prompt(base_prompt_name)
        base_messages = base_template.format(**kwargs)
        
        # Get few-shot examples
        examples = self.get_few_shot_examples(domain)[:max_examples]
        
        if not examples:
            return base_messages
        
        # Insert examples into system prompt
        if base_messages and base_messages[0].get('role') == 'system':
            system_content = base_messages[0]['content']
            
            # Add examples section
            examples_text = "\n\nHere are some examples:\n\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"Example {i}:\n"
                examples_text += f"Question: {example.get('question', '')}\n"
                examples_text += f"Context: {example.get('context', '')}\n"
                examples_text += f"Answer: {example.get('answer', '')}\n\n"
            
            base_messages[0]['content'] = system_content + examples_text
        
        return base_messages


class DynamicPromptBuilder:
    """Build prompts dynamically based on context and requirements"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def build_retrieval_prompt(
        self, 
        query: str, 
        context_types: List[str],
        domain: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build context-aware retrieval prompt"""
        # Start with base retrieval prompt
        base_prompt = "enhanced_rag" if len(context_types) > 1 else "basic_rag"
        
        # Get domain-specific prompt if available
        if domain:
            domain_prompt = self.prompt_manager.get_domain_prompt(domain, "retrieval")
            if domain_prompt:
                return domain_prompt.format(query=query, context_types=context_types)
        
        # Use base prompt with dynamic context description
        context_description = self._build_context_description(context_types)
        
        template = self.prompt_manager.get_prompt(base_prompt)
        messages = template.format(
            question=query,
            context=context_description
        )
        
        return messages
    
    def build_concept_extraction_prompt(
        self, 
        text: str, 
        domain: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Build domain-aware concept extraction prompt"""
        # Get base template
        template = self.prompt_manager.get_prompt("concept_extraction")
        
        # Modify for domain if specified
        if domain and domain in self.prompt_manager.domains:
            domain_config = self.prompt_manager.domains[domain]
            categories = domain_config.get('embedding_categories', [])
            
            # Create domain-specific instruction
            domain_instruction = f"\nFor {domain} domain, focus on these categories: {', '.join(categories)}"
            
            # Modify system prompt
            messages = template.format(summary=text)
            if messages and messages[0].get('role') == 'system':
                messages[0]['content'] += domain_instruction
            
            return messages
        
        return template.format(summary=text)
    
    def _build_context_description(self, context_types: List[str]) -> str:
        """Build description of available context types"""
        type_descriptions = {
            'metadata': 'Paper metadata (title, authors, dates)',
            'summary': 'Paper summary (comprehensive overview)',
            'concepts': 'Relevant concepts (definitions and explanations)', 
            'content': 'Document content (specific passages)',
            'memory': 'Memory system information (structured knowledge)'
        }
        
        descriptions = []
        for ctx_type in context_types:
            if ctx_type in type_descriptions:
                descriptions.append(f"- {type_descriptions[ctx_type]}")
        
        return "Available context sources:\n" + "\n".join(descriptions)


class PromptVersionManager:
    """Manage different versions of prompts for A/B testing and rollbacks"""
    
    def __init__(self, base_path: str = "./config/prompts"):
        self.base_path = Path(base_path)
        self.versions: Dict[str, Dict[str, PromptTemplate]] = {}
        self.active_version = "v1"
    
    def load_version(self, version: str) -> None:
        """Load specific version of prompts"""
        version_file = self.base_path / f"prompts_{version}.yaml"
        
        if not version_file.exists():
            raise ConfigurationError(f"Prompt version file not found: {version_file}")
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            version_prompts = {}
            if 'prompts' in data:
                for name, template_data in data['prompts'].items():
                    version_prompts[name] = SimplePromptTemplate(name, template_data)
            
            self.versions[version] = version_prompts
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load prompt version {version}: {e}")
    
    def set_active_version(self, version: str) -> None:
        """Set active prompt version"""
        if version not in self.versions:
            self.load_version(version)
        
        self.active_version = version
    
    def get_prompt(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """Get prompt from specific version"""
        version = version or self.active_version
        
        if version not in self.versions:
            self.load_version(version)
        
        if name not in self.versions[version]:
            raise ValidationError(f"Prompt '{name}' not found in version {version}")
        
        return self.versions[version][name]
    
    def compare_versions(self, prompt_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare prompt between two versions"""
        try:
            prompt1 = self.get_prompt(prompt_name, version1)
            prompt2 = self.get_prompt(prompt_name, version2)
            
            return {
                "prompt_name": prompt_name,
                "version1": version1,
                "version2": version2,
                "system_prompt_changed": prompt1.system_prompt != prompt2.system_prompt,
                "human_prompt_changed": prompt1.human_prompt != prompt2.human_prompt,
                "metadata_changed": prompt1.metadata != prompt2.metadata
            }
        except Exception as e:
            return {"error": str(e)}


# Global prompt manager instance
_global_prompt_manager = None


def get_prompt_manager(prompts_file: Optional[str] = None) -> PromptManager:
    """Get global prompt manager instance"""
    global _global_prompt_manager
    
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(prompts_file)
    
    return _global_prompt_manager


# Convenience functions

def format_prompt(name: str, **kwargs) -> List[Dict[str, str]]:
    """Format prompt using global prompt manager"""
    return get_prompt_manager().format_prompt(name, **kwargs)


def validate_prompt_variables(name: str, **kwargs) -> List[str]:
    """Validate prompt variables using global prompt manager"""
    return get_prompt_manager().validate_prompt(name, **kwargs)