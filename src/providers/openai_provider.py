"""
OpenAI provider implementation for LLM and Embedding services.
"""

import time
from typing import List, Dict, Any, Union
import tiktoken
from openai import OpenAI

from ..interfaces import LLMClient, EmbeddingClient, LLMResponse, EmbeddingResponse
from ..utils.retry import retry_with_backoff
from ..utils.errors import APIError, RateLimitError


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = config.get('api_key') or config.get('openai_api_key')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        
        # Model-specific configurations
        self.model_configs = {
            'gpt-4o-mini': {
                'max_tokens': 128000,
                'cost_per_1k_input': 0.00015,
                'cost_per_1k_output': 0.0006
            },
            'gpt-4o': {
                'max_tokens': 128000,
                'cost_per_1k_input': 0.005,
                'cost_per_1k_output': 0.015
            },
            'gpt-3.5-turbo': {
                'max_tokens': 16385,
                'cost_per_1k_input': 0.0005,
                'cost_per_1k_output': 0.0015
            }
        }
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response from messages"""
        start_time = time.time()
        
        try:
            # Prepare request
            request_kwargs = self._prepare_request_kwargs(**kwargs)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **request_kwargs
            )
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost_usd = self._calculate_cost(
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0
                }
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            else:
                raise APIError(f"OpenAI API error: {e}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate with automatic retry logic"""
        return self.generate(messages, **kwargs)
    
    def estimate_cost(self, text: str) -> float:
        """Estimate cost for processing text"""
        token_count = self.count_tokens(text)
        
        # Estimate input/output split (rough approximation)
        input_tokens = int(token_count * 0.8)  # Assume 80% input
        output_tokens = int(token_count * 0.2)  # Assume 20% output
        
        return self._calculate_cost(input_tokens, output_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
    
    @property
    def max_context_length(self) -> int:
        """Maximum context length for the model"""
        return self.model_configs.get(self.model, {}).get('max_tokens', 4096)
    
    def _prepare_request_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters"""
        request_kwargs = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'timeout': kwargs.get('timeout', self.timeout)
        }
        
        # Add other supported parameters
        for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
            if param in kwargs:
                request_kwargs[param] = kwargs[param]
        
        return request_kwargs
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        model_config = self.model_configs.get(self.model, {
            'cost_per_1k_input': 0.001,
            'cost_per_1k_output': 0.002
        })
        
        input_cost = (input_tokens / 1000) * model_config['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * model_config['cost_per_1k_output']
        
        return input_cost + output_cost


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embedding client implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize OpenAI client
        api_key = config.get('api_key') or config.get('openai_api_key')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        
        # Model-specific configurations
        self.model_configs = {
            'text-embedding-3-small': {
                'dimensions': 1536,
                'max_tokens': 8191,
                'cost_per_1k_tokens': 0.00002
            },
            'text-embedding-3-large': {
                'dimensions': 3072,
                'max_tokens': 8191,
                'cost_per_1k_tokens': 0.00013
            },
            'text-embedding-ada-002': {
                'dimensions': 1536,
                'max_tokens': 8191,
                'cost_per_1k_tokens': 0.0001
            }
        }
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def embed_query(self, text: str, **kwargs) -> List[float]:
        """Embed a single query"""
        response = self.embed_documents([text], **kwargs)
        return response.embeddings[0]
    
    def embed_documents(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """Embed multiple documents"""
        start_time = time.time()
        
        try:
            # Batch texts to respect API limits
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Make API call
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=kwargs.get('dimensions', self.dimensions)
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if response.usage:
                    total_tokens += response.usage.total_tokens
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            cost_usd = self._calculate_cost(total_tokens)
            
            return EmbeddingResponse(
                embeddings=all_embeddings,
                model=self.model,
                tokens_used=total_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata={
                    'batch_count': (len(texts) + self.batch_size - 1) // self.batch_size,
                    'dimensions': self.dimensions
                }
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            else:
                raise APIError(f"OpenAI embedding API error: {e}")
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def embed_with_retry(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """Embed with automatic retry logic"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embed_documents(texts, **kwargs)
    
    def estimate_cost(self, texts: Union[str, List[str]]) -> float:
        """Estimate cost for embedding texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        total_tokens = sum(self.count_tokens(text) for text in texts)
        return self._calculate_cost(total_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for embedding"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation
        return len(text) // 4
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage"""
        model_config = self.model_configs.get(self.model, {
            'cost_per_1k_tokens': 0.0001
        })
        
        return (tokens / 1000) * model_config['cost_per_1k_tokens']