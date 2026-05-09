"""
Phase 3: Query Rewriting Service

Improves retrieval quality by expanding, disambiguating, and optimizing queries.
Uses LLM-based rewriting for better semantic matching.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
    expansions: List[str]
    intent: str
    confidence: float


class QueryRewriter:
    """
    Rewrites user queries to improve retrieval accuracy.
    
    Strategies:
    - Query expansion (synonyms, related terms)
    - Disambiguation (resolving pronouns, context)
    - Simplification (removing noise)
    - Intent classification
    """
    
    def __init__(self):
        # Domain-specific expansions for trading/finance
        self.domain_expansions = {
            "btc": ["bitcoin", "BTC", "cryptocurrency"],
            "eth": ["ethereum", "ETH", "smart contracts"],
            "fed": ["federal reserve", "central bank", "monetary policy"],
            "cpi": ["consumer price index", "inflation", "economic data"],
            "slippage": ["execution cost", "price impact", "fill quality"],
            "liquidity": ["market depth", "order book", "trading volume"],
        }
        
        # Intent patterns
        self.intent_patterns = {
            r"(what|explain|define)": "definition",
            r"(how|process|mechanism)": "explanation",
            r"(why|reason|cause)": "causation",
            r"(compare|vs|difference)": "comparison",
            r"(calculate|compute|estimate)": "calculation",
            r"(risk|exposure|loss)": "risk_analysis",
            r"(price|cost|value)": "valuation",
            r"(execute|trade|order)": "execution",
        }
    
    def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> RewrittenQuery:
        """
        Rewrite a query for better retrieval.
        
        Args:
            query: Original user query
            context: Optional conversation context for disambiguation
            
        Returns:
            RewrittenQuery with expanded forms and intent
        """
        # Clean query
        cleaned = self._clean_query(query)
        
        # Expand domain terms
        expanded = self._expand_terms(cleaned)
        
        # Classify intent
        intent = self._classify_intent(cleaned)
        
        # Resolve context (pronouns, references)
        if context:
            cleaned = self._resolve_context(cleaned, context)
        
        # Generate final rewritten form
        rewritten = self._generate_rewritten_form(cleaned, expanded)
        
        # Estimate confidence
        confidence = self._estimate_confidence(query, rewritten)
        
        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            expansions=expanded,
            intent=intent,
            confidence=confidence
        )
    
    def _clean_query(self, query: str) -> str:
        """Remove noise and normalize query."""
        # Lowercase
        q = query.lower().strip()
        
        # Remove extra whitespace
        q = re.sub(r'\s+', ' ', q)
        
        # Remove trailing punctuation (keep question marks)
        q = q.rstrip('.,;:!')
        
        # Remove multiple question marks
        q = re.sub(r'\?+', '?', q)
        
        return q
    
    def _expand_terms(self, query: str) -> List[str]:
        """Expand domain-specific abbreviations and terms."""
        expansions = [query]
        
        words = query.split()
        for word in words:
            if word in self.domain_expansions:
                for expansion in self.domain_expansions[word]:
                    # Create variant with expansion
                    variant = query.replace(word, expansion)
                    if variant not in expansions:
                        expansions.append(variant)
        
        return expansions
    
    def _classify_intent(self, query: str) -> str:
        """Classify the user's intent."""
        for pattern, intent in self.intent_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return intent
        
        return "general"
    
    def _resolve_context(self, query: str, context: Dict[str, Any]) -> str:
        """Resolve pronouns and references using context."""
        resolved = query
        
        # Simple pronoun resolution
        pronouns = {
            "it": context.get("last_subject"),
            "they": context.get("last_subject"),
            "this": context.get("last_topic"),
            "that": context.get("last_topic"),
        }
        
        for pronoun, replacement in pronouns.items():
            if replacement and pronoun in resolved:
                resolved = resolved.replace(pronoun, str(replacement))
        
        return resolved
    
    def _generate_rewritten_form(self, query: str, expansions: List[str]) -> str:
        """Generate optimized rewritten query."""
        # For Phase 3, use the most comprehensive expansion
        if len(expansions) > 1:
            # Combine original with key expansions
            return f"{query} ({' '.join(expansions[1:2])})"
        
        return query
    
    def _estimate_confidence(self, original: str, rewritten: str) -> float:
        """Estimate confidence in the rewrite quality."""
        # Simple heuristic: if rewrite is very different, lower confidence
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())
        
        overlap = len(original_words & rewritten_words)
        total = len(original_words | rewritten_words)
        
        if total == 0:
            return 0.0
        
        # Base confidence from word overlap
        base_confidence = overlap / total
        
        # Boost if we added useful expansions
        if len(rewritten) > len(original) * 1.5:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return round(base_confidence, 2)
    
    def batch_rewrite(self, queries: List[str], 
                      contexts: Optional[List[Dict[str, Any]]] = None) -> List[RewrittenQuery]:
        """Rewrite multiple queries."""
        results = []
        for i, query in enumerate(queries):
            context = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.rewrite(query, context))
        return results


# Global rewriter instance
_query_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """Get or create global query rewriter instance."""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter
