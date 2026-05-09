"""
Query Rewriter - Transforms user queries for optimal retrieval

Handles:
- Query expansion
- Ambiguity resolution  
- Context-aware rewriting
- Multi-turn conversation handling
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
    expansions: List[str]
    confidence: float
    reasoning: str


class QueryRewriter:
    """
    Rewrites queries to improve retrieval quality.
    
    Strategies:
    - HyDE (Hypothetical Document Embeddings)
    - Query expansion with synonyms
    - Disambiguation of pronouns/references
    - Temporal context injection
    """
    
    def __init__(self, llm_client=None, embedding_model=None):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.rewrite_history: List[Dict[str, Any]] = []
    
    def rewrite(
        self, 
        query: str, 
        context: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        domain: Optional[str] = None
    ) -> RewrittenQuery:
        """
        Rewrite a query for better retrieval performance.
        
        Args:
            query: Original user query
            context: Additional context about the query
            conversation_history: Previous turns in conversation
            domain: Specific domain (e.g., "trading", "compliance")
            
        Returns:
            RewrittenQuery with expanded forms and reasoning
        """
        # Resolve pronouns and references from conversation history
        resolved_query = self._resolve_references(query, conversation_history)
        
        # Expand with synonyms and related terms
        expanded_queries = self._expand_query(resolved_query, domain)
        
        # Generate hypothetical document for HyDE approach
        hyde_query = self._generate_hyde(resolved_query, domain)
        
        # Select best strategy based on query type
        final_rewrite = self._select_best_form(
            resolved_query, 
            expanded_queries, 
            hyde_query
        )
        
        result = RewrittenQuery(
            original=query,
            rewritten=final_rewrite,
            expansions=expanded_queries,
            confidence=self._estimate_confidence(query, final_rewrite),
            reasoning=self._generate_reasoning(query, final_rewrite)
        )
        
        self.rewrite_history.append({
            "original": query,
            "rewritten": final_rewrite,
            "context": context,
        })
        
        return result
    
    def _resolve_references(
        self, 
        query: str, 
        history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Resolve pronouns like 'it', 'they', 'that' to actual entities."""
        if not history:
            return query
        
        # Simple heuristic: replace pronouns with last mentioned entity
        # In production, use coreference resolution model
        resolved = query
        last_subject = None
        
        for turn in reversed(history[-5:]):  # Look at last 5 turns
            if turn.get("role") == "user":
                # Extract potential subject from previous query
                words = turn["content"].split()
                if len(words) > 2:
                    last_subject = words[0]  # Simplified extraction
                    break
        
        if last_subject:
            resolved = resolved.replace("it", last_subject)
            resolved = resolved.replace("they", last_subject)
            resolved = resolved.replace("that policy", f"the {last_subject} policy")
        
        return resolved
    
    def _expand_query(self, query: str, domain: Optional[str]) -> List[str]:
        """Generate query variations with synonyms and related terms."""
        expansions = [query]
        
        # Domain-specific expansions
        domain_synonyms = {
            "trading": {
                "execution": ["fill", "order completion", "trade execution"],
                "slippage": ["price impact", "execution variance"],
                "liquidity": ["market depth", "order book"],
            },
            "compliance": {
                "policy": ["rule", "regulation", "guideline"],
                "violation": ["breach", "non-compliance", "infraction"],
            }
        }
        
        if domain and domain in domain_synonyms:
            for term, synonyms in domain_synonyms[domain].items():
                if term in query.lower():
                    for synonym in synonyms:
                        expansions.append(query.replace(term, synonym))
        
        # Add acronym expansions if present
        acronym_map = {
            "CPI": "Consumer Price Index",
            "BTC": "Bitcoin",
            "SLA": "Service Level Agreement",
        }
        
        for acronym, full_form in acronym_map.items():
            if acronym in query.upper():
                expansions.append(query.replace(acronym, full_form))
        
        return list(set(expansions))  # Remove duplicates
    
    def _generate_hyde(self, query: str, domain: Optional[str]) -> str:
        """
        Generate Hypothetical Document Embedding query.
        
        Creates a fake "document" that would answer the query,
        then uses its embedding for retrieval.
        """
        # In production, this would call LLM to generate hypothetical doc
        # For now, use template-based approach
        
        templates = {
            "what": f"This document explains {query}",
            "how": f"This guide describes the procedure for {query}",
            "why": f"This analysis provides reasons for {query}",
            "when": f"This timeline shows when {query}",
        }
        
        for question_word, template in templates.items():
            if query.lower().startswith(question_word):
                return template
        
        return f"Information about: {query}"
    
    def _select_best_form(
        self, 
        original: str, 
        expansions: List[str], 
        hyde: str
    ) -> str:
        """Select the most promising query formulation."""
        # Heuristic: prefer expanded queries for short inputs
        if len(original.split()) < 4 and len(expansions) > 1:
            return expansions[1]  # Return first expansion
        
        # Use HyDE for complex questions
        if len(original.split()) > 6:
            return hyde
        
        return original
    
    def _estimate_confidence(self, original: str, rewritten: str) -> float:
        """Estimate confidence in the rewrite quality."""
        if original == rewritten:
            return 0.9  # High confidence in original
        
        # More changes = lower confidence
        change_ratio = abs(len(rewritten) - len(original)) / max(len(original), 1)
        confidence = max(0.5, 1.0 - change_ratio)
        
        return round(confidence, 2)
    
    def _generate_reasoning(self, original: str, rewritten: str) -> str:
        """Generate explanation for why query was rewritten."""
        if original == rewritten:
            return "Original query was already optimal"
        
        reasons = []
        if len(rewritten) > len(original):
            reasons.append("expanded for clarity")
        if rewritten != original.lower():
            reasons.append("resolved references")
        
        return "; ".join(reasons) if reasons else "improved retrieval matching"
