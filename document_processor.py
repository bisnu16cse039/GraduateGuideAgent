"""
Document processing utilities for handling long context documents
"""

import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class CVSummary(BaseModel):
    """Structured CV summary"""
    education: List[str] = Field(description="Degrees, institutions, and years")
    research_interests: List[str] = Field(description="Research interests and areas")
    technical_skills: List[str] = Field(description="Technical skills and tools")
    publications: List[str] = Field(description="Key publications with brief descriptions")
    experience: List[str] = Field(description="Relevant research/work experience")
    awards: List[str] = Field(description="Awards, honors, and achievements")

class PaperSummary(BaseModel):
    """Structured paper summary"""
    title: str = Field(description="Paper title")
    main_contribution: str = Field(description="Main contribution in 1-2 sentences")
    methodology: List[str] = Field(description="Key methods/techniques used")
    findings: List[str] = Field(description="Key findings or results")
    relevance_keywords: List[str] = Field(description="Keywords for relevance matching")

class DocumentProcessor:
    """Process long documents intelligently"""
    
    def __init__(self, cheap_llm, expensive_llm=None):
        self.cheap_llm = cheap_llm
        self.expensive_llm = expensive_llm or cheap_llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def process_cv(self, cv_text: str) -> Tuple[CVSummary, str]:
        """Extract structured information from CV"""
        logger.info("Processing CV for structured extraction...")
        
        parser = PydanticOutputParser(pydantic_object=CVSummary)
        
        # Split CV into chunks for processing
        chunks = self.text_splitter.split_text(cv_text)
        
        # Process each chunk with cheap LLM
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = ChatPromptTemplate.from_template("""
            Extract key information from this CV section:
            
            {cv_chunk}
            
            Focus on:
            - Education details (degrees, institutions, years)
            - Research interests and areas
            - Technical skills
            - Publications (include brief description)
            - Relevant experience
            - Awards/honors
            
            {format_instructions}
            """)
            
            try:
                result = self.cheap_llm.invoke(
                    prompt.format_messages(
                        cv_chunk=chunk,
                        format_instructions=parser.get_format_instructions()
                    )
                )
                chunk_summaries.append(result.content)
            except Exception as e:
                logger.warning(f"Error processing CV chunk {i}: {e}")
                continue
        
        # Merge summaries with expensive LLM
        if len(chunk_summaries) > 1:
            merge_prompt = ChatPromptTemplate.from_template("""
            Merge these CV section summaries into a single coherent profile:
            
            {summaries}
            
            Combine duplicate information and ensure completeness.
            
            {format_instructions}
            """)
            
            final_summary = self.expensive_llm.invoke(
                merge_prompt.format_messages(
                    summaries="\n\n---\n\n".join(chunk_summaries),
                    format_instructions=parser.get_format_instructions()
                )
            )
            
            # Parse the final summary
            cv_summary = parser.parse(final_summary.content)
        else:
            # If only one chunk, parse it directly
            cv_summary = parser.parse(chunk_summaries[0] if chunk_summaries else "{}")
        
        # Create a concise text representation
        concise_text = self._create_concise_cv_text(cv_summary)
        
        return cv_summary, concise_text
    
    def process_paper(self, paper_text: str, max_chars: int = 1500) -> Tuple[PaperSummary, str]:
        """Process academic paper into structured summary"""
        logger.info("Processing paper for summarization...")
        
        parser = PydanticOutputParser(pydantic_object=PaperSummary)
        
        # Take the most informative parts of the paper
        abstract_section = self._extract_section(paper_text, ["abstract", "summary"], max_chars=800)
        intro_section = self._extract_section(paper_text, ["introduction", "1.", "1 "], max_chars=500)
        conclusion_section = self._extract_section(paper_text, ["conclusion", "discussion"], max_chars=500)
        
        combined_text = f"{abstract_section}\n\n{intro_section}\n\n{conclusion_section}"
        
        prompt = ChatPromptTemplate.from_template("""
        Summarize this academic paper:
        
        {paper_content}
        
        Extract:
        - Paper title (if found)
        - Main contribution (1-2 sentences)
        - Key methodologies used
        - Main findings or results
        - Keywords for relevance matching
        
        {format_instructions}
        """)
        
        result = self.cheap_llm.invoke(
            prompt.format_messages(
                paper_content=combined_text[:max_chars],
                format_instructions=parser.get_format_instructions()
            )
        )
        
        paper_summary = parser.parse(result.content)
        
        # Create concise text representation
        concise_text = (
            f"Title: {paper_summary.title}\n"
            f"Contribution: {paper_summary.main_contribution}\n"
            f"Methods: {', '.join(paper_summary.methodology[:3])}\n"
            f"Key Findings: {'; '.join(paper_summary.findings[:2])}"
        )
        
        return paper_summary, concise_text
    
    def _extract_section(self, text: str, markers: List[str], max_chars: int = 1000) -> str:
        """Extract a section from text based on markers"""
        text_lower = text.lower()
        
        for marker in markers:
            start_idx = text_lower.find(marker.lower())
            if start_idx != -1:
                # Find the actual start (considering case)
                section = text[start_idx:start_idx + max_chars]
                # Try to end at a sentence boundary
                last_period = section.rfind('.')
                if last_period > max_chars * 0.7:  # If we have at least 70% of desired length
                    section = section[:last_period + 1]
                return section.strip()
        
        # If no marker found, return the beginning
        return text[:max_chars].strip()
    
    def _create_concise_cv_text(self, cv_summary: CVSummary) -> str:
        """Create a concise text representation of CV summary"""
        parts = []
        
        if cv_summary.education:
            parts.append(f"Education: {'; '.join(cv_summary.education[:2])}")
        
        if cv_summary.research_interests:
            parts.append(f"Research Interests: {', '.join(cv_summary.research_interests[:5])}")
        
        if cv_summary.technical_skills:
            parts.append(f"Skills: {', '.join(cv_summary.technical_skills[:8])}")
        
        if cv_summary.publications:
            parts.append(f"Key Publications: {'; '.join(cv_summary.publications[:3])}")
        
        if cv_summary.experience:
            parts.append(f"Experience: {'; '.join(cv_summary.experience[:2])}")
        
        return "\n".join(parts)


class ProgressiveContextBuilder:
    """Build context progressively based on token limits"""
    
    def __init__(self, max_tokens: int = 6000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def build_context(
        self,
        cv_summary: CVSummary,
        cv_text: str,
        professor_summary: str,
        paper_summaries: List[Tuple[PaperSummary, str]],
        background_summary: str
    ) -> Dict[str, Any]:
        """Build context within token limits"""
        
        context_parts = []
        current_tokens = 0
        included_sections = []
        
        # Priority 1: Core student profile (from CV summary)
        core_profile = self._build_student_profile(cv_summary, background_summary)
        tokens = len(self.encoding.encode(core_profile))
        if current_tokens + tokens < self.max_tokens:
            context_parts.append(core_profile)
            current_tokens += tokens
            included_sections.append("student_profile")
        
        # Priority 2: Professor summary
        if professor_summary:
            tokens = len(self.encoding.encode(professor_summary))
            if current_tokens + tokens < self.max_tokens * 0.8:  # Leave room for papers
                context_parts.append(f"\nPROFESSOR RESEARCH:\n{professor_summary}")
                current_tokens += tokens
                included_sections.append("professor_research")
        
        # Priority 3: Most relevant papers (based on alignment)
        if paper_summaries:
            papers_added = 0
            for paper_summary, paper_text in paper_summaries:
                if papers_added >= 3:  # Limit to 3 papers max
                    break
                    
                tokens = len(self.encoding.encode(paper_text))
                if current_tokens + tokens < self.max_tokens:
                    context_parts.append(f"\nPAPER {papers_added + 1}:\n{paper_text}")
                    current_tokens += tokens
                    papers_added += 1
                    included_sections.append(f"paper_{papers_added}")
        
        # Priority 4: Additional CV details if space allows
        if current_tokens < self.max_tokens * 0.7:
            additional_cv = self._get_additional_cv_details(cv_summary)
            tokens = len(self.encoding.encode(additional_cv))
            if current_tokens + tokens < self.max_tokens:
                context_parts.append(f"\nADDITIONAL DETAILS:\n{additional_cv}")
                current_tokens += tokens
                included_sections.append("additional_cv")
        
        return {
            "context": "\n".join(context_parts),
            "token_count": current_tokens,
            "included_sections": included_sections
        }
    
    def _build_student_profile(self, cv_summary: CVSummary, background: str) -> str:
        """Build core student profile"""
        profile_parts = ["STUDENT PROFILE:"]
        
        if background:
            profile_parts.append(f"Background: {background[:300]}")
        
        if cv_summary.education:
            profile_parts.append(f"Education: {'; '.join(cv_summary.education)}")
        
        if cv_summary.research_interests:
            profile_parts.append(f"Research Interests: {', '.join(cv_summary.research_interests)}")
        
        if cv_summary.technical_skills:
            profile_parts.append(f"Technical Skills: {', '.join(cv_summary.technical_skills[:10])}")
        
        if cv_summary.publications:
            profile_parts.append(f"Selected Publications: {'; '.join(cv_summary.publications[:3])}")
        
        return "\n".join(profile_parts)
    
    def _get_additional_cv_details(self, cv_summary: CVSummary) -> str:
        """Get additional CV details for context"""
        details = []
        
        if cv_summary.experience and len(cv_summary.experience) > 2:
            details.append(f"Additional Experience: {'; '.join(cv_summary.experience[2:])}")
        
        if cv_summary.awards:
            details.append(f"Awards/Honors: {'; '.join(cv_summary.awards)}")
        
        if cv_summary.publications and len(cv_summary.publications) > 3:
            details.append(f"Other Publications: {'; '.join(cv_summary.publications[3:])}")
        
        return "\n".join(details)