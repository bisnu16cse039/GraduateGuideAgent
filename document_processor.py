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
    education: List[str] = Field(description="Degrees, institutions, and years", default=[])
    research_interests: List[str] = Field(description="Research interests and areas", default=[])
    technical_skills: List[str] = Field(description="Technical skills and tools", default=[])
    publications: List[str] = Field(description="Key publications with brief descriptions", default=[])
    experience: List[str] = Field(description="Relevant research/work experience", default=[])
    awards: List[str] = Field(description="Awards, honors, and achievements", default=[])

class PaperSummary(BaseModel):
    """Structured paper summary"""
    title: str = Field(description="Paper title", default="Unknown Paper")
    main_contribution: str = Field(description="Main contribution in 1-2 sentences", default="")
    methodology: List[str] = Field(description="Key methods/techniques used", default=[])
    findings: List[str] = Field(description="Key findings or results", default=[])
    relevance_keywords: List[str] = Field(description="Keywords for relevance matching", default=[])

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
        """Extract structured information from CV with better error handling"""
        logger.info("Processing CV for structured extraction...")
        
        if not cv_text or len(cv_text.strip()) < 50:
            logger.warning("CV text is too short or empty")
            empty_summary = CVSummary()
            return empty_summary, "No CV content provided"
        
        try:
            # Use a simpler extraction approach without complex JSON parsing
            cv_summary = self._extract_cv_info_simple(cv_text)
            concise_text = self._create_concise_cv_text(cv_summary)
            
            return cv_summary, concise_text
            
        except Exception as e:
            logger.error(f"Error processing CV: {e}")
            # Return basic summary if processing fails
            basic_summary = self._create_basic_cv_summary(cv_text)
            basic_text = self._create_concise_cv_text(basic_summary)
            return basic_summary, basic_text
    
    def _extract_cv_info_simple(self, cv_text: str) -> CVSummary:
        """Simple CV extraction without complex JSON parsing"""
        
        # Create a simple prompt that asks for structured text output
        prompt = f"""
        Analyze this CV and extract key information. Provide your response in the following format:

        EDUCATION:
        - [List education entries, one per line]

        RESEARCH_INTERESTS:
        - [List research interests, one per line]

        TECHNICAL_SKILLS:
        - [List technical skills, one per line]

        PUBLICATIONS:
        - [List key publications with brief descriptions, one per line]

        EXPERIENCE:
        - [List relevant experience, one per line]

        AWARDS:
        - [List awards and honors, one per line]

        CV Content:
        {cv_text[:3000]}
        """
        
        try:
            result = self.cheap_llm.invoke(prompt)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            # Parse the structured response
            cv_summary = self._parse_structured_response(response_text)
            return cv_summary
            
        except Exception as e:
            logger.error(f"Error in simple CV extraction: {e}")
            return self._create_basic_cv_summary(cv_text)
    
    def _parse_structured_response(self, response_text: str) -> CVSummary:
        """Parse the structured response into CVSummary"""
        
        sections = {
            'education': [],
            'research_interests': [],
            'technical_skills': [],
            'publications': [],
            'experience': [],
            'awards': []
        }
        
        current_section = None
        
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.upper().startswith('EDUCATION:'):
                current_section = 'education'
            elif line.upper().startswith('RESEARCH_INTERESTS:'):
                current_section = 'research_interests'
            elif line.upper().startswith('TECHNICAL_SKILLS:'):
                current_section = 'technical_skills'
            elif line.upper().startswith('PUBLICATIONS:'):
                current_section = 'publications'
            elif line.upper().startswith('EXPERIENCE:'):
                current_section = 'experience'
            elif line.upper().startswith('AWARDS:'):
                current_section = 'awards'
            elif line.startswith('-') and current_section:
                # Add item to current section
                item = line[1:].strip()
                if item:
                    sections[current_section].append(item)
        
        return CVSummary(**sections)
    
    def _create_basic_cv_summary(self, cv_text: str) -> CVSummary:
        """Create a basic CV summary using keyword matching"""
        text_lower = cv_text.lower()
        
        # Extract education
        education = []
        education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'ph.d']
        for line in cv_text.split('\n'):
            if any(keyword in line.lower() for keyword in education_keywords):
                education.append(line.strip())
                if len(education) >= 3:
                    break
        
        # Extract skills
        skills = []
        skill_keywords = ['python', 'java', 'c++', 'javascript', 'machine learning', 'data analysis', 'tensorflow', 'pytorch']
        for keyword in skill_keywords:
            if keyword in text_lower:
                skills.append(keyword.title())
        
        # Extract research interests
        research_interests = []
        if 'research' in text_lower or 'interest' in text_lower:
            for line in cv_text.split('\n'):
                if 'research' in line.lower() or 'interest' in line.lower():
                    clean_line = line.strip()
                    if len(clean_line) > 10 and len(clean_line) < 100:
                        research_interests.append(clean_line)
                        if len(research_interests) >= 3:
                            break
        
        return CVSummary(
            education=education[:3],
            research_interests=research_interests[:5],
            technical_skills=skills[:10],
            publications=[],
            experience=[],
            awards=[]
        )
    
    def process_paper(self, paper_text: str, max_chars: int = 1500) -> Tuple[PaperSummary, str]:
        """Process academic paper into structured summary"""
        logger.info("Processing paper for summarization...")
        
        if not paper_text or len(paper_text.strip()) < 100:
            empty_summary = PaperSummary()
            return empty_summary, "No paper content provided"
        
        try:
            # Take the most informative parts of the paper
            abstract_section = self._extract_section(paper_text, ["abstract", "summary"], max_chars=800)
            intro_section = self._extract_section(paper_text, ["introduction", "1.", "1 "], max_chars=500)
            conclusion_section = self._extract_section(paper_text, ["conclusion", "discussion"], max_chars=500)
            
            combined_text = f"{abstract_section}\n\n{intro_section}\n\n{conclusion_section}"
            
            # Simple extraction without JSON parsing
            prompt = f"""
            Analyze this academic paper and provide a summary in the following format:

            TITLE: [Extract or infer the paper title]
            MAIN_CONTRIBUTION: [Main contribution in 1-2 sentences]
            METHODOLOGY: [List key methods used, separated by commas]
            FINDINGS: [List key findings, separated by commas]
            KEYWORDS: [List relevant keywords, separated by commas]

            Paper content:
            {combined_text[:max_chars]}
            """
            
            result = self.cheap_llm.invoke(prompt)
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            paper_summary = self._parse_paper_response(response_text)
            
            # Create concise text representation
            concise_text = (
                f"Title: {paper_summary.title}\n"
                f"Contribution: {paper_summary.main_contribution}\n"
                f"Methods: {', '.join(paper_summary.methodology[:3])}\n"
                f"Key Findings: {'; '.join(paper_summary.findings[:2])}"
            )
            
            return paper_summary, concise_text
            
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            basic_summary = PaperSummary(
                title="Research Paper",
                main_contribution="Paper analysis failed",
                methodology=["Unknown"],
                findings=["Analysis incomplete"],
                relevance_keywords=["research"]
            )
            return basic_summary, "Paper processing failed"
    
    def _parse_paper_response(self, response_text: str) -> PaperSummary:
        """Parse paper response into PaperSummary"""
        
        title = "Unknown Paper"
        main_contribution = ""
        methodology = []
        findings = []
        keywords = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('TITLE:'):
                title = line.replace('TITLE:', '').strip()
            elif line.startswith('MAIN_CONTRIBUTION:'):
                main_contribution = line.replace('MAIN_CONTRIBUTION:', '').strip()
            elif line.startswith('METHODOLOGY:'):
                methods_str = line.replace('METHODOLOGY:', '').strip()
                methodology = [m.strip() for m in methods_str.split(',') if m.strip()]
            elif line.startswith('FINDINGS:'):
                findings_str = line.replace('FINDINGS:', '').strip()
                findings = [f.strip() for f in findings_str.split(',') if f.strip()]
            elif line.startswith('KEYWORDS:'):
                keywords_str = line.replace('KEYWORDS:', '').strip()
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
        
        return PaperSummary(
            title=title,
            main_contribution=main_contribution,
            methodology=methodology[:5],
            findings=findings[:5],
            relevance_keywords=keywords[:10]
        )
    
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
        
        return "\n".join(parts) if parts else "No CV information available"


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
        if professor_summary and professor_summary.strip():
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
            if additional_cv:
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
        
        return "\n".join(details) if details else ""