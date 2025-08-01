"""
GraduateGuide Agent - AI Assistant for Graduate School Advisor Outreach
Built with LangGraph for workflow orchestration and flexible LLM configuration
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, asdict
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import re

from langgraph.graph import StateGraph, END
#from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from document_processor import DocumentProcessor, ProgressiveContextBuilder, CVSummary, PaperSummary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Classes
@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str  # 'openai' or 'google'
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000

@dataclass
class EmailConfig:
    """Email configuration for SMTP"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True

# Pydantic Models for Structured Output
class ProfileInsights(BaseModel):
    """Extracted insights from user's CV and background"""
    research_interests: List[str] = Field(description="Key research areas of interest")
    technical_skills: List[str] = Field(description="Technical skills and expertise")
    publications: List[str] = Field(description="Notable publications or projects")
    academic_background: str = Field(description="Educational background summary")
    relevant_experience: List[str] = Field(description="Relevant work or research experience")

class ProfessorResearch(BaseModel):
    """Professor's research analysis"""
    research_areas: List[str] = Field(description="Professor's main research areas")
    recent_work_summary: str = Field(description="Summary of recent work")
    key_methodologies: List[str] = Field(description="Key research methodologies used")
    alignment_points: List[str] = Field(description="Points of alignment with student interests")
    notable_findings: List[str] = Field(description="Notable research findings or contributions")

class EmailQuality(BaseModel):
    """Email quality assessment"""
    tone_score: int = Field(ge=1, le=10, description="Professional tone score (1-10)")
    structure_score: int = Field(ge=1, le=10, description="Email structure score (1-10)")
    personalization_score: int = Field(ge=1, le=10, description="Personalization score (1-10)")
    citation_accuracy: bool = Field(description="Accuracy of paper citations")
    overall_quality: int = Field(ge=1, le=10, description="Overall quality score (1-10)")
    feedback: str = Field(description="Specific feedback for improvement")
    passes_threshold: bool = Field(description="Whether email meets quality threshold")

# State Management
class GraduateGuideState(TypedDict):
    """State object for the GraduateGuide workflow"""
    # User inputs
    cv_content: str
    background_summary: str
    target_filters: Dict[str, Any]
    professor_url: str
    paper_urls: List[str]
    
    # Processing results
    profile_insights: Optional[ProfileInsights]
    professor_research: Optional[ProfessorResearch]
    email_draft: str
    quality_assessment: Optional[EmailQuality]
    
    # New fields for document processing
    cv_summary: Optional[CVSummary]
    cv_concise: str
    processed_context: str
    context_metadata: Dict[str, Any]
    
    # Workflow control
    iteration_count: int
    max_iterations: int
    approved_by_user: bool
    email_sent: bool
    user_approval: Optional[str]  # Add this field
    
    # Logging
    audit_log: List[Dict[str, Any]]
    error_messages: List[str]

class GraduateGuideAgent:
    """Main GraduateGuide Agent using LangGraph"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the agent with configuration"""
        self.config = self._load_config(config_path)
        self.llm = self._initialize_llm()
        
        # Initialize document processors
        self.cheap_llm = self._initialize_cheap_llm()
        self.doc_processor = DocumentProcessor(self.cheap_llm, self.llm)
        self.context_builder = ProgressiveContextBuilder(max_tokens=6000)
        
        self.workflow = self._build_workflow()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4-turbo-preview",
                "api_key": "your-api-key-here",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your-email@gmail.com",
                "password": "your-app-password",
                "use_tls": True
            },
            "quality_threshold": 7,
            "max_iterations": 3
        }
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        
        if provider == "openai":
            return ChatOpenAI(
                model=llm_config.get("model_name", "gpt-4-turbo-preview"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 2000)
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=llm_config.get("model_name", "gemini-1.5-flash"),
                google_api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_output_tokens=llm_config.get("max_tokens", 2000)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_cheap_llm(self):
        """Initialize cheaper LLM for document processing"""
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-3.5-turbo",  # Use cheaper model
                api_key=llm_config.get("api_key"),
                temperature=0.3,  # Lower temperature for extraction
                max_tokens=1000
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Already a cheaper model
                google_api_key=llm_config.get("api_key"),
                temperature=0.3,
                max_output_tokens=1000
            )
        else:
            return self.llm  # Fallback to main LLM
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GraduateGuideState)
        
        # Add nodes
        workflow.add_node("extract_profile", self.extract_profile_insights)
        workflow.add_node("analyze_professor", self.analyze_professor_research)
        workflow.add_node("generate_email", self.generate_email_draft)
        workflow.add_node("evaluate_quality", self.evaluate_email_quality)
        workflow.add_node("await_approval", self.await_user_approval)
        workflow.add_node("send_email", self.send_email)
        workflow.add_node("log_completion", self.log_completion)
        
        # Define workflow edges
        workflow.set_entry_point("extract_profile")
        workflow.add_edge("extract_profile", "analyze_professor")
        workflow.add_edge("analyze_professor", "generate_email")
        workflow.add_edge("generate_email", "evaluate_quality")
        
        # Conditional routing based on quality assessment
        workflow.add_conditional_edges(
            "evaluate_quality",
            self.should_regenerate,
            {
                "regenerate": "generate_email",
                "approve": "await_approval",
                "max_iterations": "await_approval"
            }
        )
        
        # User approval routing
        workflow.add_conditional_edges(
            "await_approval",
            self.is_approved,
            {
                "send": "send_email",
                "revise": "generate_email",
                "cancel": "log_completion"
            }
        )
        
        workflow.add_edge("send_email", "log_completion")
        workflow.add_edge("log_completion", END)
        
        return workflow.compile()
    
    def extract_profile_insights(self, state: GraduateGuideState) -> GraduateGuideState:
        """Extract insights from user's CV and background"""
        logger.info("Extracting profile insights...")
        
        try:
            # Process CV using document processor
            cv_summary, cv_concise = self.doc_processor.process_cv(state["cv_content"])
            
            # Store structured summary
            state["cv_summary"] = cv_summary
            state["cv_concise"] = cv_concise
            
            # Convert to ProfileInsights for compatibility
            state["profile_insights"] = ProfileInsights(
                research_interests=cv_summary.research_interests,
                technical_skills=cv_summary.technical_skills,
                publications=cv_summary.publications,
                academic_background="; ".join(cv_summary.education),
                relevant_experience=cv_summary.experience
            )
            
            state["audit_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "profile_extraction",
                "status": "success",
                "cv_tokens": len(self.doc_processor.encoding.encode(cv_concise))
            })
            
        except Exception as e:
            logger.error(f"Error extracting profile insights: {e}")
            state["error_messages"].append(f"Profile extraction failed: {str(e)}")
        
        return state
    
    def analyze_professor_research(self, state: GraduateGuideState) -> GraduateGuideState:
        """Analyze professor's research from URL and papers"""
        logger.info("Analyzing professor's research...")
        
        # Scrape professor's profile page
        profile_content = self._scrape_webpage(state["professor_url"])
        
        # Process papers
        paper_summaries = []
        for paper_url in state["paper_urls"][:5]:  # Limit to 5 papers
            paper_content = self._scrape_webpage(paper_url)
            if paper_content:
                try:
                    paper_summary, paper_concise = self.doc_processor.process_paper(paper_content)
                    paper_summaries.append((paper_summary, paper_concise))
                except Exception as e:
                    logger.warning(f"Error processing paper {paper_url}: {e}")
        
        # Build context progressively
        context_info = self.context_builder.build_context(
            cv_summary=state.get("cv_summary"),
            cv_text=state.get("cv_concise", ""),
            professor_summary=profile_content[:1500],  # Limit professor profile
            paper_summaries=paper_summaries,
            background_summary=state["background_summary"]
        )
        
        # Store processed information
        state["processed_context"] = context_info["context"]
        state["context_metadata"] = {
            "token_count": context_info["token_count"],
            "included_sections": context_info["included_sections"]
        }
        
        # Now analyze with full context
        parser = PydanticOutputParser(pydantic_object=ProfessorResearch)
        prompt = ChatPromptTemplate.from_template("""
        Based on the following context, analyze the professor's research and alignment with the student:
        
        {full_context}
        
        Extract and analyze:
        - Main research areas and themes
        - Summary of recent work and contributions  
        - Key methodologies and approaches used
        - Points of alignment with the student's interests
        - Notable findings or innovations
        
        {format_instructions}
        """)
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "full_context": context_info["context"],
                "format_instructions": parser.get_format_instructions()
            })
            
            state["professor_research"] = result
            state["audit_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "professor_analysis",
                "status": "success",
                "context_tokens": context_info["token_count"],
                "papers_processed": len(paper_summaries)
            })
            
        except Exception as e:
            logger.error(f"Error analyzing professor research: {e}")
            state["error_messages"].append(f"Professor analysis failed: {str(e)}")
        
        return state
    
    def generate_email_draft(self, state: GraduateGuideState) -> GraduateGuideState:
        """Generate personalized email draft"""
        logger.info(f"Generating email draft (iteration {state['iteration_count'] + 1})")
        
        # Use the processed context for email generation
        prompt = ChatPromptTemplate.from_template("""
        Generate a professional, personalized email to a professor for graduate school inquiry.
        
        Context and Information:
        {processed_context}
        
        Previous Quality Feedback (if any): {previous_feedback}
        
        Email Guidelines:
        1. Professional but not overly formal tone
        2. Clear subject line starting with "Subject: "
        3. Brief introduction of student (2-3 sentences)
        4. Specific reference to 1-2 of professor's recent papers or work
        5. Clear explanation of research alignment (2-3 specific points)
        6. Mention 1-2 relevant skills or experiences
        7. Specific request (meeting, discussion, application guidance)
        8. Professional closing
        9. Keep to 300-400 words maximum
        
        The email should demonstrate genuine understanding of the professor's work and clear alignment.
        """)
        
        try:
            previous_feedback = ""
            if state["quality_assessment"]:
                previous_feedback = state["quality_assessment"].feedback
            
            result = self.llm.invoke(prompt.format_messages(
                processed_context=state.get("processed_context", ""),
                previous_feedback=previous_feedback
            ))
            
            state["email_draft"] = result.content
            state["iteration_count"] += 1
            state["audit_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "email_generation",
                "iteration": state["iteration_count"],
                "status": "success",
                "context_sections": state.get("context_metadata", {}).get("included_sections", [])
            })
            
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            state["error_messages"].append(f"Email generation failed: {str(e)}")
        
        return state
    
    def evaluate_email_quality(self, state: GraduateGuideState) -> GraduateGuideState:
        """Evaluate the quality of the generated email"""
        logger.info("Evaluating email quality...")
        
        parser = PydanticOutputParser(pydantic_object=EmailQuality)
        prompt = ChatPromptTemplate.from_template("""
        Evaluate the quality of this graduate school inquiry email:
        
        Email Draft:
        {email_draft}
        
        Professor's Research Context:
        {professor_context}
        
        Evaluate on these criteria (1-10 scale):
        1. Professional tone and language
        2. Clear structure and flow
        3. Personalization and specific references to professor's work
        4. Accuracy of any citations or references
        5. Overall quality and effectiveness
        
        Quality threshold: {quality_threshold}/10
        
        Provide specific feedback for improvement if scores are below threshold.
        
        {format_instructions}
        """)
        
        chain = prompt | self.llm | parser
        
        try:
            research = state["professor_research"]
            quality_threshold = self.config.get("quality_threshold", 7)
            
            result = chain.invoke({
                "email_draft": state["email_draft"],
                "professor_context": research.recent_work_summary if research else "",
                "quality_threshold": quality_threshold,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Set passes_threshold based on overall quality
            result.passes_threshold = result.overall_quality >= quality_threshold
            
            state["quality_assessment"] = result
            state["audit_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "quality_evaluation",
                "overall_score": result.overall_quality,
                "passes": result.passes_threshold
            })
            
        except Exception as e:
            logger.error(f"Error evaluating email quality: {e}")
            state["error_messages"].append(f"Quality evaluation failed: {str(e)}")
        
        return state
    
    def should_regenerate(self, state: GraduateGuideState) -> str:
        """Determine if email should be regenerated based on quality"""
        if not state["quality_assessment"]:
            return "approve"  # If evaluation failed, proceed to approval
        
        if state["quality_assessment"].passes_threshold:
            return "approve"
        
        if state["iteration_count"] >= state["max_iterations"]:
            return "max_iterations"
        
        return "regenerate"
    
    def await_user_approval(self, state: GraduateGuideState) -> GraduateGuideState:
        """Present email to user for approval"""
        logger.info("Awaiting user approval...")
        
        # print("\n" + "="*50)
        # print("GENERATED EMAIL DRAFT")
        # print("="*50)
        # print(state["email_draft"])
        # print("="*50)
        
        # if state["quality_assessment"]:
        #     print(f"\nQuality Score: {state['quality_assessment'].overall_quality}/10")
        #     if state["quality_assessment"].feedback:
        #         print(f"Feedback: {state['quality_assessment'].feedback}")
        
        # print("\nOptions:")
        # print("1. Send email")
        # print("2. Revise email")
        # print("3. Cancel")
        
        # choice = input("\nYour choice (1-3): ").strip()
        
        # if choice == "1":
        #     state["approved_by_user"] = True
        # elif choice == "2":
        #     state["approved_by_user"] = False
        #     # Reset iteration count for user-requested revision
        #     if state["iteration_count"] >= state["max_iterations"]:
        #         state["iteration_count"] = 0
        # else:
        #     state["approved_by_user"] = None  # Cancel
        
        return state
    
    def is_approved(self, state: GraduateGuideState) -> str:
        """Check user approval status from Streamlit"""
        # Instead of CLI, get approval from state (set by Streamlit)
        approval = state.get("user_approval")
        if approval == "Send email":
            return "send"
        elif approval == "Revise email":
            return "revise"
        else:
            return "cancel"
    
    def send_email(self, state: GraduateGuideState) -> GraduateGuideState:
        """Send the approved email via SMTP"""
        logger.info("Sending email...")
        
        try:
            email_config = self.config.get("email", {})
            
            # Extract subject and body from email draft
            lines = state["email_draft"].split('\n')
            subject_line = lines[0] if lines[0].startswith('Subject:') else "Graduate School Inquiry"
            if subject_line.startswith('Subject:'):
                subject = subject_line.replace('Subject:', '').strip()
                body = '\n'.join(lines[1:]).strip()
            else:
                subject = "Graduate School Inquiry"
                body = state["email_draft"]
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config.get("username")
            msg['To'] = input("Enter professor's email address: ")
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config.get("smtp_server"), email_config.get("smtp_port")) as server:
                if email_config.get("use_tls"):
                    server.starttls()
                server.login(email_config.get("username"), email_config.get("password"))
                server.send_message(msg)
            
            state["email_sent"] = True
            state["audit_log"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "email_sent",
                "recipient": msg['To'],
                "subject": subject,
                "status": "success"
            })
            
            logger.info("Email sent successfully!")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            state["error_messages"].append(f"Email sending failed: {str(e)}")
            state["email_sent"] = False
        
        return state
    
    def log_completion(self, state: GraduateGuideState) -> GraduateGuideState:
        """Log completion and cleanup"""
        logger.info("Workflow completed")
        
        state["audit_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "workflow_completed",
            "email_sent": state.get("email_sent", False),
            "total_iterations": state["iteration_count"]
        })
        os.makedirs("logs", exist_ok=True)
        # Save audit log to file
        log_filename = f"logs/graduate_guide_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_filename, 'w') as f:
            json.dump(state["audit_log"], f, indent=2)
        
        logger.info(f"Audit log saved to {log_filename}")
        return state
    
    def _scrape_webpage(self, url: str) -> str:
        """Scrape content from a webpage"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:500]  # Limit content length
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def run(self, cv_content: str, background_summary: str, target_filters: Dict[str, Any],
            professor_url: str, paper_urls: List[str]) -> Dict[str, Any]:
        """Run the complete GraduateGuide workflow"""
        
        initial_state = GraduateGuideState(
            cv_content=cv_content,
            background_summary=background_summary,
            target_filters=target_filters,
            professor_url=professor_url,
            paper_urls=paper_urls,
            profile_insights=None,
            professor_research=None,
            email_draft="",
            quality_assessment=None,
            cv_summary=None,
            cv_concise="",
            processed_context="",
            context_metadata={},
            iteration_count=0,
            max_iterations=self.config.get("max_iterations", 3),
            approved_by_user=False,
            email_sent=False,
            audit_log=[],
            error_messages=[]
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "email_sent": final_state.get("email_sent", False),
            "final_draft": final_state.get("email_draft", ""),
            "iterations": final_state.get("iteration_count", 0),
            "errors": final_state.get("error_messages", []),
            "audit_log": final_state.get("audit_log", [])
        }

# Usage Example
if __name__ == "__main__":
    # Initialize the agent
    agent = GraduateGuideAgent("config.json")
    
    # Example usage
    cv_content = """
    John Doe
    PhD Student in Computer Science
    University of Example
    
    Research Interests: Machine Learning, Natural Language Processing, Computer Vision
    
    Education:
    - MS Computer Science, University of Example (2022)
    - BS Computer Science, State University (2020)
    
    Publications:
    - "Deep Learning Approaches to Text Classification" (2023)
    - "Computer Vision for Medical Imaging" (2022)
    
    Skills: Python, TensorFlow, PyTorch, OpenCV, NLTK
    """
    
    background_summary = """
    I am a PhD student passionate about applying machine learning to real-world problems,
    particularly in healthcare and natural language understanding. I have experience
    with deep learning frameworks and have published work in text classification and
    medical imaging applications.
    """
    
    target_filters = {
        "country": "USA",
        "universities": ["Stanford", "MIT", "CMU"]
    }
    
    professor_url = "https://example-university.edu/faculty/professor-smith"
    paper_urls = [
        "https://arxiv.org/abs/example1",
        "https://arxiv.org/abs/example2"
    ]
    
    # Run the workflow
    result = agent.run(
        cv_content=cv_content,
        background_summary=background_summary,
        target_filters=target_filters,
        professor_url=professor_url,
        paper_urls=paper_urls
    )
    
    print("Workflow Results:")
    print(f"Email sent: {result['email_sent']}")
    print(f"Iterations: {result['iterations']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")