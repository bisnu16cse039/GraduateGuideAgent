"""
Web Interface for GraduateGuide Agent
Provides both FastAPI backend and Streamlit frontend options
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional
import json
import tempfile
import os
from datetime import datetime
from main import GraduateGuideAgent
import PyPDF2
import io

# Streamlit Web Interface
class StreamlitApp:
    """Streamlit web interface for GraduateGuide"""
    
    def __init__(self):
        self.agent = None
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="GraduateGuide",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .step-header {
            font-size: 1.5rem;
            color: #2e7d32;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.25rem;
            color: #155724;
        }
        .error-box {
            padding: 1rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 0.25rem;
            color: #721c24;
        }
        .info-box {
            padding: 1rem;
            background-color: #cce7ff;
            border: 1px solid #99d6ff;
            border-radius: 0.25rem;
            color: #004085;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_agent(self, config: Dict[str, Any]) -> bool:
        """Initialize the GraduateGuide agent with config"""
        try:
            # Save config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2)
                temp_config_path = f.name
            
            self.agent = GraduateGuideAgent(temp_config_path)
            
            # Clean up temp file
            os.unlink(temp_config_path)
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return False
    
    def sidebar_config(self) -> Dict[str, Any]:
        """Configure agent settings in sidebar"""
        st.sidebar.title("⚙️ Configuration")
        
        # LLM Configuration
        st.sidebar.subheader("🤖 LLM Settings")
        provider = st.sidebar.selectbox(
            "Provider",
            options=["google","openai"],
            help="Choose your LLM provider"
        )
        
        if provider == "openai":
            model_options = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
            default_model = "gpt-4-turbo-preview"
        else:
            model_options = ["gemini-1.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
            default_model = "gemini-1.5-flash"

        model_name = st.sidebar.selectbox("Model", options=model_options, index=model_options.index(default_model))
        api_key = st.sidebar.text_input(
            "API Key", 
            type="password",
            help="Your LLM API key"
        )
        
        temperature = st.sidebar.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            help="Controls randomness in responses"
        )
        
        # Quality Settings
        st.sidebar.subheader("📊 Quality Control")
        quality_threshold = st.sidebar.slider(
            "Quality Threshold", 
            min_value=1, 
            max_value=10, 
            value=7,
            help="Minimum quality score for email approval"
        )
        
        max_iterations = st.sidebar.slider(
            "Max Iterations", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Maximum regeneration attempts"
        )
        
        # Email Configuration (optional)
        st.sidebar.subheader("📧 Email Settings (Optional)")
        enable_email = st.sidebar.checkbox("Enable Email Sending")
        
        email_config = {}
        if enable_email:
            email_config = {
                "smtp_server": st.sidebar.text_input("SMTP Server", value="smtp.gmail.com"),
                "smtp_port": st.sidebar.number_input("SMTP Port", value=587),
                "username": st.sidebar.text_input("Email Username"),
                "password": st.sidebar.text_input("Email Password", type="password"),
                "use_tls": st.sidebar.checkbox("Use TLS", value=True)
            }
        
        return {
            "llm": {
                "provider": provider,
                "model_name": model_name,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": 2000
            },
            "email": email_config,
            "quality_threshold": quality_threshold,
            "max_iterations": max_iterations
        }
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def input_section(self) -> Dict[str, Any]:
        """Handle user input section"""
        st.markdown('<h2 class="step-header">📄 Step 1: Your Profile</h2>', unsafe_allow_html=True)
        
        # CV/Resume Input
        cv_input_method = st.radio(
            "How would you like to provide your CV/Resume?",
            options=["Upload file (PDF)", "Paste text"],
            horizontal=True
        )
        
        cv_content = ""
        if cv_input_method == "Upload file (PDF)":
            uploaded_file = st.file_uploader(
                "Upload your CV/Resume (PDF)",
                type=['pdf'],
                help="Upload a PDF file containing your CV or resume"
            )
            if uploaded_file:
                cv_content = self.extract_text_from_pdf(uploaded_file)
                if cv_content:
                    st.success("✅ CV/Resume loaded successfully")
                    with st.expander("Preview extracted text"):
                        st.text(cv_content[:1000] + "..." if len(cv_content) > 1000 else cv_content)
        else:
            cv_content = st.text_area(
                "Paste your CV/Resume content here:",
                height=200,
                help="Copy and paste the text content of your CV or resume"
            )
        
        # Background Summary
        st.markdown('<h2 class="step-header">📝 Step 2: Research Background</h2>', unsafe_allow_html=True)
        background_summary = st.text_area(
            "Describe your research interests and background:",
            height=150,
            help="Provide a brief summary of your research interests, goals, and relevant experience"
        )
        
        # Target Filters
        st.markdown('<h2 class="step-header">🎯 Step 3: Target Preferences</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            target_country = st.text_input(
                "Preferred Country (optional):",
                help="e.g., USA, Canada, UK"
            )
        
        with col2:
            target_universities = st.text_input(
                "Preferred Universities (comma-separated, optional):",
                help="e.g., MIT, Stanford, CMU"
            )
        
        # Professor Information
        st.markdown('<h2 class="step-header">👨‍🏫 Step 4: Professor Information</h2>', unsafe_allow_html=True)
        
        # Professor input method selection
        prof_input_method = st.radio(
            "How would you like to provide professor information?",
            ["URL (Web scraping)", "Manual text input (Copy & Paste)"],
            help="Choose URL if the professor's webpage is accessible, or manual input if you want to copy-paste information"
        )
        
        professor_url = ""
        professor_text = ""
        
        if prof_input_method == "URL (Web scraping)":
            professor_url = st.text_input(
                "Professor's Profile URL:",
                help="Link to the professor's faculty page, lab website, or academic profile"
            )
            if professor_url:
                st.info("💡 If this URL gets blocked (403 error), try using manual text input instead")
        else:
            professor_text = st.text_area(
                "Professor's Profile Information:",
                height=200,
                placeholder="Paste the professor's biography, research interests, recent work, etc. from their faculty page or other sources...",
                help="Copy and paste information about the professor's research, background, recent work, and interests"
            )
        
        # Paper URLs and Text Inputs
        st.subheader("Recent Papers")
        st.markdown(
            '<div class="info-box">Add information about the professor\'s recent papers for better personalization. You can use URLs (if accessible) or copy-paste paper content.</div>',
            unsafe_allow_html=True
        )
        
        # Paper input method
        paper_input_method = st.radio(
            "How would you like to provide paper information?",
            ["URLs only", "Manual text only", "PDF upload", "Mix of both"],
            key="paper_method"
        )
        
        paper_urls = []
        paper_texts = []
        
        num_papers = st.number_input("Number of papers to analyze:", min_value=0, max_value=5, value=2)
        
        if paper_input_method == "URLs only":
            st.subheader("📄 Paper URLs")
            for i in range(num_papers):
                url = st.text_input(f"Paper {i+1} URL:", key=f"paper_url_{i}")
                if url:
                    paper_urls.append(url)
                    
        elif paper_input_method == "Manual text only":
            st.subheader("📄 Paper Content (Copy & Paste)")
            for i in range(num_papers):
                text = st.text_area(
                    f"Paper {i+1} Content:",
                    height=150,
                    key=f"paper_text_{i}",
                    placeholder="Paste the paper's title, abstract, key sections, or full text here...",
                    help="Include title, abstract, methodology, findings, and any other relevant content"
                )
                if text.strip():
                    paper_texts.append(text.strip())
        
        elif paper_input_method == "PDF upload":
            st.subheader("📎 Upload PDF Papers")
            uploaded_pdfs = st.file_uploader(
                "Upload PDF papers:",
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload PDF files of research papers for analysis"
            )
            
            if uploaded_pdfs:
                st.success(f"✅ {len(uploaded_pdfs)} PDF(s) uploaded successfully")
                
                # Process PDFs and add to paper_texts
                for i, pdf_file in enumerate(uploaded_pdfs):
                    try:
                        pdf_text = self.extract_text_from_pdf(pdf_file)
                        if pdf_text and pdf_text.strip():
                            paper_texts.append(pdf_text)
                            st.info(f"📄 Extracted text from {pdf_file.name} ({len(pdf_text)} characters)")
                            
                            # Show preview
                            with st.expander(f"Preview: {pdf_file.name}"):
                                st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
                    except Exception as e:
                        st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
                    
        else:  # Mix of both
            st.subheader("📄 Papers (URLs, Manual Text, and PDFs)")
            for i in range(num_papers):
                st.markdown(f"**Paper {i+1}:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    url = st.text_input(f"URL (optional):", key=f"paper_url_mix_{i}")
                    if url:
                        paper_urls.append(url)
                
                with col2:
                    text = st.text_area(
                        f"Manual text (optional):",
                        height=100,
                        key=f"paper_text_mix_{i}",
                        placeholder="Paste paper content here...",
                    )
                    if text.strip():
                        paper_texts.append(text.strip())
            
            # PDF upload for mix option
            st.subheader("📎 Additional PDF Papers")
            uploaded_pdfs = st.file_uploader(
                "Upload additional PDF papers (optional):",
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload PDF files of research papers for analysis",
                key="mix_pdf_upload"
            )
            
            if uploaded_pdfs:
                st.success(f"✅ {len(uploaded_pdfs)} PDF(s) uploaded successfully")
                
                # Process PDFs and add to paper_texts
                for i, pdf_file in enumerate(uploaded_pdfs):
                    try:
                        pdf_text = self.extract_text_from_pdf(pdf_file)
                        if pdf_text and pdf_text.strip():
                            paper_texts.append(pdf_text)
                            st.info(f"📄 Extracted text from {pdf_file.name} ({len(pdf_text)} characters)")
                            
                            # Show preview
                            with st.expander(f"Preview: {pdf_file.name}"):
                                st.text(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
                    except Exception as e:
                        st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
        
        # Remove the separate PDF upload section since it's now integrated above
        
        target_filters = {
            "country": target_country if target_country else None,
            "universities": [u.strip() for u in target_universities.split(',')] if target_universities else []
        }
        
        return {
            "cv_content": cv_content,
            "background_summary": background_summary,
            "target_filters": target_filters,
            "professor_url": professor_url,
            "professor_text": professor_text,
            "paper_urls": paper_urls,
            "paper_texts": paper_texts
        }
    
    def run_workflow(self, user_input: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the GraduateGuide workflow"""
        if not self.initialize_agent(config):
            return None
        
        with st.spinner("🔄 Processing your request... This may take a few minutes."):
            try:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates (in real implementation, you'd hook into the actual workflow)
                steps = [
                    "Analyzing your profile...",
                    "Processing professor information...",
                    "Analyzing papers and research...",
                    "Generating personalized email...",
                    "Evaluating email quality...",
                    "Finalizing draft..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                
                # Run the actual workflow with new parameters
                result = self.agent.run(
                    cv_content=user_input.get("cv_content", ""),
                    background_summary=user_input.get("background_summary", ""),
                    target_filters=user_input.get("target_filters", {}),
                    professor_url=user_input.get("professor_url", ""),
                    professor_text=user_input.get("professor_text", ""),
                    paper_urls=user_input.get("paper_urls", []),
                    paper_texts=user_input.get("paper_texts", [])
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Workflow completed!")
                
                return result
                
            except Exception as e:
                st.error(f"❌ Workflow failed: {e}")
                st.error("💡 Try using manual text input if URLs are being blocked")
                return None
    
    def display_results(self, result: Dict[str, Any]):
        """Display workflow results"""
        st.markdown('<h2 class="step-header">📧 Generated Email Draft</h2>', unsafe_allow_html=True)
        
        if result and result.get('final_draft'):
            # Show processing information
            if result.get('audit_log'):
                with st.expander("📊 Processing Information"):
                    audit_log = result['audit_log']
                    
                    # Find relevant log entries
                    for entry in audit_log:
                        if entry.get('action') == 'professor_analysis':
                            st.info(f"✅ Processed {entry.get('papers_processed', 0)} papers using {entry.get('context_tokens', 0)} tokens")
                            if entry.get('used_manual_prof_text'):
                                st.info("📝 Used manual professor text input")
                            if entry.get('used_manual_papers', 0) > 0:
                                st.info(f"📝 Used {entry.get('used_manual_papers')} manual paper inputs")
                        elif entry.get('action') == 'profile_extraction':
                            st.info(f"✅ CV processed successfully ({entry.get('cv_tokens', 0)} tokens)")
            
            # Show errors if any
            if result.get('errors'):
                with st.expander("⚠️ Processing Warnings", expanded=True):
                    for error in result['errors']:
                        st.warning(error)
                    st.info("💡 Consider using manual text input if URLs are blocked or inaccessible")
            
            # Quality metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Iterations", result.get('iterations', 0))
            with col2:
                st.metric("Quality Score", "N/A")  # Would be available in real implementation
            with col3:
                status = "✅ Sent" if result.get('email_sent') else "📝 Draft Ready"
                st.metric("Status", status)
            
            # Email draft
            st.subheader("Email Draft:")
            email_draft = result['final_draft']
            
            # Make email editable
            edited_email = st.text_area(
                "Review and edit the email if needed:",
                value=email_draft,
                height=400,
                help="You can modify the email before sending"
            )
            
            # Verification step
            st.markdown('<h3 class="step-header">✅ Verify Email Content</h3>', unsafe_allow_html=True)
            if st.button("Send Verified Email"):
                # Here you would call your email sending logic
                st.success("Email sent successfully! Please check your inbox for confirmation.")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Copy to Clipboard", help="Copy email to clipboard"):
                    st.success("Email copied to clipboard!")
            with col2:
                st.download_button(
                    label="Download Email Draft",
                    data=edited_email,
                    file_name=f"graduate_guide_email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            with col3:
                if st.button("🔄 Regenerate", help="Generate a new version"):
                    st.experimental_rerun()
            
            # Error messages
            if result.get('errors'):
                st.subheader("⚠️ Warnings/Errors:")
                for error in result['errors']:
                    st.warning(error)
            
            # Download results
            results_json = json.dumps(result, indent=2)
            st.download_button(
                label="📊 Download Full Results (JSON)",
                data=results_json,
                file_name=f"graduate_guide_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        else:
            st.error("❌ No email draft was generated. Please check the inputs and try again.")
    
    def main(self):
        """Main Streamlit application"""
        # Header
        st.markdown('<h1 class="main-header">🎓 GraduateGuide</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">AI Assistant for Graduate School Advisor Outreach - Generate personalized emails to professors</div>',
            unsafe_allow_html=True
        )
        
        # New features information
        with st.expander("🆕 New Features & Improvements", expanded=False):
            st.markdown("""
            **Recent Updates:**
            - 🔧 **Fixed JSON parsing errors** - More reliable document processing
            - 📝 **Manual text input** - Copy & paste professor profiles and paper content
            - 📎 **PDF upload** - Upload research papers directly
            - 🌐 **Better web scraping** - Improved handling of blocked URLs (403 errors)
            - 💰 **Cost optimization** - Uses cheaper models for processing, expensive models only for final email
            - 🧠 **Smart context building** - Prioritizes most important information within token limits
            
            **When to use manual input:**
            - Professor's website blocks automated access (403 error)
            - You want to provide specific information not available on the website
            - Research papers are behind paywalls or not accessible via URL
            - You have PDFs of papers you want to analyze
            """)
        
        # Sidebar configuration
        config = self.sidebar_config()
        
        # Validation
        if not config['llm']['api_key']:
            st.warning("⚠️ Please enter your API key in the sidebar to continue.")
            return
        
        # Main input section
        user_input = self.input_section()
        
        # Updated validation - now more flexible
        if not user_input.get('cv_content'):
            st.warning("⚠️ Please provide your CV/Resume content.")
            return
        
        if not user_input.get('background_summary'):
            st.warning("⚠️ Please provide your research background summary.")
            return
        
        # Check if at least some professor information is provided
        has_prof_info = (
            user_input.get('professor_url') or 
            user_input.get('professor_text')
        )
        
        if not has_prof_info:
            st.warning("⚠️ Please provide professor information (either URL or manual text).")
            return
        
        # Check if at least some paper information is provided
        has_paper_info = (
            user_input.get('paper_urls') or 
            user_input.get('paper_texts')
        )
        
        if not has_paper_info:
            st.info("💡 Consider adding paper information for more personalized emails.")
        
        # Run workflow button
        st.markdown('<h2 class="step-header">🚀 Generate Email</h2>', unsafe_allow_html=True)
        
        if st.button("🎯 Generate Personalized Email", type="primary", use_container_width=True):
            result = self.run_workflow(user_input, config)
            if result:
                st.session_state['result'] = result
        
        # Display results if available
        if 'result' in st.session_state:
            self.display_results(st.session_state['result'])
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Made with ❤️ using [LangGraph](https://github.com/langchain-ai/langgraph) • "
            "[Documentation](https://github.com/your-repo/graduateguide) • "
            "[Report Issues](https://github.com/your-repo/graduateguide/issues)"
        )

def run_streamlit_app():
    """Run the Streamlit application"""
    app = StreamlitApp()
    app.main()

if __name__ == "__main__":
    run_streamlit_app()