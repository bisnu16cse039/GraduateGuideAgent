#!/usr/bin/env python3
"""
Command Line Interface for GraduateGuide Agent
Provides an interactive CLI for using the GraduateGuide system
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from graduate_guide import GraduateGuideAgent
import PyPDF2
import io

class GraduateGuideCLI:
    """Command Line Interface for GraduateGuide"""
    
    def __init__(self):
        self.agent = None
        self.config_path = "config.json"
    
    def initialize_agent(self, config_path: str = None):
        """Initialize the GraduateGuide agent"""
        if config_path:
            self.config_path = config_path
        
        try:
            self.agent = GraduateGuideAgent(self.config_path)
            print(f"✅ GraduateGuide agent initialized with config: {self.config_path}")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            sys.exit(1)
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file (supports txt, pdf)"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == '.pdf':
            return self._read_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")
    
    def collect_user_input(self) -> Dict[str, Any]:
        """Collect input from user interactively"""
        print("\n🎓 Welcome to GraduateGuide!")
        print("Let's help you create a personalized outreach email to a professor.\n")
        
        # Step 1: CV/Resume input
        print("📄 Step 1: CV/Resume Input")
        cv_choice = input("Do you have a CV/Resume file? (y/n): ").lower().strip()
        
        if cv_choice == 'y':
            cv_path = input("Enter the path to your CV/Resume file (PDF or TXT): ").strip()
            try:
                cv_content = self.read_file(cv_path)
                print("✅ CV/Resume loaded successfully")
            except Exception as e:
                print(f"❌ Error reading CV: {e}")
                cv_content = input("Please paste your CV content here:\n")
        else:
            cv_content = input("Please paste your CV/Resume content here:\n")
        
        # Step 2: Background summary
        print("\n📝 Step 2: Research Background")
        background_summary = input("Provide a brief summary of your research interests and background:\n")
        
        # Step 3: Target filters
        print("\n🎯 Step 3: Target Preferences")
        target_country = input("Preferred country (optional): ").strip() or None
        target_universities = input("Preferred universities (comma-separated, optional): ").strip()
        target_unis_list = [u.strip() for u in target_universities.split(',')] if target_universities else []
        
        target_filters = {
            "country": target_country,
            "universities": target_unis_list
        }
        
        # Step 4: Professor information
        print("\n👨‍🏫 Step 4: Professor Information")
        professor_url = input("Professor's profile URL (lab website, faculty page, etc.): ").strip()
        
        print("Enter URLs of the professor's recent papers (one per line, press Enter twice when done):")
        paper_urls = []
        while True:
            url = input().strip()
            if not url:
                break
            paper_urls.append(url)
        
        if not paper_urls:
            print("⚠️  No paper URLs provided. The email may be less personalized.")
        
        return {
            "cv_content": cv_content,
            "background_summary": background_summary,
            "target_filters": target_filters,
            "professor_url": professor_url,
            "paper_urls": paper_urls
        }
    
    def run_interactive(self):
        """Run the interactive CLI interface"""
        if not self.agent:
            self.initialize_agent()
        
        try:
            # Collect user input
            user_input = self.collect_user_input()
            
            print("\n🚀 Starting GraduateGuide workflow...")
            print("This may take a few minutes as we analyze your profile and the professor's research.\n")
            
            # Run the agent
            result = self.agent.run(**user_input)
            
            # Display results
            self.display_results(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Workflow cancelled by user.")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
    
    def run_batch(self, input_file: str):
        """Run in batch mode with JSON input file"""
        if not self.agent:
            self.initialize_agent()
        
        try:
            with open(input_file, 'r') as f:
                batch_data = json.load(f)
            
            print(f"📦 Running batch processing from: {input_file}")
            
            if isinstance(batch_data, list):
                # Process multiple entries
                for i, entry in enumerate(batch_data):
                    print(f"\n📧 Processing entry {i+1}/{len(batch_data)}")
                    result = self.agent.run(**entry)
                    self.display_results(result, entry_num=i+1)
            else:
                # Process single entry
                result = self.agent.run(**batch_data)
                self.display_results(result)
                
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    def display_results(self, result: Dict[str, Any], entry_num: int = None):
        """Display workflow results"""
        prefix = f"[Entry {entry_num}] " if entry_num else ""
        
        print(f"\n{'='*60}")
        print(f"{prefix}GRADUATEGUIDE RESULTS")
        print(f"{'='*60}")
        
        if result['email_sent']:
            print("✅ Email sent successfully!")
        else:
            print("📧 Email draft ready (not sent)")
        
        print(f"🔄 Iterations: {result['iterations']}")
        
        if result['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in result['errors']:
                print(f"   - {error}")
        
        if result['final_draft']:
            print(f"\n📧 Final Email Draft:")
            print("-" * 40)
            print(result['final_draft'])
            print("-" * 40)
        
        # Save results to file
        timestamp = result['audit_log'][-1]['timestamp'] if result['audit_log'] else "unknown"
        output_file = f"graduateguide_result_{timestamp.replace(':', '-')}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    def test_configuration(self):
        """Test the configuration and LLM connectivity"""
        print("🔧 Testing GraduateGuide configuration...")
        
        try:
            # Test agent initialization
            self.initialize_agent()
            
            # Test LLM connectivity
            test_message = "Hello, this is a test message."
            response = self.agent.llm.invoke(test_message)
            
            print("✅ Configuration test passed!")
            print(f"✅ LLM Provider: {self.agent.config['llm']['provider']}")
            print(f"✅ Model: {self.agent.config['llm']['model_name']}")
            print(f"✅ LLM Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            print("\nPlease check:")
            print("1. Your config.json file exists and is valid")
            print("2. Your API keys are correctly set")
            print("3. You have internet connectivity")
    
    def create_sample_config(self):
        """Create a sample configuration file"""
        sample_config = {
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4-turbo-preview",
                "api_key": "your-openai-api-key-here",
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
        
        config_file = "config_sample.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"📝 Sample configuration created: {config_file}")
        print("Please copy this to config.json and update with your actual API keys and email settings.")
    
    def show_help(self):
        """Show detailed help information"""
        help_text = """
🎓 GraduateGuide Help

USAGE:
  python cli.py [command] [options]

COMMANDS:
  run            Run interactive mode (default)
  batch <file>   Run batch processing from JSON file
  test           Test configuration and LLM connectivity  
  config         Create sample configuration file
  help           Show this help message

INTERACTIVE MODE:
  The interactive mode will guide you through:
  1. Uploading/pasting your CV/Resume
  2. Describing your research background
  3. Setting target preferences
  4. Providing professor's profile and paper URLs
  5. Reviewing and approving the generated email

BATCH MODE:
  Create a JSON file with the following structure:
  {
    "cv_content": "Your CV content...",
    "background_summary": "Your research background...",
    "target_filters": {"country": "USA", "universities": ["MIT", "Stanford"]},
    "professor_url": "https://example.edu/faculty/prof-smith",
    "paper_urls": ["https://arxiv.org/abs/1234.5678"]
  }

CONFIGURATION:
  The system uses config.json for settings. Key configurations:
  
  LLM Provider (OpenAI or Google):
  - Set provider to "openai" or "google"
  - Add appropriate API key
  - Choose model name (e.g., "gpt-4-turbo-preview" or "gemini-pro")
  
  Email Settings:
  - Configure SMTP server details for sending emails
  - Use app passwords for Gmail
  
  Quality Control:
  - Set quality_threshold (1-10) for email approval
  - Set max_iterations for regeneration attempts

EXAMPLES:
  python cli.py                    # Interactive mode
  python cli.py batch input.json   # Batch processing
  python cli.py test              # Test configuration
  python cli.py config            # Create sample config

For more information, visit: https://github.com/your-repo/graduateguide
        """
        print(help_text)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GraduateGuide - AI Assistant for Graduate School Advisor Outreach",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command', 
        nargs='?', 
        default='run',
        choices=['run', 'batch', 'test', 'config', 'help'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input file for batch processing'
    )
    
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    args = parser.parse_args()
    
    cli = GraduateGuideCLI()
    
    if args.command == 'run':
        cli.run_interactive()
    elif args.command == 'batch':
        if not args.input_file:
            print("❌ Batch mode requires an input file")
            print("Usage: python cli.py batch <input_file.json>")
            sys.exit(1)
        cli.run_batch(args.input_file)
    elif args.command == 'test':
        cli.test_configuration()
    elif args.command == 'config':
        cli.create_sample_config()
    elif args.command == 'help':
        cli.show_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()