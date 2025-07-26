# GraduateGuide Agent

**GraduateGuide** is an AI-powered assistant to help students generate personalized outreach emails to professors for graduate school applications. It leverages LLMs (OpenAI or Google Gemini), workflow orchestration with LangGraph, and provides both CLI and web (Streamlit) interfaces.

---

## Features

- **Interactive CLI**: Guided prompts to collect your profile, research interests, and target professor details.
- **Web Interface**: User-friendly Streamlit app for uploading CVs, configuring settings, and generating emails.
- **Automated Workflow**: Uses LangGraph to orchestrate profile analysis, professor research, email drafting, and quality evaluation.
- **Personalization**: Scrapes professor profiles and papers for tailored email content.
- **Quality Control**: Automated scoring and feedback for email drafts.
- **Email Sending**: Optional SMTP integration to send emails directly.
- **Batch Processing**: Run multiple inputs from a JSON file.

---

## Project Structure

```
.
├── agent_cli.py         # Command-line interface for GraduateGuide
├── app.py               # Streamlit web application
├── main.py              # Core agent logic and workflow (LangGraph)
├── config.json          # Main configuration file (LLM, email, etc.)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
├── docker-compose.yml   # Docker setup (incomplete, see below)
├── .gitignore
├── .python-version
└── README.md
```

---

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/your-repo/graduateguide.git
   cd graduateguide
   ```

2. **Set up Python environment**

   - Python 3.12 recommended (see `.python-version`)
   - Create a virtual environment:

     ```sh
     python3.12 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

---

## Configuration

- Copy `config_sample.json` (generated via CLI) to `config.json` and fill in your API keys and email credentials.
- Example config fields:
  - LLM provider (`openai` or `google`)
  - Model name (e.g., `gpt-4-turbo-preview`, `gemini-pro`)
  - Email SMTP settings (optional, for sending emails)
  - Quality thresholds and workflow settings

---

## Usage

### 1. Command-Line Interface

Run interactively:

```sh
python agent_cli.py
```

Batch mode (process multiple entries from a JSON file):

```sh
python agent_cli.py batch input.json
```

Test configuration:

```sh
python agent_cli.py test
```

Generate a sample config:

```sh
python agent_cli.py config
```

### 2. Web Interface (Streamlit)

```sh
streamlit run app.py
```

- Upload your CV (PDF or text), enter your background, and professor details.
- Configure LLM and email settings in the sidebar.
- Generate, review, and download or send your personalized email.

---

## Core Logic

- The main workflow is implemented in [`main.py`](main.py) as [`GraduateGuideAgent`](main.py).
- CLI logic is in [`agent_cli.py`](agent_cli.py).
- Streamlit web app is in [`app.py`](app.py).
- Configuration is loaded from [`config.json`](config.json).

---

## Dependencies

See [`requirements.txt`](requirements.txt) for all dependencies, including:

- LangGraph, LangChain, OpenAI, Google Generative AI
- PyPDF2, BeautifulSoup4, Requests, Pandas
- Streamlit (for web UI)
- Email and utility libraries

---

## Docker

A `docker-compose.yml` is present but incomplete. You may need to add a `Dockerfile` or update the compose file for full containerization.

---

## Logging & Output

- Audit logs and results are saved as JSON files after each run.
- Email drafts and workflow details are available for review and download.

---

## Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/your-repo/graduateguide).

---

## License

MIT License (see `LICENSE` file if present).

---

## Acknowledgements

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/).
