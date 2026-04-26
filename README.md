# AI Business Analyst Multi-Agent System
Multi-agent AI pipeline using LangGraph that autonomously researches and reports on business queries. A Planner breaks down the query, a Research agent fetches data via Serper API and Selenium scraping, an Analysis agent generates insights and SWOT using DeepSeek LLM, a Critic loops back if needed, and a Report agent delivers the final output.


#  AI Business Research Agent

A multi-agent AI pipeline built with **LangGraph** that autonomously researches, analyzes, and reports on any business query using web search, live scraping, and the DeepSeek LLM.

## 🧠 How It Works

The system chains five specialized agents in a directed graph:

| Agent | Role |
|---|---|
| **Planner** | Breaks the user's query into discrete research tasks |
| **Research** | Searches the web via Serper API and scrapes pages with Selenium |
| **Analysis** | Synthesizes findings into insights, trends, and a SWOT analysis |
| **Critic** | Evaluates the analysis and decides if more research is needed |
| **Report** | Produces a professional business report with recommendations |

The Critic can loop the pipeline back to the Research agent if the analysis is deemed insufficient — creating a self-improving feedback cycle.

## 🛠️ Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful multi-agent orchestration
- **[DeepSeek API](https://platform.deepseek.com/)** — LLM for planning, analysis, and reporting
- **[Serper API](https://serper.dev/)** — Google Search results
- **[Selenium + ChromeDriver](https://www.selenium.dev/)** — dynamic web scraping fallback
- **Python 3.10+**

## 📦 Installation

```bash
git clone https://github.com/yourusername/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
```

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
SERPER_API_KEY=your_serper_api_key
```

## 🚀 Usage

```bash
python agent.py
```

You'll be prompted to enter a business query, for example:
