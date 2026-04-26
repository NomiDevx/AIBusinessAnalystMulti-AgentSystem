import os
import requests
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

load_dotenv()

# ==============================
# 🔐 API KEYS
# ==============================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# ==============================
# 🧾 STATE
# ==============================
class AgentState(TypedDict):
    user_query: str
    tasks: List[str]
    research_data: List[str]
    analysis: str
    feedback: str
    needs_more_data: bool
    final_report: str


# ==============================
# 🤖 DEEPSEEK ONLY
# ==============================
def call_deepseek(prompt: str) -> str:
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"DEEPSEEK ERROR: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# ==============================
# 🌐 SERPER SEARCH
# ==============================
def search_serper(query: str):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"q": query}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return []

    results = response.json()

    links = []
    snippets = []

    for item in results.get("organic", [])[:3]:
        links.append(item.get("link"))
        snippets.append(item.get("snippet", ""))

    return links, snippets


# ==============================
# 🕷️ SELENIUM SCRAPER
# ==============================
def scrape_with_selenium(urls: List[str]) -> List[str]:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    scraped_data = []

    for url in urls:
        try:
            driver.get(url)
            text = driver.find_element("tag name", "body").text
            scraped_data.append(text[:2000])  # limit text
        except:
            continue

    driver.quit()
    return scraped_data


# ==============================
# 🧠 AGENTS
# ==============================

# 1️⃣ Planner
def planner_agent(state: AgentState) -> AgentState:
    prompt = f"""
    Break this into research tasks:
    {state['user_query']}

    Return Python list only.
    """

    response = call_deepseek(prompt)

    try:
        tasks = eval(response)
    except:
        tasks = [state["user_query"]]

    return {**state, "tasks": tasks}


# 2️⃣ Research (SERPER + Selenium)
def research_agent(state: AgentState) -> AgentState:
    all_data = []

    for task in state["tasks"]:
        links, snippets = search_serper(task)

        all_data.extend(snippets)

        # 🔥 Selenium fallback
        if links:
            scraped = scrape_with_selenium(links)
            all_data.extend(scraped)

    return {**state, "research_data": all_data}


# 3️⃣ Analysis
def analysis_agent(state: AgentState) -> AgentState:
    data = "\n".join(state["research_data"][:5000])

    prompt = f"""
    Analyze this data:

    {data}

    Provide:
    - Key Insights
    - Trends
    - SWOT Analysis
    """

    analysis = call_deepseek(prompt)

    return {**state, "analysis": analysis}


# 4️⃣ Critic
def critic_agent(state: AgentState) -> AgentState:
    prompt = f"""
    Review this analysis:

    {state['analysis']}

    Respond:
    needs_more_data: true/false
    feedback: explanation
    """

    response = call_deepseek(prompt)

    needs_more_data = "true" in response.lower()

    return {
        **state,
        "feedback": response,
        "needs_more_data": needs_more_data
    }


# 5️⃣ Report
def report_agent(state: AgentState) -> AgentState:
    prompt = f"""
    Create a professional business report:

    Query: {state['user_query']}
    Analysis: {state['analysis']}

    Include:
    - Executive Summary
    - Insights
    - Recommendations
    """

    report = call_deepseek(prompt)

    return {**state, "final_report": report}


# ==============================
# 🔁 GRAPH
# ==============================
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "critic")

    def decide(state):
        return "research" if state["needs_more_data"] else "report"

    graph.add_conditional_edges("critic", decide)
    graph.add_edge("report", "__end__")

    return graph.compile()


# ==============================
# 🚀 RUN
# ==============================
if __name__ == "__main__":
    app = build_graph()

    user_query = input("Enter your business query: ")

    result = app.invoke({
        "user_query": user_query,
        "tasks": [],
        "research_data": [],
        "analysis": "",
        "feedback": "",
        "needs_more_data": False,
        "final_report": ""
    })

    print("\n📊 FINAL REPORT:\n")
    print(result["final_report"])