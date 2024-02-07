from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain.llms import Ollama
ollama_llm = Ollama(model="openhermes")


from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()


# Define your CrewAI agents and tasks
# Define Agents
fact_checking_agent = Agent(
    role='Fact-Checking Agent',
    goal='Verify the factual accuracy of the news article or statement.',
    backstory='Expert in data verification and fact-checking, skilled in discerning truth from fiction in news reporting.',
    tools=[search_tool],
    verbose=True,
    llm=ollama_llm
)

political_analyst_agent = Agent(
    role='Political Analyst Agent',
    goal='Provide context and political analysis on Pakistan.',
    backstory='Specializes in South Asian geopolitics, focusing on Pakistan.',
    verbose=True,
    llm=ollama_llm
)

media_bias_analyst_agent = Agent(
    role='Media Bias Analyst Agent',
    goal='Assess potential biases in the news source.',
    backstory='Expert in media studies, focusing on detecting biases in news reporting.',
    verbose=True,
    llm=ollama_llm
)

public_sentiment_analyst_agent = Agent(
    role='Public Sentiment Analyst Agent',
    goal='Gauge public reaction to the news.',
    backstory='Skilled in analyzing public opinion and sentiment on social media and online forums.',
    tools=[search_tool],
    verbose=True,
    llm=ollama_llm
)
# Define a function to integrate the tools with CrewAI
def analyze_news_article(content):

    fact_checking_task = Task(
        description=f'Analyze the news article: {content} for factual accuracy. Final answer must be a detailed report on factual findings.',
        agent=fact_checking_agent
    )

    political_analysis_task = Task(
        description=f'Analyze the political context of the news article: {content}. Final answer must include an assessment of the current political situation and its credibility.',
        agent=political_analyst_agent
    )

    media_bias_analysis_task = Task(
        description=f'Evaluate the news source: {content} for biases and report on potential influences on the article\'s narrative. Final answer must include an analysis of media bias.',
        agent=media_bias_analyst_agent
    )

    public_sentiment_analysis_task = Task(
        description=f'Analyze public reaction to the news: {content} on social media and forums. Final answer must summarize public sentiment.',
        agent=public_sentiment_analyst_agent
    )


    crew = Crew(
        agents=[fact_checking_agent, political_analyst_agent, media_bias_analyst_agent, public_sentiment_analyst_agent],  # Other agents
        tasks=[fact_checking_task,political_analysis_task, media_bias_analysis_task, public_sentiment_analysis_task],    # Other tasks
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result

#  usage
final_result = analyze_news_article("Feb 8 2024 Elections have been cancelled in Pakistan. The government has declared a state of emergency. The military has taken over the country.")
print(final_result)