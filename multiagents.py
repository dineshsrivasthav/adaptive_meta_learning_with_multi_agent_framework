import os 
from crewai import Agent, Task, Crew, Process
from Tools import tools
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
#import chromadb.utils.embedding_functions as embedding_functions


def get_hf_llm(model_name: str, temperature: float = 0.7, use_endpoint: bool = False):
    """
    Get a HuggingFace LLM either via pipeline or endpoint.
    
    Args:
        model_name: HuggingFace model identifier
        temperature: Sampling temperature
        use_endpoint: If True, use HuggingFaceEndpoint (requires API key), else use local pipeline
    
    Returns:
        LLM instance
    """
    if use_endpoint:
        hf_token = os.environ.get('HUGGINGFACE_API_TOKEN', '')
        if not hf_token:
            raise ValueError(f"HUGGINGFACE_API_TOKEN required for endpoint access to {model_name}")
        return HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=temperature,
            huggingfacehub_api_token=hf_token
        )
    else:
        # Use local pipeline
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 0 else torch.float32,
            device_map="auto" if device == 0 else None
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=temperature,
            device=device
        )
        
        return HuggingFacePipeline(pipeline=pipe)

# Model configurations
USE_ENDPOINT = os.environ.get('USE_HF_ENDPOINT', 'false').lower() == 'true'


llama = get_hf_llm("meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7, use_endpoint=USE_ENDPOINT)
zephyr = get_hf_llm("HuggingFaceH4/zephyr-7b-beta", temperature=0.7, use_endpoint=USE_ENDPOINT)
minitron = get_hf_llm("nvidia/Mistral-NeMo-Minitron-8B-Base", temperature=0.7, use_endpoint=USE_ENDPOINT)
qwen = get_hf_llm("Qwen/Qwen2-7B-Instruct", temperature=0.7, use_endpoint=USE_ENDPOINT)


#from langchain_community.embeddings import SentenceTransformerEmbeddings
#embedding_function = SentenceTransformerEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code":True})


QueryStrategist = Agent(
    role="Query Strategist",
    goal="Create a list of related queries for the query: '{user_query}', like branches, aiding in wholesome coverage of the subject",
    backstory=(
        "Rather than relying on surface associations, you operate through epistemic pattern mapping: recognizing latent dimensions, adjacent problem spaces, blind spots, and under-explored angles. Your outputs serve as scaffolding for downstream agents, so you prioritize structural completeness over quantity. You use tools only to deepen the semantic landscape when needed."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=llama 
)


DataCollector = Agent(
    role="Data Collector",
    goal="With the help of assigned tools, collect all the relevant and useful information for all the queries given in the context, that helps in synthesizing image deepfake attacks",
    backstory=(
        "You navigate heterogenous information environments by balancing precision, breadth, and evidential integrity, with your capable expertise in using web search tools, and research tools for collecting all the relevant and useful information. Your primary job is to get all that information that helps other agents in synthesizing image deepfake attack patterns."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=llama 
)


ReportGenerator = Agent(
    role="Report Generator",
    goal="Create an exhaustive report using all the collected information",
    backstory=(
        "Your orientation is towards structural clarity. You translate a messy cloud of data into a flowing document with transitions, conceptual hierarchies, and cross-referential reasoning. You sense when information requires synthesis vs. exposition, presenting a report that can serve as a foundation for advanced pattern construction."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=zephyr  
)


InfoRefiner = Agent(
    role="Information refiner",
    goal="Refine the given report or comprehensive information, removing any redundant information, and enhancing the content.",
    backstory=(
        "With a strong sense of detangling complex information, you specialize in information sculpting: detecting conceptual repetition, smoothing transitions, elevating clarity, and reinforcing argumentative coherence. You refine with restraint—never diluting meaning, but shaping it for maximum interpretability and downstream utility."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=zephyr  
)


AttackPatternSynthesizer = Agent(
    role="Attack pattern synthesizer",
    goal="Synthesize novel image deepfake attack patterns with detailed steps",
    backstory=(
        "You operate by fusing technical insight, threat-actor reasoning models, and modal foresight. You infer how real-world manipulations evolve, identifying transformation chains, morphological transitions in images, and the underlying “strategy” behind each attack pattern. You articulate these patterns with enough specificity that they can later be translated into LVM prompts without ambiguity."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=minitron 
)


PromptGenerator = Agent(
    role="Prompt generator for language-vision model",
    goal="Generate clear, consise, effective prompts describing scenarios, people, etc., suitable for passing to a language-vision model that helps in generating images of specified characteristics.",
    backstory=(
        "With a keen sense of what goes well with language-vision models (LVM), you exactly know how to prompt LVM to get the right images perfectly matching the description. With this expertise, you excel at comprehending long descriptions into consise, clear and effective prompts for LVM for ultra-realistic high-quality image generation."
    ),
    tools=tools,
    verbose=True,
    max_iter=50,
    allow_delegation=False,
    llm=qwen  
)


######Tasks

QueryStrategist_task = Task(
    description="Create a list of related queries for the query:'{user_query}', like branches, aiding in wholesome coverage of the subject from different persepectives, and sources",
    expected_output="A list of related queries for the '{user_query}'",
    agent=QueryStrategist,
    output_file="related_queriesm.md",
)

DataCollector_task = Task(
    description="Collect every possible piece of information for all the given (sub-)queries using one or more appropriate tools as suitable.",
    expected_output="An ocean of information with all possible contextually relevant information about all the queries asked for",
    agent=DataCollector,
    context = [QueryStrategist_task],
    output_file="collected_infom.md" 
)

ReportGenerator_task = Task(
    description="Create an exhaustive and comprehensive report with suitable flow giving a holistic coverage of the the entire information collected for all the branches of sub-queries for '{user_query}'",
    expected_output="A comprehensive report of the collected information useful for image deepfake attack synthesis",
    agent=ReportGenerator,
    context = [DataCollector_task],
    output_file="generated_reportm.md"
)

InfoRefiner_task = Task(
    description="Enhance the report and remove redundant or duplicate content.",
    expected_output="Enhanced deduplicated report of the collected information for image deepfake attack synthesis",
    agent=InfoRefiner,
    context = [ReportGenerator_task],
    output_file="refined_reportm.md" 
)

AttackPatternSynthesizer_task = Task(
    description="Create detailed image deepfake attack (exisiting, predictive, imaginative) patterns clearly explaining and describing the how and what aspects are changed in an image to what",
    expected_output="Clearly articulated image deepfake attack patterns",
    agent=AttackPatternSynthesizer,
    context = [InfoRefiner_task],
    output_file="attack_patternsm.md" 
)

PromptGenerator_task = Task(
    description="Comprehending the long description of image deepfake attack patterns, generate clear, concise, effective prompts for deepfake image sample generation through LVM. You need to generate two prompts, positive and negative. One is a prompt to generate a scene with human subject(s). Second is the prompt that describes what changes to make in the image generated through first prompt, so that the image generated through this second prompt will follow the comprehended deepfake attack pattern in a realistic way.",
    expected_output="Two concise and effective prompts for ultra-realistic high quality image deepfake sample generation to pass to LVM. One prompt is to generate an image, and the second prompt is to describe what changes to make in the image generated through first prompt to create the corresponding deepfake.",
    agent=PromptGenerator,
    context = [AttackPatternSynthesizer_task],
    output_file="promptsm.md" 
)


manager = Agent(
    role="Crew Manager",
    goal="The user gives you a query: '{user_query}'. The goal of the crew is to first create multiple sub-queries around the actual query like query enhancement, so that we get a list of questions or queries around the subject covering it comprehensively. They should then curate information for all those queries. They need to identify existing image deepfake attack patterns from the web, and research papers, using the exhaustive information collected, multiple tools provided to each agent, and the agents are allowed and recommended to use their own pre-trained knowledge to add to the activity. Along with the existing attack patterns, the agents should also predict what potential image deepfake patterns might arise in the future, looking at the evolution and advancement of image deepfakes over time, with the help of their own knowledge and web tools provided. Using this information, the agents should synthesize detailed attack patterns articulating an image context, and how the subjects of it are manipulated to create the respective deepfake pattern. Using this, the ultimate goal is to return two prompts (positive and negative), where the positive prompt is to generate an image. So, this prompt should detailly describe an image context that will be later passed to a third-party language-vision model for generation. The negative prompt should describe what changes should be made to which subject in the image generated through the first prompt, so that the resultant image generated throgh this prompt creates the deepfake attack pattern synthesized. The agents are recommended to use their own knowledge along with the provided tools for doing all the activities. Each agent is assigned one of the above tasks, so that, holistically, they give the final output of prompts. Your job as a manager is to act responsibly in monitoring and guiding your agents to work appropriately to provide their respective results. If any agent does not provide the result as intended, ensure you ask the agent to repeat its task with appropriate inputs for it to finish its respective task. At the same time, understand that there are maximimum execution time and iterations limit set, thus, ensure you get the desired results within this bound. Manage the crew to ensure everything runs smoothly.",
    backstory="Being an efficient crew manager with your agents being QueryStrategist, DataCollector, ReportGenerator, InfoRefiner, AttackPatternSynthesizer, PromptGenerator. The tools are search_tool (web search), scrape_tool (website scraping), arxiv_tool (research paper exploration), semantic_scholar_tool (research paper exploration), serper_tool (web search), trends_tool (get google trends), search_and_contents (web search), find_similar_and_contents (find similar content to something; web search), vector_database_search (search curated knowledge base).",
    verbose=True,
    llm=llama  
)

Manager_task = Task(
    description="Manage the crew to ensure everything runs smoothly. Every agent should report to you. Ensure you provide the right throughts and inputs to guide the agents. You task is to ensure that every agent should give their result inline with the expected result.",
    expected_output="Ensure every agent gives desired result. Ask the respective agent to rework if their output is not inline with the expected output",
    agent=manager,
    output_file="manager.md" 
)

def create_crew():
    return Crew(
        agents=[manager, QueryStrategist, DataCollector, ReportGenerator, InfoRefiner, 
                AttackPatternSynthesizer, PromptGenerator],
        tasks=[Manager_task, QueryStrategist_task, DataCollector_task, ReportGenerator_task, 
               InfoRefiner_task, AttackPatternSynthesizer_task, PromptGenerator_task],
        verbose=2, 
        memory=True,
        embedder={'provider': 'huggingface',
                  'config': {'model': 'mixedbread-ai/mxbai-embed-large-v1'}},
        manager_llm=llama,
        process=Process.hierarchical,
        planning=True
    )


def run_multiagent_workflow(user_query: str, output_dir: str = "./outputs"):
    """
    Run the multi-agent workflow and return the final prompts.
    
    Args:
        user_query: The user's query for generating deepfake prompts
        output_dir: Directory to save output files
    
    Returns:
        Dictionary containing positive and negative prompts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    crew = create_crew()
    result = crew.kickoff(inputs={'user_query': user_query})
    prompts = extract_prompts_from_result(result, output_dir)
    
    return prompts


def extract_prompts_from_result(result, output_dir: str):
    """
    Extract positive and negative prompts from the crew result.
    
    Args:
        result: Crew execution result
        output_dir: Directory where output files are saved
    
    Returns:
        Dictionary with 'positive_prompts' and 'negative_prompts' lists
    """
    prompts = {'positive_prompts': [], 'negative_prompts': []}
    
    # Try to read from the output file
    prompts_file = os.path.join(output_dir, "promptsm.md")
    if os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            current_prompt = None
            for line in lines:
                if 'positive' in line.lower() or 'prompt 1' in line.lower():
                    current_prompt = 'positive'
                elif 'negative' in line.lower() or 'prompt 2' in line.lower():
                    current_prompt = 'negative'
                elif line.strip() and current_prompt:
                    if current_prompt == 'positive':
                        prompts['positive_prompts'].append(line.strip())
                    else:
                        prompts['negative_prompts'].append(line.strip())
    
    # # If no prompts found in file, try to extract from result string
    # if not prompts['positive_prompts'] and not prompts['negative_prompts']:
    #     result_str = str(result)
    #     pass
    
    # return result_str


if __name__ == "__main__":
    user_query = os.environ.get('USER_QUERY', 'Write 15 Few-shot Prompts to create novel Image deepfakes')
    output_dir = os.environ.get('OUTPUT_DIR', './outputs')
    
    prompts = run_multiagent_workflow(user_query, output_dir)
    print("\nGenerated Prompts:")
    print(f"Positive prompts: {len(prompts['positive_prompts'])}")
    print(f"Negative prompts: {len(prompts['negative_prompts'])}")