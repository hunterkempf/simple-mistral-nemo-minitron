import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

model_name = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=500,
    device="cuda"
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
Context: You are a helpful bot who reads texts and answers questions about them. Answer the user's query.
Query: {question}
Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()
print("Starting LLM Model...")
print(" To Exit hit cntrl-c or type exit() in the prompt input")
while(1):
    user_query = input("Ask a question:")
    if user_query.startswith("exit()"):
        break
    print("Running Mistral Nemo Minitron LLM...")
    output = llm_chain.invoke({"question": user_query})
    print(output.split("Context:")[0])