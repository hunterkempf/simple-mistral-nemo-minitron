import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

model_name = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)#,attn_implementation="flash_attention_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.special_tokens_map)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
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
<extra_id_0>System
You are a helpful bot who reads users questions and answers them. Answer the user's query with what you know. If you do not know the answer reply with "I do not know that".

<extra_id_1>User
{question}
<extra_id_1>Assistant
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm.bind(stop="<extra_id_1>",stop_strings=["<extra_id_1>"])#| StrOutputParser()
print("Starting LLM Model...")
print(" To Exit hit cntrl-c or type exit() in the prompt input")
while(1):
    user_query = input("Ask a question:")
    if user_query.startswith("exit()"):
        break
    print("Running Mistral Nemo Minitron LLM...")
    #output = llm_chain.invoke({"question": user_query},stop_strings=["<extra_id_1>"])#stop=["<extra_id_1>"])
    #print(output)#.split("Context:")[0]
    for chunk in llm_chain.stream({"question": user_query}):
        print(chunk, end="", flush=True)