import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

#model_name = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)#,attn_implementation="flash_attention_2"
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
<end_of_turn>System
You are a helpful bot who reads users questions and answers them. Answer the user's query with what you know. If you do not know the answer reply with "I do not know that".

<end_of_turn>User
{question}
<end_of_turn>Assistant
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm| StrOutputParser()
llm_chain.invoke({"question": "test"})
print(f"GPU's Available: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print("Starting LLM Model...")
print(" To Exit hit cntrl-c or type exit() in the prompt input")
print(" Running Qwen2.5-7B-Instruct...")
while(1):
    user_query = input("\nAsk a question:")
    if user_query.startswith("exit()"):
        break
    for chunk in llm_chain.stream({"question": user_query}):
        print(chunk.replace("<end_of_turn>",""), end="", flush=True)