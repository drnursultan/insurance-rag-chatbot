from .config import LLM_PROVIDER, OPENAI_MODEL

def load_llm():
  if LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
  else:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    # Choose a lightweight model
    name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(
      name, device_map="auto", torch_dtype="auto"
    )

    gen = pipeline(
      "text-generation",
      model=mdl,
      tokenizer=tok,
      max_new_tokens=256,
      do_sample=False,
    )
    return HuggingFacePipeline(pipeline=gen)
