CODE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_code.py")).read()
TEXT = ("The river Danube rises in the Black Forest in Germany and flows for about 2,850 kilometres "
        "through ten countries before it empties into the Black Sea in Romania and Ukraine. "
        "Bratislava, Vienna, Budapest and Belgrade all sit on its banks. In the Middle Ages the river "
        "was a trade route for salt, wine and timber, and today it carries barges, cruise ships and "
        "freight between the North Sea and the Black Sea through the Rhine-Main-Danube canal. ") * 3
PROMPTS = {
  "code": "Rewrite the following Python code so that every function has a docstring and type hints. "
          "Return the full code only.\n\n" + CODE,
  "quote": "Repeat the following text exactly, then add one sentence summarizing it.\n\n" + TEXT,
  "chat": "Explain in three sentences why the sky is blue.",
}


def enc(tok, p):
    m = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, add_generation_prompt=True)
