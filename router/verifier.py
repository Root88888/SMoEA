# -*- coding: utf-8 -*-
"""
router/verifier.py

【送審裁決】LLM 是非題驗證器：對送審樣本的每個候選單位，把「單位說明書
＋單位示例＋查詢」組成是非題，讀首 token 的 Yes/No logit 換算 p_yes；
決策層取 p_yes 最大的候選，達 θ（預設 0.5）才路由、否則拒絕。

prompt 文字、Yes/No 變體表、logit 換算式全部照 v5b 原文收編——prompt
是系統行為的一部分，不可改寫。裁決模型預設 Llama-3.1-8B-Instruct
（configs 的 verifier.model_path，可填本地路徑）。

【資產】單位說明書 assets/unit_descriptions.json：
  {"descriptions": {"0": {"description": "..."}, ...}}
說明書由 LLM 生成＋人工校訂，是建置產物；本模組只讀不寫。
"""

import math


SYSTEM_PROMPT_VERIFY = (
    "You are a precise NLP task verifier. Given a task description and a user "
    "query, you judge whether the query is an instance of that exact task. "
    "You answer with exactly one word: Yes or No."
)

VERIFY_PROMPT = """Task description:
{desc}

Example input of this task:
{ex}

User query:
{query}

Is the user query an instance of the exact task described above (same operation, same kind of input)? Answer Yes or No.

Answer:"""

YES_VARIANTS = ["Yes", "yes", " Yes", " yes", "YES"]
NO_VARIANTS = ["No", "no", " No", " no", "NO"]


def clip_head_tail(text, head_chars, tail_chars):
    """長文本頭尾截取（中段以 ... 略去），保留任務格式的頭尾特徵。"""
    if len(text) <= head_chars + tail_chars + 5:
        return text
    return text[:head_chars] + "\n...\n" + text[-tail_chars:]


def load_llm(model_path):
    """載入裁決 LLM（float16、device_map=auto）。照 v5 原文。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}", flush=True)
    return tok, model


def first_token_ids(tok, variants):
    ids = set()
    for s in variants:
        t = tok.encode(s, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


def make_scorer(tok, model):
    """回傳 p_yes(system, user)：Yes/No 首 token logit 的二類 softmax。"""
    import torch
    yes_ids = first_token_ids(tok, YES_VARIANTS)
    no_ids = first_token_ids(tok, NO_VARIANTS)

    def p_yes(system, user):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(ids).logits[0, -1].float()
        ly = lg[yes_ids].max().item()
        ln = lg[no_ids].max().item()
        return 1.0 / (1.0 + math.exp(ln - ly))   # softmax 二類
    return p_yes


def make_fake_scorer(seed=0):
    """【僅供管線自測】決定性偽 scorer：對 prompt 全文 crc32 回傳偽 p_yes
    （跨執行穩定、隨 query 變異）。不進任何正式評測；eval 腳本以
    --fake_verifier 明示啟用。"""
    import zlib

    def p_yes(system, user):
        h = zlib.crc32(f"{seed}|{user}".encode()) % 1000
        return h / 1000.0
    return p_yes


def score_candidate(scorer, description, unit_example, query,
                    query_clip=2000):
    """對單一 (查詢, 候選單位) 打 p_yes。"""
    user = VERIFY_PROMPT.format(desc=description, ex=unit_example,
                                query=query[:query_clip])
    return round(scorer(SYSTEM_PROMPT_VERIFY, user), 4)
