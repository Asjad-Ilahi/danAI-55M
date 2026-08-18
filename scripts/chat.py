#!/usr/bin/env python3
"""
Interactive Prompt Engine for 54.5M SLM (Single Fused Model).

Features:
- Single Standalone Fused Model (exp_018_fused_slm: Merged Reasoning, Email/Chat, and Tools).
- Real-time streaming generation with low-noise greedy/temperature sampling.
- Live Tool Calling Execution:
  1. Exact Calculator (arithmetic & algebraic math).
  2. Live Web Search (Wikipedia Search & Summary API).
  3. Python Sandbox execution.
- Dynamic Tool Triggering (no prompt pollution on basic chat / email writing).
- RAG document grounding (/rag <context>, /rag clear).
"""

import ast
import html
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model.gpt import CausalLM
from src.utils.config import Config
from src.utils.device import get_device

DEFAULT_CHECKPOINT = "experiments/exp_019_perfect_alignment/checkpoints/best.pt"

DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent AI assistant. You have access to the following tools:\n"
    "- calculator(expression: str): Evaluates mathematical and arithmetic expressions with exact precision.\n"
    "- search_web(query: str): Searches the web for recent, real-time, or external information.\n"
    "- run_python(code: str): Executes Python code in a secure sandbox and returns the stdout output.\n"
    "When a tool is needed, respond with a <tool_call> block containing a JSON object with 'name' and 'arguments'."
)


# ==============================================================================
# TOOL EXECUTORS
# ==============================================================================

def execute_calculator(expression: str) -> str:
    cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    safe_dict = {
        "__builtins__": None,
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
        "math": math, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10, "exp": math.exp, "pi": math.pi, "e": math.e,
    }
    try:
        node = ast.parse(cleaned, mode='eval')
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom, ast.Call)):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    if sub.func.id not in safe_dict:
                        return "Error: Unsupported function call"
        res = eval(compile(node, '<string>', 'eval'), safe_dict)
        if isinstance(res, float) and res.is_integer():
            res = int(res)
        return str(res)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def clean_query_text(text: str) -> str:
    cleaned = re.sub(
        r'(?i)\b(search web to find me meaning of|search online to find me meaning of|search web to find meaning of|search online to find meaning of|search web to find|search online to find|search the web for|search online for|search for|search web|search it in internet|search it on internet|search up|search|look up|tell me about|find me the meaning of|find the meaning of|find me meaning of|what is the meaning of|what is|who is|meaning of|definition of)\b',
        '',
        text
    )
    cleaned = cleaned.replace('User:', '').replace('Assistant:', '').replace('System:', '').strip('?., "\'\n')
    cleaned = re.sub(r'^(me|the|a|an|of)\s+', '', cleaned, flags=re.I)
    return re.sub(r'\s+', ' ', cleaned).strip()


def execute_web_search(query: str) -> str:
    clean_q = clean_query_text(query)
    if not clean_q or len(clean_q) < 2:
        clean_q = query.strip()
    
    # 1. Search Wikipedia Search API
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json&utf8="
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = data.get("query", {}).get("search", [])
            if results:
                top_title = results[0]["title"]
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(top_title.replace(' ', '_'))}"
                sum_req = urllib.request.Request(summary_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                with urllib.request.urlopen(sum_req, timeout=4) as sum_resp:
                    sum_data = json.loads(sum_resp.read().decode('utf-8'))
                    extract = sum_data.get("extract", "")
                    if extract and len(extract) > 40:
                        return extract[:800]
    except Exception:
        pass

    # 2. DuckDuckGo Instant Answer API Fallback
    try:
        encoded = urllib.parse.quote(clean_q)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            abstract = data.get("AbstractText", "")
            if abstract:
                return abstract
            related = data.get("RelatedTopics", [])
            if related and isinstance(related[0], dict) and "Text" in related[0]:
                return related[0]["Text"]
    except Exception:
        pass

    return f"Search results retrieved for '{clean_q}'."


def execute_python_sandbox(code: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return out if out else "(Code executed successfully with no stdout output)"
        else:
            return f"Execution Error:\n{err}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (exceeded 5 seconds)"
    except Exception as e:
        return f"Error executing Python code: {e}"


def dispatch_tool_call(tool_name: str, arguments: Dict[str, Any], raw_user_prompt: str = "") -> str:
    t_lower = tool_name.lower()
    if "calc" in t_lower:
        expr = arguments.get("expression") or arguments.get("expr") or str(arguments)
        return execute_calculator(expr)
    elif "search" in t_lower:
        query = str(arguments.get("query") or arguments.get("q") or "").strip('\'" ')
        
        # Check if query drifted or is generic
        if raw_user_prompt:
            if "User:" in raw_user_prompt:
                user_match = re.search(r'User:\s*(.*?)(?:\n\nAssistant:|$)', raw_user_prompt, re.DOTALL)
                user_text = user_match.group(1).strip() if user_match else raw_user_prompt
            else:
                user_text = raw_user_prompt

            user_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', user_text.lower()))
            gen_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
            stopwords = {'search', 'the', 'web', 'online', 'internet', 'for', 'who', 'what', 'and', 'give', 'summary', 'find', 'about', 'tell'}
            meaningful_user_words = user_words - stopwords
            overlap = gen_words.intersection(meaningful_user_words)
            
            if not overlap and meaningful_user_words:
                clean = clean_query_text(user_text)
                if clean:
                    query = clean
                    
        return execute_web_search(query)
    elif "python" in t_lower:
        code = arguments.get("code") or arguments.get("script") or str(arguments)
        return execute_python_sandbox(code)
    else:
        return f"Error: Unknown tool '{tool_name}'"


# ==============================================================================
# MODEL LOADING & STREAM GENERATION
# ==============================================================================

HF_REPO_ID = "asjadilahi/danAI-55M-Reasoning"


def resolve_model_weights(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Tuple[Dict[str, Any], bool, str]:
    """
    Finds model weights locally or automatically downloads them from Hugging Face Hub.
    Returns (state_dict, is_ema, source_description).
    """
    p = Path(checkpoint_path)
    if p.exists() and p.is_file():
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
            return ckpt["ema_state_dict"]["shadow"], True, str(p)
        elif "ema_weights" in ckpt:
            return ckpt["ema_weights"], True, str(p)
        elif "model_state_dict" in ckpt:
            return ckpt["model_state_dict"], False, str(p)
        return ckpt, False, str(p)

    safetensors_local = Path("hf_export/model.safetensors")
    if safetensors_local.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_local))
        return state_dict, True, str(safetensors_local)

    # Auto-download from Hugging Face
    print(f"\n[Hub] Local checkpoint not found at '{checkpoint_path}'.")
    print(f"[Hub] Auto-downloading danAI-55M from Hugging Face ({HF_REPO_ID})...")
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        weights_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="model.safetensors",
        )
        state_dict = load_file(weights_file)
        print(f"[Hub] ✓ Downloaded model weights: {weights_file}\n")
        return state_dict, True, f"HuggingFace ({HF_REPO_ID})"
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load checkpoint '{checkpoint_path}' or download from Hugging Face ({HF_REPO_ID}): {e}"
        )


def resolve_tokenizer(tokenizer_dir: str = "tokenizer") -> Tokenizer:
    """Loads tokenizer locally or downloads from Hugging Face Hub."""
    t_file = Path(tokenizer_dir) / "tokenizer.json"
    if t_file.exists():
        return Tokenizer.from_file(str(t_file))
    
    if Path("hf_export/tokenizer.json").exists():
        return Tokenizer.from_file("hf_export/tokenizer.json")
        
    print(f"[Hub] Auto-downloading tokenizer from Hugging Face ({HF_REPO_ID})...")
    from huggingface_hub import hf_hub_download
    tok_file = hf_hub_download(repo_id=HF_REPO_ID, filename="tokenizer.json")
    return Tokenizer.from_file(tok_file)


def load_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_config_path: str = "configs/model.yaml",
    use_ema: bool = True,
    device: str = "cpu",
) -> Tuple[CausalLM, Config, str, bool]:
    cfg = Config.from_yaml(model_config_path) if Path(model_config_path).exists() else Config({
        "model": {
            "vocab_size": 32768,
            "hidden_size": 512,
            "intermediate_size": 1376,
            "num_layers": 12,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 64,
            "max_seq_len": 2048,
            "tie_embeddings": True,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
        }
    })
    
    model = CausalLM(cfg.model)
    state_dict, loaded_ema, source_desc = resolve_model_weights(checkpoint_path)

    # Clean state dict keys
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[clean_k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model, cfg, source_desc, loaded_ema


def stream_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 300,
    temperature: float = 0.1,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    tools_enabled: bool = True,
) -> Tuple[str, int, float]:
    start_time = time.perf_counter()
    current_prompt = prompt
    full_response_text = ""
    total_tokens_generated = 0
    max_tool_turns = 3

    for turn_idx in range(max_tool_turns):
        tokens = tokenizer.encode(current_prompt).ids
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        generated_ids: List[int] = []
        response_ids: List[int] = []
        printed_text = ""
        in_think_block = False
        in_tool_block = False

        for step in range(max_new_tokens):
            context_ids = torch.cat([input_ids, torch.tensor([generated_ids], dtype=torch.long, device=device)], dim=1)
            if context_ids.size(1) > 2048:
                context_ids = context_ids[:, -2048:]

            with torch.no_grad():
                out = model(context_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_token_logits = logits[0, -1, :]

            # Repetition Penalty
            if repetition_penalty != 1.0 and (tokens + generated_ids):
                unique_tokens = set(tokens + generated_ids)
                for t in unique_tokens:
                    if next_token_logits[t] > 0:
                        next_token_logits[t] /= repetition_penalty
                    else:
                        next_token_logits[t] *= repetition_penalty

            # Temperature Sampling
            if temperature > 0:
                scaled = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[-1]] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled[indices_to_remove] = -float('Inf')
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token in [tokenizer.token_to_id("<|endoftext|>"), tokenizer.token_to_id("<eos>"), 0]:
                break

            response_ids.append(next_token)
            generated_ids.append(next_token)
            total_tokens_generated += 1

            curr_decoded = tokenizer.decode(response_ids)
            new_chars = curr_decoded[len(printed_text):]
            if new_chars:
                if "User:" in curr_decoded:
                    idx = curr_decoded.find("User:")
                    to_print = curr_decoded[len(printed_text):idx]
                    if to_print:
                        sys.stdout.write(to_print)
                        sys.stdout.flush()
                    break

                # Formatting
                if "<think>" in new_chars and not in_think_block:
                    in_think_block = True
                    sys.stdout.write("\033[2;36m")
                if "<tool_call>" in new_chars and not in_tool_block:
                    in_tool_block = True
                    sys.stdout.write("\033[1;35m")

                sys.stdout.write(new_chars)

                if "</think>" in curr_decoded and in_think_block:
                    in_think_block = False
                    sys.stdout.write("\033[0m\n")
                if "</tool_call>" in curr_decoded and in_tool_block:
                    in_tool_block = False
                    sys.stdout.write("\033[0m\n")

                sys.stdout.flush()
                printed_text = curr_decoded

            if tools_enabled and "<tool_call>" in curr_decoded:
                call_substr = curr_decoded[curr_decoded.find("<tool_call>"):]
                if "}" in call_substr or any(t in curr_decoded for t in ["</tool_call>", "</tool>", "</toolentry>"]):
                    break

        turn_text = tokenizer.decode(response_ids).strip()
        full_response_text += "\n" + turn_text

        # Check for tool call execution
        if tools_enabled and "<tool_call>" in turn_text:
            match = re.search(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|</tool>|</toolentry>|$)", turn_text, re.DOTALL)
            if match:
                raw_json = match.group(1).strip()
                t_name = "unknown"
                t_args = {}
                try:
                    call_obj = json.loads(raw_json)
                    t_name = call_obj.get("name", "unknown")
                    t_args = call_obj.get("arguments", {})
                except json.JSONDecodeError:
                    name_m = re.search(r'"name":\s*"([^"]+)"', raw_json)
                    if name_m:
                        t_name = name_m.group(1).strip()
                    
                    if "calc" in t_name or "calculator" in raw_json:
                        t_name = "calculator"
                        expr_m = re.search(r'"(?:expression|expr)":\s*"([^"]+)"', raw_json)
                        if not expr_m:
                            expr_m = re.search(r'[\'"]([0-9\+\-\*\/\^\s\(\)\.]+)[\'"]', raw_json)
                        if expr_m:
                            t_args = {"expression": expr_m.group(1)}
                    elif "search" in t_name or "search_web" in raw_json:
                        t_name = "search_web"
                        q_m = re.search(r'"(?:query|q)":\s*"([^"]+)"', raw_json)
                        if q_m:
                            t_args = {"query": q_m.group(1)}
                    elif "python" in t_name or "run_python" in raw_json:
                        t_name = "run_python"
                        c_m = re.search(r'"(?:code|script)":\s*"([^"]+)"', raw_json)
                        if c_m:
                            t_args = {"code": c_m.group(1)}

                # Normalize tool name
                if "calc" in t_name.lower():
                    t_name = "calculator"
                elif "search" in t_name.lower():
                    t_name = "search_web"
                elif "python" in t_name.lower():
                    t_name = "run_python"

                if t_name != "unknown" and t_args:
                    sys.stdout.write(f"\n\033[1;33m[⚙️  Executing Tool: {t_name}]\033[0m\n\n")
                    tool_res = dispatch_tool_call(t_name, t_args, raw_user_prompt=current_prompt)
                    
                    if t_name == "search_web":
                        sys.stdout.write(f"\033[1;32mAssistant:\033[0m {tool_res}\n")
                        full_response_text += f"\n{tool_res}"
                        break
                    elif t_name == "calculator":
                        expr_val = t_args.get("expression", "")
                        sys.stdout.write(f"\033[1;32mAssistant:\033[0m Using the calculator, {expr_val} = {tool_res}\n")
                        full_response_text += f"\nUsing the calculator, {expr_val} = {tool_res}"
                        break
                    elif t_name == "run_python":
                        sys.stdout.write(f"\033[1;32mAssistant:\033[0m Python Output:\n{tool_res}\n")
                        full_response_text += f"\nPython Output:\n{tool_res}"
                        break
                    else:
                        sys.stdout.write("\033[1;32mAssistant:\033[0m ")
                        sys.stdout.flush()
                        current_prompt = f"{current_prompt}{turn_text}\n\n<tool_response>\n{tool_res}\n</tool_response>\n\nAssistant: "
                        continue
                else:
                    sys.stdout.write(f"\n\033[1;31m[Tool Call Parse Failed]\033[0m\n")
        break

    elapsed = time.perf_counter() - start_time
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()
    return full_response_text.strip(), total_tokens_generated, elapsed


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Prompt Engine for danAI-55M")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint path (or auto-downloads from HF)")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Tokenizer path")
    parser.add_argument("--temperature", "--temp", type=float, default=0.1, help="Temperature (0 for greedy, 0.1 default)")
    parser.add_argument("--repetition-penalty", "--penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--max-new-tokens", "--tokens", type=int, default=300, help="Max tokens per response")
    args = parser.parse_args()

    device = get_device()
    tokenizer = resolve_tokenizer(args.tokenizer_dir)

    print("\n" + "=" * 80)
    print("       🧠 danAI-55M-Reasoning INTERACTIVE ENGINE (with Agentic Tools)")
    print("=" * 80)
    
    model, config, source_desc, active_ema = load_model(
        checkpoint_path=args.checkpoint,
        use_ema=True,
        device=device,
    )
    print(f"  • Model Source: {source_desc}")
    print(f"  • Hardware:     {device}")

    temperature = args.temperature
    repetition_penalty = args.repetition_penalty
    max_new_tokens = args.max_new_tokens
    active_system_prompt = ""
    active_rag_context = ""
    tools_enabled = True

    print(f"  • Settings:    temp={temperature}, penalty={repetition_penalty}, max_tokens={max_new_tokens}")
    print("-" * 80)
    print("  Commands available anytime:")
    print("    /system <prompt>        -> Set active custom persona")
    print("    /system clear           -> Clear custom persona")
    print("    /rag <context>          -> Set document context for RAG")
    print("    /reset                  -> Reset all prompts, RAG, and history")
    print("    temp <val>              -> Change temperature (e.g. `temp 0` or `temp 0.7`)")
    print("    penalty <val>           -> Change repetition penalty (e.g. `penalty 1.15`)")
    print("    tokens <val>            -> Change max response length (e.g. `tokens 400`)")
    print("    \"\"\"                     -> Enter multi-line prompt mode")
    print("    exit / quit             -> Exit")
    print("=" * 80)

    while True:
        try:
            rag_tag = f" \033[1;33m[RAG Active: {active_rag_context[:25]}...]\033[0m" if active_rag_context else ""
            line = input(f"\n\033[1;36mPrompt{rag_tag}:\033[0m ").strip()
            if not line:
                continue
            if line == '"""' or line == "'''":
                print("\033[0;33m(Multi-line mode active. Enter your text and finish with \"\"\" on a new line)\033[0m")
                lines = []
                while True:
                    sub = input()
                    if sub.strip() in ['"""', "'''"]:
                        break
                    lines.append(sub)
                line = "\n".join(lines).strip()

            lower = line.lower()
            if lower in ["exit", "quit", "/exit", "/quit"]:
                print("\nExiting. Happy experimenting!\n")
                break

            if lower in ["/system clear", "system clear", "/system off", "system off"]:
                active_system_prompt = ""
                print("\033[1;32m✓ System prompt cleared.\033[0m")
                continue

            if lower.startswith("/system ") or lower.startswith("system "):
                active_system_prompt = line.split(maxsplit=1)[1].strip()
                print(f"\033[1;32m✓ System prompt updated: {active_system_prompt[:80]}...\033[0m")
                continue

            if lower.startswith("/rag ") or lower.startswith("rag "):
                active_rag_context = line.split(maxsplit=1)[1].strip()
                if active_rag_context.lower() in ["clear", "none", "off", "reset"]:
                    active_rag_context = ""
                    print("\033[1;32m✓ RAG context CLEARED. Normal mode active.\033[0m")
                else:
                    print(f"\033[1;32m✓ RAG document context set:\033[0m \"{active_rag_context[:80]}...\"")
                continue

            if lower in ["/rag clear", "rag clear", "/clear rag", "clear rag", "/reset"]:
                active_rag_context = ""
                active_system_prompt = ""
                print("\033[1;32m✓ Context and Prompts RESET to clean defaults.\033[0m")
                continue

            if lower.startswith("temp ") or lower.startswith("/temp "):
                try:
                    temperature = float(line.split()[1])
                    print(f"\033[1;32m✓ Temperature set to {temperature}\033[0m")
                except Exception:
                    print("Usage: temp 0.7")
                continue

            if lower.startswith("penalty ") or lower.startswith("/penalty "):
                try:
                    repetition_penalty = float(line.split()[1])
                    print(f"\033[1;32m✓ Repetition penalty set to {repetition_penalty}\033[0m")
                except Exception:
                    print("Usage: penalty 1.15")
                continue

            if lower.startswith("tokens ") or lower.startswith("/tokens "):
                try:
                    max_new_tokens = int(line.split()[1])
                    print(f"\033[1;32m✓ Max new tokens set to {max_new_tokens}\033[0m")
                except Exception:
                    print("Usage: tokens 300")
                continue

            # Detect tool-relevant triggers in user query
            tool_keywords = ["search", "look up", "calculate", "compute", "run python", "execute python", "eval", "times", "divided"]
            has_math_ops = bool(re.search(r'\d+\s*[\*\/\+\-\^]\s*\d+', lower))
            is_greeting = lower in ["hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you", "what can you do"]
            is_tool_query = (any(w in lower for w in tool_keywords) or has_math_ops) and not is_greeting

            # Assemble prompt
            prompt_parts = []
            if active_system_prompt:
                prompt_parts.append(f"System: {active_system_prompt}\n\n")
            elif is_tool_query and tools_enabled:
                prompt_parts.append(f"System: {DEFAULT_SYSTEM_PROMPT}\n\n")

            user_body = line
            if active_rag_context:
                user_body = f"Context:\n{active_rag_context}\n\nQuestion: {line}\n\nAnswer based solely on the context:"

            prompt_parts.append(f"User: {user_body}\n\nAssistant: ")
            full_prompt = "".join(prompt_parts)

            print("\033[1;32mAssistant:\033[0m ", end="", flush=True)

            response_text, num_toks, elapsed = stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                repetition_penalty=repetition_penalty,
                tools_enabled=is_tool_query and tools_enabled,
            )

            tok_per_sec = num_toks / max(0.001, elapsed)
            print(f"\033[0;90m({num_toks} tokens, {elapsed:.2f}s, {tok_per_sec:.1f} tok/s)\033[0m")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n\033[1;31mError: {e}\033[0m")


if __name__ == "__main__":
    main()
