"""Talk to the trained model. It runs bash and the editor by itself."""

import argparse
from pathlib import Path

import torch
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import (
    STOCK_TOKENIZER,
    Editor,
    clip,
    elide_old_observations,
    generate,
    load_config,
    observation_message,
    parse_call,
    run_bash,
)


def show(text, limit=40):
    lines = text.splitlines() or [""]
    print("\n".join(lines[:limit]))
    if len(lines) > limit:
        print(f"... {len(lines) - limit} more lines")


def tool_label(name, parameters):
    if name == "bash":
        return parameters.get("command", "")
    if name == "str_replace_editor":
        return f"{parameters.get('command', '')} {parameters.get('path', '')}"
    return name


def solve(model, tokenizer, config, root, task, max_calls, max_new_tokens):
    templates = config["templates"]
    history = [
        {"role": "system", "content": templates["system_template"], "message_type": "system_prompt"},
        {
            "role": "user",
            "content": Template(templates["instance_template"]).render(working_dir=str(root), problem_statement=task),
            "message_type": "observation",
        },
    ]
    editor = Editor(root)
    timeout = config["tools"]["execution_timeout"]
    keep_observations = config["history_processors"][0]["n"]
    for step in range(1, max_calls + 1):
        prompt = elide_old_observations(history, keep_observations)
        answer = generate(
            model,
            tokenizer,
            [{"role": item["role"], "content": item["content"]} for item in prompt],
            max_new_tokens,
        )
        print(f"\n[{step}]")
        show(answer.strip() or "(empty reply)")
        parsed, error = parse_call(answer)
        if error:
            output = error
            print("\nno tool call")
        else:
            name, parameters = parsed
            print(f"\n$ {tool_label(name, parameters)}")
            if name == "submit":
                print("submitted")
                return
            try:
                output = run_bash(root, parameters.get("command", ""), timeout) if name == "bash" else editor(parameters)
            except (ValueError, IndexError, OSError) as exc:
                output = str(exc)
        output = clip(output, templates["max_observation_length"])
        print()
        show(output or "Your command ran successfully and did not produce any output.")
        history.append({"role": "assistant", "content": answer, "message_type": "action"})
        history.append({"role": "user", "content": observation_message(config, output), "message_type": "observation"})
    print(f"\nstopped after {max_calls} calls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trainer_outputs/runpod-checkpoint-338")
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    parser.add_argument("--max-calls", type=int, default=75)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    root = args.dir.resolve()
    config = load_config()
    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = AutoTokenizer.from_pretrained(STOCK_TOKENIZER).chat_template
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    model.config.use_cache = True
    model.to("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"working directory: {root}")
    print("Describe a task. Empty line to quit.")
    while True:
        try:
            task = input("\ntask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            break
        try:
            solve(model, tokenizer, config, root, task, args.max_calls, args.max_new_tokens)
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
