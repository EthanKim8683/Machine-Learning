"""Local stand-in for SWE-agent's swesmith_infer.yaml loop.

SWE-agent sends role/content messages to an OpenAI-compatible server. That server
applies the stock Qwen2.5-Coder chat template. The template saved on the trained
checkpoint is the training template, which adds generation tags, so this harness
does not use it.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path(__file__).with_name("swesmith_infer.yaml")
STOCK_TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
FN_REGEX = re.compile(r"<function=([^>]+)>\n(.*?)</function>", re.DOTALL)
PARAM_REGEX = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
MISSING_TOOL_CALL = """\
Your last output did not use any tool calls!
Please make sure your output includes exactly _ONE_ function call!
If you think you have already resolved the issue, please submit your changes by running the `submit` command.
If you think you cannot solve the problem, please run `submit`.
Else, please continue with a new tool call!"""
MULTIPLE_TOOL_CALLS = """\
Your last output included multiple tool calls!
Please make sure your output includes a thought and exactly _ONE_ function call."""


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())["agent"]
    return config


def observation_message(config, output):
    template = config["templates"]["next_step_no_output_template"] if output == "" else config["templates"]["next_step_template"]
    return Template(template).render(observation=output)


def elide_old_observations(history, keep):
    observation_indices = [index for index, message in enumerate(history) if message["message_type"] == "observation"]
    omit = set(observation_indices[1:-keep] if keep else observation_indices[1:])
    elided = []
    for index, message in enumerate(history):
        if index not in omit:
            elided.append(message)
            continue
        lines = message["content"].count("\n") + 1
        elided.append({**message, "content": f"Old environment output: ({lines} lines omitted)"})
    return elided


def parse_call(text):
    matches = list(FN_REGEX.finditer(text))
    if not matches:
        return None, MISSING_TOOL_CALL
    if len(matches) > 1:
        return None, MULTIPLE_TOOL_CALLS
    match = matches[0]
    name = match.group(1).strip()
    if name == "execute_bash":
        name = "bash"
    if name == "finish":
        name = "submit"
    parameters = {key: re.sub(r"^\n|\n$", "", value) for key, value in PARAM_REGEX.findall(match.group(2))}
    if name not in {"bash", "submit", "str_replace_editor"}:
        return None, f"Your action could not be parsed properly: Command '{name}' not found in list of available commands."
    if "view_range" in parameters and not re.fullmatch(r"\[\d+,\s*-?\d+\]", parameters["view_range"]):
        return None, f"Your action could not be parsed properly: view_range must be in the format [<start>, <end>], got {parameters['view_range']}."
    return (name, parameters), None


def numbered(text, start=1):
    return "\n".join(f"{number:6d}|{line}" for number, line in enumerate(text.splitlines(), start))


class Editor:
    def __init__(self, root):
        self.root = root
        self.undo = {}

    def resolve(self, raw_path):
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path
        path.relative_to(self.root)
        return path

    def view(self, path, view_range):
        if path.is_dir():
            lines = []
            for child in sorted(path.rglob("*")):
                if any(part.startswith(".") or part == "__pycache__" for part in child.relative_to(path).parts):
                    continue
                if len(child.relative_to(path).parts) <= 2:
                    lines.append(str(child))
            return "\n".join(lines)
        file_text = path.read_text(errors="replace")
        if view_range is None:
            return numbered(file_text)
        start, end = json.loads(view_range)
        rows = file_text.splitlines()
        end = len(rows) if end == -1 else end
        return numbered("\n".join(rows[start - 1:end]), start)

    def __call__(self, parameters):
        command = parameters.get("command")
        path = self.resolve(parameters["path"])
        if command == "view":
            if not path.exists():
                return f"Path not found: {path}"
            return self.view(path, parameters.get("view_range"))
        if command == "create":
            if path.exists():
                return "File already exists."
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parameters.get("file_text", ""))
            return f"File created successfully at: {path}"
        if command == "str_replace":
            old, new = parameters.get("old_str", ""), parameters.get("new_str", "")
            file_text = path.read_text(errors="replace")
            count = file_text.count(old)
            if count != 1:
                return f"old_str matched {count} times; replacement not performed."
            self.undo.setdefault(path, []).append(file_text)
            path.write_text(file_text.replace(old, new, 1))
            return "The file has been edited."
        if command == "insert":
            rows = path.read_text(errors="replace").splitlines()
            self.undo.setdefault(path, []).append("\n".join(rows) + ("\n" if rows else ""))
            rows.insert(int(parameters["insert_line"]), parameters.get("new_str", ""))
            path.write_text("\n".join(rows) + "\n")
            return "The file has been edited."
        if command == "undo_edit":
            path.write_text(self.undo[path].pop())
            return "Last edit to the file has been undone."
        return f"Unsupported editor command: {command}"


def run_bash(root, command, timeout):
    try:
        completed = subprocess.run(["bash", "-lc", command], cwd=root, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds."
    output = (completed.stdout + completed.stderr).strip()
    if output == "" and completed.returncode == 0:
        return ""
    if completed.returncode != 0:
        return f"{output}\nExit code: {completed.returncode}".strip()
    return output


def clip(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<response clipped>"


def generate(model, tokenizer, messages, max_new_tokens):
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=False)


def checkout(repo, commit, destination):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    subprocess.check_call(["git", "init"], cwd=destination)
    subprocess.check_call(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=destination)
    subprocess.check_call(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
    subprocess.check_call(["git", "checkout", "FETCH_HEAD"], cwd=destination)


def run_instance(model, tokenizer, config, row, workspace, max_calls, max_new_tokens):
    root = workspace / row["instance_id"]
    checkout(row["repo"], row["base_commit"], root)
    templates = config["templates"]
    history = [
        {"role": "system", "content": templates["system_template"], "message_type": "system_prompt"},
        {
            "role": "user",
            "content": Template(templates["instance_template"]).render(working_dir=str(root), problem_statement=row["problem_statement"]),
            "message_type": "observation",
        },
    ]
    editor = Editor(root)
    timeout = config["tools"]["execution_timeout"]
    keep_observations = config["history_processors"][0]["n"]
    transcript = []
    for _ in range(max_calls):
        prompt = elide_old_observations(history, keep_observations)
        answer = generate(model, tokenizer, [{"role": item["role"], "content": item["content"]} for item in prompt], max_new_tokens)
        parsed, error = parse_call(answer)
        if error:
            output = error
        else:
            name, parameters = parsed
            if name == "submit":
                transcript.append({"assistant": answer, "function": name})
                break
            output = run_bash(root, parameters.get("command", ""), timeout) if name == "bash" else editor(parameters)
        output = clip(output, templates["max_observation_length"])
        user_content = observation_message(config, output)
        history.append({"role": "assistant", "content": answer, "message_type": "action"})
        history.append({"role": "user", "content": user_content, "message_type": "observation"})
        transcript.append({"assistant": answer, "function": None if error else name, "observation": user_content})
    diff = subprocess.run(["git", "-C", str(root), "diff"], text=True, capture_output=True).stdout
    return {"instance_id": row["instance_id"], "transcript": transcript, "diff": diff}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trainer_outputs/runpod-checkpoint-338")
    parser.add_argument("--instance", action="append", required=True)
    parser.add_argument("--max-calls", type=int, default=75)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/swe-smith-harness"))
    args = parser.parse_args()
    config = load_config()
    dataset = {row["instance_id"]: row for row in load_dataset("SWE-bench/SWE-bench_Lite", split="test")}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = AutoTokenizer.from_pretrained(STOCK_TOKENIZER).chat_template
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    model.config.use_cache = True
    model.to("mps" if torch.backends.mps.is_available() else "cpu")
    results = [run_instance(model, tokenizer, config, dataset[instance_id], args.workspace, args.max_calls, args.max_new_tokens) for instance_id in args.instance]
    output = args.workspace / "results.json"
    output.write_text(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
