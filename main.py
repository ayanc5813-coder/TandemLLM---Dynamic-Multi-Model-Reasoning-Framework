import os
import re
import time
import math
import pickle
import pprint
import logging
import statistics
from collections import Counter
from openai import OpenAI
from datasets import load_dataset
from google.colab import userdata

# ==========================================
# CONFIGURATION 
# ==========================================
class Args:
    # Model IDs from OpenRouter
    big_model_name = "nvidia/nemotron-3-super-120b-a12b:free"
    small_model_name = "liquid/lfm-2.5-1.2b-instruct:free" 
    
    # Dataset and Problem settings
    dataset_name = "aime24"  # Choices: "aime24", "gpqa", "aime25"
    problem_id = 0         # Which specific problem in the dataset to solve
    repeat_id = 0          
    
    # Algorithm parameters
    confident_threshold = 0.8  # Trigger big model if small model is >80% hyper-confident
    start_step = 2             # How many initial steps the big model should plan
    rectify_step = 1           # How many steps to let big model fix a confused small model
    token_budget = 8192        # Max total tokens allowed
    output_dir = "./results"

args = Args()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# API CLIENT SETUP (OpenRouter)
# ==========================================
# Fetch the key from Colab Secrets
try:
    OPENROUTER_API_KEY = userdata.get('OPENROUTER_KEY')
except Exception:
    raise ValueError("Please add OPENROUTER_KEY to your Colab Secrets (the 🔑 icon on the left).")

model_names = {
    args.big_model_name: args.big_model_name,
    args.small_model_name: args.small_model_name,
}

clients = {}
for model_name in model_names.keys():
    clients[model_name] = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://colab.research.google.com/", 
            "X-Title": "TrigReason_Colab", 
        }
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_avg_score(scores):
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    return dict(Counter(scores))

def get_first_user_msg(problem, options=None):
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt.format(problem=problem)
    else:
        system_prompt = """
        What is the correct answer to the following problem? Please reason step by step. 
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}

        Problem: {problem}.
        Choices: 
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        return system_prompt.format(
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )

def generate_new_step(problem, steps_so_far, model_name, options=None, stop_token="\n\n"):
    client = clients[model_name]
    
    if steps_so_far == []: 
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else: 
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=512,
        stop=[stop_token],
        extra_body=extra_body, 
        logprobs=True,       
        top_logprobs=1,          
    )

    choice = response.choices[0]
    step_str = choice.message.content
    
    # Handle missing usage stats gracefully
    num_output_tokens = response.usage.completion_tokens if response.usage else len(step_str.split())
    finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:", "Final Answer"]])

    ratio_below_threshold = 0.0
    token_ppls = []

    # OpenRouter logprobs support varies by model.
    if hasattr(choice, 'logprobs') and choice.logprobs is not None and choice.logprobs.content is not None:
        token_logprobs = [t.logprob for t in choice.logprobs.content]
        token_ppls = [math.exp(-lp) for lp in token_logprobs]
        count_below = sum(1 for ppl in token_ppls if ppl < 1.05)
        ratio_below_threshold = count_below / len(token_ppls) if token_ppls else 0.0
    else:
        logging.warning(f"Warning: logprobs not available for {model_name}. Cannot compute PPL threshold.")

    return step_str, finished, num_output_tokens, ratio_below_threshold, token_ppls

def get_dataset_local(dataset_name):
    logging.info(f"Downloading/Loading dataset: {dataset_name}...")
    if dataset_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    elif dataset_name == "aime25":
        dataset = load_dataset("math-ai/aime25", "gpqa_diamond")["train"]
    else:
        raise NotImplementedError
    return dataset

def has_hesitation(text: str) -> bool:
    hesitation_patterns = [
        r"\bwait\b", r"\bhmm\b", r"\b(?:um|uh|er|ah)\b", r"\bmaybe\b", r"\bperhaps\b",
        r"\bcould be\b", r"\bmight be\b", r"\bpossibly\b", r"\b(on the other hand)\b",
        r"\b(alternatively)\b", r"\b(another possibility)\b", r"\b(or perhaps)\b",
        r"\b(actually)\b", r"\bnow that I think about it\b", r"\bI think I made a mistake\b",
        r"\blet me reconsider\b", r"\bnot sure\b", r"\bI'm not entirely sure\b",
        r"\bthis might be wrong\b", r"\bI could be mistaken\b", r"\bunless I'm wrong\b",
        r"\bthinking\b", r"\bunsure\b", r"\bconfused\b", r"\bdebatable\b"
    ]
    pattern = '|'.join(hesitation_patterns)
    return bool(re.search(pattern, text, re.IGNORECASE))

# ==========================================
# MAIN EXECUTION
# ==========================================
args.output_dir = f'{args.output_dir}/TrigReason_{args.start_step}_{args.rectify_step}_{args.confident_threshold}_{args.dataset_name}_{args.small_model_name.replace("/", "_")}_{args.big_model_name.replace("/", "_")}'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.dataset = get_dataset_local(args.dataset_name)

if args.dataset_name == "aime24":
    problem = args.dataset["problem"][args.problem_id]
    ans = args.dataset["answer"][args.problem_id]
    options = None
elif args.dataset_name == "gpqa":
    problem = args.dataset["Question"][args.problem_id]
    ans = "B"
    options = {
        "A": args.dataset["Incorrect Answer 1"][args.problem_id],
        "B": args.dataset["Correct Answer"][args.problem_id],
        "C": args.dataset["Incorrect Answer 2"][args.problem_id],
        "D": args.dataset["Incorrect Answer 3"][args.problem_id],
    }
elif args.dataset_name == "aime25":
    problem = args.dataset["question"][args.problem_id]
    ans = args.dataset["answer"][args.problem_id]
    options = None

problem_id_str = f"{args.dataset_name}_{args.problem_id}"
output_filename = os.path.join(args.output_dir, f"{args.problem_id}_{args.repeat_id}")

logging.info(f"Starting Problem: {problem}")

steps_so_far = []
step_id = 0
metadata_list = []
rectify_step = 0

cur = time.time()
try:
    while True:
        time.sleep(3)
        warning_flag = False
        
        # 1. trigger a planning using a big model
        if step_id < args.start_step:
            logging.info(f"[Step {step_id}] Priming with BIG model...")
            step_str, finished, num_output_tokens, _, _ = generate_new_step(problem, steps_so_far, args.big_model_name, options=options)
            base_model_step, num_output_tokens_base = step_str, num_output_tokens
            small_model_step, num_output_tokens_small = "", None
            
        # 2. generate a reasoning step using a small model
        else:
            logging.info(f"[Step {step_id}] Generating with SMALL model...")
            step_str, finished, num_output_tokens, ratio, ppls = generate_new_step(problem, steps_so_far, args.small_model_name, options=options)
            small_model_step, num_output_tokens_small = step_str, num_output_tokens
            base_model_step, num_output_tokens_base = None, None
            
            # 3. trigger when over-confident or rectify
            if ratio > args.confident_threshold or rectify_step > 0:
                logging.info(f"[Step {step_id}] Intervening! Generating fallback with BIG model...")
                step_str, finished, num_output_tokens, _, _ = generate_new_step(problem, steps_so_far, args.big_model_name, options=options)
                base_model_step, num_output_tokens_base = step_str, num_output_tokens
                num_output_tokens_small = None
                
                if ratio > args.confident_threshold:
                    logging.info(f"   -> Rejected small model: hyper-confident ratio {ratio:.2f}")
                else:
                    logging.info(f"   -> Rejected small model: currently rectifying.")
                    rectify_step -= 1
            else:
                logging.info(f"   -> Accepted small model: ratio {ratio:.2f}")
                if step_id > 10:
                    if has_hesitation(step_str) and has_hesitation(metadata_list[-1]['step_str']) and has_hesitation(metadata_list[-2]['step_str']):
                        rectify_step = args.rectify_step
                
        if "</think>" in step_str and not finished:
            logging.warning(f"Warning: step_str had a </think>, removing.")
            step_str = step_str.replace("</think>", "")
            warning_flag = True
        
        # 4. append and repeat
        steps_so_far.append(step_str)
        print(f"\n--- STEP {step_id} OUTPUT ---\n{step_str}\n---------------------------\n")
         
        metadata = {
            "step_id": step_id,
            "step_str": step_str,
            "small_model_step": small_model_step,
            "num_output_tokens_small": num_output_tokens_small,
            "base_model_step": base_model_step,
            "num_output_tokens_base": num_output_tokens_base,
            "final_num_output_tokens": num_output_tokens_base if num_output_tokens_base is not None else num_output_tokens_small,
        }
        if warning_flag:
            metadata["warning"] = "step_str had a </think>"
        metadata_list.append(metadata)
        step_id += 1
        
        if len(steps_so_far) > 2:
            finished = finished or steps_so_far[-1] == steps_so_far[-2] 
        
        if finished or sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
            logging.info(f"Problem ID {args.problem_id} Repeat ID {args.repeat_id} Total time: {time.time() - cur:.1f} seconds")
            total_tokens = sum([m["final_num_output_tokens"] for m in metadata_list])
            small_tokens = sum([m["num_output_tokens_small"] for m in metadata_list if m["num_output_tokens_small"] is not None]) 
            small_percent = small_tokens / total_tokens if total_tokens > 0 else 0
            
            logging.info(f"Total tokens count {total_tokens}") 
            logging.info(f"Small_tokens count {small_tokens}") 
            logging.info(f"Small model tokens percentage {small_percent:.2%}") 
             
            if sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
                metadata_list[-1]["stop_reason"] = "budget"
            else:
                metadata_list[-1]["stop_reason"] = "finished"
            break

except ValueError as e:
    logging.error(f"ValueError caught: {e}")

# Save the outputs
os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_list, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_list, stream=f)

print(f"\n✅ Finished! Results saved to {output_filename}.txt")
