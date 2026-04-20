# TandemLLM: Dynamic Multi-Model Reasoning Framework

TandemLLM is a dynamic, cost-efficient reasoning framework designed to solve complex logic and math problems (like GPQA and AIME) by orchestrating two Large Language Models (LLMs) simultaneously. 

Instead of relying solely on a massive, expensive model to generate an entire response, this framework uses a **"Big Model"** (e.g., 120B parameters) for strategic planning and error correction, while offloading the bulk of the step-by-step computational work to a fast, cheap **"Small Model"** (e.g., 1.5B - 8B parameters).

## 🧠 Core Mechanics: How it Works

The framework generates answers step-by-step, evaluating the small model's performance at every interval using three core triggers:

1. **Strategic Priming:** The Big Model is forced to generate the first few logical steps to ensure the foundational equations and logic of the problem are set up correctly.
2. **Cognitive Offloading & Confidence Thresholds:** Control is handed to the Small Model to execute the intermediate steps. The framework analyzes the mathematical probability (`logprobs`) of every token the small model generates. If the small model becomes suspiciously over-confident (a common sign of hallucination or repetitive loops in smaller models), the step is rejected and the Big Model steps in.
3. **Heuristic Intervention:** The framework monitors the small model's internal monologue for hesitation patterns (e.g., *"wait"*, *"hmm"*, *"I think I made a mistake"*). If the small model exhibits confusion across multiple steps, an intervention is triggered, and the Big Model takes over to course-correct.

## 🚀 Features
* **OpenRouter Integration:** Swap models instantly using OpenRouter's API.
* **Cost-Saving:** Achieve 120B+ parameter reasoning quality at a fraction of the API cost.
* **Built-in Rate Limit Handling:** Automatically catches `429` errors and applies an exponential backoff retry loop (perfect for using `:free` tier models).
* **Hugging Face Dataset Support:** Out-of-the-box support for loading `aime24`, `aime25`, and `gpqa` datasets.

## 🛠️ Setup & Installation

**1. Install Dependencies**
```bash
pip install openai datasets transformers tqdm
```
**2. API Keys**
You will need an OpenRouter API key to route the models.
If you plan to use gated datasets like gpqa, you will also need a Hugging Face access token.

**Output**
The script automatically generates a results/ directory containing:

A .txt file with a highly readable log of every step, token counts, and which model generated what.

A .pickle file containing the raw metadata for further data analysis.
