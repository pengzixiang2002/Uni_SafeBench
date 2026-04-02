# Does Synergy Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models

<p align="center">
    <img src="./images/logo.png" width="30%" height="30%">
</p>
<p align="center">
  <a href="http://arxiv.org/abs/2604.00547">[📖 arXiv Paper]</a> &nbsp;
</p>



---

- **[2026/04/01]** Revised manuscript is now available on arXiv to establish chronological priority and address recent concerns regarding potential plagiarism.

- **[2026/03/24]** Following the receipt of professional peer-review feedback, we formally **withdrew** the submission from ICML 2026 to incorporate significant enhancements and additional experiments.

- **[2026/01/29]** The core methodology, dataset, and evaluations were originally **submitted to ICML 2026 (Paper ID: 12029)**. To strictly comply with the double-blind review policy, the work was not previously posted on arXiv.

  > **Note:** We maintain comprehensive, time-stamped system records of the original January submission (ID: 12029) to resolve any disputes regarding academic priority.

<p align="center">
    <img src="images/framework.png" width="80%">
</p>


## 📋 Overview

**Uni-SafeBench** addresses the critical need fo safety evaluation in the era of Unified Multimodal Large Language Models (U-MLLMs). Unlike previous benchmarks that focus on single modalities, Uni-SafeBench provides:

* **Multi-Task Evaluation**: Seamlessly handles understanding (VQA) and generation (T2I, Editing) tasks.
* **Automated Judging**: A robust `Uni-Judger` toolchain powered by GPT-4o/Qwen-VL for consistent safety scoring.
* **Rich Scenarios**: Includes complex "Composite-Unsafe" cases where safe inputs combine to create unsafe intents.

## 🖼️ Dataset Examples

<p align="center">
    <img src="images/case.jpg" width="85%">
</p>


## 🗂️ Project Structure

```text
Uni-SafeBench/
├── Uni-Judger/              # 🛠️ Core evaluation toolkit
│   ├── safety_checker.py    # Main safety evaluation script
│   ├── intention.py         # Intent extraction from samples
│   ├── cate.py              # Category-based safety statistics
│   ├── statistics.py        # Cross-model statistics aggregation
│   ├── sum.py               # Result aggregation and summarization
│   └── utils.py             # Utility functions
├── data/                    # 📊 Benchmark datasets
│   ├── VQA/                 # Visual Question Answering tasks
│   │   ├── safety_I_safety_T/
│   │   ├── safety_I_unsafety_T/
│   │   ├── unsafety_I_safety_T/
│   │   └── unsafety_I_unsafety_T/
│   ├── T2I/                 # Text-to-Image generation
│   ├── Text-Guided Image Editing/
│   └── Text Generation/
└── gen_wcloud.py            # Word cloud visualization generator
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible endpoint for Judge models)

### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/Uni-SafeBench.git](https://github.com/yourusername/Uni-SafeBench.git)
cd Uni-SafeBench

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Set up your API credentials for the Judge Model:

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="your-base-url-here"  # Optional

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

## 📖 Usage

### 1. Safety Evaluation (Uni-Judger)

Run the automated safety checker on your model outputs:

Bash

```bash
python Uni-SafeBench/Uni-Judger/safety_checker.py \
    --input <path-to-input-jsonl> \
    --output <path-to-output-jsonl> \
    --model_name "Qwen3-VL-8B-Instruct" \
    --stats_output <path-to-stats-json>
```

**Input Format (JSONL):**

```json
{
  "id": 1,
  "prompt": "User prompt...",
  "output_image": "path/to/generated_image.jpg",
  "generated_text": "Model response...",
  "category": "Violence",
  "intention": "Harmful intent description"
}
```

### 2. Intention Extraction

Extract safety-related intentions from raw prompts to assist the judge:

```bash
python Uni-SafeBench/Uni-Judger/intention.py \
    --input <input-jsonl> \
    --output <output-jsonl> \
    --model "Qwen3-VL-8B-Instruct" \
    --image_root <path-to-images>
```

### 3. Statistical Analysis

Generate comprehensive reports after evaluation:

Bash

```bash
# Category-based Statistics
python Uni-SafeBench/Uni-Judger/cate.py

# Cross-model Statistics
python Uni-SafeBench/Uni-Judger/statistics.py

# Result Aggregation
python Uni-SafeBench/Uni-Judger/sum.py \
    --input <numeric-results-jsonl> \
    --output <aggregated-jsonl> \
    --stats <stats-json>
```

## ⚖️ License

```Text
Uni-SafeBench is only used for academic research. Commercial use in any form is prohibited.
The copyright of all images belongs to the image owners.
If there is any infringement in Uni-SafeBench, please email pengzixiang@iie.ac.cn and we will remove it immediately.
Without prior approval, you cannot distribute, publish, copy, disseminate, or modify Uni-SafeBench in whole or in part. 
You must strictly comply with the above restrictions.
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact **[pengzixiang@iie.ac.cn]**.