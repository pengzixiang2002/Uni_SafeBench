# Does Synergy Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models

<p align="center">
    <img src="./images/logo.png" width="30%" height="30%">
</p>

<p align="center">
  <a href="http://arxiv.org/abs/2604.00547">[📖 arXiv Paper]</a> &nbsp;
  <a href="https://pengzixiang2002.github.io/Uni-SafeBench.github.io/">[🌐 Project Page]</a> &nbsp;
  <a href="https://github.com/pengzixiang2002/Uni_SafeBench">[💻 Code]</a> &nbsp;
  <a href="https://huggingface.co/datasets/Hades2002/Uni-SafeBench">[🤗 Dataset]</a>
</p>




---

## News

- **[2026/07/13]** We release the Uni-SafeBench dataset on Hugging Face. 🤗

- **[2026/07/10]** Uni-SafeBench has been accepted by **ACM Multimedia 2026**.

- **[2026/04/01]** The paper is available on arXiv.

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
git clone https://github.com/pengzixiang2002/Uni_SafeBench.git
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

## 📚 Citation

If you find Uni-SafeBench useful for your research or applications, please cite our paper using the following BibTeX:

```bibtex
@article{peng2026does,
  title={Does Unification Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models},
  author={Peng, Zixiang and Xu, Yongxiu and Zhang, Qinyi and Shen, Jiexun and Zhang, Yifan and Xu, Hongbo and Wang, Yubin and Gou, Gaopeng},
  journal={arXiv preprint arXiv:2604.00547},
  year={2026}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact **[pengzixiang@iie.ac.cn]**.