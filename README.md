# 🚀 AutoMathText-V2: A 2.46 Trillion Token AI-Curated STEM Pretraining Dataset

[![arXiv](https://img.shields.io/badge/arXiv-2402.07625-b31b1b.svg)](https://arxiv.org/abs/2402.07625)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://iiis-ai.github.io/AutoMathText-V2) 
[![Technical Report](https://img.shields.io/badge/Technical-Report-blue)](https://iiis-ai.github.io/AutoMathText-V2/AutoMathText-V2.pdf)
[![License: CC-BY-SA-4.0](https://img.shields.io/badge/License-AutoMathText-yellow.svg)](https://github.com/iiis-ai/AutoMathText-V2/blob/master/LICENSE)
[![AutoMathText-V2](https://img.shields.io/badge/Huggingface-Datasets-blue)](https://huggingface.co/datasets/OpenSQZ/AutoMathText-V2) 

📊 **AutoMathText-V2** consists of **2.46 trillion tokens** of high-quality, deduplicated text spanning web content, mathematics, code, reasoning, and bilingual data. This dataset was meticulously curated using a **three-tier deduplication pipeline** and **AI-powered quality assessment** to provide superior training data for large language models.

Our dataset combines **50+ premium data sources** with advanced processing techniques including **semantic deduplication**, **contamination detection**, and **intelligent text cleaning** to deliver exceptional model performance across diverse domains.

## 🎯 What makes AutoMathText-V2 special?

- **🔢 STEM Concentration**: Specially optimized for STEM content (especially Math) 
- **🔍 Triple Deduplication**: Exact → Fuzzy (MinHash+LSH) → Semantic (GTE embeddings) 
- **🤖 AI Quality Assessment**: Qwen2-based classifier with multi-source score fusion 
- **🧹 Advanced Text Cleaning**: All text data was processed using **Ultimate Data Cleaner v7.5.0.5**, which provides robust, high-performance cleaning tailored for web-scraped and scientific data. 
- **🛡️ Contamination Prevention**: Automatic test set leak detection and removal 

## 📚 Dataset Composition

### Token Distribution by Domain

| Domain | Token Count | Percentage | Description |
|--------|-------------|------------|-------------|
| **🏆 Nemotron CC High** | 1,468.3B | 59.7% | High quality CommonCrawl data |
| **🌐 DCLM** | 314.2B | 12.8% | DCLM baseline web content |
| **💻 RefineCode** | 279.4B | 11.4% | GitHub repositories (Academic Use Only) |
| **⭐ Nemotron CC Medium-High** | 254.5B | 10.3% | Medium-high quality CommonCrawl data |
| **📚 FineWeb Edu** | 117.4B | 4.8% | Educational web content |
| **🌏 Chinese** | 112.18B | 4.6% | Chinese general content |
| **🧠 Reasoning QA** | 86.2B | 3.5% | Instruction-following and complex reasoning tasks |
| **🔢 Math Web** | 68.3B | 2.8% | Mathematics and scientific content |
| **📊 MegaMath** | 28.5B | 1.2% | Specialized mathematical collections |
| **🔄 Translation** | 1.61B | 0.1% | English-Chinese translation pairs |
| **Total** | **2,460.71B** | **100%** | Complete dataset |


### 🔥 Complete Data Sources by Domain (52 Premium Datasets)

#### **📍 DCLM Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| DCLM-Baseline | `DCLM/dclm-baseline-1.0` | High-quality web content from DCLM |

#### **📚 FineWeb Edu Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | Educational web content (0-5 quality scale) |

#### **🌏 FineWeb Edu Chinese Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| FineWeb-Edu-Chinese | `opencsg/Fineweb-Edu-Chinese-V2.1` | Chinese educational content (3.4-5.0 scale) |

#### **🔢 Math Web Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| AutoMathText | `math-ai/AutoMathText` | Math/Code/ArXiv content with lm_q1q2_score |
| FineMath | `HuggingFaceTB/finemath` | High-quality mathematics content (0-5 scale) |
| Open-Web-Math-Pro | `gair-prox/open-web-math-pro` | Mathematical web pages |
| InfiMM-WebMath-40B | `Infi-MM/InfiMM-WebMath-40B` | Multimodal mathematical content |

#### **🏆 Nemotron CC High Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| Nemotron-CC (High) | `nvidia/nemotron-cc` | High-quality CommonCrawl subset |

#### **⭐ Nemotron CC Medium-High Domain** 
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| Nemotron-CC (Medium-High) | `nvidia/nemotron-cc` | Medium-high quality CommonCrawl subset |

#### **💻 RefineCode Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| RefineCode | `m-a-p/RefineCode` | GitHub repositories (Academic Use Only) |

#### **🧠 Reasoning QA Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| OPC-Annealing-Corpus | `OpenCoder-LLM/opc-annealing-corpus` | Code training corpus |
| OPC-SFT-Stage1 | `OpenCoder-LLM/opc-sft-stage1` | Instruction following data (stage 1) |
| OPC-SFT-Stage2 | `OpenCoder-LLM/opc-sft-stage2` | Instruction following data (stage 2) |
| Magpie-Reasoning-V2-250K-CoT-QwQ | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ` | Chain-of-thought reasoning (QwQ) |
| Magpie-Reasoning-V1-150K-CoT-QwQ | `Magpie-Align/Magpie-Reasoning-V1-150K-CoT-QwQ` | Chain-of-thought reasoning (QwQ) |
| Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B | `Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B` | Advanced reasoning (DeepSeek-R1) |
| Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | Advanced reasoning (DeepSeek-R1) |
| General-Instruction-Augmented-Corpora | `instruction-pretrain/general-instruction-augmented-corpora` | General instruction synthesis |
| FT-Instruction-Synthesizer-Collection | `instruction-pretrain/ft-instruction-synthesizer-collection` | Fine-tuning instruction synthesis |
| Code-Feedback-Filtered-Instruction | `m-a-p/CodeFeedback-Filtered-Instruction` | Code QA with feedback |
| XCoder-80K | `banksy235/XCoder-80K` | Code instruction data |
| Orca-Math-Word-Problems-200K | `microsoft/orca-math-word-problems-200k` | Math word problems |
| Meta-Math-QA | `meta-math/MetaMathQA` | Mathematical QA dataset |
| Numina-Math-CoT | `AI-MO/NuminaMath-CoT` | Math chain-of-thought |
| Scale-Quest-Math | `dyyyyyyyy/ScaleQuest-Math` | Mathematical problem solving |
| Calc-Ape210K | `MU-NLPC/Calc-ape210k` | Chinese math problems |
| MathInstruct | `TIGER-Lab/MathInstruct` | Math instruction data |
| MathScaleQA-2M | `fdqerq22ds/MathScaleQA-2M` | Large-scale math QA |
| Gretel-Math-GSM8K-V1 | `gretelai/gretel-math-gsm8k-v1` | GSM8K style problems |
| Open-Math-Instruct-2 | `nvidia/OpenMathInstruct-2` | Open math instructions |
| Stack-Math-QA | `math-ai/StackMathQA` | Stack Exchange math QA |
| OpenR1-Math-220K | `open-r1/OpenR1-Math-220k` | Advanced math reasoning |
| Natural-Reasoning | `facebook/natural_reasoning` | Natural language reasoning |
| Math-Code-Instruct | `MathLLMs/MathCodeInstruct` | Math with code instructions |
| Math-Code-Instruct-Plus | `MathLLMs/MathCodeInstruct-Plus` | Enhanced math-code instructions |
| Open-Orca | `Open-Orca/OpenOrca` | General instruction following |
| SlimOrca-Deduped-Cleaned-Corrected | `Open-Orca/slimorca-deduped-cleaned-corrected` | Cleaned instruction data |
| Orca-AgentInstruct-1M-V1-Cleaned | `mlabonne/orca-agentinstruct-1M-v1-cleaned` | Agent instruction data |
| FOL-NLI | `tasksource/FOL-nli` | First-order logic reasoning |
| Infinity-Instruct | `BAAI/Infinity-Instruct` | Multi-domain instructions |
| Llama-Nemotron-Post-Training-Dataset-V1 | `nvidia/Llama-Nemotron-Post-Training-Dataset-v1` | Post-training dataset |
| Codeforces-CoTs | `open-r1/codeforces-cots` | Competitive programming |
| Reasoning-V1-20M | `glaiveai/reasoning-v1-20m` | Large-scale reasoning data |
| Lean-STaR-Plus | `ScalableMath/Lean-STaR-plus` | Lean formal proofs (enhanced) |
| Lean-STaR-Base | `ScalableMath/Lean-STaR-base` | Lean formal proofs (base) |
| Lean-CoT-Plus | `ScalableMath/Lean-CoT-plus` | Lean chain-of-thought (enhanced) |
| Lean-CoT-Base | `ScalableMath/Lean-CoT-base` | Lean chain-of-thought (base) |
| Lean-Github | `internlm/Lean-Github` | Lean repository code |
| Lean-Workbook | `internlm/Lean-Workbook` | Lean problem workbook |
| DeepSeek-Prover-V1 | `deepseek-ai/DeepSeek-Prover-V1` | Formal proof verification |

#### **🔄 Translation Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| UN-PC | `Helsinki-NLP/un_pc` | English-Chinese translation pairs |
| UN-PC-Reverse | `Helsinki-NLP/un_pc` | Chinese-English translation pairs |

#### **📊 MegaMath Domain**
| Source | HuggingFace Dataset | Description |
|--------|-------------------|-------------|
| MegaMath-QA | `LLM360/MegaMath` | Large-scale mathematical QA |
| MegaMath-Translated-Code | `LLM360/MegaMath` | Mathematical code translations |
| MegaMath-Text-Code-Block | `LLM360/MegaMath` | Mixed math text and code blocks |

**Total: 52 Premium Data Sources** with official HuggingFace dataset links covering web content, mathematics, code, reasoning, formal proofs, and bilingual data.

## 🛠️ Processing Pipeline

### 1. **Data Extraction & Standardization**
```python
{
    "domain_prefix": "lbty.org",
    "id": "117b6a7d-5126-41fe-9bc2-d276e98632e6",
    "meta": "{\"domain\": \"dclm\", \"ori_score\": 0.043276190757751465, \"source\": \"dclm_baseline\"}",
    "text": "Sabine Expedition\n\nThe Sabine Expedition was an expedition approved by the United States Congress in 1806...",
    "tokens": 145,  # Token count using Qwen2.5 tokenizer
    "url": "[https://lbty.org/american-indian-battles/sabine-expedition/](https://lbty.org/american-indian-battles/sabine-expedition/)",
    "score": 0.19072403013706207
}
````

### 2\. **Three-Tier Deduplication**

#### 🎯 **Exact Deduplication**

  - SHA256 content hashing
  - Priority-based duplicate resolution
  - **Result**: \~30% exact duplicates removed

#### 🔄 **Fuzzy Deduplication** 

  - MinHash Locality Sensitive Hashing (LSH)
  - Jaccard similarity threshold: 0.9
  - Connected components clustering
  - **Result**: \~20% near-duplicates removed

#### 🧠 **Semantic Deduplication**

  - `Alibaba-NLP/gte-multilingual-base` embeddings
  - K-means clustering (k=100,000)  
  - Cosine similarity threshold: 0.007
  - **Result**: \~10% semantic duplicates removed

### 3\. **🤖 AI Quality Assessment**

**Qwen2-Based Classifier Architecture**:

  - Fine-tuned regression head for quality scoring
  - Multi-source score normalization and fusion
  - MSE loss with sigmoid activation

### 4\. **🧹 Advanced Text Cleaning**

All text data was processed using **Ultimate Data Cleaner v7.5.0.5**, which provides robust, high-performance cleaning tailored for web-scraped and scientific data.

**Key Features Used:**

  - **Advanced LaTeX & Code Protection**: protect complex nested LaTeX environments (`\begin{}...\end{}`), inline math (`$...$`), commands, and markdown code fences. 
  - **Quality Heuristics**: Removes corrupted samples with excessive repetition, severe bracket imbalances, etc. 

### 5\. **🛡️ Contamination Detection**

**Test Set Protection**:

  - Math dataset test questions
  - GSM8K evaluation problems  
  - Exact string matching with preprocessing
  - Automatic filtering during data extraction

## 🚀 How to Use

### Loading with Datasets

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("OpenSQZ/AutoMathText-V2", streaming=True)

# Load specific domain
math_data = load_dataset("OpenSQZ/AutoMathText-V2", name="math_web", streaming=True)
```

### 💻 RefineCode Content Download

**Important**: For the RefineCode domain, only metadata is included in the dataset. The actual code content was removed to reduce storage requirements. To access the full code content, use the `blob_id` field from the metadata to download from AWS S3:

```python
import os
import json
import boto3
from smart_open import open
from datasets import load_dataset

# Setup AWS credentials
session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)
s3 = session.client("s3")

def download_code_content(blob_id, src_encoding):
    """Download code content from AWS S3 using blob_id"""
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    
    try:
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            content = fin.read().decode(src_encoding)
        return {"content": content}
    except Exception as e:
        return {"content": None, "error": str(e)}

# Load RefineCode domain
refinecode_data = load_dataset("OpenSQZ/AutoMathText-V2", name="refinecode", streaming=True)

# Process each sample to download content
for sample in refinecode_data:
    # Parse metadata to extract blob_id and encoding
    meta = json.loads(sample["meta"])
    blob_id = meta.get("blob_id")
    src_encoding = meta.get("src_encoding", "utf-8")
    
    if blob_id:
        # Download the actual code content
        code_data = download_code_content(blob_id, src_encoding)
        
        # Combine metadata with downloaded content
        full_sample = {
            **sample,
            "code_content": code_data["content"]
        }
        
        print(f"Downloaded content for {sample['id']}")
        print(f"Content length: {len(code_data['content']) if code_data['content'] else 0}")
        break
```

**Requirements**:

  - AWS credentials with access to Software Heritage S3 bucket
  - `smart_open` library: `pip install smart_open[s3]`
  - `boto3` library: `pip install boto3`

**Note**: This download method is required only for the RefineCode domain. All other domains contain the full text content directly in the dataset.

## 🌐 Dataset Structure & Configurations

### Directory Structure

The dataset is organized by domain with quality-based token splits:

```
AutoMathText-V2/
├── dclm/                  # DCLM baseline web content
│   ├── 0-10/             # Bottom 10% quality tokens (score-based)
│   ├── 10-20/            # 10-20% quality tokens
│   ├── 20-30/            # 20-30% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── fineweb_edu/           # FineWeb educational content
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── 10-20/            # 10-20% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── fineweb_edu_chinese/   # Chinese educational content
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── math_web/              # Mathematics and scientific content
│   ├── 0-10/  .          # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── megamath/              # Specialized math collections
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── nemotron_cc_high/      # High quality Nemotron CommonCrawl
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── nemotron_cc_medium_high/ # Medium-high quality Nemotron CommonCrawl
│   ├── 0-10/            . # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── reasoning_qa/          # Instruction and reasoning data
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
├── refinecode/            # GitHub code repositories (Academic Use Only)
│   ├── 0-10/             # Bottom 10% quality tokens
│   ├── ...               # Additional percentile ranges
│   └── 90-100/           # Top 10% highest quality tokens
└── translation/           # English-Chinese translation pairs
    ├── 0-10/             # Bottom 10% quality tokens
    ├── ...               # Additional percentile ranges
    └── 90-100/           # Top 10% highest quality tokens
```

### Quality-Based Token Distribution

Each domain is divided into **10 quality percentiles** (0-10, 10-20, ..., 90-100) based on:

  - **Token count**: Equal number of tokens per percentile bucket
  - **Quality scores**: AI classifier scores from Qwen2-based quality assessment
  - **Percentile ranking**: Higher percentiles contain higher quality content

### Available Configurations

  - **Domain-specific configs**: Load individual domains (`dclm`, `fineweb_edu`, `math_web`, `reasoning_qa`, etc.)
  - **Quality-filtered configs**: Load specific quality ranges (e.g., `dclm/90-100` for top quality DCLM content)
  - **Nemotron variants**: Choose between `nemotron_cc_high` and `nemotron_cc_medium_high` based on quality needs
  - **Combined configs**: Mix domains and quality levels based on training requirements
  - **Custom sampling**: Select percentile ranges across multiple domains for balanced training

### Language Distribution

  - **English**: \~95% of content
  - **Chinese**: \~5% of content

## 🔬 Technical Deep Dive

For detailed technical documentation, including:

  - Complete processing pipeline specifications  
  - Deduplication algorithm details
  - Quality classifier training procedures
  - Contamination detection methodology

Please refer to our [Technical Documentation](https://iiis-ai.github.io/AutoMathText-V2) and [GitHub Repository](https://github.com/iiis-ai/AutoMathText-V2).

## 🤝 Contributing

We welcome contributions to improve dataset quality and processing techniques:

  - 🐛 **Bug Reports**: Issues with data quality or processing
  - 💡 **Feature Requests**: New data sources or processing improvements  
  - 📚 **Documentation**: Help improve our guides and examples
  - 🔬 **Research**: Collaborate on quality assessment and deduplication methods

## 📜 Licensing & Citation

### License

Released under **AutoMathText Data Agreement for Model Training** (See [LICENSE](https://github.com/iiis-ai/AutoMathText-V2/blob/master/LICENSE)). 

### Citation

```bibtex
@misc{automathtext_v2_2025,
  title={AutoMathText-V2: A 2.46 Trillion Token AI-Curated STEM Pretraining Dataset},
  author={Li, Chao and Zhang, Yifan and Yuan, Yang and Yao, Andrew C},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/OpenSQZ/AutoMathText-V2},
  note={A 2.46T token multi-domain dataset with fine-grained deduplication and AI-powered quality assessment.}
}

@article{zhang2025autonomous,
  title={Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts},
  author={Zhang, Yifan and Luo, Yifan and Yuan, Yang and Yao, Andrew C},
  journal={The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025 Findings)},
  year={2025}
}
```
