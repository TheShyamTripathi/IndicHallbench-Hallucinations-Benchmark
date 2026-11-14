# IndicHallBench: A Multilingual Benchmark for Fact-Level Hallucination Evaluation in Indic Languages

<p align="center">
  <img src="https://img.shields.io/badge/Research-NLP-blue" alt="Research"/>
  <img src="https://img.shields.io/badge/Domain-Hallucination%20Detection-purple" alt="Hallucination Detection"/>
  <img src="https://img.shields.io/badge/Languages-6-green" alt="Multilingual"/>
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange" alt="Under Development"/>
  <img src="https://img.shields.io/badge/Platform-Python-yellow" alt="Python"/>
</p>

<p align="center">
  <b>Evaluating factual reliability, claim-level consistency, and hallucination behavior in Large Language Models across Indic languages.</b>
</p>

<p align="center">
  <a href="https://github.com/TheShyamTripathi/IndicHallbench-Hallucinations-Benchmark">GitHub Repository</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#annotation-framework">Annotation Framework</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#citation">Citation</a>
</p>

---

## 1. Overview

**IndicHallBench** is a multilingual benchmark initiative designed to investigate and evaluate factual hallucinations in Large Language Models (LLMs), with particular emphasis on Indic languages and fact-level evaluation.

Although modern LLMs demonstrate increasingly strong capabilities in natural language understanding and generation, their responses can contain fabricated facts, unsupported statements, factual inconsistencies, and partially incorrect information. These problems become particularly challenging in multilingual settings, where language-specific resources, factual coverage, and reliable evaluation methodologies remain unevenly developed.

IndicHallBench aims to address this challenge through a structured benchmark that brings together domain-specific factual resources, multilingual datasets, LLM-generated responses, and fact-level annotation artifacts.

The benchmark is being developed to support systematic investigation of:

- Factual correctness at the individual claim or fact level.
- Hallucination behavior across different LLMs.
- Multilingual factual reliability.
- Domain-specific hallucination patterns.
- Consistency between generated responses and reference facts.
- Annotation-driven evaluation of factual claims.
- Reproducible benchmarking and comparative analysis.

Rather than treating an entire response as a single unit of evaluation, IndicHallBench is designed around a more granular perspective in which factual statements can be examined against relevant reference information.

The project is under active development. Dataset statistics, annotation protocols, evaluation definitions, and release information will be updated as the benchmark progresses through validation and finalization.

## 2. Research Objectives

IndicHallBench is being developed around the following objectives:

1. **Multilingual factual evaluation:** Establish a benchmark covering multiple languages relevant to Indic-language research.

2. **Fact-level analysis:** Support evaluation at a finer granularity than response-level correctness alone.

3. **Domain diversity:** Incorporate factual resources from multiple knowledge domains to examine differences in hallucination behavior.

4. **Cross-model comparison:** Organize responses from different LLMs to facilitate comparative factuality analysis.

5. **Annotation-driven evaluation:** Develop a structured annotation workflow for identifying and examining factual inconsistencies.

6. **Reproducibility:** Provide the data-processing, annotation, validation, and analysis infrastructure required for repeatable experiments.

7. **Extensibility:** Enable future additions of languages, domains, models, factual resources, and evaluation approaches without compromising previous benchmark versions.

## 3. Dataset

### 3.1 Language Coverage

The current repository contains multilingual dataset artifacts covering the following languages:

| Language | Language code | Dataset artifact |
|---|---|---|
| English | `en` | `dataset_en.json` |
| Hindi | `hi` | `dataset_hi.json` |
| Hindi–English | `hi-en` | `dataset_hi-en.json` |
| Nepali | `ne` | `dataset_ne.json` |
| Tamil | `ta` | `dataset_ta.json` |
| Telugu | `te` | `dataset_te.json` |

These files are maintained under:

```text
data/processed/multilingual/
```

The multilingual dataset includes individual language-specific files and a consolidated cleaned dataset.

**Note:** The presence of a language-specific dataset does not necessarily establish complete translation equivalence or identical record coverage across languages. Such properties will be documented through the final dataset statistics and validation reports.

### 3.2 Knowledge Domains

The current source dataset organization includes ten knowledge domains:

| No. | Domain | Source dataset | Fact bank |
|---|---|---|---|
| 1 | Culture | `dataset_culture.json` | `fact_bank_culture.json` |
| 2 | Economics | `dataset_economics.json` | `fact_bank_economics.json` |
| 3 | Education | `dataset_education.json` | `fact_bank_education.json` |
| 4 | Geography | `dataset_geography.json` | `fact_bank_geography.json` |
| 5 | Government | `dataset_government.json` | `fact_bank_government.json` |
| 6 | Health | `dataset_health.json` | `fact_bank_health.json` |
| 7 | History | `dataset_history.json` | `fact_bank_history.json` |
| 8 | Science | `dataset_science.json` | `fact_bank_science.json` |
| 9 | Sports | `dataset_sports.json` | `fact_bank_sports.json` |
| 10 | Technology | `dataset_technology.json` | `fact_bank_technology.json` |

The domain-specific resources are organized under:

```text
data/raw/domain_datasets/
data/raw/fact_banks/
```

### 3.3 Dataset Components

The repository currently maintains several categories of data artifacts:

| Component | Description |
|---|---|
| Domain-specific datasets | Original domain-organized dataset files |
| Fact banks | Domain-specific factual reference resources |
| Merged datasets | Consolidated dataset artifacts |
| Multilingual datasets | Language-specific and cleaned multilingual data |
| LLM responses | Model-specific generated response collections |
| Annotation inputs | Reference and response files used in annotation |
| Pilot annotations | Earlier annotation pilot artifacts |
| Working annotations | Intermediate annotation outputs |
| Final annotations | Consolidated annotation outputs in JSONL and CSV formats |
| Manual-review data | Records prepared for manual inspection |

The complete dataset composition, record counts, language distribution, domain distribution, and annotation statistics will be reported in the finalized dataset documentation.

## 4. LLM Response Collection

The current repository contains model-response collections associated with six LLM families:

- ChatGPT
- Claude
- DeepSeek
- Gemini
- Qwen
- Z.ai

The corresponding response files are maintained under:

```text
data/processed/llm_responses/
```

Available artifacts include:

```text
model_responses_clean.json
responses_chatgpt.json
responses_claude.json
responses_deepseek.json
responses_gemini.json
responses_qwen.json
responses_zai.json
```

These resources provide the foundation for examining factual response behavior across different language models.

For reproducible comparisons, future dataset releases should document the exact model identifiers, model versions, generation dates, prompts, and generation configurations wherever available.

Model-family names alone should not be interpreted as sufficient experimental metadata.

## 5. Methodology

IndicHallBench is organized around a data preparation, response collection, annotation, and evaluation workflow.

The following describes the intended high-level research organization. Detailed operational definitions and implementation-specific procedures will be maintained in the methodology documentation.

### 5.1 Dataset Preparation

Domain-specific datasets and their corresponding fact banks are maintained as separate resources.

The preparation pipeline includes dataset extraction, merging, multilingual data processing, and consolidation.

Relevant implementation files include:

```text
src/indichallbench/extraction/
src/indichallbench/preprocessing/
```

### 5.2 Multilingual Data Processing

The repository maintains language-specific dataset artifacts and a consolidated cleaned multilingual dataset.

Processing scripts support dataset merging and question extraction.

The objective is to maintain traceability between the source datasets and their processed counterparts.

### 5.3 LLM Response Organization

Generated responses are maintained in model-specific collections alongside consolidated response data.

This organization supports future model-wise and language-wise comparisons.

### 5.4 Fact-Level Annotation

The annotation framework contains:

- Annotation scripts.
- Optimized annotation implementation.
- Annotation prompts.
- Pilot annotation records.
- Working annotation outputs.
- Final annotation outputs.
- Manual-review artifacts.
- Annotation metrics.

The annotation pipeline is maintained under:

```text
annotation/
```

The exact annotation labels, decision rules, treatment of partial correctness, and adjudication procedures will be documented in the finalized annotation guidelines.

### 5.5 Validation and Quality Assurance

The benchmark is being organized to support systematic validation of dataset and annotation artifacts.

Planned validation areas include:

- Structural and schema validation.
- Missing-value detection.
- Duplicate record identification.
- Record identifier consistency.
- Annotation completeness.
- Reference-to-response consistency.
- Reproducibility of data-processing operations.

Structural validity and factual correctness are distinct evaluation concerns and should be assessed separately.

### 5.6 Evaluation and Analysis

The evaluation infrastructure is intended to support factuality and hallucination analysis at multiple levels, including:

- Fact-level analysis.
- Response-level analysis.
- Language-wise comparisons.
- Domain-wise comparisons.
- Model-wise comparisons.
- Annotation distribution analysis.

The exact metrics and their mathematical definitions will be published with the corresponding evaluation implementation and methodology.

## 6. Annotation Framework

The annotation infrastructure is maintained independently from the source dataset and response-generation resources.

Current organization:

```text
annotation/
├── notebooks/
│   ├── Data_Annotation_manual.ipynb
│   └── Data_Annotation_Merge.ipynb
│
├── prompts/
│   └── annotation_prompt.txt
│
├── reports/
│   └── annotation_metrics.jsonl
│
└── scripts/
    ├── annotate.py
    └── annotate_optimized.py
```

The associated annotation data is maintained under:

```text
data/annotations/
├── final/
│   ├── final_annotations.csv
│   └── final_annotations.jsonl
│
├── manual_review/
│
├── pilot/
│   └── annotations_pilot_v3.jsonl
│
└── working/
    ├── annotations.jsonl
    ├── annotation_errors.jsonl
    ├── references.json
    └── responses.json
```

The distinction between pilot, working, manual-review, and final annotation artifacts is intentional.

Only validated and documented annotation outputs should be considered part of an official benchmark release.

The final annotation schema and annotation guidelines will specify the exact meaning of each field and label.

## 7. Repository Structure

The repository follows a modular organization that separates datasets, annotation resources, processing scripts, documentation, and research utilities.

```text
IndicHallBench/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── .github/
│   └── workflows/
│
├── annotation/
│   ├── notebooks/
│   ├── prompts/
│   ├── reports/
│   └── scripts/
│
├── data/
│   ├── annotations/
│   │   ├── final/
│   │   ├── manual_review/
│   │   ├── pilot/
│   │   └── working/
│   │
│   ├── interim/
│   ├── processed/
│   │   ├── llm_responses/
│   │   └── multilingual/
│   │
│   ├── raw/
│   │   ├── domain_datasets/
│   │   └── fact_banks/
│   │
│   ├── samples/
│   └── schemas/
│
├── docs/
│
├── experiments/
│   ├── configurations/
│   ├── logs/
│   └── results/
│
├── notebooks/
│
├── prompts/
│
├── scripts/
│
├── src/
│   └── indichallbench/
│       ├── annotation/
│       ├── evaluation/
│       ├── extraction/
│       ├── preprocessing/
│       ├── validation/
│       └── visualization/
│
├── tests/
│
├── validation/
│   ├── reports/
│   └── test_cases/
│
└── visualizations/
    ├── dashboards/
    ├── figures/
    └── tables/
```

Some directories are intended for additional validation, evaluation, testing, and visualization implementations that are under development.

## 8. Installation and Environment

IndicHallBench is being developed primarily using Python.

A Python 3.10 environment is recommended as the initial development baseline.

Clone the repository:

```bash
git clone https://github.com/TheShyamTripathi/IndicHallbench-Hallucinations-Benchmark.git

cd IndicHallbench-Hallucinations-Benchmark
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies once the project dependency files have been finalized:

```bash
pip install -r requirements.txt
```

For development installations, once the Python package configuration is available:

```bash
pip install -e ".[dev]"
```

**Environment note:** Dependency versions and optional packages will be documented in the finalized environment configuration. Not all modules in the repository necessarily require the same dependencies.

## 9. Usage

IndicHallBench is intended to provide reusable data-processing and evaluation components.

The following illustrates the planned workflow:

### 9.1 Dataset Exploration

Start by examining the language-specific and domain-specific dataset files.

```python
import json

dataset_path = "data/processed/multilingual/all_dataset_clean.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("Dataset loaded successfully.")
print("Container type:", type(dataset).__name__)

if isinstance(dataset, list):
    print("Number of records:", len(dataset))
```

### 9.2 Inspecting Annotation Records

```python
import json

annotation_path = (
    "data/annotations/final/final_annotations.jsonl"
)

with open(annotation_path, "r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index >= 5:
            break

        record = json.loads(line)
        print(record)
```

### 9.3 Exploring LLM Responses

```python
import json

response_path = (
    "data/processed/llm_responses/responses_chatgpt.json"
)

with open(response_path, "r", encoding="utf-8") as file:
    responses = json.load(file)

print("Response data loaded.")
print("Container type:", type(responses).__name__)
```

These examples demonstrate basic data inspection. The exact loading logic should be adjusted if a particular artifact uses a different JSON structure.

### 9.4 Validation and Evaluation

Dedicated validation and evaluation commands will be documented after their implementations and test cases have been completed.

Researchers should use the validation reports associated with each published benchmark version rather than assuming that all development-stage files have passed validation.

## 10. Reproducibility and Versioning

Reproducibility is a central design principle of IndicHallBench.

Future public releases will use versioned snapshots to ensure that researchers can identify the exact dataset and software artifacts used in their experiments.

The proposed versioning convention follows Semantic Versioning:

| Version component | Intended use |
|---|---|
| MAJOR | Incompatible changes to data schema or benchmark definitions |
| MINOR | New domains, languages, records, or compatible benchmark capabilities |
| PATCH | Corrections and compatible fixes |

For each official release, the project aims to document:

- Dataset version.
- Software version.
- Dataset composition.
- Annotation methodology.
- Data schema.
- Validation results.
- Evaluation definitions.
- Environment and dependencies.
- Known limitations.
- Changelog.

Official dataset and software archives will be maintained through Zenodo when their respective releases are published.

## 11. Intended Research Applications

IndicHallBench is intended to support research in:

- Multilingual hallucination detection.
- LLM factuality evaluation.
- Fact-level consistency assessment.
- Indic-language NLP.
- Cross-lingual factual reliability.
- Comparative LLM evaluation.
- Domain-specific hallucination analysis.
- Annotation-driven benchmark development.
- Factual verification and evidence-grounded generation.

Potential extensions include retrieval-augmented factuality evaluation, temporal fact verification, human-versus-model annotation analysis, and multimodal hallucination assessment.

These are prospective research directions, not claims that all such capabilities are currently implemented.

## 12. Limitations and Responsible Use

IndicHallBench is an evolving research resource. Its limitations should be considered when designing experiments and interpreting results.

Important considerations include:

1. **Language coverage:** The presence of a language in the dataset does not establish comprehensive coverage of all linguistic varieties, dialects, or cultural contexts.

2. **Domain coverage:** The ten current knowledge domains do not represent the entirety of human knowledge.

3. **Temporal factuality:** Facts may change over time. Evaluations should account for the time at which reference information was considered valid.

4. **Reference quality:** The reliability of fact-level evaluation depends on the quality, provenance, and completeness of reference information.

5. **Model variability:** LLM outputs may change across model versions, prompting configurations, and generation settings.

6. **Annotation subjectivity:** Some factual claims may require contextual interpretation or adjudication.

7. **Translation effects:** Translation and multilingual processing can introduce semantic differences that influence evaluation.

8. **Benchmark limitations:** Results obtained using IndicHallBench should not automatically be generalized to all languages, domains, models, or real-world applications.

IndicHallBench is intended for research and evaluation. Benchmark scores should not be interpreted as a complete measure of the reliability or safety of an LLM.

## 13. Data Access and Licensing

The GitHub repository serves as the central location for source code, documentation, and benchmark development.

The complete dataset and associated release artifacts will be made available through the designated archival and distribution platforms, subject to applicable licensing and redistribution permissions.

**Dataset availability:** DOI to be assigned following the first official dataset release.

**Software availability:** DOI to be assigned following the first official software release.

Please consult the repository license files and dataset documentation before redistributing any material.

Third-party source material, translations, and model-generated outputs may be subject to additional terms and restrictions. Availability in this repository should not be interpreted as unrestricted permission to redistribute third-party content.

## 14. Citation

If you use IndicHallBench in academic research, please cite the corresponding version of the dataset and software.

A provisional citation format is:

```text
Tripathi, S. (2026).
IndicHallBench: A Multilingual Benchmark for
Fact-Level Hallucination Evaluation in Indic Languages.
Dataset and accompanying software.
```

The official bibliographic citation and DOI will be added after the first public release.

For reproducible research, cite the exact dataset version used in your experiments. Cite the software release separately when using its processing, validation, or evaluation implementation.

## 15. Contribution

Contributions that improve the benchmark's reliability, documentation, multilingual coverage, and reproducibility are welcome.

Potential contribution areas include:

- Additional Indic-language resources.
- New factual domains.
- Dataset validation.
- Annotation quality assurance.
- Evaluation methodologies.
- Visualization and statistical analysis.
- Documentation and reproducibility improvements.
- Benchmark extensions.

Before contributing, please review the project documentation and ensure that submitted material has appropriate provenance and redistribution permissions.

Contribution guidelines will be maintained in `CONTRIBUTING.md`.

## 16. Project Roadmap

The project is being developed incrementally.

| Milestone | Description | Status |
|---|---|---|
| Dataset organization | Domain-specific and multilingual data organization | In progress |
| LLM response collection | Model-specific response organization | Existing artifacts |
| Annotation framework | Annotation scripts, prompts, and annotation outputs | Existing artifacts |
| Annotation completion | Complete fact-level annotation and review | In progress |
| Dataset documentation | Dataset card, schema, and methodology | Planned |
| Validation framework | Reproducible structural and quality validation | Planned |
| Evaluation framework | Fact-level evaluation and comparative analysis | Planned |
| Visualization framework | Reproducible benchmark statistics and figures | Planned |
| First public release | Versioned dataset and software archives | Planned |
| DOI registration | Permanent Zenodo identifiers | Planned |
| Benchmark extensions | Additional models, domains, and languages | Future work |

The roadmap will be updated as implementation and validation progress.

## 17. Acknowledgements

IndicHallBench is developed as a research initiative focused on improving the evaluation of factual reliability in multilingual Large Language Models, particularly in Indic-language settings.

Acknowledgements for data sources, annotation contributors, infrastructure, and other research support will be included in the corresponding official release documentation.

## 18. Maintainer and Contact

**Project:** IndicHallBench

**Researcher and Maintainer:** Shyam Tripathi

**GitHub:** [TheShyamTripathi](https://github.com/TheShyamTripathi)

**Repository:** [IndicHallBench – Hallucinations Benchmark](https://github.com/TheShyamTripathi/IndicHallbench-Hallucinations-Benchmark)

For questions, research collaboration, reproducibility issues, or benchmark contributions, please open an issue in the GitHub repository.

---

<p align="center">
  <b>IndicHallBench</b><br/>
  Toward systematic, reproducible, and multilingual evaluation of factual reliability in Large Language Models.
</p>