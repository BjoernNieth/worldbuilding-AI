# AI in Worldbuilding Study

Project description goes here 

## Project Overview

The project consists of four main scripts that form a complete experimental pipeline:

1. **`run_pipeline.py`** - Main experiment runner that generates narrative text using AI models
2. **`sentence_clf_dataset.py`** - Sentence classification tool for analyzing generated text
3. **`Get_Similarity_Scores.py`** - Calculates similarity scores between generated text and reference corpora
4. **`Package_Results.py`** - Aggregates results from multiple experiments into combined datasets

## Prerequisites

- Python 3.8+
- Hugging Face account and token with access granted to the models you want to use
- OpenAI account with API key
- Packages:
    - vLLM: Follow the instructions on https://docs.vllm.ai/en/latest/getting_started/installation/index.html.
    - Afterwards use your package manager to install pandas

## Usage

### 1. Run Pipeline (`run_pipeline.py`)

This is the main script that runs narrative generation experiments using various AI models.

```bash
python run_pipeline.py --data-path <path_to_datasets> --output-dir <output_directory> --huggingface-token <your_hf_token> --config-file-path <path_to_config.json> [--test-mode <True/False>]
```

**Parameters:**
- `--data-path`: Path to the datasets directory (contains `beginnings_de.csv`, `gutenberg_en_sample_1000_clean.csv`, etc.)
- `--output-dir`: Directory where experiment results will be saved
- `--huggingface-token`: Your Hugging Face authentication token
- `--config-file-path`: Path to experiment configuration JSON file (see `Experiment_Configs/` directory for examples)
- `--test-mode`: Optional boolean flag to run in test mode with smaller dataset samples (default: False)

**Example:**
```bash
python run_pipeline.py --data-path "./datasets" --output-dir "./results" --huggingface-token "hf_your_token_here" --config-file-path "./Experiment_Configs/Narrative_En_ChatGPT_Batch.json"
```

**Output:** Creates experiment directories with generated narratives and configuration files.

If you are using the OpenAI API, you have to set your API key before calling the script:
``` bash
export OPENAI_API_KEY='yourkey'"
```

### 2. Sentence Classification (`sentence_clf_dataset.py`)

Analyzes generated text using sentence classification models to categorize content.

```bash
python sentence_clf_dataset.py <csv_path> --model_weights_dir <path> --text_columns <column1> <column2> ... --language <en/ger> [--batch_size <size>]
```

**Parameters:**
- `csv_path`: Path to the input CSV file containing generated text
- `--model_weights_dir`: Path to the directory the model weights are downloaded to.
- `--text_columns`: List of column names to process
- `--language`: Language for classification - 'en' for English or 'ger' for German (default: 'en')
- `--batch_size`: Batch size for prediction model (default: 500)

**Example:**
```bash
python sentence_clf_dataset.py "./results/experiment1/output.csv" --text_columns generated_chapter_1 generated_chapter_2 --language en --batch_size 1000
```

**Output:** Creates `Model_Output_sentence_classification.csv` with classification results.

### 3. Get Similarity Scores (`Get_Similarity_Scores.py`)

Calculates ROUGE-n (n=13) and ROUGE-L similarity scores between generated texts and the reference text in the original corpus. The ROUGE-n score is calcualted between the beginning of the original text and the full original text. The beginning will have the same length as the according generated story.

```bash
python Get_Similarity_Scores.py <input_dir> <corpus_dir> --text_columns <column1> <column2> ...
```

**Parameters:**
- `input_dir`: Path to directory containing the CSV file with generated text
- `corpus_dir`: Path to directory containing reference corpus text files
- `--text_columns`: List of text column names in the CSV to analyze (required)

**Example:**
```bash
python Get_Similarity_Scores.py "./results/experiment1" "./datasets/reference_corpus" --text_columns generated_chapter_1 generated_chapter_2
```

**Output:** Creates `Similarity_Scores.csv` with ROUGE scores and similarity metrics.

### 4. Package Results (`Package_Results.py`)

Aggregates results from multiple experiments into combined datasets for analysis and stores them as a parquet file.

```bash
python Package_Results.py <input_dir1> <input_dir2> ... --output_dir <output_directory>
```

**Parameters:**
- `input_dirs`: List of input directory paths containing experiment results
- `--output_dir`: Path to output directory for combined results (required)

**Example:**
```bash
python Package_Results.py "./results/experiment1" "./results/experiment2" "./results/experiment3" --output_dir "./combined_results"
```

**Output:** Creates `combined_results.parquet` and `combined_similarity_scores.parquet` files.

## Experiment Configuration

The pipeline uses JSON configuration files to define experiments. See the `Experiment_Configs/` directory for examples:

- `Narrative_En_ChatGPT_Batch.json` - ChatGPT experiments in English
- `Narrative_Ger_Llama_70B_final.json` - Llama experiments in German
- And more...

Each config file specifies:
- Dataset to use
- Model configuration
- Prompt templates and system messages
- Generation parameters
- Experiment metadata

## Typical Workflow

1. **Setup**: Prepare your datasets and configure experiment parameters
2. **Generate**: Run `run_pipeline.py` to generate narrative text with AI models
3. **Classify**: Use `sentence_clf_dataset.py` to analyze the generated content
4. **Score**: Run `Get_Similarity_Scores.py` to compute overlap with reference corpus
5. **Aggregate**: Use `Package_Results.py` to combine results from multiple experiments

## Visualization
The notebooks used for the visualizations from the paper are in the subfolder "visualisation". The scripts assume, that the results files are in the same directory. 

## Project Structure

```
├── datasets/                  # Input datasets
├── Experiment_Configs/        # Configuration files for experiments
├── src/                       # Source code modules
│   ├── datasets/              # Dataset handling
│   ├── models/                # AI model implementations
│   ├── prompter/              # Prompting strategies
│   └── utils/                 # Utility functions
├── visualization/             # Notebooks producing the visualization of the paper
├── run_pipeline.py            # Main experiment runner
├── sentence_clf_dataset.py    # Sentence classification tool
├── Get_Similarity_Scores.py   # Similarity scoring
└── Package_Results.py         # Result aggregation
```

