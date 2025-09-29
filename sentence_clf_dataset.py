import os
import gdown
import argparse
import pandas as pd
from transformers import AutoTokenizer
from transformers import logging
from transformers import pipeline, AutoConfig
import torch
import pandas as pd
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Process a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file. Assumes all related experiments are in the same parent directory for aggregated results.")
    parser.add_argument("--model_weights_dir", type=str, help="Path to the model weights directory. If directory is empty, the weight will be downloaded.")
    parser.add_argument(
        "--text_columns", "-c", 
        nargs="+", 
        type=str, 
        help="List of column names to process"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["en", "ger"],
        default="en",
        help="Language to use for sentence classification: 'en' for English or 'ger' for German (default: 'en')"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=500,
        help="Batch size for sentence prediction model (default: 5000)"
    )
    return parser.parse_args()

def remove_incomplete_last_sentence(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return text
    last = sentences[-1].strip()
    # If last sentence does not end with ., !, or ?, remove it
    if not re.search(r'[.!?]["\']?\s*$', last):
        sentences = sentences[:-1]
    return ' '.join(sentences)



if __name__ == "__main__":
    print("Setting up the sentence classification script...")
    args = parse_args()

    # Parse the arguments
    output_dir = os.path.dirname(args.csv_path)
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    batch_size = args.batch_size

    # Select the model depending on the language
    model_weights_dir = args.model_weights_dir
    if args.language == "en":
        model_weights_dir = os.path.join(model_weights_dir, "Rohrbach_Tagger_En")
        model_weight_gid = "1aVOT57euDZ2QpDb5_--f0DFgaw80IeE9"
    elif args.language == "ger":
        model_weights_dir = os.path.join(model_weights_dir, "Rohrbach_Tagger_Ger")
        model_weight_gid = "1-OIgf-F7lNsSkDv3fuqVN3tpCNv2LgiF"

    # Check if the model weights are downloaded
    if not os.path.exists(model_weights_dir):
        print(f"Model weights directory {model_weights_dir} does not exist. Downloading weights...")
        if not os.path.exists(model_weights_dir):
            gdown.download_folder(id=model_weight_gid, output=model_weights_dir)
    print(f"Using model weights from: {model_weights_dir}")
    
    # Load the model configuration
    print("Loading model configuration...")
    label2id = {"perceived_space": 0, "action_space": 1, "visual_space": 2, "descriptive_space":3, "no_space":4}
    id2label = {v : k for k, v in label2id.items()}
    config = AutoConfig.from_pretrained(model_weights_dir, label2id=label2id, id2label=id2label)
    if args.language == "ger":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    elif args.language == "en":
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    
    # Check if gpu is available 
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline('text-classification', model = model_weights_dir , tokenizer = tokenizer, config = config, device = device, batch_size=120)
     
    # Load the dataset
    print(f"Processing CSV file: {args.csv_path}")
    dataset = pd.read_csv(args.csv_path, index_col=0)

    if os.path.exists(os.path.join(output_dir, f"{base_name}_sentence_classification.csv")):
        result_df = pd.read_csv(os.path.join(output_dir, f"{base_name}_sentence_classification.csv"), index_col=0)
        dataset = dataset[~dataset.index.isin(result_df.index.unique())]


    print("Splitting the corpus into sentences...")
    text_columns = args.text_columns

    # For each column, split into sentences and remove incomplete last sentence
    for col in text_columns:
        dataset[col] = dataset[col].apply(remove_incomplete_last_sentence)

    # Join all sentences from all columns for each row
    dataset["sentence"] = dataset[text_columns].agg(" ".join, axis=1).apply(sent_tokenize)
    dataset["document_id"] = dataset.index
    dataset = dataset.explode("sentence")
    
    print("Total samples in dataset:", len(dataset.index))
    # Filter out empty, None, or non-string sentences
    print(f"Total sentences before filtering: {len(dataset)}")
    dataset = dataset[dataset["sentence"].notna()]  # Remove NaN values
    dataset = dataset[dataset["sentence"].astype(str).str.strip() != ""]  # Remove empty strings
    dataset = dataset[dataset["sentence"].astype(str) != "nan"]  # Remove string "nan"
    print(f"Total sentences after filtering: {len(dataset)}")
    
    dataset["sentence_id"] = dataset.groupby("document_id").cumcount()
    dataset = dataset.set_index(["document_id", "sentence_id"]).drop(labels=text_columns, axis=1)

    # Apply the sentence classification model
    print("Applying the sentence classification model...")
    print(f"Sample sentences: {dataset['sentence'].head().tolist()}")
    with torch.no_grad():
        results = pipe(dataset["sentence"].to_list(), batch_size=batch_size, truncation=True)
    # Convert results to a DataFrame
    result_df = pd.concat([dataset, pd.DataFrame(results, index=dataset.index)], axis=1)


    # Get a cummulative word count for each document
    result_df['word_count'] = result_df['sentence'].str.split().str.len()
    result_df = result_df.sort_values(by=['document_id', 'sentence_id'])
    result_df['cumulative_word_count'] = result_df.groupby('document_id')['word_count'].cumsum()


    print("Processing results...")
    if os.path.exists(os.path.join(output_dir, f"{base_name}_sentence_classification.csv")):
        result_df[["label", "score", "cumulative_word_count", "sentence"]].to_csv(os.path.join(output_dir, f"{base_name}_sentence_classification.csv"), mode='a', index=True)
    else:
        # Write out the result csv
        result_df[["label", "score", "cumulative_word_count", "sentence"]].to_csv(os.path.join(output_dir, f"{base_name}_sentence_classification.csv"), mode='w', index=True)
   