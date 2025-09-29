from rouge_score import rouge_scorer, tokenizers, scoring
import collections
import six
import argparse
import os
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import concurrent.futures



# This code is directly adopted from rouge_score from https://github.com/google-research/google-research/tree/master/rouge
# All credit should go to the original authors.
def _create_ngrams(tokens, n):
  """Creates ngrams from the given list of tokens.

  Args:
    tokens: A list of tokens from which ngrams are created.
    n: Number of tokens to use, e.g. 2 for bigrams.
  Returns:
    A dictionary mapping each bigram to the number of occurrences.
  """

  ngrams = collections.Counter()
  for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
    ngrams[ngram] += 1
  return ngrams

## I added the option to extract the counts 
def _score_ngrams(target_ngrams, prediction_ngrams):
  """Compute n-gram based rouge scores.

  Args:
    target_ngrams: A Counter object mapping each ngram to number of
      occurrences for the target text.
    prediction_ngrams: A Counter object mapping each ngram to number of
      occurrences for the prediction text.
  Returns:
    A Score object containing computed scores.
  """

  intersection_ngrams_count = 0
  for ngram in six.iterkeys(target_ngrams):
    intersection_ngrams_count += min(target_ngrams[ngram],
                                     prediction_ngrams[ngram])
  target_ngrams_count = sum(target_ngrams.values())
  prediction_ngrams_count = sum(prediction_ngrams.values())

  precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
  recall = intersection_ngrams_count / max(target_ngrams_count, 1)
  fmeasure = scoring.fmeasure(precision, recall)

  return intersection_ngrams_count, max(prediction_ngrams_count, 1),  max(target_ngrams_count, 1), scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

#######################################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Process a CSV and a corpus directory with specified text columns.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing the CSV file")
    parser.add_argument("corpus_dir", type=str, help="Path to the corpus directory containing text files")
    parser.add_argument(
        "--text_columns", "-c",
        nargs="+",
        type=str,
        required=True,
        help="List of text column names in the CSV"
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

def get_rouge_l(reference, generated, use_stemmer=False):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, generated)
    return scores['rougeL']


def get_rouge_n(reference, generated, n, use_stemmer=False):
    _tokenizer = tokenizers.DefaultTokenizer(use_stemmer)
    target_tokens = _tokenizer.tokenize(reference)
    prediction_tokens = _tokenizer.tokenize(generated)
    target_ngrams = _create_ngrams(target_tokens, n)
    prediction_ngrams = _create_ngrams(prediction_tokens, n)
    intersection_ngrams_count, generated_ngrams_count, target_ngrams_count, scores = _score_ngrams(target_ngrams, prediction_ngrams)
    return intersection_ngrams_count, generated_ngrams_count, target_ngrams_count, scores

def process_row(row_data):
    try:
        idx, row, corpus_dir, text_columns = row_data

        # English files in beginnings_sample_1000_en are named differently
        if "beginnings_sample_1000_en" in corpus_dir:
            if "PG" in idx:
                original_file = os.path.join(corpus_dir, f"{idx}_raw.txt")
            else:
                original_file = os.path.join(corpus_dir, f"PG{idx}_raw.txt")
        else:
            original_file = os.path.join(corpus_dir, f"{idx}")
        with open(original_file, "r", encoding="utf-8") as f:
            original_text = f.read()    
        generated_text = ""
        for col in text_columns:
            if col in row.index:
                generated_text += remove_incomplete_last_sentence(row[col])

        print("Calculating scores for:", idx)
        same_length_original = remove_incomplete_last_sentence(original_text[:len(generated_text)])
        rouge_l_same = get_rouge_l(same_length_original, generated_text)

        intersection_ngrams_count_full, generated_ngrams_count_full, target_ngrams_count_full, rouge_13_full = get_rouge_n(original_text, generated_text, 13)
        intersection_ngrams_count_same, generated_ngrams_count_same, target_ngrams_count_same, rouge_13_same = get_rouge_n(same_length_original, generated_text, 13)

        print(f"Processed {idx}")
        return {
            "filename": idx,
            "rouge_l_precision_same": rouge_l_same.precision,
            "rouge_l_recall_same": rouge_l_same.recall,
            "rouge_l_fmeasure_same": rouge_l_same.fmeasure,
            "13gram_overlap_count_full": intersection_ngrams_count_full,
            "13gram_generated_count_full": generated_ngrams_count_full,
            "13gram_target_count_full": target_ngrams_count_full,
            "13gram_overlap_rate_recall_full": rouge_13_full.recall,
            "13gram_overlap_rate_precision_full": rouge_13_full.precision,
            "13gram_overlap_rate_fmeasure_full": rouge_13_full.fmeasure,
            "13gram_generated_count_same": generated_ngrams_count_same,
            "13gram_target_count_same": target_ngrams_count_same,
            "13gram_overlap_count_same": intersection_ngrams_count_same,
            "13gram_overlap_rate_recall_same": rouge_13_same.recall,
            "13gram_overlap_rate_precision_same": rouge_13_same.precision,
            "13gram_overlap_rate_fmeasure_same": rouge_13_same.fmeasure
        }
    except Exception as e:
        print(f"Error processing {idx}: {e}")
        return {
            "filename": idx,
            "rouge_l_precision_same": None,
            "rouge_l_recall_same": None,
            "rouge_l_fmeasure_same": None,
            "13gram_overlap_count_full": None,
            "13gram_generated_count_full": None,
            "13gram_target_count_full": None,
            "13gram_overlap_rate_recall_full": None,
            "13gram_overlap_rate_precision_full": None,
            "13gram_overlap_rate_fmeasure_full": None,
            "13gram_generated_count_same": None,
            "13gram_target_count_same": None,
            "13gram_overlap_count_same": None,
            "13gram_overlap_rate_recall_same": None,
            "13gram_overlap_rate_precision_same": None,
            "13gram_overlap_rate_fmeasure_same": None
        }


if __name__ == "__main__":
    args = parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Corpus directory: {args.corpus_dir}")
    print(f"Text columns: {args.text_columns}")

    df = pd.read_csv(os.path.join(args.input_dir, "Model_Output.csv"), index_col=0)
    df = df.astype(str)
    df.index = df.index.astype(str)

    # Prepare data for parallel processing
    row_data_list = [(idx, row, args.corpus_dir, args.text_columns) for idx, row in df.iterrows()]
    
    # Process in parallel using ThreadPoolExecutor (more stable than ProcessPoolExecutor)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results_list = list(executor.map(process_row, row_data_list))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(args.input_dir, "Similarity_Scores.csv"), index=False)
