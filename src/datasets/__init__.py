from .Narrative_Beginnings_Ger import Narrative_Beginnings_Ger
from .Narrative_Beginnings_En import Narrative_Beginnings_En
from datetime import datetime

datasets = {
    "Narrative_Beginnings_Ger": Narrative_Beginnings_Ger,
    "Narrative_Beginnings_En": Narrative_Beginnings_En
}



cutoff_dates = {
    # Before "Attention is All you Need" NeurIPS 2017
    "Pre-Transformer": datetime(2017, 12, 3),
    # Before BERT paper
    "Pre-LLMs": datetime(2018, 10, 10),
    # Before GPT3 + safety margin
    "Pre-GPT3": datetime(2020, 1, 1),
    # Before ChatGPT was made public
    "Pre-ChatGPT": datetime(2022, 11, 29)
}

def get_dataset(dataset_name, data_path, test_mode, cutoff="Pre-GPT3"):
    return datasets[dataset_name](data_path, cutoff_dates[cutoff], is_test=test_mode)
