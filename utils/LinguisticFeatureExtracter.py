import spacy

languages = ["EN"]

def _ExtractLinguisticFeaturesEnglish(text):
    tagger = spacy.load("en_core_web_sm")
    tokens = tagger(text)
    features = {
        "past_tense_verbs": 0,
        "present_tense_verbs": 0,
        "perfect_aspect": 0,
        "personal_pronouns": 0,
        "nominalizations": 0,  # Words ending with -tion, -ment, etc.
        "conjunctions": 0,
        "place_adverbials": 0,
        "time_adverbials": 0
    }

    # Define nominalization suffixes
    nominal_suffixes = ("tion", "ment", "ness", "ity", "ance", "ence")

    # Adverbial examples
    place_adverbs = {"here", "there", "nearby"}
    time_adverbs = {"now", "then", "today", "yesterday", "since"}

    # Extract features
    for token in tokens:
        # Tense and aspect
        if token.tag_ in ["VBD"]:  # Past tense
            features["past_tense_verbs"] += 1
        if token.tag_ in ["VBZ", "VBP"]:  # Present tense
            features["present_tense_verbs"] += 1
        if token.tag_ == "VBN" and token.head.lemma_ == "have":  # Perfect aspect
            features["perfect_aspect"] += 1

        # Pronouns
        if token.pos_ == "PRON":
            features["personal_pronouns"] += 1

        # Nominalizations
        if token.pos_ == "NOUN" and token.text.endswith(nominal_suffixes):
            features["nominalizations"] += 1

        # Conjunctions
        if token.pos_ == "CCONJ":
            features["conjunctions"] += 1

        # Adverbials
        if token.pos_ == "ADV":
            if token.text.lower() in place_adverbs:
                features["place_adverbials"] += 1
            if token.text.lower() in time_adverbs:
                features["time_adverbials"] += 1

    return features


def ExtractLinguisticFeatures(text, language):
    assert language in languages

    if language == "EN":
        return _ExtractLinguisticFeaturesEnglish(text)


text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
print(ExtractLinguisticFeatures(text, "EN"))