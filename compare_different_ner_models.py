""""
This script is designed for evaluating Named Entity Recognition (NER) models, 
particularly focusing on a comparative analysis between a baseline model and a newly proposed model. 
It employs the WNUT2017 dataset to provide a consistent evaluation framework, 
ensuring that both models are assessed under the same conditions.

The script operates by generating NER predictions from each model, 
followed by a meticulous process of aligning and mapping these predictions to 
the standardized tagging scheme of the dataset. This is crucial for addressing the variances 
in tokenization between the models' outputs and the actual data annotations, with a special 
focus on correctly handling subtokens and other tokenization nuances.

A significant feature of this evaluation is the manual mapping of model-specific entity tags 
to those in the dataset, facilitating an apples-to-apples comparison in the subsequent analysis. 
The core of the evaluation lies in computing key metrics such as precision, recall, and the F1 score for both models, 
alongside generating a comprehensive classification report that offers a detailed breakdown of each model's 
performance across different entity types.

This methodical approach allows for an in-depth comparative study, illuminating 
the strengths and potential areas for improvement in the NER capabilities of each model. 
Through this script, users can gain valuable insights into the relative performance of the models, 
guiding future developments and refinements in NER model implementations.
"""

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch

# Load the WNUT2017 test dataset
wnut2017_dataset = load_dataset("wnut_17", split='test')

# Mapping dictionary for entity tags
entity_mapping = {
    'LOC': 'LOCATION',
    'ORG': 'ORGANIZATION',
    'PER': 'PERSON',
    # Add more mappings as needed
}

# dataset tag to model tag mapping for BERT model
bert_manual_mappings = {
    'B-person': 'B-PER',
    'I-person': 'I-PER',
    'B-location': 'B-LOC',
    'I-location': 'I-LOC',
    'B-corporation': 'B-ORG',
    'I-corporation': 'I-ORG',
    'B-group': 'B-ORG',  # Assuming groups can be mapped to organizations
    'I-group': 'I-ORG',
    'B-creative-work': 'B-MISC',  # Assuming creative works can be considered miscellaneous
    'I-creative-work': 'I-MISC',
    'B-product': 'B-MISC',  # Assuming products can be considered miscellaneous
    'I-product': 'I-MISC',
    'O': 'O'  # 'O' tag typically remains the same
}

# dataset tag to model tag mapping for RoBERTa model
roberta_manual_mapping = {
    'B-person': 'B-person',
    'I-person': 'I-person',
    'B-product': 'B-product',
    'I-product': 'I-product',
    'B-group': 'B-group',
    'I-group': 'I-group',
    'B-location': 'B-location',
    'I-location': 'I-location',
    'B-creative-work': 'B-creative-work',
    'I-creative-work': 'I-creative-work',
    'B-corporation': 'B-corporation',
    'I-corporation': 'I-corporation',
    'O': 'O'
}

def compute_metrics(predictions, labels):
    """
    Compute precision, recall, and F1 metrics for NER predictions.

    Args:
        predictions (list): List of predicted labels.
        labels (list): List of true labels.

    Returns:
        A dictionary with precision, recall, and F1-score.
    """
    # Convert y_pred and y_label to list of lists format
    flat_predictions = [[label for label in seq] for seq in predictions]
    flat_labels = [[label for label in seq] for seq in labels]

    # Compute precision, recall, and F1 score
    precision = precision_score(flat_labels, flat_predictions)
    
    recall = recall_score(flat_labels, flat_predictions)
    f1 = f1_score(flat_labels, flat_predictions)

    # Optionally, generate a classification report
    report = classification_report(flat_labels, flat_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

def merge_entities(predicted_entities):
    # Merge consecutive 'I-' entities with their corresponding 'B-' entity and process tags
    merged_entities = []
    current_entity = None
    for word, label in predicted_entities:
        if label.startswith('B-'):
            # Start a new entity
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = (word, label.replace('B-', '').upper())  # Remove 'B-' and process tag
        elif label.startswith('I-'):
            # Append to the current entity and process tag
            if current_entity:
                current_entity = (current_entity[0] + word.replace('##', ''), current_entity[1])  # Merge words and remove '##'
        else:
            # Non-entity token, ignore
            pass

    # Append the last entity if exists
    if current_entity:
        merged_entities.append(current_entity)
    # Convert abbreviated entity tags to full forms using the mapping dictionary
    merged_entities_mapped = []
    for word, tag in merged_entities:
        if tag in entity_mapping:
            tag = entity_mapping[tag]
        merged_entities_mapped.append((word, tag))

    return merged_entities_mapped

def align_tokens(actual_tokens, pre_tokens, pre_labels):
    """
    Aligns predicted labels with tokens from a baseline tokenizer.

    This function addresses the tokenization differences between two models by aligning the labels
    from the model's predictions (which may use a different tokenizer) with the tokens generated
    by a baseline tokenizer. It ensures that each token from the baseline tokenizer has an 
    associated label, which is crucial for accurately evaluating the model's performance.

    The function handles sub-token elements (e.g., tokens prefixed with '##' in BERT tokenizers) 
    by merging them to reconstruct the original tokens, thus ensuring correct label assignment.

    Parameters:
    - actual_tokens (list of str): Tokens from the baseline tokenizer.
    - pre_tokens (list of str): Tokens from the model's tokenizer, which may include sub-tokens.
    - pre_labels (list of str): Labels assigned by the model, corresponding to pre_tokens.

    Returns:
    - list of str: Aligned labels for each token in actual_tokens.

    Example:
    actual_tokens = ["New", "York", "City"]
    pre_tokens = ["New", "York", "Ci", "##ty"]
    pre_labels = ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]
    aligned_labels = align_tokens(actual_tokens, pre_tokens, pre_labels)
    # aligned_labels will be ["B-LOC", "I-LOC", "I-LOC"]
    """
    aligned_labels = ['O'] * len(actual_tokens)  # Initialize with 'O' labels
    pre_index = 0  # Index for predicted tokens

    for actual_index, actual_token in enumerate(actual_tokens):
        if pre_index >= len(pre_tokens):
            break  # Stop if we've processed all predicted tokens

        reconstructed_token = ""
        while pre_index < len(pre_tokens):
            part_of_token = pre_tokens[pre_index].replace("##", "").replace("Ġ", "")
            reconstructed_token += part_of_token

            # Move to the next predicted token if it's part of the current actual token
            if actual_token.startswith(reconstructed_token):
                if actual_token == reconstructed_token:
                    aligned_labels[actual_index] = pre_labels[pre_index]
                    pre_index += 1  # Only move to the next predicted token if we fully matched
                    break
                pre_index += 1
            else:
                # This handles cases where the predicted tokenization moves beyond the current actual token
                break

    return aligned_labels

def map_model_tags_to_dataset_tags(predicted_tags, mapping):
    return [mapping.get(tag, 'O') for tag in predicted_tags]

# Function to create a mapping between model tags and dataset tags
def create_mapping(model_tags, dataset_tags):
    mapping = {}
    
    # Create a simplified mapping of just the entity types to full dataset tags
    entity_to_dataset_tag = {}
    for tag in dataset_tags:
        if '-' in tag:
            entity = tag.split('-')[-1]
            entity_to_dataset_tag[entity] = tag

    # Map model tags to dataset tags
    for tag in model_tags:
        if tag == 'O':
            mapping[tag] = 'O'
            continue

        prefix, entity = tag.split('-')
        
        # Find the corresponding dataset tag with the same entity
        if entity in entity_to_dataset_tag:
            mapped_tag = entity_to_dataset_tag[entity]
            # Ensure the prefix in the dataset tag matches the model tag
            if mapped_tag.startswith(prefix):
                mapping[tag] = mapped_tag
            else:
                # If no exact match, use the 'O' tag
                mapping[tag] = 'O'
        else:
            mapping[tag] = 'O'

    return mapping

model_name_to_manual_mapping = {
    'dslim/bert-base-NER': bert_manual_mappings,
    'dslim/bert-base-NER-uncased': bert_manual_mappings,
    'Babelscape/wikineural-multilingual-ner': bert_manual_mappings,
    'tner/roberta-large-wnut2017': roberta_manual_mapping,
    'dbmdz/bert-large-cased-finetuned-conll03-english': bert_manual_mappings,
}

def check_model_tags(model_name, dataset):
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model_ner_tags = set(model.config.id2label.values())
        dataset_ner_tags = set(dataset.features['ner_tags'].feature.names)
        manual_tags = set(model_name_to_manual_mapping[model_name].values())
        if model_ner_tags != manual_tags:
            print(f"Error: The model, {model_name} NER tags do not match the manually defined tags.")
            print("Model NER Tags:", model_ner_tags, " for model:", model_name)
            print("Manual NER Tags:", manual_tags)
        else:
            print(f"The model, {model_name}, NER tags match the manually defined tags.")

def evaluate_model_w_orig_word_list(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    manual_sub_mappings = model_name_to_manual_mapping[model_name]
    check_model_tags(model_name, wnut2017_dataset)

    predictions = []
    labels = []
    for example in dataset:
        tokens = example['tokens']
        encoded_input = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

        with torch.no_grad():
            output = model(**encoded_input)

        prediction = output.logits.argmax(-1).squeeze().tolist()
        word_ids = encoded_input.word_ids()

        # Initialize an empty list to hold the predictions for each original word
        word_level_prediction = ['O'] * len(tokens)  # Assuming 'O' is the default label

        for idx, word_id in enumerate(word_ids):
            if word_id is not None and (idx == 0 or word_id != word_ids[idx - 1]):
                # Assign the label of the first subtoken to the entire word
                word_level_prediction[word_id] = model.config.id2label[prediction[idx]]

        assert len(tokens) == len(word_level_prediction), "Length mismatch between tokens and predictions"
        predictions.append(word_level_prediction)
        actual_labels = [dataset.features['ner_tags'].feature.int2str(tag) for tag in example['ner_tags']]
        mapped_actual_labels = map_model_tags_to_dataset_tags(actual_labels, manual_sub_mappings)   
        labels.append(mapped_actual_labels)
    metrics = compute_metrics(predictions, labels)
    return metrics

def evaluate_model(model_name, dataset):
    # Load the model and tokenizer using the pipeline
    nlp = pipeline("ner", model=model_name, tokenizer=model_name)

    predictions = []
    labels = []

    manual_sub_mappings = model_name_to_manual_mapping[model_name]
    check_model_tags(model_name, wnut2017_dataset)

    for example in dataset:
        tokens = example['tokens']
        text = " ".join(tokens) if isinstance(tokens, list) else tokens
        ner_results = nlp(text)

        # Tokenize the text using the model's tokenizer to get a list of tokens that should align with ner_results
        tokenized_text = nlp.tokenizer.tokenize(text)

        # Initialize predicted labels with 'O'
        pre_labels = ['O'] * len(tokenized_text)
        
        
        # Update predicted labels based on ner_results
        for entity in ner_results:
            # Adjust the index to align with tokenized_text
            index = entity['index'] - 1  # Adjusting the index to match the tokenized text
            entity_word = entity['word'].replace("##", "")
            entity_tokens = nlp.tokenizer.tokenize(entity['word'])

            # Find the start index of the entity word in the tokenized text
            for i in range(index, min(index + 2, len(tokenized_text))):
                if tokenized_text[i].endswith(entity_word):
                    pre_labels[i] = entity['entity']
                    break

        # Align and map the predicted labels to the dataset labels
        aligned_pre_labels = align_tokens(tokens, tokenized_text, pre_labels)
        mapped_pre_labels = map_model_tags_to_dataset_tags(aligned_pre_labels, manual_sub_mappings)

        predictions.append(mapped_pre_labels)
        actual_labels = [dataset.features['ner_tags'].feature.int2str(tag) for tag in example['ner_tags']]
        labels.append(actual_labels)
    metrics = compute_metrics(predictions, labels)
    return metrics

def main():
    # Model names
    models = [
        'dslim/bert-base-NER',
        'dslim/bert-base-NER-uncased',
        'Babelscape/wikineural-multilingual-ner',
        'tner/roberta-large-wnut2017',
        'dbmdz/bert-large-cased-finetuned-conll03-english',
    ]

    # Evaluate each model
    for model in models:
        metrics = evaluate_model_w_orig_word_list(model, wnut2017_dataset)
        print(f"Metrics for {model}: {metrics}")

if __name__ == "__main__":
    main()
