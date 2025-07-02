import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

from nltk.corpus import stopwords
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
import string
import torch

# Function to turn list of string tokens into a sentence
def tokens_to_sentence(tokens):
    return ' '.join(tokens)

def encode_texts_in_batches_pooled(texts, tokenizer, model, batch_size=32):
    # Prepare a list to store the pooled outputs
    pooled_outputs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        # Select the batch slice
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the texts in the batch
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        # Get the model's output for the batch
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        # Collect the pooled_output from the batch
        pooled_outputs.append(outputs.pooler_output)
    
    # Concatenate the pooled outputs from all batches into a single tensor
    pooled_outputs_tensor = torch.cat(pooled_outputs, dim=0)
    
    return pooled_outputs_tensor

def encode_texts_in_batches_whole(texts, tokenizer, model, batch_size=32, max_length=32):
    # Prepare a list to store the sequence outputs
    sequence_outputs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i + batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        # Pad the sequences to the same max_length for all batches (if not already)
        sequence_output = outputs.last_hidden_state
        padding_needed = max_length - sequence_output.shape[1]
        if padding_needed > 0:
            # Assuming the model's embedding dimension is in sequence_output.shape[2]
            padding = torch.zeros((sequence_output.shape[0], padding_needed, sequence_output.shape[2]), dtype=sequence_output.dtype)
            sequence_output = torch.cat([sequence_output, padding], dim=1)
    
        sequence_outputs.append(sequence_output)

    sequence_outputs_tensor = torch.cat(sequence_outputs, dim=0)
    return sequence_outputs_tensor

def encode_sentences_whole(sentences, tokenizer, model, batch_size=64, max_length=32):
    sentences_outputs = []
    sentences_lengths = [len(sentence_list) for sentence_list in sentences]
    sentences_ids = [[id for id, _ in sentence_list] for sentence_list in sentences]
    unfoleded_sentences = [sentence for sentence_list in sentences for _, sentence in sentence_list]

    for i in tqdm(range(0, len(unfoleded_sentences), batch_size), desc="Encoding batches"):
        batch_texts = unfoleded_sentences[i:i + batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        # Pad the sequences to the same max_length for all batches (if not already)
        sequence_output = outputs.last_hidden_state
        padding_needed = max_length - sequence_output.shape[1]
        if padding_needed > 0:
            # Assuming the model's embedding dimension is in sequence_output.shape[2]
            padding = torch.zeros((sequence_output.shape[0], padding_needed, sequence_output.shape[2]), dtype=sequence_output.dtype)
            sequence_output = torch.cat([sequence_output, padding], dim=1)
    
        sentences_outputs.append(sequence_output)

    sequence_outputs_tensor = torch.cat(sentences_outputs, dim=0)
    folded_sentences_tensors = []
    start = 0
    for index, length in enumerate(sentences_lengths):
        ids = sentences_ids[index]
        new_sentences = []
        for i in range(length):
            new_sentences.append((ids[i], sequence_outputs_tensor[start + i]))
        folded_sentences_tensors.append(new_sentences)
        start += length
    
    return folded_sentences_tensors
    

# Function to preprocess and tokenize a sentence
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def preprocess_training_examples(sentences_dataset):
    preprocessed_dataset = []
    for _, data in tqdm(enumerate(sentences_dataset), total=len(sentences_dataset), desc="Preprocessing sentences"):
        claim_tokens = preprocess_sentence(data['claim'])
        sentence_tokens = preprocess_sentence(data['sentence'])
        preprocessed_dataset.append({'claim': claim_tokens, 'sentence': sentence_tokens, 'relevant': data['relevant']})
    return preprocessed_dataset

def preprocess_training_examples_triplet(sentences_dataset):
    preprocessed_dataset = []
    for _, data in tqdm(enumerate(sentences_dataset), total=len(sentences_dataset), desc="Preprocessing sentences"):
        claim_tokens = preprocess_sentence(data['claim'])
        evidence_tokens = preprocess_sentence(data['evidence'])
        irrelevant_tokens = preprocess_sentence(data['irrelevant'])
        preprocessed_dataset.append({'claim': claim_tokens, 'evidence': evidence_tokens, 'irrelevant': irrelevant_tokens})
    return preprocessed_dataset

def preprocess_testing_examples(sentences_dataset):
    preprocessed_dataset = []
    for _, data in tqdm(enumerate(sentences_dataset), total=len(sentences_dataset), desc="Preprocessing sentences"):
        claim_tokens = preprocess_sentence(data['claim'])
        sentences_tokens = [(i, preprocess_sentence(sentence)) for i, sentence in data['sentences']]
        preprocessed_dataset.append({'claim': claim_tokens, "label": data["label"], "sentences": sentences_tokens})
    return preprocessed_dataset