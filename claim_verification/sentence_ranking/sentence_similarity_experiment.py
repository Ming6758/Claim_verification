import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
import load_process_data as lpd
import vectorize as vec
from tqdm.auto import tqdm

# Load the pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the fine-tuned model weights
model.load_state_dict(torch.load('minilm-finetune.ckpt'))

# Set the model to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the dataset and the wikipages index
dataset = lpd.load_fever_dataset()
with open('wikipages_index.json', 'r') as f:
    wikipages_index = json.load(f)

# Extract and preprocess the testing examples
X_valid = dataset["labelled_dev"]
sentence_dataset = lpd.extract_testing_examples(wikipages_index, X_valid, num_of_examples=1000)
preprocessed_dataset = vec.preprocess_testing_examples(sentence_dataset)

# Save preprocessed_dataset
with open('preprocessed_dataset_test.json', 'w') as f:
    json.dump(preprocessed_dataset, f)

# load preprocessed_dataset
with open('preprocessed_dataset_test.json', 'r') as f:
    preprocessed_dataset = json.load(f)
preprocessed_dataset = preprocessed_dataset[:1500]

# Calculate the mean rank and NDCG for the testing examples
triplets = []
for i in range(len(preprocessed_dataset)):
    claim = preprocessed_dataset[i]['claim']
    sentence = [(id, s) for id, s in preprocessed_dataset[i]['sentences']]
    label = preprocessed_dataset[i]['label']
    triplets.append({'claim': claim, 'sentences': sentence, 'label': label})

mean_rank = 0
ndcg = 0
for i in tqdm(range(len(triplets)), desc="Calculating mean rank"):
    claim = triplets[i]['claim']
    sentences = triplets[i]['sentences']
    label = triplets[i]['label']

    # Tokenize the claim and the sentences
    claim_tokenized = tokenizer(claim, padding=True, truncation=True, return_tensors="pt")
    claim_tokenized = {key: value.to(device) for key, value in claim_tokenized.items()}
    sentences_tokenized = tokenizer([s for _, s in sentences], padding=True, truncation=True, return_tensors="pt")
    
    # Turn the dict sentences_tokenized into a list of dicts
    sentences_tokenized = [
    {key: value[i].unsqueeze(0).to(device) for key, value in sentences_tokenized.items()}
    for i in range(len(sentences_tokenized['input_ids']))]

    # Calculate the similarity between the claim and the sentences
    claim_embedding = model(**claim_tokenized).pooler_output
    sentence_embeddings = [model(**inputs).pooler_output for inputs in sentences_tokenized]
    sentence_embeddings_with_id = [(id, sentence) for id, sentence in zip([id for id, _ in sentences], sentence_embeddings)]
    similarities = []
    for id, sentence_embedding in sentence_embeddings_with_id:
        similarity = F.cosine_similarity(claim_embedding, sentence_embedding)
        similarities.append((id, similarity.item()))
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    num_candidates = len(similarities)
    
    for i, (id, similarity) in enumerate(similarities):
        if id == label:
            ndcg += 1 / torch.log2(torch.tensor(i + 2))
            mean_rank += i + 1
            break
        
mean_rank /= len(triplets)
ndcg /= len(triplets)
print("Mean rank:", mean_rank)
print("NDCG:", ndcg.item())
    