from transformers import AutoModel, AutoTokenizer
import json
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

num_layers_to_unfreeze = 6

# Load the pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Adjust how many layers to unfreeze
for param in model.parameters():
    param.requires_grad = False
num_layers = len(model.encoder.layer)
print(f"Number of layers: {num_layers}")
print(f"Unfreezing the last {num_layers_to_unfreeze} layers")

# Unfreeze the last num_layers_to_unfreeze layers
for i in range(num_layers - num_layers_to_unfreeze, num_layers):
    for param in model.encoder.layer[i].parameters():
        param.requires_grad = True

# Load the fine-tuned model weights
with open('preprocessed_dataset.json', 'r') as f:
    preprocessed_dataset = json.load(f)
preprocessed_dataset = preprocessed_dataset[:50000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

# Pre-tokenize the dataset
def pretokenize_dataset():
    # Pretokenize the dataset
    tokenized_claims = tokenizer([item['claim'] for item in preprocessed_dataset], padding=True, truncation=True, return_tensors="pt")
    tokenized_evidences = tokenizer([item['evidence'] for item in preprocessed_dataset], padding=True, truncation=True, return_tensors="pt")
    tokenized_irrelevants = tokenizer([item['irrelevant'] for item in preprocessed_dataset], padding=True, truncation=True, return_tensors="pt")

    # Move the tokenized dataset to the device
    tokenized_claims = {key: value.to(device) for key, value in tokenized_claims.items()}
    tokenized_evidences = {key: value.to(device) for key, value in tokenized_evidences.items()}
    tokenized_irrelevants = {key: value.to(device) for key, value in tokenized_irrelevants.items()}

    # Turn mapping of list to list of mappings
    tokenized_claims = [{key: value[i] for key, value in tokenized_claims.items()} for i in range(len(tokenized_claims['input_ids']))]
    tokenized_evidences = [{key: value[i] for key, value in tokenized_evidences.items()} for i in range(len(tokenized_evidences['input_ids']))]
    tokenized_irrelevants = [{key: value[i] for key, value in tokenized_irrelevants.items()} for i in range(len(tokenized_irrelevants['input_ids']))]

    tokenized_dataset = []
    for i in range(len(preprocessed_dataset)):
        tokenized_dataset.append({
            'claim': tokenized_claims[i],
            'evidence': tokenized_evidences[i],
            'irrelevant': tokenized_irrelevants[i]
        })

    return tokenized_dataset

tokenized_dataset = pretokenize_dataset()

# Define the dataset and the dataloader
class TokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['claim'], self.data[idx]['evidence'], self.data[idx]['irrelevant']

dataset = TokenizedDataset(tokenized_dataset)

# Define the training loop
triplet_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(5):  # Number of epochs
    total_loss = 0
    for batch in tqdm(triplet_loader, desc=f"Epoch {epoch+1}", unit="batch", total=len(triplet_loader)):
        anchors, positives, negatives = batch
        
        # Forward pass
        anchor_embeddings = model(**anchors).pooler_output
        positive_embeddings = model(**positives).pooler_output
        negative_embeddings = model(**negatives).pooler_output
        
        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        total_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(triplet_loader):.2f}")

# Save the model
torch.save(model.state_dict(), "minilm-finetune.ckpt")