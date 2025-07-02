import json
import load_process_data as lpd
import vectorize as vec
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from tqdm.auto import tqdm
import torch.nn.functional as F

dataset = lpd.load_fever_dataset()
# wikipages_index = lpd.load_fever_wikipages()

# Save wikipages_index
# with open('wikipages_index.json', 'w') as f:
#     json.dump(wikipages_index, f)

# load wikipages_index
with open('wikipages_index.json', 'r') as f:
    wikipages_index = json.load(f)

X_train = dataset["train"]
sentence_dataset = lpd.extract_training_examples_triplet(wikipages_index, X_train, num_of_examples=50000)
preprocessed_dataset = vec.preprocess_training_examples_triplet(sentence_dataset)

# Save preprocessed_dataset
with open('preprocessed_dataset.json', 'w') as f:
    json.dump(preprocessed_dataset, f)

# load preprocessed_dataset
with open('preprocessed_dataset.json', 'r') as f:
    preprocessed_dataset = json.load(f)

preprocessed_dataset = preprocessed_dataset[:50000]

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Process claims, evidences and irrelevant sentences separatelyt
claims = [item['claim'] for item in preprocessed_dataset]
evidences = [item['evidence'] for item in preprocessed_dataset]
irrelevant = [item['irrelevant'] for item in preprocessed_dataset]

# Get tensors for claims, evidences and irrelevant sentences
claim_tensors = vec.encode_texts_in_batches_whole(claims, tokenizer, model, batch_size=64, max_length=16)
with open('claim_tensors_whole.pt', 'wb') as f:
    torch.save(claim_tensors, f)
evidence_tensors = vec.encode_texts_in_batches_whole(evidences, tokenizer, model, batch_size=64, max_length=32)
with open('evidence_tensors_whole.pt', 'wb') as f:
    torch.save(evidence_tensors, f)
irrelevant_tensors = vec.encode_texts_in_batches_whole(irrelevant, tokenizer, model, batch_size=64, max_length=32)
with open('irrelevant_tensors_whole.pt', 'wb') as f:
    torch.save(irrelevant_tensors, f)

with open('claim_tensors_whole.pt', 'rb') as f:
    claim_tensors = torch.load(f)
with open('evidence_tensors_whole.pt', 'rb') as f:
    evidence_tensors = torch.load(f)
with open('irrelevant_tensors_whole.pt', 'rb') as f:
    irrelevant_tensors = torch.load(f)

vectorized_dataset = []
for i in range(len(preprocessed_dataset)):
    vectorized_dataset.append({
        'claim': claim_tensors[i],
        'evidence': evidence_tensors[i],
        'irrelevant': irrelevant_tensors[i],
    })

class VectorizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['claim'], self.data[idx]['evidence'], self.data[idx]['irrelevant']

dataset = VectorizedDataset(vectorized_dataset)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveTokenEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=312, num_classes=16):
        super(AdaptiveTokenEmbeddingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, embedding_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 1)) 
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 1)) 
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = AdaptiveTokenEmbeddingCNN()
model.to(device)

# Loss and optimizer
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for claims, evidences, irrelevants in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]'):
        # Forward pass
        claims, evidences, irrelevants = claims.to(device), evidences.to(device), irrelevants.to(device)
        claims_embedding = model(claims)
        evidences_embedding = model(evidences)
        irrelevants_embedding = model(irrelevants)

        loss = criterion(claims_embedding, evidences_embedding, irrelevants_embedding)
            
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    running_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for claims, evidences, irrelevants in val_loader:
        claims, evidences, irrelevants = claims.to(device), evidences.to(device), irrelevants.to(device)
        claims_embedding = model(claims)
        evidences_embedding = model(evidences)
        irrelevants_embedding = model(irrelevants)

        similarity_evidence = F.cosine_similarity(claims_embedding, evidences_embedding)
        similarity_irrelevant = F.cosine_similarity(claims_embedding, irrelevants_embedding)

        c = similarity_evidence > similarity_irrelevant
        correct += c.sum().item()
        total += c.size(0)

    print(f'Accuracy of the model on the validation set: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model_cnn_triplet.ckpt')

# Load the model
model = AdaptiveTokenEmbeddingCNN()
model.load_state_dict(torch.load('model_cnn_triplet.ckpt'))