import fireworks.client
import json
import llm
import load_process_data as lpd
import vectorize as vec
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from tqdm.auto import tqdm
import torch.nn.functional as F

fireworks.client.api_key = 'oczYTQbZX6VmiTM185LY2X2vmU1LmGdyBm6AdTg6YKiKYQg0'

dataset = lpd.load_fever_dataset()
# wikipages_index = lpd.load_fever_wikipages()

# Save wikipages_index
# with open('wikipages_index.json', 'w') as f:
#     json.dump(wikipages_index, f)

# load wikipages_index
# with open('wikipages_index.json', 'r') as f:
#     wikipages_index = json.load(f)

# X_train = dataset["train"]
# X_valid = dataset["labelled_dev"]
# X_test = dataset["paper_test"]
# sentence_dataset = lpd.extract_training_examples(wikipages_index, X_train, num_of_examples=10000)
# preprocessed_dataset = vec.preprocess_training_examples(sentence_dataset)

# Save preprocessed_dataset
# with open('preprocessed_dataset.json', 'w') as f:
#     json.dump(preprocessed_dataset, f)

# load preprocessed_dataset
with open('preprocessed_dataset.json', 'r') as f:
    preprocessed_dataset = json.load(f)

preprocessed_dataset = preprocessed_dataset[:25000]

# Initialize the BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# Process claims and sentences separately
# claims = [vec.tokens_to_sentence(item['claim']) for item in preprocessed_dataset]
# sentences = [vec.tokens_to_sentence(item['sentence']) for item in preprocessed_dataset]

# Get tensors for claims and sentences
# claim_tensors = vec.encode_texts_in_batches_whole(claims, tokenizer, model, batch_size=64, max_length=16)
# sentence_tensors = vec.encode_texts_in_batches_whole(sentences, tokenizer, model, batch_size=64, max_length=32)

# with open('claim_tensors_whole.pt', 'wb') as f:
#     torch.save(claim_tensors, f)
# with open('sentence_tensors_whole.pt', 'wb') as f:
#     torch.save(sentence_tensors, f)

with open('claim_tensors_whole.pt', 'rb') as f:
    claim_tensors = torch.load(f)
with open('sentence_tensors_whole.pt', 'rb') as f:
    sentence_tensors = torch.load(f)

vectorized_dataset = []
for i in range(len(preprocessed_dataset)):
    vectorized_dataset.append({
        'claim': claim_tensors[i],
        'sentence': sentence_tensors[i],
        'relevant': preprocessed_dataset[i]['relevant']
    })

class VectorizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['claim'], self.data[idx]['sentence'], torch.tensor(self.data[idx]['relevant'], dtype=torch.float)

dataset = VectorizedDataset(vectorized_dataset)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---------------------------------------------------
## Use pooled input for NN classifier
# class FeedForwardClassifier(nn.Module):
#     def __init__(self, input_size):
#         super(LogisticRegression, self).__init__()
#         self.linears = nn.Sequential(
#             nn.Linear(input_size * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
    
#     def forward(self, claim, sentence):
#         combined = torch.cat((claim, sentence), 1) # Concatenate claim and sentence vectors
#         return torch.sigmoid(self.linears(combined))
# model = FeedForwardClassifier(input_size=768)

# ---------------------------------------------------
## Use sequence input for RNN classifier
# class SimpleRNNClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super(SimpleRNNClassifier, self).__init__()
#         # Define the RNN layer for the first sequence
#         self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
#         # Define the RNN layer for the second sequence (can be the same as rnn1 if you prefer)
#         self.rnn2 = nn.RNN(input_size, hidden_size, batch_first=True)
        
#         # Define the fully connected layer for classification
#         self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because we concatenate the outputs
        
#         # Activation function
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq1, seq2):
#         # Assuming seq1 and seq2 are packed sequences or have been padded to the maximum length
#         # Process the first sequence
#         _, h_n1 = self.rnn1(seq1)
#         # Process the second sequence
#         _, h_n2 = self.rnn2(seq2)
        
#         # Concatenate the final hidden states from both sequences
#         combined = torch.cat((h_n1[-1], h_n2[-1]), dim=1)
        
#         # Pass through the fully connected layer
#         out = self.fc(combined)
        
#         # Apply sigmoid to get outputs between 0 and 1
#         out = self.sigmoid(out)
        
#         return out
# model = SimpleRNNClassifier(input_size=768, hidden_size=256)

# ---------------------------------------------------
## Use interactive RNN classifier
# class InteractiveRNNClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super(InteractiveRNNClassifier, self).__init__()
#         self.hidden_size = hidden_size
        
#         # RNN for the first sequence
#         self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        
#         # RNN for the second sequence. Note: input_size is doubled to allow concatenating the hidden state from rnn1
#         self.rnn2 = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)
        
#         # Fully connected layer for classification
#         self.fc = nn.Linear(hidden_size, output_size)
        
#         # Sigmoid activation for binary classification
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq1, seq2):
#         # Process the first sequence
#         _, h_n1 = self.rnn1(seq1)
        
#         # Prepare the second sequence input by concatenating the hidden state of seq1 with seq2 at each timestep
#         # This step assumes seq2 is padded or truncated to a specific length for simplicity
#         seq2_mod = torch.cat([seq2, h_n1[-1].repeat(seq2.size(1), 1, 1).transpose(0, 1)], dim=2)
        
#         # Process the modified second sequence, initializing its hidden state with the final state of the first RNN
#         _, h_n2 = self.rnn2(seq2_mod)
        
#         # Pass the final hidden state of the second sequence through the fully connected layer
#         out = self.fc(h_n2[-1])
        
#         # Apply sigmoid activation
#         out = self.sigmoid(out)
        
#         return out
# model = InteractiveRNNClassifier(input_size=768, hidden_size=128)

# ---------------------------------------------------
# class ConcatLSTMClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1):
#         super(ConcatLSTMClassifier, self).__init__()
#         # Define the RNN layer
#         self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
#         # Define the fully connected layer for classification
#         self.fc = nn.Linear(hidden_size, output_size)
        
#         # Define a sigmoid activation function for binary classification
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq1, seq2):
#         # Concatenate the sequences along the sequence length dimension
#         concatenated_seq = torch.cat((seq1, seq2), dim=1)  # Assuming seq1 and seq2 have dimensions (batch_size, seq_len, input_size)
        
#         # Process the concatenated sequence with the RNN
#         _, h_n = self.rnn(concatenated_seq)
        
#         # Use the final hidden state to pass through the fully connected layer
#         out = self.fc(h_n[-1])
        
#         # Apply sigmoid activation to get the output between 0 and 1
#         out = self.sigmoid(out)
        
#         return out
# model = ConcatLSTMClassifier(input_size=768, hidden_size=128)

# class ConcatCNNClassifier(nn.Module):
#     def __init__(self, input_channels, output_size=1):
#         super(ConcatCNNClassifier, self).__init__()
#         # Define the CNN layers
#         self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
#         # Calculate the size of the output after the conv and pool layers to properly define the fully connected layer
#         # Assuming a fixed length for the concatenated sequences
#         self.num_flatten_features = 64 * (48 // 2 // 2)  # Example calculation; adjust based on your sequence lengths and architecture
        
#         self.fc1 = nn.Linear(self.num_flatten_features, 512)
#         self.fc2 = nn.Linear(512, output_size)
        
#         # Define a sigmoid activation function for binary classification
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq1, seq2):
#         # Concatenate the sequences along the sequence length dimension and add a channel dimension
#         concatenated_seq = torch.cat((seq1, seq2), dim=1)  # Adjust dimensions as necessary based on your data
#         concatenated_seq = concatenated_seq.permute(0, 2, 1)  # Permute to have (batch_size, channels, seq_length)
        
#         # Apply the CNN layers
#         x = self.pool(F.relu(self.conv1(concatenated_seq)))
#         x = self.pool(F.relu(self.conv2(x)))
        
#         # Flatten the output for the fully connected layer
#         x = x.view(-1, self.num_flatten_features)
        
#         # Pass through the fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         # Apply sigmoid activation to get the output between 0 and 1
#         x = self.sigmoid(x)
        
#         return x
# model = ConcatCNNClassifier(input_channels=768)

class ConcatCNNClassifierNew(nn.Module):
    def __init__(self, input_channels, output_size=1):
        super(ConcatCNNClassifierNew, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
       
        self.num_flatten_features = 128 * (48 // 2 // 2)
       
        self.fc1 = nn.Linear(self.num_flatten_features, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)
       
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, seq1, seq2):
        concatenated_seq = torch.cat((seq1, seq2), dim=1)
        concatenated_seq = concatenated_seq.permute(0, 2, 1)
       
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(concatenated_seq))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
       
        x = x.view(-1, self.num_flatten_features)
       
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
       
        x = self.sigmoid(x)
       
        return x
model = ConcatCNNClassifierNew(input_channels=768)

# ## Use transformer classifier
# class SimpleTransformer(nn.Module):
#     def __init__(self, d_model=768, nhead=2, num_layers=2, dim_feedforward=2048, dropout=0.1):
#         super(SimpleTransformer, self).__init__()
        
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
#         self.fc = nn.Linear(d_model, 1)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, seq1, seq2):
#         input_seq = torch.cat((seq1, seq2), dim=1)
#         output = self.encoder(input_seq)
#         mean_output = torch.mean(output, dim=1)
#         logits = self.fc(mean_output)
#         probs = self.sigmoid(logits)
        
#         return probs
# model = SimpleTransformer(d_model=768, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# # Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# # Train the model
num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    for claims, sentences, labels in tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]'):
        # Forward pass
        claims, sentences, labels = claims.to(device), sentences.to(device), labels.to(device)
        outputs = model(claims, sentences)
        loss = criterion(outputs.squeeze(-1).squeeze(0), labels)
            
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    running_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')

# # Evaluate the model
with torch.no_grad():
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total = 0
    for claims, sentences, labels in val_loader:
        claims, sentences, labels = claims.to(device), sentences.to(device), labels.to(device)
        outputs = model(claims, sentences)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    print(f'Accuracy of the model on the validation set: {100 * correct / total:.2f}%')
    print(f"Precison: {true_positives / (true_positives + false_positives):.2f}")
    print(f"Recall: {true_positives / (true_positives + false_negatives):.2f}")
    print(f"F1 Score: {2 * (true_positives / (true_positives + false_positives)) * (true_positives / (true_positives + false_negatives)) / ((true_positives / (true_positives + false_positives)) + (true_positives / (true_positives + false_negatives))):.2f}")

# # Save the model
torch.save(model.state_dict(), 'model_rnn.ckpt')

# # Load the model
model = ConcatCNNClassifierNew(input_channels=768)
model.load_state_dict(torch.load('model_rnn.ckpt'))