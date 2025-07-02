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
# with open('wikipages_index.json', 'r') as f:
#     wikipages_index = json.load(f)

# X_train = dataset["train"]
# X_valid = dataset["labelled_dev"]
# X_test = dataset["paper_test"]
# sentence_dataset = lpd.extract_testing_examples(wikipages_index, X_valid, num_of_examples=1000)
# preprocessed_dataset = vec.preprocess_testing_examples(sentence_dataset)

# Save preprocessed_dataset
# with open('preprocessed_dataset_test.json', 'w') as f:
#     json.dump(preprocessed_dataset, f)

# load preprocessed_dataset
with open('preprocessed_dataset_test.json', 'r') as f:
    preprocessed_dataset = json.load(f)
preprocessed_dataset = preprocessed_dataset[:500]

# Initialize the BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Process claims, evidences and irrelevant sentences separatelyt
claims = [item['claim'] for item in preprocessed_dataset]
sentences = [[(i, s) for i, s in item['sentences']] for item in preprocessed_dataset]

claim_tensors = vec.encode_texts_in_batches_whole(claims, tokenizer, model, batch_size=64, max_length=16)
sentences_tensors = vec.encode_sentences_whole(sentences, tokenizer, model, max_length=32)

# Save tensors
with open('claim_tensors_test.pt', 'wb') as f:
    torch.save(claim_tensors, f)
with open('sentences_tensors_test.pt', 'wb') as f:
    torch.save(sentences_tensors, f)

# Load tensors
with open('claim_tensors_test.pt', 'rb') as f:
    claim_tensors = torch.load(f)
with open('sentences_tensors_test.pt', 'rb') as f:
    sentences_tensors = torch.load(f)

vectorized_dataset = []
for i in range(len(preprocessed_dataset)):
    vectorized_dataset.append({
        'claim': claim_tensors[i],
        'sentences': sentences_tensors[i],
        'label': preprocessed_dataset[i]['label']
    })

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
model.load_state_dict(torch.load('model_cnn_triplet.ckpt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    model.eval()
    mean_rank = 0
    ndcg = 0
    total = 0
    for data in tqdm(vectorized_dataset, desc="Evaluating"):
        claim = data['claim']
        sentences_list = data['sentences']
        label = data['label']
        claim = claim.to(device).unsqueeze(0)
        sentences = [sentence.to(device).unsqueeze(0) for _, sentence in sentences_list]
        ids = [i for i, _ in sentences_list]
        claim_embedding = model(claim)
        sentences_embedding = [model(sentence) for sentence in sentences]
        
        similarity = [F.cosine_similarity(claim_embedding, sentence_embedding, dim=1) for sentence_embedding in sentences_embedding]
        similarity_with_id = list(zip(similarity, ids))
        similarity_with_id.sort(reverse=True)
        for pos, (_, i) in enumerate(similarity_with_id):
            if i == label:
                mean_rank += pos + 1
                ndcg += 1 / torch.log2(torch.tensor(pos + 2))
                break
        total += 1


    print(f"Mean rank: {mean_rank/total}")
    print(f"NDCG: {ndcg/total}")