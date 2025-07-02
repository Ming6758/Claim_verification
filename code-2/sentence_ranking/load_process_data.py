from datasets import load_dataset
from tqdm.auto import tqdm
import random
import gc

# Load the FEVER dataset
def load_fever_dataset():
    dataset = load_dataset("fever", "v1.0", trust_remote_code=True)
    return dataset

# Load the FEVER wikipages and build an index
def load_fever_wikipages():
    wiki_dump = load_dataset("fever", "wiki_pages", trust_remote_code=True)
    wikipages = wiki_dump["wikipedia_pages"]

    wikipages_index = {}
    for _, example in tqdm(enumerate(wikipages), total=len(wikipages), desc="Building index"):
        wikipages_index[example['id']] = example

    del wiki_dump, wikipages
    gc.collect()

    return wikipages_index

# Preprocess: generate training examples with relevancy scores as labels
def extract_training_examples(wikipages_index, train_dataset, num_of_examples=10000):
  returned_dataset = []
  for i, data in tqdm(enumerate(train_dataset), desc="Extracting training examples", total=len(train_dataset)):
    if len(returned_dataset) > num_of_examples:
      break
    evidence_sentence_id = data["evidence_sentence_id"]
    if evidence_sentence_id != -1:
      claim = data["claim"]
      try:
        page_content = wikipages_index[data['evidence_wiki_url']]["lines"]
      except:
        continue
      lines = page_content.split('\n')
      sentences = [line.split('\t', maxsplit=1) for line in lines]
      sentences = {int(l[0]): l[1] for l in sentences}
      sentence_id = train_dataset[i]['evidence_sentence_id']
      if sentence_id < len(sentences):
        evidence_sentence = sentences[sentence_id]
      else:
        continue

      irrelevant_sentence = sentences[sentence_id-1] if sentence_id > 0 else sentences[1]

      returned_dataset.append({"claim": claim, "sentence": evidence_sentence, "relevant": 1})
      if irrelevant_sentence != "":
        returned_dataset.append({"claim": claim, "sentence": irrelevant_sentence, "relevant": 0})

  return returned_dataset

# Preprocess: generate training examples with claim, evidence and irrelevant sentences
def extract_training_examples_triplet(wikipages_index, train_dataset, num_of_examples=10000):
  returned_dataset = []
  for i, data in tqdm(enumerate(train_dataset), desc="Extracting training examples", total=len(train_dataset)):
    if len(returned_dataset) > num_of_examples:
      break
    evidence_sentence_id = data["evidence_sentence_id"]
    if evidence_sentence_id != -1:
      claim = data["claim"]
      try:
        page_content = wikipages_index[data['evidence_wiki_url']]["lines"]
      except:
        continue
      lines = page_content.split('\n')
      sentences = [line.split('\t', maxsplit=1) for line in lines]
      sentences = {int(l[0]): l[1] for l in sentences}
      sentence_id = train_dataset[i]['evidence_sentence_id']
      if sentence_id < len(sentences):
        evidence_sentence = sentences[sentence_id]
      else:
        continue

      irrelevant_sentences = [sentences[i] for i in range(len(sentences)) if i != sentence_id]
      irrelevant_sentences = [irrelevant_sentences[i] for i in range(len(irrelevant_sentences)) if irrelevant_sentences[i] != ""]

      # Randomly select an irrelevant sentence
      irrelevant_sentence_1 = random.choice(irrelevant_sentences)
      returned_dataset.append({"claim": claim, "evidence": evidence_sentence, "irrelevant": irrelevant_sentence_1})

      if len(irrelevant_sentences) > 1:
        irrelevant_sentence_2 = random.choice([irrelevant_sentences[i] for i in range(len(irrelevant_sentences)) if irrelevant_sentences[i] != irrelevant_sentence_1])
        returned_dataset.append({"claim": claim, "evidence": evidence_sentence, "irrelevant": irrelevant_sentence_2})

      if len(irrelevant_sentences) > 2:
        irrelevant_sentence_3 = random.choice([irrelevant_sentences[i] for i in range(len(irrelevant_sentences)) if irrelevant_sentences[i] != irrelevant_sentence_1 and irrelevant_sentences[i] != irrelevant_sentence_2])
        returned_dataset.append({"claim": claim, "evidence": evidence_sentence, "irrelevant": irrelevant_sentence_3})

  return returned_dataset

# Preprocess: generate testing examples with claim, evidence and irrelevant sentences
def extract_testing_examples(wikipages_index, test_dataset, num_of_examples=100):
  returned_dataset = []
  for _, data in tqdm(enumerate(test_dataset), desc="Extracting testing examples", total=len(test_dataset)):
    if len(returned_dataset) > num_of_examples:
      break
    if data["evidence_sentence_id"] != -1:
      try:
        page_content = wikipages_index[data['evidence_wiki_url']]["lines"]
      except:
        continue
      lines = page_content.split('\n')
      lines = [line[1:].replace('\t', '') for line in lines]
      lines = [(i, line) for i, line in enumerate(lines) if line != ""]
      returned_dataset.append({"claim": data["claim"], "label": data["evidence_sentence_id"], "sentences": lines})

  return returned_dataset