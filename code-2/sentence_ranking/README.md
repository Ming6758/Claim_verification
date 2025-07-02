# Sentence ranker model

In this folder, there are code for finetuning the model all-miniLM-L6-V2 for the purpose of evidence retrieval. It loads data from the FEVER dataset and uses triplet loss to finetune the model.

You can run `sentence_similarity_finetune.py` to run finetuning yourself, or you can run `sentence_similarity_experiment.py` to load the finetuned model and test it on the test set of FEVER.