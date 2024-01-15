import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

def similarity_check(path_to_teacher_folder):
    sim_rec = []
    dataset = load_dataset("stsb_multi_mt", name="en", split="train")

    teacher=path_to_teacher_folder

    tokenizer = AutoTokenizer.from_pretrained(teacher)
    model = AutoModel.from_pretrained(teacher)

    sentences1_list = dataset['sentence1']
    sentences2_list = dataset['sentence2']

    inputs1 = tokenizer(sentences1_list, padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sentences2_list, padding=True, truncation=True, return_tensors="pt")


    with torch.no_grad():
        s1_emb = model(**inputs1, output_hidden_states=True, return_dict=True).pooler_output
        s2_emb= model(**inputs2, output_hidden_states=True, return_dict=True).pooler_output
        #embeddings2 = model2(**inputs2, output_hidden_states=True, return_dict=True).pooler_output

    for i in range(len(dataset)):
        sim_rec.append(1 - cosine(s1_emb[i], s2_emb[i]))
    
    return sim_rec


teacher1="no_seed_teacher_ensemble/T1_expnum3-warmup_steps3550-distillation_stopping_steps8000"
teacher2="no_seed_teacher_ensemble/T2_expnum3-warmup_steps3550-distillation_stopping_steps8000"

sim_rec1 = similarity_check(teacher1)
sim_rec2 = similarity_check(teacher2)

# Calculate Spearman correlation coefficient
spearman_corr, _ = spearmanr(sim_rec1, sim_rec2)

print(f"Spearman correlation coefficient: {spearman_corr}")
