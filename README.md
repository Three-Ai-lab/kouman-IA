# Kouman IA (F-llama-3.2-1B-Baoulé)

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Tree--AI--lab-yellow?logo=huggingface)](https://huggingface.co/Tree-AI-lab/F-llama-3-2-1B-Baoule-1000_step)
[![Organization](https://img.shields.io/badge/Organization-Tree--AI--lab-blue)](https://github.com/Three-Ai-lab)

**Kouman IA** est le modèle de langage (LLM) basé sur **Meta Llama 3.2 1B**, fine-tuné spécifiquement pour la langue **Baoulé** (Côte d'Ivoire) par l'équipe **Tree AI Lab**.

Ce dépôt fait partie de l'écosystème **Kouman**, aux côtés des modèles de reconnaissance vocale (**ASR**) et de synthèse vocale (**TTS**).

---

## Architecture & Caractéristiques

- **Modèle de base :** Meta Llama 3.2 1B
- **Langue cible :** Baoulé (Côte d'Ivoire)
- **Étapes d'entraînement :** 1000 steps de fine-tuning continu (Beyond Chinchilla Omega Baoulé)
- **Poids du modèle :** Hébergés sur [Hugging Face Hub](https://huggingface.co/Tree-AI-lab/F-llama-3-2-1B-Baoule-1000_step)

---

## Installation & Utilisation rapide

```bash
pip install torch transformers accelerate
```

### Script d'inférence en Python (`inference.py`)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Tree-AI-lab/F-llama-3-2-1B-Baoule-1000_step"

print("Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Exemple d'amorce en Baoulé
prompt = "N'anhouan"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Contenu du dépôt

- `inference.py` : Script prêt à l'emploi pour exécuter des inférences.
- `config.json` & `generation_config.json` : Paramètres d'architecture et de génération du modèle.
- `tokenizer.json` & `tokenizer_config.json` : Tokenizer adapté au vocabulaire Baoulé.
- `Beyond_Chinchilla_Omega_Baoule.pdf` : Papier de recherche et documentation technique.
- Captures & logos officiels du projet.

---

## Citation & Contact

Développé par **Tree AI Lab** (Côte d'Ivoire).  
Pour toute question ou collaboration, visitez notre organisation [Three-Ai-lab sur GitHub](https://github.com/Three-Ai-lab).
