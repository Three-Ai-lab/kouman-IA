"""
Script d'inference pour le modele F-llama-3-2-1B-Baoule-1000_step
Developpe par Tree AI Lab (Cote d'Ivoire)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "."  # Utilise les fichiers locaux ou "Tree-AI-lab/F-llama-3-2-1B-Baoule-1000_step"

def main():
    print("Chargement du tokenizer et du modele Baoule...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution sur : {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    texte_entree = "N'anhouan"  # Exemple d'amorce en baoule
    inputs = tokenizer(texte_entree, return_tensors="pt").to(device)
    
    print("Generation en cours...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nResultat genere :")
    print("-" * 50)
    print(reponse)
    print("-" * 50)

if __name__ == "__main__":
    main()
