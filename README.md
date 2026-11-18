# OCR et compréhension de documents : Tesseract + LayoutLMv3 sur FUNSD

Projet de **Vision par ordinateur avancée** (3e année cycle ingénieur en informatique, GL-AL, groupe 01, ISSAT Sousse, 2026–2027).

**Auteures :** Amira ABIDI · Sarra BEN REJEB

À partir de l'image scannée d'un formulaire, le système :

1. lit le texte avec un OCR (mots + boîtes englobantes) ;
2. classe chaque mot avec **LayoutLMv3** en `HEADER` (titre), `QUESTION`, `ANSWER` (réponse) ou `O` (autre) ;
3. regroupe les mots en entités et relie chaque question à sa réponse ;
4. produit un **JSON structuré**.

```
image → prétraitement → OCR (mots + boîtes) → LayoutLMv3 (étiquetage) → entités → liaison question→réponse → JSON
```


## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Projet_Vision_OCR_LayoutLMv3.ipynb` | Notebook complet : entraînement, évaluation, ablation, comparaison d'OCR, pipeline bout-en-bout, démo Gradio |
| `results.json` | Résultats chiffrés exportés par le notebook |
| `requirements_versions.txt` | Versions exactes des bibliothèques utilisées |
| `Presentation_Projet_Vision_OCR_Abidi_BenRejeb.pptx` | Présentation de la soutenance |

## Données

- **Dataset :** FUNSD (*Form Understanding in Noisy Scanned Documents*), version Hugging Face [`nielsr/funsd-layoutlmv3`](https://huggingface.co/datasets/nielsr/funsd-layoutlmv3).
- 199 formulaires annotés (149 entraînement, 50 test), 7 étiquettes BIO (`O`, `B/I-HEADER`, `B/I-QUESTION`, `B/I-ANSWER`), environ 146 mots par document.
- Découpage utilisé : 126 entraînement / 23 validation (15 %) / 50 test officiel.
- Licence : jeu de données destiné à la recherche et à l'enseignement, usage non commercial. À vérifier sur la page du dataset.

## Modèle

- Base : [`microsoft/layoutlmv3-base`](https://huggingface.co/microsoft/layoutlmv3-base), environ 125 M de paramètres, 12 blocs Transformer.
- Entrées : `input_ids` (B, 512), `bbox` (B, 512, 4, valeurs 0–1000), `pixel_values` (B, 3, 224, 224).
- Séquence interne : (B, 709, 768) = 512 tokens de texte + 197 tokens d'image. Logits : (B, 512, 7).
- Perte : entropie croisée par token (sous-tokens à −100 ignorés).
- Entraînement : boucle PyTorch écrite à la main, AdamW (lr 5e-5, weight decay 0,01), warmup 10 %, fp16, batch 4, 15 epochs, meilleure epoch choisie sur le F1 de validation, graine 42.

## Résultats (jeu de test FUNSD, 50 documents)

| Expérience | Résultat |
|---|---|
| F1 entités, mots annotés | **0,892** (précision 0,884, rappel 0,900) |
| F1 par classe | ANSWER 0,914 · QUESTION 0,906 · HEADER 0,616 |
| Ablation : sans layout (boîtes à zéro) | 0,702 (le layout apporte +19 points) |
| OCR, F1 sur les mots | Tesseract psm3 0,621 · psm11 0,659 · **psm11 + prétraitement 0,696** · EasyOCR 0,520 |
| Bout-en-bout, F1 entités | mots annotés 0,889 → **OCR réel 0,268** |

Limite principale : les erreurs d'OCR se propagent au modèle de structure. Autres limites : peu de données d'entraînement (surapprentissage), classe `HEADER` faible, formulaires en anglais uniquement, 512 tokens maximum, liaison question → réponse heuristique.

## Contributions par rapport à l'implémentation d'origine

- Comparaison d'OCR (Tesseract psm3 / psm11, EasyOCR) et prétraitement (agrandissement ×2 + binarisation d'Otsu).
- Ablation avec / sans layout.
- Pipeline bout-en-bout avec un vrai OCR.
- Liaison question → réponse (heuristique géométrique : réponse à droite ou en dessous, la plus proche).
- Évaluation au niveau des entités (similarité `difflib` ≥ 80 %).
- Démo interactive Gradio.

## Installation et exécution

Le plus simple est d'ouvrir le notebook dans **Google Colab** avec un GPU (Exécution > Modifier le type d'exécution > GPU T4).

```bash
apt-get install -y tesseract-ocr
pip install "datasets==3.6.0" transformers seqeval pytesseract easyocr gradio gcsfs --upgrade
```

1. Exécuter les cellules dans l'ordre. Pour un test rapide, mettre `EPOCHS = 2`.
2. La dernière cellule de démo lance l'interface Gradio (lien public valable environ une semaine).
3. Pour publier le modèle sur Hugging Face, décommenter les lignes `push_to_hub` de la section « Sauvegarde, publication et reproductibilité ».

## Environnement testé

Python 3.13.16 · PyTorch 2.14.1+cu130 · Transformers 5.23.0 · Datasets 3.6.0 · seqeval 1.2.2 · pytesseract 0.3.13 (Tesseract 5.3.4) · EasyOCR 1.7.2 · Gradio 6.32.0 · GPU NVIDIA Tesla T4 (Colab). Liste complète : `requirements_versions.txt`.

## Références

- Y. Huang et al., *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking*, ACM Multimedia 2022, [arXiv:2204.08387](https://arxiv.org/abs/2204.08387).
- G. Jaume, H. K. Ekenel, J.-P. Thiran, *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents*, 2019, [arXiv:1905.13538](https://arxiv.org/abs/1905.13538).
- Documentation Hugging Face Transformers, Tesseract OCR, EasyOCR.

