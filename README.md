# 📩 Tri automatique de tickets support

Application de classification automatique de tickets IT/support client, développée dans le cadre du **projet final NLP** (Licence 3 Data Science, Big Data & AI — ISI, Dakar).

Le modèle analyse un message entrant et prédit automatiquement sa catégorie parmi 8 possibles, affiche le niveau de confiance de la prédiction, et suggère une réponse automatique adaptée.

## Démo

👉 [Tester l'application](https://tri-tickets-support-dozau9vqktsdrh5c4nup25.streamlit.app)

## Sommaire

- [Fonctionnement](#fonctionnement)
- [Dataset](#dataset)
- [Approche technique](#approche-technique)
- [Résultats](#résultats)
- [Limites connues](#limites-connues)
- [Utilisation en local](#utilisation-en-local)
- [Structure du projet](#structure-du-projet)

## Fonctionnement

1. L'utilisateur colle le contenu d'un ticket/message (en français ou en anglais)
2. Si le texte est en français, il est automatiquement traduit en anglais (langue d'entraînement du modèle)
3. Le modèle DistilBERT fine-tuné prédit la catégorie et son score de confiance
4. Une réponse automatique type est suggérée en français selon la catégorie détectée

## Dataset

**IT Service Ticket Classification Dataset** (Kaggle, [adisongoh](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset))

- 47 837 tickets IT/support
- 8 catégories : Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights
- Textes courts et pré-anonymisés (identifiants/pièces jointes remplacés par des tokens génériques)
- Fort déséquilibre de classes (Hardware : 13 617 exemples vs Administrative rights : 1 760)

## Approche technique

| Étape | Détail |
|---|---|
| Baseline | TF-IDF (1-2 grams) + LinearSVC, `class_weight="balanced"` |
| Modèle avancé | `distilbert-base-uncased` fine-tuné, 3 epochs |
| Tokenisation | max_length=128 |
| Métrique de sélection | F1 macro (pertinent vu le déséquilibre de classes) |
| Support multilingue | Traduction FR→EN (`deep-translator`) avant classification |

## Résultats

| Modèle | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| Baseline TF-IDF + LinearSVC | 86.0% | 0.860 | 0.860 |
| **DistilBERT fine-tuné** | **88.8%** | **0.887** | **0.888** |

Le modèle avancé améliore la classification sur presque toutes les catégories, avec un gain particulièrement net sur les classes minoritaires (Administrative rights : 0.78 → 0.82 F1).

## Limites connues

- **Confusion résiduelle Hardware ↔ HR Support/Miscellaneous** : ces catégories partagent un vocabulaire générique une fois les données anonymisées, ce qui entraîne des erreurs de classification sur des cas ambigus même pour DistilBERT.
- **Support multilingue par traduction, pas par modèle natif** : un premier essai avec `distilbert-base-multilingual-cased` (mBERT) a montré un collapse vers la classe majoritaire sur du texte français non vu à l'entraînement (confiance élevée mais fausse). La solution retenue (traduction FR→EN + modèle anglais) est plus fiable mais dépend d'un service de traduction externe.
- **Déséquilibre de classes** : malgré `class_weight="balanced"`, la classe la plus rare (Administrative rights, 1 760 exemples) reste la moins bien prédite.
- **Léger surapprentissage à partir de l'epoch 3** : la validation loss remonte légèrement alors que le F1 macro continue de s'améliorer — un early stopping plus strict serait une piste d'amélioration.

## Utilisation en local

```bash
git clone https://github.com/Ahma2re/tri-tickets-support.git
cd tri-tickets-support
pip install -r requirements.txt
streamlit run app.py
```

## Structure du projet
├── app.py # Application Streamlit
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier

Le modèle fine-tuné est hébergé sur le Hugging Face Hub : [Ahma2re/ticket-classifier-distilbert](https://huggingface.co/Ahma2re/ticket-classifier-distilbert)

Le notebook complet (exploration, baseline, fine-tuning, évaluation) est disponible dans ce même repo : `notebook.ipynb`.

---

*Projet réalisé par Ahmady Touré — L3 Data Science Big Data & AI, ISI Dakar*
