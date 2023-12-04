# Local-LLMs

## Présentation
Local-LLMs est un projet répertiorant les informations nécessaires pour permettre l'exécution de modèles de langage à grande échelle (LLMs) en local avec [llama_cpp_python](https://github.com/abetlen/llama-cpp-python), utilisant les GPUs sous MacOS (avec puce Silicon) et Linux (avec support CUDA). Ce projet comprend deux notebooks Jupyter principaux :

1. **Local_LLM_Linux_CUDA.ipynb** - Pour les utilisateurs Linux avec support CUDA (fonctionne sur google colab).
2. **Local_LLM_MacOS_Silicon.ipynb** - Pour les utilisateurs MacOS avec puce Silicon.

## Fonctionnalités Clés
- **Utilisation du GPU pour une inférence rapide** : Profitez de la puissance de votre GPU pour accélérer l'inférence.
- **Compatibilité avec divers modèles** : La capacité à exécuter des modèles dépend de la quantité de RAM (sur Mac) ou de VRAM (sur systèmes Linux). 
- **Installation simplifiée** : Les notebooks sont auto-suffisants et gèrent l'installation des dépendances nécessaires.

## Modèles Quantifiés
Nous utilisons des modèles quantifiés (de 2 à 8 bits), offrant un compromis entre précision et efficacité en termes de ressources. Bien que légèrement moins précis que les modèles non quantifiés, ils demandent moins de ressources pour fonctionner.

### Sources des Modèles
Les modèles sont disponibles sur Huggingface, en particulier via le compte TheBloke qui propose près de 3000 modèles quantifiés : [Huggingface/TheBloke](https://huggingface.co/TheBloke).

### Bases de Modèles Disponibles
- **Llama 2** (Facebook) : 7, 13, 70 milliards de paramètres.
- **CodeLlama** (Facebook) : 7, 13, 34 milliards de paramètres.
- **Falcon** (Arabie Saoudite) : 7, 40, 180 milliards de paramètres.
- **Mistral** (France - startup Mistral) : 7 milliards de paramètres.

Ces modèles existent en plusieurs versions finetuned adaptées à divers usages (Instruct, Orca, OpenHermes, etc.).

### Quantification des Modèles
Nous nous concentrons sur les modèles quantifiés selon la méthode GGUF, qui est le format de quantification développé par [llama-cpp](https://github.com/ggerganov/llama.cpp), parmi d'autres méthodes telles que GPTQ et AWQ.

### Templates de Prompt
Le template de prompt pour chaque modèle est disponible dans la description du modèle sur [Huggingface/TheBloke](https://huggingface.co/TheBloke) ou dans la documentation du modèle non quantifié sur Hugging Face.

### Évaluation et Conseils d'Utilisation
TheBloke fournit une évaluation de chaque modèle quantifié, ainsi que des conseils d'utilisation basés sur les ressources disponibles (RAM/VRAM, taille du modèle, perte de qualité due à la quantification).

## Installation et Configuration
Il est conseillé de créer un environnement virtuel pour une gestion plus aisée des dépendances :

```bash
python -m venv venv
source venv/bin/activate
```

## Remarques Importantes
- Ce projet ne prend pas en compte les systèmes Windows : l'installation de [llama_cpp_python](https://github.com/abetlen/llama-cpp-python) diffère quelque peu (voir la documentation de la bibliothèque). 
- La performance et la compatibilité des modèles dépendent fortement de la configuration matérielle de l'utilisateur.
