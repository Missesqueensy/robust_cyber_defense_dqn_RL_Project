# Étude d'Ablation $2^3$ sur DQN pour la Cyberdéfense Autonome (Yawning-Titan)

> **Base logicielle :** Ce projet étend l'environnement de simulation défensive **Yawning-Titan** (DSTL). La documentation d'origine de Yawning-Titan est conservée dans [`README_YAWNING_TITAN.md`](./README_YAWNING_TITAN.md).

Étude expérimentale menée dans le cadre du module **Apprentissage par Renforcement appliqué à la cybersécurité**. Ce dépôt regroupe nos implémentations des extensions de DQN (Double DQN, Dueling DQN, Prioritized Experience Replay), l'orchestrateur de la campagne factorielle $2^3$, les résultats bruts nettoyés sur 5 graines aléatoires et les figures d'analyse.

---

## 1. Question de recherche & Variations expérimentales

* **Question de recherche :** Parmi les trois améliorations classiques de DQN — **Double Q-Learning** (van Hasselt et al., 2016), **Dueling Network Architecture** (Wang et al., 2016) et **Prioritized Experience Replay — PER** (Schaul et al., 2016) — lesquelles améliorent réellement la défense autonome d'un réseau simulé face à une propagation d'attaque sous alertes bruitées, seules ou combinées ?
* **Variations personnelles du groupe (Double axe de généralisation Zero-Shot) :**
  1. **Généralisation structurelle inter-topologies ($\epsilon = 0$) :** les agents entraînés sur un réseau de référence sont évalués sans réentraînement sur quatre topologies distinctes (`star`, `chain`, `ring`, `bridge`) afin de mesurer leur robustesse face aux changements de connectivité du graphe.
  2. **Passage à l'échelle (*Scalability* — augmentation du nombre d'hôtes) :** évaluation de la capacité de généralisation des agents lorsque la taille du réseau augmente au-delà de la configuration d'entraînement ($N = 10$ hôtes), en testant l'extension vers des réseaux plus larges à **$N = 20$ et $N = 50$ hôtes** (espace d'états et d'actions élargi et propagation plus étendue).

---

## 2. Structure de nos ajouts dans le dépôt

* `README.md` : Documentation principale de l'étude
* `README_YAWNING_TITAN.md` : Documentation d'origine de Yawning-Titan
* `src/yawning_titan/my_game/` :
  * `defense_env.py` : Wrapper Gymnasium de cyberdéfense autour de Yawning-Titan
  * `make_env.py` : Générateur des topologies (`star`, `chain`, `ring`, `bridge`) et tailles $N$
  * `make_game_mode.py` & `game_modes.json` : Configuration du scénario (taux FP/FN, propagation)
  * `my_network.py` : Réseau Q standard (MLP)
  * `dueling_dqn.py` : Architecture Dueling DQN (Valeur $V(s)$ + Avantage $A(s,a)$)
  * `per_buffer.py` : Replay Buffer priorisé (SumTree + Importance Sampling)
  * `reward.py` : Fonction de récompense (compromis sécurité / disponibilité)
  * `baselines.py` : Politiques de référence (Aléatoire et Heuristique)
  * `run_ablation.py` : Orchestrateur de la campagne $2^3$ (8 configs $\times$ 5 seeds)
* `results/` :
  * `training_curves_clean.csv` : 24 000 épisodes (8 configs $\times$ 5 seeds $\times$ 600 épisodes)
  * `zero_shot_eval_clean.csv` : 160 évaluations Zero-Shot (8 configs $\times$ 5 seeds $\times$ 4 topologies)
* `figures/` :
  * `fig1_dqn_base_learning_curve.png` : Courbe d'apprentissage DQN de base (5 graines)
  * `fig2_ablation_2x3_learning_curves.png` : Comparaison des 8 configurations d'ablation $2^3$
  * `fig3_zero_shot_generalization.png` : Généralisation Zero-Shot par topologie (IC 95 %)

---

## 3. Protocole expérimental & Hyperparamètres

| Paramètre | Valeur | Description |
| :--- | :--- | :--- |
| **Plan factoriel** | $2^3 = 8$ configurations | `DQN_Base`, `Double`, `Dueling`, `PER`, `Double_Dueling`, `Double_PER`, `Dueling_PER`, `Full` |
| **Graines (`seeds`)** | `0, 1, 2, 3, 4` | 40 entraînements indépendants au total (5 graines par configuration) |
| **Budget d'entraînement** | 600 épisodes / run | Horizon maximal fixé avec gestion explicite troncature vs état terminal |
| **Exploration ($\epsilon$-greedy)** | $\epsilon : 1.0 \rightarrow 0.05$ | Décroissance linéaire sur l'entraînement ; $\epsilon = 0.0$ strict en évaluation |
| **Évaluation Zero-Shot** | 4 topologies | `star`, `chain`, `ring`, `bridge` (sans mise à jour des poids) |
| **Variation d'échelle ($N$)** | $N = 10 \rightarrow \{20, 50\}$ | Extension expérimentale de la taille du réseau (passage à 20 et 50 hôtes) |
| **PER (si activé)** | $\alpha = 0.6$, $\beta : 0.4 \rightarrow 1.0$ | Priorisation proportionnelle à $|\delta_t| + \epsilon$ avec poids d'importance |

---

## 4. Installation et Reproduction complète

```bash
# 1. Créer un environnement virtuel Python 3.10
python -m venv yt_venv
.\yt_venv\Scripts\Activate.ps1   # Sous Linux/macOS : source yt_venv/bin/activate

# 2. Installer gym==0.21.0 et Yawning-Titan en mode éditable
pip install "pip<24.1" "setuptools<58" "wheel<0.38"
pip install "gym==0.21.0" --no-build-isolation
pip install "setuptools>=65,<70" "wheel"
pip install -e . --no-build-isolation
pip install torch pandas matplotlib

# 3. Lancer les références et la campagne complète d'ablation 2^3
cd src/yawning_titan/my_game
python baselines.py
python run_ablation.py