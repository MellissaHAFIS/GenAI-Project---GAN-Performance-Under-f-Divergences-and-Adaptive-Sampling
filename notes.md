# Project Organisation and File Roles

## Project Organisation (High Level)
- Root contains the training, generation, model and utility code plus metadata:
  - `train.py` — training entrypoint
  - `generate.py` — sample generation entrypoint
  - `model.py` — network definitions
  - `utils.py` — training & I/O helper functions
  - `requirements.txt` — pinned dependencies
  - `README.md` — usage / cluster instructions
  - `checkpoints/` — saved model weights (e.g. `G.pth`)
  - `scripts/` — shell wrappers for running on the cluster (`train.sh`, `generate.sh`)
- Typical workflow:
  1. Create virtualenv and install `requirements.txt`.
  2. Run `train.py` (or `scripts/train.sh`) to train and produce checkpoints.
  3. Run `generate.py` (or `scripts/generate.sh`) to load `checkpoints/G.pth` and produce images in `samples/`.

## File-by-File Explanation and How Each File Works

### `model.py`
- Purpose: defines the neural network architectures used by the GAN.
- Contents:
  - `Generator(nn.Module)`:
    - Input: 100-d latent vector.
    - Fully connected layers: 100 -> 256 -> 512 -> 1024 -> g_output_dim (default used is 784).
    - Activations: LeakyReLU between hidden layers, final `tanh` to output in [-1, 1].
    - Output: flattened MNIST image of size `g_output_dim` (28*28 = 784).
  - `Discriminator(nn.Module)`:
    - Input: flattened image (`d_input_dim`, e.g. 784).
    - Fully connected layers: d_input_dim -> 1024 -> 512 -> 256 -> 1.
    - Activations: LeakyReLU for hidden layers, final `sigmoid` to produce probability.
- Role: encapsulates forward computation for generator and discriminator used during training and inference.

### `train.py`
- Purpose: orchestrates dataset loading, device selection, training loop, and checkpoint saving.
- Key behaviour:
  - Device selection: prefers CUDA, then Apple MPS, otherwise CPU.
  - Data pipeline: MNIST dataset with `ToTensor()` and `Normalize(mean=0.5, std=0.5)`.
  - DataLoader: uses `num_workers=4` and `pin_memory=True` for performance.
  - Model instantiation: constructs `Generator` and `Discriminator`, moves to device; wraps in `DataParallel` if multiple GPUs requested.
  - Loss and optimizers: `BCELoss` and Adam optimizers for G and D.
  - Training loop: for each batch
    - Flattens images to 784 vector and calls helper functions `D_train` and `G_train` (from `utils.py`) to perform one step for D and G.
    - Checkpoints saved every 10 epochs via `save_models`.
  - CLI: accepts `--epochs`, `--lr`, `--batch_size`, `--gpus`.

### `utils.py`
- Purpose: helper functions used by `train.py` and `generate.py`.
- Functions:
  - `D_train(x, G, D, D_optimizer, criterion, device)`:
    - Trains discriminator for one mini-batch:
      - Computes loss on real images (`y_real=1`) and fake images produced by `G(z)` (`y_fake=0`).
      - Backpropagates summed D loss and steps `D_optimizer`.
      - Returns scalar D loss.
  - `G_train(x, G, D, G_optimizer, criterion, device)`:
    - Trains generator for one mini-batch:
      - Samples z, generates fake images, gets D(G(z)), uses target `y=1` to train G to fool D.
      - Backpropagates G loss and steps `G_optimizer`.
      - Returns scalar G loss.
  - `save_models(G, D, folder)`:
    - Saves `G.state_dict()` and `D.state_dict()` into `folder` as `G.pth` and `D.pth`.
  - `load_model(G, folder, device)`:
    - Loads `G.pth` into the passed `Generator` instance.
    - Strips `'module.'` prefixes from keys (DataParallel compatibility) and loads on `map_location=device`.
- Role: isolates training step logic and model I/O.

### `generate.py`
- Purpose: load trained generator and produce image samples.
- Key behaviour:
  - Device selection like `train.py`.
  - Instantiates `Generator(g_output_dim=784)` and calls `utils.load_model` to load `checkpoints/G.pth`.
  - Optionally wraps model in `DataParallel` if multiple CUDA devices are present.
  - In `torch.no_grad()` loop:
    - Samples z ~ N(0,1) with shape `(batch_size, 100)`.
    - Calls `model(z)` -> output shaped `(batch_size, 784)`, reshapes to `(batch_size, 28, 28)`.
    - Saves each sample to `samples/{n}.png` using `torchvision.utils.save_image`, until 10000 images are generated.
  - CLI: accepts `--batch_size` (default 2048).

### `requirements.txt`
- Purpose: pinned Python package versions used by the project (numpy, scikit-learn, scipy, tqdm, torch, torchvision).
- Role: used to reproducibly create the Python environment.

### `README.md`
- Purpose: usage and deployment instructions, especially for the MesoNet/Juliet cluster.
- Role: documents how to set up virtual environment, install dependencies, run `scripts/train.sh` / `scripts/generate.sh` and notes about supported hardware (CUDA/MPS/CPU).

### `checkpoints/G.pth` (and `checkpoints/D.pth` if present)
- Purpose: binary saved state_dict for the trained generator (and discriminator).
- Role: used by `generate.py` to produce images without retraining.

## Notes about Integration and Dataflow (Compact)
- Training flow: `train.py` -> DataLoader yields real x -> `D_train` uses x and G(z) to update D -> `G_train` uses D(G(z)) to update G -> periodic `save_models`.
- Generation flow: `generate.py` -> loads `G.pth` via `load_model` -> sample z and produce images saved to `samples/`.

If you want, I can:
- produce a one-page diagram summarising the dataflow and file responsibilities, or
- check for small issues (e.g., recommended default batch sizes, where `DataParallel` is applied) and propose minimal fixes.

---

# Plan de realisation du projet ForgeGAN
### **Objectif Global**

Comparer 5 méthodes différentes de génération sur le dataset **MNIST** en utilisant une architecture de générateur **fixe et identique** pour toutes. L'évaluation se fera sur la qualité (FID, Precision) et la diversité (Recall).

---

### **Phase 1 : Mise en Place et Baseline (Fondations)**

Avant de coder les méthodes complexes, assurez-vous que l'environnement de base fonctionne.

1. **Préparation des Données & Environnement :**
* Charger le dataset **MNIST**.
* Configurer l'environnement (PyTorch/TensorFlow).
* **Crucial :** Isoler le code de l'architecture du Générateur () fourni dans le dépôt. Vous ne devez **jamais** modifier ses couches (layers), seulement ses poids via l'entraînement.


2. **Implémentation des Métriques (Le Juge) :**
* C'est la priorité absolue. Sans métriques fiables, vous ne pouvez rien comparer.
* **FID (Fréchet Inception Distance) :** Mesure la distance entre la distribution réelle et générée. (Plus bas est mieux). *Note : Sur MNIST, assurez-vous d'utiliser un extracteur de caractéristiques adapté (souvent un petit CNN pré-entraîné sur MNIST plutôt qu'InceptionV3 standard qui est pour ImageNet).*
* **Precision & Recall (P&R) :**
* *Precision :* Qualité des images (est-ce que mes images ressemblent à des vrais chiffres ?).
* *Recall :* Diversité (est-ce que je génère *tous* les types de chiffres ou seulement des "1" ?).

3. **Baseline 1 : Original GAN :**
* Lancer l'entraînement du code fourni dans le dépôt sans modification.
* Calculer FID, Precision, Recall. Ce seront vos **scores de référence**.

---

### **Phase 2 : Méthodes de Troncature (Truncation)**

Ces méthodes modifient la façon dont on *échantillonne* le bruit  (vecteur latent) sans nécessairement changer l'entraînement.

4. **Baseline 2 : Hard Truncation :**
* **Principe :** Lors de la génération (inférence), au lieu d'échantillonner  depuis une Normale , on re-échantillonne les valeurs qui dépassent un seuil (ex: ).
* **Implémentation :** Coder une fonction d'échantillonnage `sample_truncated_normal(threshold)`.
* **Test :** Générer des images avec le modèle "Original GAN" mais avec ce nouveau sampling.


5. **Baseline 3 : Soft Truncation :**
* **Principe :** Souvent interprété comme une optimisation du vecteur  pour trouver des régions latentes qui produisent de meilleurs résultats, ou une modification douce de la distribution de  (ex: température).
* **Action :** Vérifiez la définition exacte attendue (parfois c'est une interpolation linéaire vers la moyenne). Si non spécifié, implémentez une méthode de "Latent Optimization" simple.



---

### **Phase 3 : Implémentation des Papiers Choisis (Le Cœur du Projet)**

C'est ici que vous apportez votre valeur ajoutée.

**Papier A : f-GAN (Variational Divergence Minimization)**


* **Le Défi :** Changer la *fonction de perte* (Loss Function) sans toucher à l'architecture.
* **Action :**
* Remplacer la perte standard (BCE / Minimax) par la fonction objective du f-GAN : .


* Choisir une divergence spécifique (ex: **Pearson ** ou **Kullback-Leibler** qui ont bien performé sur MNIST dans le papier ).
* Adapter la dernière couche du Discriminateur () : Le papier f-GAN exige une fonction d'activation spécifique en sortie du discriminateur selon la divergence choisie (ex: pas de Sigmoid pour certaines divergences, voir Table 2 du papier ).


* **Attention :** Le discriminateur *peut* être modifié (seul le générateur est "sacré").

7. **Papier B : Discriminator Gradient Flow (DGF)**
* **Le Défi :** C'est une méthode de raffinement *post-hoc* ou itérative.
* **Principe :** Une fois le générateur entraîné, on ne se contente pas de faire . On utilise le discriminateur  pour améliorer . On met à jour  en suivant le gradient qui maximise le score du discriminateur : .
* **Action :**
* Prendre votre modèle "Original GAN" (ou le f-GAN).
* Implémenter une boucle de raffinement lors de la génération.
* Comparer les images avant et après raffinement.

---

### **Phase 4 : Expérimentation et Comparaison**
#### 1. Faut-il ajouter le Gradient Flow sur le Baseline ?

**OUI, absolument.** Le "Baseline" (Original GAN) utilise la divergence de Jensen-Shannon de manière implicite via la `BCELoss`. Tester le Gradient Flow (DGflow) sur le baseline te donnera un point de comparaison crucial pour savoir si le gain de performance vient de la technique de sampling elle-même ou de la nouvelle Loss (Pearson/KL).

#### 2. Plan d'Expérience

Pour chaque modèle entraîné (Baseline, fgan_pearson, fgan_kl, fgan_js),on doit tester **tous** les types de sampling. Cela permettra de remplir un tableau croisé. Voici comment organiser les tests pour le rapport :

#### Groupe A : Baseline (Original GAN / JS Implicit)

1. **Normal Sampling** (La référence absolue).
2. **Hard Truncation** (Pour voir si on gagne en FID au prix du Recall).
3. **DGflow (Gradient Flow)** :  `baseline` dans args de generate.py.

#### Groupe B : f-GAN Pearson (La variante "Quadratique")

1. **Normal Sampling**.
2. **Hard Truncation**.
3. **Soft Truncation**.
4. **DGflow (Gradient Flow)** : *Attention :* On doit s'attendre a ce que  Pearson soit instable ici. Si les scores explosent (NaN), c'est un résultat en soi à noter dans ton rapport ! Parce que (explication mathematique) la divergence de Pearson est très sensible aux ratios de densité, et peut diverger si le discriminateur devient trop confiant.
Donc on peut re-tester avec step_size plus petit comme 0.001. 
Explication detaillee en annexe : "Pourquoi Pearson est instable avec DGflow ?"

#### Groupe C : f-GAN KL (La variante "Informationnelle")

1. **Normal Sampling**.
2. **Hard Truncation**.
3. **Soft Truncation**.
4. **DGflow (Gradient Flow)** : C'est souvent ici que DGflow brille le plus car la divergence KL est très liée au ratio de densité utilisé dans le papier original.

#### 3. Tableau de comparaison Final (Structure cible)

Voici à quoi devrait ressembler ton tableau final pour ton rendu. Il permet de comparer l'impact de la **Loss** (lignes) et du **Sampling** (colonnes).

| Modèle Entraîné | Sampling | FID ↓ | Precision ↑ | Recall ↑ | Observation |
| --- | --- | --- | --- | --- | --- |
| **1. Vanilla GAN** | Normal | ... | ... | ... | **Baseline** |
|  | Hard Truncation | ... | ... | ... | Trade-off Q/D |
|  | Soft Truncation | ... | ... | ... | Trade-off Q/D |
|  | DGflow | ... | ... | ... | Raffinement du Baseline |
| **2. f-GAN Pearson** | Normal | ... | ... | ... | Impact de la Loss $\chi^2$ |
|  | DGflow | ... | ... | ... | Stabilité à vérifier |
|  | Hard Truncation | ... | ... | ... | Trade-off Q/D |
|  | Soft Truncation | ... | ... | ... | Trade-off Q/D |
| **3. f-GAN KL** | Normal | ... | ... | ... | Impact de la Loss KL |
|  | DGflow | ... | ... | ... | Souvent le meilleur duo |
|  | Hard Truncation | ... | ... | ... | Trade-off Q/D |
|  | Soft Truncation | ... | ... | ... | Trade-off Q/D |
| **4. f-GAN JS** | Normal | ... | ... | ... | Devrait être proche du Baseline |
|  | DGflow | ... | ... | ... | Raffinement JS |
|  | Hard Truncation | ... | ... | ... | Trade-off Q/D |
|  | Soft Truncation | ... | ... | ... | Trade-off Q/D |

#### 4. Conseils pour l'exécution

1. **Priorité de génération :** Pour chaque modèle, génère d'abord en `normal`. Si le modèle `normal` est déjà mauvais (ex: Mode Collapse), le `DGflow` ne pourra pas faire de miracle.
2. **Le "Recall" est ton juge :** La Truncation (Hard/Soft) va booster ta **Precision** mais va sûrement faire chuter ton **Recall** (diversité). Le **DGflow**, s'il est réussi, est censé augmenter la **Precision** SANS trop détruire le **Recall**. C'est l'argument principal du papier.
3. **Organisation des dossiers :**
* `samples/samples_normal_baseline/`
* `samples/samples_dgflow_kl/`
* etc.
* Utiliser le script `evaluate_all.py` pour boucler sur ces dossiers et calculer les scores.
4. **Visualisation Qualitative :**
* Générer une grille d'images pour chaque méthode.
* Montrer les cas d'échec (mode collapse, artefacts).

---

### **Phase 5 : Analyse (Ce qui fera votre note)**

Ne vous contentez pas de donner les chiffres, expliquez-les.

* **Qualité vs Diversité :** Est-ce que la *Hard Truncation* améliore le FID (qualité) mais détruit le Recall (diversité) ? C'est le comportement attendu.
  
**Impact du f-GAN :** Est-ce que la divergence choisie (ex: KL) a stabilisé l'entraînement par rapport au GAN original (Jensen-Shannon) ?.

* **Effet du Gradient Flow :** Est-ce que le raffinement "nettoie" les chiffres mal formés ?

### Annexe : Pourquoi Pearson est instable avec DGflow ?
Bien que le papier ne précise pas explicitement pourquoi la divergence de **Pearson $\chi^2$** a été écartée, on peut déduire plusieurs raisons techniques et théoriques basées sur la nature du flux de gradient et les propriétés mathématiques de cette divergence.

Voici les raisons probables pour lesquelles les auteurs ont préféré la **KL**, la **JS** et la **log D** :

### 1. La forme de la dérivée $f'(r)$
Dans l'algorithme DGflow, la mise à jour dépend de $\nabla_z f'(r)$. Comparons les dérivées :

*   **KL :** $f'(r) = \log(r) + 1$ (Croissance logarithmique, très stable).
*   **Pearson $\chi^2$ :** Pour $f(r) = (r-1)^2$, on a **$f'(r) = 2(r-1)$**.

La dérivée de Pearson est **linéaire** par rapport au ratio $r$. Comme le ratio $r = \exp(-d(x))$ peut devenir extrêmement grand (si le discriminateur est très confiant), la mise à jour $2(\exp(-d(x)) - 1)$ peut provoquer des **gradients explosifs** massifs dans l'espace latent. 

> "However, our initial image experiments showed that refining directly in high-dimensional data-spaces with the stale estimate is problematic; error is accumulated at each time-step..." <alphaxiv-paper-citation paper="2012.00780" title="Methodology" page="4" first="However, our initial" last="at each time-step" />

### 2. Sensibilité aux "Outliers" (Valeurs aberrantes)
La divergence de Pearson $\chi^2$ est connue dans la littérature statistique pour être très sensible aux ratios de densité élevés. 
*   Si un échantillon généré est très loin de la distribution réelle, $r$ devient géant. 
*   Avec $f'(r)$ linéaire (Pearson), le pas de gradient sera proportionnel à ce ratio géant, "projetant" le vecteur latent très loin dans l'espace, ce qui détruit la structure de l'image.
*   À l'inverse, avec la **KL** ($\log r$), le gradient est amorti par le logarithme, ce qui rend le raffinement beaucoup plus doux et robuste.

### 3. Lien avec les méthodes existantes (DOT & DDLS)
L'un des objectifs majeurs du papier était de montrer que les méthodes d'état de l'art étaient des **cas particuliers** de DGflow :
*   **DDLS** est exactement DGflow avec une divergence **KL**.
*   **DOT** est lié au transport optimal, qui se rapproche de la limite de DGflow sous certaines conditions.

En testant la KL, la JS et la log D, les auteurs couvraient déjà les variantes les plus populaires des GAN (le GAN original utilise la JS, le KL est standard pour les EBM/Langevin). La Pearson $\chi^2$ (utilisée dans LSGAN) est moins courante pour le raffinement post-hoc.

### 4. Problème de la zone $r < 1$
Pour Pearson, $f'(r) = 2(r-1)$. 
*   Si $r < 1$ (échantillon déjà "bon"), le gradient change de signe de manière très abrupte autour de 1.
*   Les fonctions basées sur le log (KL, JS) ont une transition plus fluide qui semble mieux guidée le flux de gradient vers l'équilibre sans oscillations.

### Tableau comparatif des gradients de raffinement

| Divergence | Fonction $f'(r)$ | Comportement du Gradient |
| :--- | :--- | :--- |
| **KL (Testé)** | $\log(r) + 1$ | **Stable** : Amortit les gros ratios. |
| **JS (Testé)** | $\log(\frac{2r}{r+1})$ | **Trés Stable** : Toujours borné entre $\log(0)$ et $\log(2)$. |
| **Pearson (Non testé)** | $2r - 2$ | **Instable** : Croissance linéaire, risque d'explosion. |

**Conseil :** Si vous testez Pearson, réduisez drastiquement votre `step_size` ($\eta$) à $0.001$ ou moins pour éviter que vos vecteurs latents ne s'envolent vers l'infini !

> "The SDE can be approximately simulated via the stochastic Euler scheme... where the time interval [0, T] is partitioned into equal intervals of size η" <alphaxiv-paper-citation paper="2012.00780" title="Simulation" page="4" first="The SDE can" last="size η" />
---

# Organisation Cible du Repository

Pour intégrer les 5 méthodes et les métriques sans transformer le code existant en "usine à gaz", je vous propose une **architecture modulaire**. L'idée est de ne pas toucher aux fichiers originaux (`train.py`, `model.py`) pour garder la Baseline pure, et de créer des fichiers dédiés pour les extensions.

Voici l'arborescence recommandée :

```text
root/
│
├── checkpoints/          # Sauvegarde des modèles (.pth)
├── samples/              # Images générées
├── data/                 # Dataset MNIST (téléchargé auto)
│
├── model.py              # [INTOUCHABLE] Architecture G et D originale
├── utils.py              # Helpers originaux (chargement, I/O)
│
├── # --- BASELINE ---
├── train.py              # Script d'entraînement Original GAN
├── generate.py           # Script de génération standard
│
├── # --- MÉTRIQUES (Le Juge) ---
├── metrics.py            # [NOUVEAU] Contient les fonctions FID, Precision, Recall
|__ train_feature_extractor.py  # [NOUVEAU] Script pour entraîner un petit CNN sur MNIST pour extraire les features nécessaires au FID
│
├── # --- PARTIE 1 : f-GAN ---
├── fgan_utils.py         # [NOUVEAU] Définitions des f-divergences et fonctions d'activation conjuguées
├── train_fgan.py         # [NOUVEAU] Copie modifiée de train.py qui utilise fgan_utils
│
├── # --- PARTIE 2 : SAMPLING AVANCÉ (Truncation & Gradient Flow) ---
├── sampling_utils.py     # [NOUVEAU] Fonctions pour Hard/Soft Truncation et les Gradient Flow utils
│
├── # --- ORCHESTRATION ---
├── evaluate_all.py       # [NOUVEAU] Script principal qui charge les modèles et calcule les scores
│
├── requirements.txt
└── README.md

```

#### Détail des fichiers à créer/modifier :

**A. `metrics.py` (Priorité absolue)**

* **Contenu :** Classes ou fonctions pour calculer le FID et Precision/Recall.
* **Pourquoi :** Vous en aurez besoin partout. Isolez ce code pour ne pas le copier-coller.
* *Note :* Utilisez `pytorch-fid` ou `torch-fidelity` si possible, ou implémentez une version légère adaptée à MNIST.

**B. `fgan_utils.py**`

* **Contenu :** Dictionnaires contenant les formules mathématiques du papier f-GAN.
* Exemple : `{'pearson': (fonction_activation_D, fonction_loss_conjuguee)}`.


* **Pourquoi :** Garde `train_fgan.py` propre.

**C. `train_fgan.py**`

* **Base :** Copie de `train.py`.
* **Modifications :**
1. Importer `fgan_utils`.
2. Modifier la **Loss function** (remplacer `BCELoss`).
3. **Important :** Le papier f-GAN requiert souvent que le discriminateur n'ait *pas* de Sigmoid à la fin (sortie linéaire ).
* *Astuce :* Importez `Discriminator` de `model.py`, mais après l'instanciation, remplacez la dernière couche :


```python
D = Discriminator()
# Remplacer la dernière couche Sigmoid par Identity pour f-GAN
D.main[-1] = nn.Identity() 

```





**D. `sampling_utils.py**`

* **Contenu :**
1. `sample_hard_truncation(generator, threshold, ...)`
2. `sample_soft_truncation(...)`
3. `sample_gradient_flow(generator, discriminator, z, iterations, learning_rate)` -> C'est ici que vous implémentez la logique du papier "Discriminator Gradient Flow".



**E. `evaluate_all.py` (Le script final)**

* **Rôle :** C'est le script que vous lancerez à la fin pour remplir votre tableau de notes.
* **Logique :**
1. Charger `checkpoints/G.pth` (Original) et `checkpoints/G_fgan.pth`.
2. Générer 10k images avec la méthode "Original" -> Calculer Metrics.
3. Générer 10k images avec "Hard Truncation" -> Calculer Metrics.
4. ... idem pour les 5 méthodes.
5. Afficher le tableau final dans la console ou sauvegarder dans `results.csv`.

---
# fgan with JS and 100 epochs 

📂 Using samples directory: folder_of_samples/samples_normal_js
[f-GAN JS] -> FID: 0.0279 | Precision: 0.6746 | Recall: 0.7706

📂 Using samples directory: folder_of_samples/samples_hard_js
[f-GAN JS] -> FID: 0.1590 | Precision: 0.6517 | Recall: 0.2904

📂 Using samples directory: folder_of_samples/samples_soft_js
[f-GAN JS] -> FID: 0.0408 | Precision: 0.7088 | Recall: 0.6449

📂 Using samples directory: folder_of_samples/samples_dgflow_js
[f-GAN JS] -> FID: 0.0273 | Precision: 0.6703 | Recall: 0.7714

# fgan with JS and 400 epochs (the original disc with dropouts only)

📂 Using samples directory: folder_of_samples/samples_normal_js
[f-GAN JS] -> FID: 0.0207 | Precision: 0.7514 | Recall: 0.8297

📂 Using samples directory: folder_of_samples/samples_hard_js
[f-GAN JS] -> FID: 0.3576 | Precision: 0.8617 | Recall: 0.2519

📂 Using samples directory: folder_of_samples/samples_soft_js
[f-GAN JS] -> FID: 0.0623 | Precision: 0.8273 | Recall: 0.6476

📂 Using samples directory: folder_of_samples/samples_dgflow_js
[f-GAN JS] -> FID: 0.0203 | Precision: 0.7495 | Recall: 0.8262

# fgan with JS and 400 epochs + SN in the discriminator
📂 Using samples directory: folder_of_samples/samples_normal_js_disc4SN_400
[f-GAN JS] -> FID: 0.0110 | Precision: 0.7427 | Recall: 0.7954

📂 Using samples directory: folder_of_samples/samples_hard_js_disc4SN_400
[f-GAN JS] -> FID: 0.4316 | Precision: 0.8948 | Recall: 0.2581

📂 Using samples directory: folder_of_samples/samples_soft_js_disc4SN_400
[f-GAN JS] -> FID: 0.0517 | Precision: 0.8214 | Recall: 0.6331

📂 Using samples directory: folder_of_samples/samples_dgflow_js_disc4SN_400
[f-GAN JS] -> FID: 0.0111 | Precision: 0.7421 | Recall: 0.7941

Remark: The fact that we added Spectral normalization in the discriminator, the performance decrease. Maybe whene we add SN we should remove the dropout layer -> to be verified and tested. 

# fgan with JS and 400 epochs + SN - dropouts in the discriminator

📂 Using samples directory: folder_of_samples/samples_normal_js_disc4SN_nodrop_400
[f-GAN JS] -> FID: 0.0061 | Precision: 0.7843 | Recall: 0.7972

📂 Using samples directory: folder_of_samples/samples_hard_js_disc4SN_nodrop_400
[f-GAN JS] -> FID: 0.2373 | Precision: 0.8616 | Recall: 0.3207

📂 Using samples directory: folder_of_samples/samples_soft_js_disc4SN_nodrop_400
[f-GAN JS] -> FID: 0.0312 | Precision: 0.8529 | Recall: 0.6537

📂 Using samples directory: folder_of_samples/samples_dgflow_js_disc4SN_nodrop_400
[f-GAN JS] -> FID: 0.0056 | Precision: 0.7826 | Recall: 0.8016

In JS the performance (with SN and without the dropout) is better that (with dropout and SN) and better that the case (without SN and with dropout).

# Baseline model: Vanilla GAN with 100 epochs
📂 Using samples directory: folder_of_samples/samples_normal_Baseline_100epochs
[Baseline Generator] -> FID: 0.0638 | Precision: 0.4521 | Recall: 0.7610

# Baseline model: Vanilla GAN with 200 epochs

📂 Using samples directory: folder_of_samples/samples_normal_Baseline
[Baseline Generator] -> FID: 0.0198 | Precision: 0.6655 | Recall: 0.8128

📂 Using samples directory: folder_of_samples/samples_hard_Baseline
[Baseline Generator] -> FID: 0.4390 | Precision: 0.8204 | Recall: 0.3422

📂 Using samples directory: folder_of_samples/samples_soft_Baseline
[Baseline Generator] -> FID: 0.0538 | Precision: 0.7257 | Recall: 0.6954

📂 Using samples directory: folder_of_samples/samples_dgflow_Baseline
[Baseline Generator] -> FID: 0.0202 | Precision: 0.6589 | Recall: 0.8136

# Baseline model: Vanilla GAN with 400 epochs 

📂 Using samples directory: folder_of_samples/samples_normal_Baseline *
[Baseline Generator] -> FID: 0.0128 | Precision: 0.7216 | Recall: 0.8283
soumis le 09/03/2026: 
Leaderboard:       21.39                0.85            0.62

📂 Using samples directory: folder_of_samples/samples_hard_Baseline
[Baseline Generator] -> FID: 0.4006 | Precision: 0.8666 | Recall: 0.3110

📂 Using samples directory: folder_of_samples/samples_soft_Baseline
[Baseline Generator] -> FID: 0.0490 | Precision: 0.8061 | Recall: 0.6945

📂 Using samples directory: folder_of_samples/samples_dgflow_Baseline
[Baseline Generator] -> FID: 0.0132 | Precision: 0.7296 | Recall: 0.8288

# Vanila gan using Discriminateur + SN 400 epochs

📂 Using samples directory: folder_of_samples/samples_normal_baseline_disc4SN_400
[Baseline Generator] -> FID: 0.1705 | Precision: 0.4609 | Recall: 0.2747

📂 Using samples directory: folder_of_samples/samples_hard_baseline_disc4SN_400
[Baseline Generator] -> FID: 0.6927 | Precision: 0.7291 | Recall: 0.1977

📂 Using samples directory: folder_of_samples/samples_soft_baseline_disc4SN_400
[Baseline Generator] -> FID: 0.2612 | Precision: 0.5369 | Recall: 0.2467

📂 Using samples directory: folder_of_samples/samples_dgflow_baseline_disc4SN_400
[Baseline Generator] -> FID: 0.1718 | Precision: 0.4575 | Recall: 0.3210

Remark: by adding SN to the discriminator the the performace decreased !!!!

# Vanila gan using Discriminateur + SN + 400 epochs - dropouts

📂 Using samples directory: folder_of_samples/samples_normal_baseline_disc4SN_nodrop_400
[Baseline Generator] -> FID: 0.0862 | Precision: 0.4662 | Recall: 0.1686

📂 Using samples directory: folder_of_samples/samples_hard_baseline_disc4SN_nodrop_400
[Baseline Generator] -> FID: 0.6040 | Precision: 0.6890 | Recall: 0.2078

📂 Using samples directory: folder_of_samples/samples_soft_baseline_disc4SN_nodrop_400
[Baseline Generator] -> FID: 0.1338 | Precision: 0.4896 | Recall: 0.0945

📂 Using samples directory: folder_of_samples/samples_dgflow_baseline_disc4SN_nodrop_400
[Baseline Generator] -> FID: 0.0852 | Precision: 0.4626 | Recall: 0.1445

With SN and without dropouts in the discriminator the performance of vanilla gan is worest.

# Fgan model using: KL + Shallow disciminator v2 + SN + 200 epochs 

📂 Using samples directory: folder_of_samples/samples_normal_kl_shallow_200
[f-GAN KL] -> FID: 0.0180 | Precision: 0.6852 | Recall: 0.7285

# Fgan model using: KL + original discriminator + SN + 200 epochs 

📂 Using samples directory: folder_of_samples/samples_normal_kl_disc4SN_200
[f-GAN KL] -> FID: 0.0154 | Precision: 0.6920 | Recall: 0.7780

📂 Using samples directory: folder_of_samples/samples_hard_kl_disc4SN_200
[f-GAN KL] -> FID: 0.2170 | Precision: 0.6478 | Recall: 0.3332

📂 Using samples directory: folder_of_samples/samples_soft_kl_disc4SN_200
[f-GAN KL] -> FID: 0.0344 | Precision: 0.7325 | Recall: 0.6805

📂 Using samples directory: folder_of_samples/samples_dgflow_kl_disc4SN_200
[f-GAN KL] -> FID: 0.0153 | Precision: 0.6953 | Recall: 0.7829

# Fgan model using: KL + original discriminator + SN + 400 epochs 
📂 Using samples directory: folder_of_samples/samples_normal_kl_disc4SN_400
[f-GAN KL] -> FID: 0.0116 | Precision: 0.7345 | Recall: 0.8085

📂 Using samples directory: folder_of_samples/samples_hard_kl_disc4SN_400
[f-GAN KL] -> FID: 0.5160 | Precision: 0.8446 | Recall: 0.3333

📂 Using samples directory: folder_of_samples/samples_soft_kl_disc4SN_400
[f-GAN KL] -> FID: 0.0674 | Precision: 0.8054 | Recall: 0.6833

📂 Using samples directory: folder_of_samples/samples_dgflow_kl_disc4SN_400
[f-GAN KL] -> FID: 0.0107 | Precision: 0.7348 | Recall: 0.7982

# Fgan model using: KL + original discriminator + SN - dropout + 400 epochs 

📂 Using samples directory: folder_of_samples/samples_normal_kl_disc4SN_nodrop_400
[f-GAN KL] -> FID: 0.0067 | Precision: 0.7736 | Recall: 0.7790
soumis le 10/03/2026: 

📂 Using samples directory: folder_of_samples/samples_hard_kl_disc4SN_nodrop_400
[f-GAN KL] -> FID: 0.1628 | Precision: 0.8229 | Recall: 0.2853

📂 Using samples directory: folder_of_samples/samples_soft_kl_disc4SN_nodrop_400
[f-GAN KL] -> FID: 0.0215 | Precision: 0.8407 | Recall: 0.6519

📂 Using samples directory: folder_of_samples/samples_dgflow_kl_disc4SN_nodrop_400
[f-GAN KL] -> FID: 0.0075 | Precision: 0.7781 | Recall: 0.7927

# Fgan: Pearson  + disc SN + dropout+ 400 epochs
 Using samples directory: folder_of_samples/samples_normal_pearson_disc4SN_400
[f-GAN Pearson] -> FID: 0.0121 | Precision: 0.7205 | Recall: 0.7948

# conlusion about SN and Dropouts in discriminator 
Ater doing these empirical experiments inorder to know the best architecture of the disciminator (Dropouts only OR  SN 'necessay for KL, pearson' + dropouts OR SN and without the dropouts ) we found out this: 

The empirical results are **consistent with theory and with what is observed in the GAN literature**. 

## Short answer (decision rule)

### ✅ For **f-GAN (KL, Pearson, JS)**

**Use spectral normalization and remove dropout**

### ⚠️ For **vanilla GAN (BCE / JS baseline)**

**Spectral normalization is NOT always beneficial** and often **hurts performance**, especially in small MLP discriminators like yours.

The results already demonstrate this.


## Why the results make sense

## 1. f-GAN (KL / Pearson / JS)

### What the theory says

In f-GANs, especially **KL and Pearson**, the discriminator output appears **directly inside non-linear functions**:

* KL:
  [
  \exp(T(x) - 1)
  ]
* Pearson:
  [
  T(x)^2
  ]

So if the discriminator outputs large logits:

* KL → **exponential explosion**
* Pearson → **quadratic explosion**

This makes the training **extremely sensitive** to discriminator scale.

### What spectral normalization does here

Spectral normalization:

* constrains the Lipschitz constant of the discriminator
* keeps logits in a narrow, stable range
* prevents loss explosions

Dropout, on the other hand:

* injects noise into discriminator outputs
* increases variance **inside the exponential / square**

So combining **SN + dropout** is counterproductive.

### The empirical confirmation (very important)

We observed:

> f-GAN JS / KL
> **SN + no dropout > SN + dropout > no SN and dropout (which never works for KL and pearson)**


## 2. Vanilla GAN (BCE / JS baseline)

This is where the behavior changes.

### Why vanilla GAN behaves differently

Vanilla GAN discriminator loss is:

[
\log \sigma(D(x)) + \log(1 - \sigma(D(G(z))))
]

Important difference:

* discriminator output is **passed through a sigmoid**
* loss **saturates naturally**
* no exponential or quadratic growth

So the discriminator **does not need strong Lipschitz control**.

### What spectral normalization does here

In vanilla GAN:

* SN **weakens the discriminator**
* reduces its ability to separate real/fake
* leads to **underfitting**

Especially in **MLP-based discriminators on MNIST**, SN is often **too restrictive**.

This causes:

* weaker gradients for the generator
* slower convergence
* worse FID and Recall

### Why dropout helped vanilla GAN before

Dropout:

* adds noise
* prevents discriminator from becoming too confident
* acts as a *soft regularizer*

This can actually **help vanilla GAN**, which otherwise overfits very quickly.

### The empirical confirmation

We observed:

> Vanilla GAN
> **Adding SN makes performance worse**
> Even worse without dropout

This is **expected**.

✔ Not a bug
✔ Not an implementation issue
✔ This is a known tradeoff

## 3. Why SN + dropout is almost always bad

This combination is problematic because:

* SN already regularizes weights
* Dropout regularizes activations
* Together → **over-regularization**

Effects:

* discriminator too weak
* gradients noisy + small
* generator cannot learn effectively

---

# Final comparaison (600 epochs)
## Baseline
📂 Using samples directory: folder_of_samples/samples_normal_Baseline
[Baseline Generator] -> FID: 0.0208 | Precision: 0.7338 | Recall: 0.8256 <- 600 epochs

[Baseline Generator] -> FID: 0.0128 | Precision: 0.7216 | Recall: 0.8283 <- 400 epochs
soumis le 09/03/2026: 
Leaderboard:       21.39                0.85            0.62

📂 Using samples directory: folder_of_samples/samples_hard_Baseline
[Baseline Generator] -> FID: 0.4905 | Precision: 0.8938 | Recall: 0.2015

📂 Using samples directory: folder_of_samples/samples_soft_Baseline
[Baseline Generator] -> FID: 0.0664 | Precision: 0.8264 | Recall: 0.6528

📂 Using samples directory: folder_of_samples/samples_dgflow_Baseline
[Baseline Generator] -> FID: 0.0207 | Precision: 0.7460 | Recall: 0.8270

## JS

📂 Using samples directory: folder_of_samples/samples_normal_js -> 13/03
[f-GAN JS] -> FID: 0.0059 | Precision: 0.8025 | Recall: 0.7935
LeaderBoard:        8.51             0.86             0.64

📂 Using samples directory: folder_of_samples/samples_hard_js
[f-GAN JS] -> FID: 0.2592 | Precision: 0.8780 | Recall: 0.2496

📂 Using samples directory: folder_of_samples/samples_soft_js
[f-GAN JS] -> FID: 0.0384 | Precision: 0.8730 | Recall: 0.6227

📂 Using samples directory: folder_of_samples/samples_dgflow_js -> 14/03
[f-GAN JS] -> FID: 0.0056 | Precision: 0.8417 | Recall: 0.7504
LeaderBoard:         7.38                0.87            0.62

## KL

📂 Using samples directory: folder_of_samples/samples_normal_kl -> 11/03
[f-GAN KL] -> FID: 0.0080 | Precision: 0.7524 | Recall: 0.8211
Dans leaderboard    11.62                0.84            0.66

📂 Using samples directory: folder_of_samples/samples_hard_kl
[f-GAN KL] -> FID: 0.2399 | Precision: 0.8306 | Recall: 0.2903

📂 Using samples directory: folder_of_samples/samples_soft_kl
[f-GAN KL] -> FID: 0.0283 | Precision: 0.8316 | Recall: 0.6635

📂 Using samples directory: folder_of_samples/samples_dgflow_kl -> 12/03
[f-GAN KL] -> FID: 0.0106 | Precision: 0.7913 | Recall: 0.7328
Dans leaderboard    10.56              0.84              0.62

## Pearson

📂 Using samples directory: folder_of_samples/samples_normal_pearson -> 16/03
[f-GAN Pearson] -> FID: 0.0059 | Precision: 0.7909 | Recall: 0.7868
Dans leaderboard          9.78               0.84             0.66

📂 Using samples directory: folder_of_samples/samples_hard_pearson
[f-GAN Pearson] -> FID: 0.2184 | Precision: 0.8663 | Recall: 0.2385

📂 Using samples directory: folder_of_samples/samples_soft_pearson
[f-GAN Pearson] -> FID: 0.0264 | Precision: 0.8537 | Recall: 0.6497

📂 Using samples directory: folder_of_samples/samples_dgflow_pearson -> 15/03
[f-GAN Pearson] -> FID: 0.0060 | Precision: 0.8127 | Recall: 0.7879
Dans leaderboard           9.3               0.85             0.65

#### Pearson en changeant le eta de gdflow 
[f-GAN Pearson] -> eta = 0.01(default):
FID: 0.0097 | Precision: 0.8571 | Recall: 0.6939

[f-GAN Pearson] -> eta = 0.001:
FID: 0.0056 | Precision: 0.8041 | Recall: 0.7834

[f-GAN Pearson] -> eta = 0.0001:
FID: 0.0066 | Precision: 0.7932 | Recall: 0.7891

Ceci prouve emeriquement que dgflow doit etre adapte (dans la valeur de eta) pour pearson. Car on avait expliquer en haut que il faut "# réduisez drastiquement votre `step_size` ($\eta$) à $0.001$ ou moins pour éviter que vos vecteurs latents ne s'envolent vers l'infini" 
