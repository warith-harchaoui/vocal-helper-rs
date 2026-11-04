# Vocal Helper (Rust)

[🇫🇷](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/vocal-helper-rs/blob/main/README.md)

[![crates.io](https://img.shields.io/crates/v/vocal-helper-rs.svg)](https://crates.io/crates/vocal-helper-rs) [![Licence : BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)

Réécriture en Rust du chemin de diarisation gratuit et local de [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) — Nemotron 3 Diarization par son graphe ONNX. Pas un portage ligne à ligne : du Rust idiomatique de bout en bout.

## Pourquoi ce dépôt existe, dit honnêtement

**Pas pour la vitesse.** C'est mesuré, pas supposé. En profilant la diarisation sur deux minutes
d'audio, l'appel au graphe et le front-end chronométrés à part :

| | temps | part |
|---|---:|---:|
| graphe ONNX *(C++, identique depuis Rust)* | 90,47 s | **99,8 %** |
| front-end log-mel *(numpy, FFT en C)* | 0,06 s | 0,1 % |
| comptabilité du cache + Python | 0,10 s | 0,1 % |

Le plafond d'un portage Rust est de **0,2 % du temps** — un facteur temps réel de 0,754 au lieu de
0,755. La caisse `ort` appelle la même bibliothèque C++, et les morceaux de Python sont soit
vectorisés, soit du travail sur des tableaux de 264 × 512 à raison d'un appel par seconde environ.
Qui promet un gain de RTF ici n'a pas mesuré.

**Pour des endroits où tourner.** Sur iOS et Android, le chemin Python n'existe pas : on n'embarque
pas raisonnablement CPython et une pile scientifique dans une application mobile. C'est un argument
de *possibilité* et non de performance, et il est décisif. S'y ajoutent un binaire unique par
plateforme, aucune résolution de dépendances à l'installation, aucun interpréteur à démarrer, et
une empreinte mémoire modeste pour un service qui écoute en arrière-plan.

Même moteur, mêmes nombres, même vitesse — quelque part où Python n'accède pas.

## Portée

Tout ce qui vit **autour** du graphe est là, et c'est l'essentiel du travail. Ce qui n'est pas
encore écrit est nommé comme tel plutôt que sous-entendu.

| module | ce qu'il fait | état |
|---|---|---|
| `mel` | le front-end log-mel que le graphe attend, à la trame près | **fait**, parité prouvée |
| `cache` | la tenue du cache de locuteurs et de la file d'un pas à l'autre | **fait**, parité prouvée |
| `schedule` | le calendrier de flux et les deux plafonds qui le bornent | **fait**, parité prouvée |
| `segments` | des probabilités par trame aux tours de parole | **fait**, parité prouvée |
| `accel` | choisir un moteur d'exécution en le mesurant sur l'appareil | **fait** |
| `asr` | l'hôte de la transcription Nemotron 3.5 : front-end NeMo, fenêtres, CTC glouton, mots horodatés, balises de locuteur et d'événement | **fait**, parité prouvée sur les métadonnées et les sorties du vrai graphe |
| `onnx_meta` | lire les `metadata_props` d'un graphe sans charger ses poids | **fait** |
| liaison ONNX Runtime | piloter le graphe lui-même, par `ort`, derrière la feature `onnx` | **transcription faite** (`asr_session`, `examples/transcribe.rs` ; 0,86 % de WER d'écart au jumeau Python en int8, RTF 0,15 sur CPU) ; **diarisation faite** (`diar_session`, `examples/diarize.rs` ; probabilités à 1,7e-6 du jumeau Python sur 3 min d'AMI, compression du cache comprise) |
| chemins payants | Gladia et pyannoteAI, en lot et en direct | pas écrits |

```sh
cargo test                                  # 86 tests, parité comprise
cargo clippy --all-targets -- -D warnings
```

## Un portage ne vaut que sa parité

Le graphe ne prend pas de l'audio, il prend des trames log-mel. Sa configuration en donne la
charpente — 16 kHz, fenêtre de 400 échantillons, saut de 160, transformée sur 512, pré-accentuation
0,97, 128 bandes — mais pas les détails qui décident du résultat. Ils ont été **retrouvés en
comparant à l'extracteur du modèle**, pas devinés, et deux comptent :

- La fenêtre d'analyse de 400 points est **centrée** dans la transformée de 512, et non collée au
  début. La coller multiplie par seize l'écart aux sorties officielles — 0,530 au lieu de 0,033 —
  parce que le décalage de phase introduit change le spectre.
- Le plancher du logarithme est `2^-24`, lu dans le minimum des sorties officielles : −16,36, qui
  en est exactement le logarithme.

Les deux chemins alimentent le **même graphe** : s'ils divergent, le chemin mobile fabrique
silencieusement d'autres probabilités, et aucun test unitaire de chaque côté ne le verrait.
`tests/parite_mel.rs` compare donc à une référence produite par
`vocal_helper.nemotron.log_mel` sur quatre signaux choisis pour ce qui peut différer : un bruit
riche, un ton pur sur la charnière de l'échelle Slaney, deux tons éloignés pour détecter une fuite
entre filtres voisins, et du silence, où le plancher doit être atteint exactement. L'accord est à
`1e-3` près.

Cela fait une chaîne de deux maillons : extracteur officiel → Python (0,006 d'écart moyen) → Rust
(≤ 1e-3).

## Accélérations : mesurer l'appareil, ne pas croire le raisonnement

Le raisonnement évident sur les accélérateurs est faux, et on l'a payé. ONNX Runtime n'a pas de
moteur Metal, donc sur Apple Silicon l'accélération passe par CoreML, et le mettre en tête semblait
aller de soi. Mesuré sur le graphe de diarisation, CoreML donne un facteur temps réel de **1,257**
— plus lent que le direct, donc inutilisable — contre **0,528** pour le CPU seul. Le journal de la
session dit pourquoi : CoreML ne prend que 1193 nœuds sur 2827 et les découpe en **316
partitions**, si bien que chaque pas de flux paie autant de traversées entre deux moteurs.

Le second réglage « évident » était fautif aussi : forcer huit fils sur une machine à huit cœurs
performants mesure 0,953 contre 0,528 en laissant le runtime choisir.

La leçon n'est pas que CoreML est mauvais — il sera excellent sur un autre graphe, ou dans une
version qui partitionne mieux. La leçon est qu'un ordre codé en dur sera faux quelque part, donc la
caisse **essaie sur l'appareil** et retient le verdict. Un verdict est attaché à l'empreinte du
graphe, parce que le découpage dépend des opérateurs donc du modèle : c'est précisément l'erreur
qu'un ordre codé en dur commet.

La politique est pure — pas de modèle, pas de réseau, pas d'ONNX — donc entièrement testable, et la
liaison au runtime vit derrière un drapeau de compilation.

**Et un moteur nommé gagne toujours.** L'échelle est : le moteur que l'appelant nomme, puis
`VOCAL_HELPER_PROVIDER`, puis le verdict mesuré, puis la tête de l'ordre de départ de la
plateforme quand rien n'a pu être mesuré. Un nom est pris tel quel — jamais mesuré, jamais
comparé, jamais remplacé en silence, puisque c'est tout l'intérêt de le nommer. Un nom qui ne
correspond à rien est une **erreur**, pas un repli : un repli rendrait une faute de frappe
indétectable, et on croirait tourner sur CUDA en tournant sur le CPU. La décision porte son
autorité (`Explicit`, `Measured`, `Preference`), pour qu'un journal ne laisse jamais croire à
une mesure qui n'a pas eu lieu.

```sh
VOCAL_HELPER_PROVIDER=coreml   # épingle cette machine, sans toucher au code
```


## D'où viennent ces chiffres

La campagne de mesure derrière ces choix est consignée dans
[`doc/diarization-results.fr.md`](https://github.com/warith-harchaoui/vocal-helper/blob/main/doc/diarization-results.fr.md)
de vocal-helper, impasses comprises : quatre paramétrages de seuil réfutés, le regroupement sans
centroïde fermé, et la découverte que la référence d'AMI ne peut pas arbitrer parce qu'elle est
transcrite sur les micros-casques alors qu'on donne au moteur le mixage de salle.

## Voisinage

Fait partie du même outillage local-first que [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper) (Python), [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs), [`capture-helper-rs`](https://github.com/warith-harchaoui/capture-helper-rs), [`podcast-helper-rs`](https://github.com/warith-harchaoui/podcast-helper-rs), [`youtube-helper-rs`](https://github.com/warith-harchaoui/youtube-helper-rs), et la suite [AI Helpers](https://github.com/warith-harchaoui/ai-helpers). Réécriture indépendante, pas une liaison.

## Licence

BSD-3-Clause, voir [LICENSE](LICENSE). Les poids de modèle qu'elle charge ne sont pas les nôtres et
portent leurs propres termes — Nemotron 3 Diarization est à NVIDIA, sous OpenMDW License 1.1.

## Auteur

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)
