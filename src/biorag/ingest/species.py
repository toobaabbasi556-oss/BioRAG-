"""Heuristic filter for non-human studies.

The PubMed query used to build this corpus searches biomechanics vocabulary
("gait analysis", "joint kinematics", "ground reaction force"), all of which
veterinary and animal research uses identically. Roughly 5% of the fetched
abstracts turned out to be equine, canine, murine or invertebrate studies.

This is a KEYWORD heuristic, not a MeSH filter. It is approximate in both
directions: it will miss animal papers that never name their species in the
abstract, and it will wrongly drop a human study that merely mentions an
animal model in passing. The precise fix is `AND humans[MeSH Terms]` in the
Entrez query, which requires refetching the corpus.
"""

import re

# Word-boundary matched so "rat" does not fire on "ratio", "pig" on "pigment".
# Deliberately EXCLUDED after auditing what they removed, because each dropped
# genuine human studies:
#   "pig"  -> "Plug-in Gait" (PiG), the standard human gait model
#   "calf" -> human calf muscle, common in EMG and gait work
#   "cat"  -> spurious matches; also CAT imaging
# The unambiguous species words below cover the same papers without the damage.
_ANIMAL_TERMS = [
    "dog", "dogs", "canine", "feline",
    "horse", "horses", "equine", "pony", "ponies",
    "mouse", "mice", "murine", "rat", "rats", "rodent",
    "porcine", "swine", "sheep", "ovine",
    "bovine", "cattle", "rabbit", "rabbits",
    "primate", "macaque", "monkey", "zebrafish",
    "elegans", "drosophila", "insect", "veterinary",
    "in vivo animal", "animal model", "animal models",
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _ANIMAL_TERMS) + r")\b",
    re.IGNORECASE,
)


def is_animal_study(title: str, text: str) -> bool:
    """True if the abstract appears to study a non-human species.

    Only the title and the opening of the abstract are searched: papers state
    their species early, whereas a human study citing animal literature tends
    to do so in later discussion. This keeps false positives down.
    """
    return bool(_PATTERN.search(f"{title} {text[:600]}"))
