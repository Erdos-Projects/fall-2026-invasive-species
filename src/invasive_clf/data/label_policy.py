"""Label policy: the team's rules for which species count as invasive in the Desert Southwest.

The starting labels come from the species lists (``classifier_label`` in the ranked CSVs, see
``data/reference/``). This module holds the agreed group rules and the hand-made exceptions, so every labeling
decision lives in one reviewable place. ``build_dataset.py`` applies them in step 3.

Sources of the labels: build_hybrid.py (classifier_label = invasive / noxious only / native, noxious)
via ranked_desert_sw.csv and ranked_full_bbox.csv; the override sets below were agreed on 2026-10-29
(draft final query) and the group rules on 2026-10-30 (prompts 2 and 3).

Group rules (2026-10-30):
  * invasive                 = classifier_label "invasive" or INVASIVE_OVERRIDES            -> its own class
  * noxious only             = non-native, regulatory listing only                  -> label NONE_LABEL (hard negative)
  * native, noxious          = native to part of the region, legally listed         -> label NONE_LABEL (hard negative)
  * background species       = every other species in the place with enough photos, native or
                               introduced-but-unlisted                               -> label NONE_LABEL
"""

# ---- Hand overrides of the list labels ------------------------------------------------------------------------
# Two bool-valued tables with the same valence: True always points toward "invasive".
# Both are applied in build_dataset.py step 3 (listed candidates), right after the ranked CSVs are merged, and every
# row they change gets a note in column `label_edits` (carried into classes.csv and species_summary.csv; not into the
# final train/val/test tables). The step also names entries that changed nothing.
# They cannot be one table: whether a species is native and whether the group calls it invasive are separate
# judgments (Cyperus esculentus is not native, yet the group did not make it an invasive class).
#
# INVASIVE_OVERRIDES: True = invasive whatever the classifier_label in the ranked lists says (False would force not invasive).
#   Effect: `invasive` -> `group` = "invasive" -> `label` = the species name (its own class).
#   * Ipomoea hederacea, Ipomoea purpurea: the lists say "native, noxious" (native to part of the Southwest), but
#     iNaturalist flags every desert observation introduced; agreed as invasive in the draft final query (2026-10-29).
#   * Nicotiana glauca: already "invasive" in both ranked lists; kept only as a guard if the lists are rebuilt.
INVASIVE_OVERRIDES = {"Nicotiana glauca": True, "Ipomoea hederacea": True, "Ipomoea purpurea": True}

# INTRODUCED_OVERRIDES: True = not native in the Desert Southwest (`native_in_sw` = False), False = native.
#   Without an entry, `native_in_sw` is True exactly for the "native, noxious" bucket.
#   Effect: the `native_in_sw` metadata column only; `group` and `label` do not change.
#   * Cyperus esculentus: list bucket "native, noxious", but iNaturalist flags 99% of desert observations introduced.
#   * Ipomoea hederacea, Ipomoea purpurea: same, and they are invasive classes (see above).
#   * Phragmites australis: iNaturalist flags 100% of desert observations native (subsp. americanus). The ranked list
#     row used here already says "native, noxious", so this entry changes nothing; it is a guard if the lists are rebuilt.
INTRODUCED_OVERRIDES = {"Cyperus esculentus": True, "Ipomoea hederacea": True, "Ipomoea purpurea": True,
                        "Phragmites australis": False}

# Cal-IPC "Moderate"-only species the group decided to keep as invasive (already 'invasive' in the lists)
CALIPC_KEEP = ["Hordeum murinum", "Vinca major", "Avena fatua", "Stipa capensis",
               "Hirschfeldia incana", "Ficus carica", "Rumex acetosella"]

# Cal-IPC "Moderate"-only species expected to fall under the photo threshold; queried only to confirm
CALIPC_CONFIRM = ["Oxalis pes-caprae", "Mesembryanthemum crystallinum", "Foeniculum vulgare",
                  "Avena barbata", "Tanacetum vulgare", "Atriplex semibaccata", "Torilis arvensis"]

# Names the list resolver (notebooks/data/04_inat_counts.ipynb) could not match, fixed by hand (iNat taxon ids)
UNRESOLVED_FIX = {"Oenothera sinuosus": 78248,                 # typo for Oenothera sinuosa
                  "Physalis virginiana var. sonorae": 154825,  # Physalis virginiana
                  "Cucumis melo var. dudaim": 80640,           # the variety itself
                  "Orobanche ramosa": 804770,                  # now Phelipanche ramosa
                  "Sinapis arvensis": 1562069}                 # now Mutarda arvensis

# Candidates dropped on purpose
EXCLUDE = {"Cucumis melo var. dudaim": "variety only; resolving to species would mean every melon",
           "Ipomoea × leucantha": "hybrid, 2 photos all-time"}

# Species whose iNat ancestry contains a complex that is itself a class merge into it
# (e.g. Tamarix ramosissima -> Tamarix chinensis complex)
MERGE_INTO_COMPLEX = True

# ---- group / label mapping -------------------------------------------------------------------
# `group` keeps the fine distinction for analysis; `label` is the classification target.
GROUP_OF_LABEL04 = {"invasive": "invasive",
                    "noxious only": "listed_not_invasive",
                    "native, noxious": "native_listed"}
BACKGROUND_GROUPS = {"native": "native_background", "introduced": "introduced_unlisted"}
# This is the classification label for observations not belonging to any invasive species.
NONE_LABEL = "not_invasive"


def group_for_listed(label04, invasive):
    """Group of a species that comes from the ranked lists."""
    if invasive:
        return "invasive"
    # a class merged from several list rows carries 'a;b' here; the first bucket decides
    first = str(label04).split(";")[0]
    return GROUP_OF_LABEL04.get(first, "listed_not_invasive")


def label_for(class_name, invasive):
    """Classification target: the invasive species' name, or NONE_LABEL for every non-invasive class."""
    return class_name if invasive else NONE_LABEL


def label_dir(label):
    """Folder name under images/ for a label (spaces and the hybrid sign are awkward in paths)."""
    return label.replace(" ", "_").replace("×", "x").replace("/", "-")
