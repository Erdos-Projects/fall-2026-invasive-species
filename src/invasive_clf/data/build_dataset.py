"""Build the final dataset tables from iNaturalist: which classes, which photos, and which split each photo is in.

Usage (from the repository root)::

    python -m invasive_clf.data.build_dataset --config configs/dataset.yaml
    python -m invasive_clf.data.build_dataset --config configs/dataset.yaml --stop-after-availability   # dry run

This is the script version of the team notebook ``inat_final_query.ipynb``. The steps keep the notebook's numbering:

1. (configuration: ``configs/dataset.yaml``)
2. Open an API session (``inat_api.INat``).
3. **Listed candidates**: the two ranked species lists (``ranked_desert_sw.csv``, ``ranked_full_bbox.csv``), merged by
   iNaturalist taxon id with the Desert-SW row preferred, and the hand overrides in ``label_policy.py`` applied.
4. **Resolve** every listed candidate to a species-level class *before* any observation is fetched: inactive names
   follow their replacement, subspecies/varieties roll up to their species, species complexes become one merged
   class, genus-level entries are dropped.
5. **Background classes**: every other plant species in the place with enough photos (native, or introduced but not
   on any list), from ``/observations/species_counts``. They all carry the label ``not_invasive``.
6. **Availability**: per class, the number of photos in the upload window and after the cutoff. Classes under the
   thresholds are dropped; background classes are ranked (look-alikes of invasive genera and introduced-unlisted
   species first, then round-robin over plant families) and added until the projected size reaches ``row_budget``;
   a seeded ~10% of them is held out as ``test_unseen``.
7. **Fetch** the newest observations of each selected class (cached per class, so an interrupted run resumes).
8. **Combine** into ``metadata_all.csv`` and write per-class summaries.
9. **Split** with ``split_rules.make_split`` and write ``train.csv``, ``val.csv``, ``test.csv``, ``test_unseen.csv``.
10. (image download: separate script, ``download_images.py``)
11. **Checks** on the written tables.

Everything is written under ``paths.out_dir``. The working tables and API caches go to ``<out_dir>/meta``. If that
folder already holds a split (``meta/split_assignment_all_rows.csv``), its observer -> fold assignment is reused, so
re-running keeps every photographer on the same side.

Differences from the notebook: no Google Drive mount; settings come from the YAML file; the reuse of images from the
earlier ``inat_dataset_top_species`` folder (a time saver during development) is removed; the image download and
the example-image plot are not part of this script.

Note that iNaturalist changes over time (new uploads, deleted photos, re-identifications), so a re-run will not
reproduce the committed tables in ``data/splits/`` exactly. Use ``download_images.py`` with those tables to get
the exact images the models were trained and evaluated on.
"""
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from invasive_clf.data import inat_api, split_rules
from invasive_clf.data.label_policy import (BACKGROUND_GROUPS, CALIPC_CONFIRM, CALIPC_KEEP, EXCLUDE, INTRODUCED_OVERRIDES,
                                            INVASIVE_OVERRIDES, MERGE_INTO_COMPLEX, NONE_LABEL, UNRESOLVED_FIX,
                                            group_for_listed, label_dir, label_for)

FINAL_SPLITS = ("train", "val", "test", "test_unseen")

# Columns of the final train/val/test/test_unseen tables: what the image shows and where it came from,
# nothing about how the row was chosen.
FINAL_COLUMNS = ["image_path", "label", "invasive", "group", "scientific_name", "common_name", "taxon_id", "genus", "family",
                 "native_in_sw", "observation_id", "observation_url", "obs_taxon_name", "obs_rank", "quality_grade",
                 "num_identification_agreements", "observed_on", "month", "phenology", "created_at", "latitude", "longitude",
                 "place_guess", "positional_accuracy_m", "inat_introduced", "inat_native", "observer", "observer_id",
                 "photo_id", "photo_url", "photo_license", "photo_attribution", "photo_width", "photo_height",
                 "n_photos_in_observation"]


# ------------------------------------------------------------------------------------------------------------------
# 1 · Configuration
# ------------------------------------------------------------------------------------------------------------------
def load_config(path, root=Path(".")):
    """Read the YAML settings into a namespace with UPPER_CASE names (the names the notebook used)."""
    raw = yaml.safe_load(Path(path).read_text())
    paths = raw.pop("paths")
    C = SimpleNamespace(**{k.upper(): v for k, v in raw.items()})
    C.RANKED_DIR = root / paths["ranked_dir"]
    C.OUT_DIR = root / paths["out_dir"]
    C.PLANTAE_ID = inat_api.PLANTAE_ID
    C.BASE_FILTERS = dict(place_id=C.PLACE_ID, **C.BASE_FILTERS, created_d1=C.WINDOW_START)
    C.BACKGROUND_FLAGS = tuple(C.BACKGROUND_FLAGS)
    C.BACKGROUND_QUOTA = tuple(C.BACKGROUND_QUOTA) if C.BACKGROUND_QUOTA is not None else None
    return C


def save_run_config(C):
    """Record the settings of this run next to its outputs."""
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "meta" / "species").mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in vars(C).items()}, open(C.OUT_DIR / "run_config.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------------------------
# 3 · Listed candidates from the ranked lists
# ------------------------------------------------------------------------------------------------------------------
def listed_candidates(C):
    """One row per listed taxon id, with the list label (``label04``) and the label-policy overrides applied.

    ``invasive``: list label "invasive", or forced by INVASIVE_OVERRIDES.
    ``native_in_sw``: list bucket "native, noxious", or set by INTRODUCED_OVERRIDES.
    Every override that changed a row is recorded in ``label_edits``.
    """
    def load_ranked(path, source):
        df = pd.read_csv(path, keep_default_na=False)
        df["taxon_id"] = pd.to_numeric(df["taxon_id"], errors="coerce")
        df["source_list"] = source
        return df

    ranked = pd.concat([load_ranked(C.RANKED_DIR / "ranked_desert_sw.csv", "desert_sw"),
                        load_ranked(C.RANKED_DIR / "ranked_full_bbox.csv", "full_bbox")], ignore_index=True)

    for name, tid in UNRESOLVED_FIX.items():            # hand fixes for names the list resolver could not match
        ranked.loc[ranked["inat_name"] == name, "taxon_id"] = tid
    ranked = ranked[ranked["taxon_id"].notna()].copy()
    ranked["taxon_id"] = ranked["taxon_id"].astype(int)

    ranked["prio"] = (ranked["source_list"] != "desert_sw").astype(int)     # the Desert-SW row wins
    cand = (ranked.sort_values(["taxon_id", "prio"])
                  .groupby("taxon_id", as_index=False)
                  .agg(inat_name=("inat_name", "first"), common_name=("common_name", "first"),
                       label04=("classifier_label", "first"), native_flag_text=("native_flag", "first"),
                       ecological_sources=("ecological_sources", "first"), noxious_states=("noxious_states", "first"),
                       source_list=("source_list", lambda s: ";".join(sorted(set(s)))),
                       list_names=("inat_name", lambda s: ";".join(sorted(set(s))))))

    invasive_default = cand["label04"] == "invasive"
    native_default = cand["label04"] == "native, noxious"
    cand["invasive"] = invasive_default
    cand["native_in_sw"] = native_default
    for name, is_invasive in INVASIVE_OVERRIDES.items():
        cand.loc[cand["inat_name"] == name, "invasive"] = is_invasive
    for name, is_introduced in INTRODUCED_OVERRIDES.items():
        cand.loc[cand["inat_name"] == name, "native_in_sw"] = not is_introduced
    # record every override that changed a row, next to that row
    cand["label_edits"] = ""
    inv_edit = cand["invasive"] != invasive_default
    cand.loc[inv_edit, "label_edits"] = ("INVASIVE_OVERRIDES: invasive=" + cand.loc[inv_edit, "invasive"].astype(str)
                                         + " (list label '" + cand.loc[inv_edit, "label04"] + "')")
    nat_edit = cand["native_in_sw"] != native_default
    cand.loc[nat_edit, "label_edits"] = (cand.loc[nat_edit, "label_edits"].where(cand.loc[nat_edit, "label_edits"] == "", cand.loc[nat_edit, "label_edits"] + "; ")
                                         + "INTRODUCED_OVERRIDES: native_in_sw=" + cand.loc[nat_edit, "native_in_sw"].astype(str)
                                         + " (list bucket implied " + native_default[nat_edit].astype(str) + ")")
    cand["excluded"] = cand["inat_name"].map(EXCLUDE).fillna("")

    missing = [n for n in set(INVASIVE_OVERRIDES) | set(INTRODUCED_OVERRIDES) | set(CALIPC_KEEP) | set(CALIPC_CONFIRM) if n not in set(cand["inat_name"])]
    assert not missing, f"names in the policy lists but not in the ranked CSVs: {missing}"
    assert cand.loc[cand["inat_name"].isin(CALIPC_KEEP), "invasive"].all()

    print(len(cand), "candidate taxa;", (cand["excluded"] == "").sum(), "after exclusions")
    print(cand["label04"].value_counts().to_string())
    print("\ninvasive:", cand["invasive"].sum(), "| native_in_sw:", cand["native_in_sw"].sum())
    print("\noverrides that changed a row (label_policy.py):")
    print(cand.loc[cand["label_edits"] != "", ["inat_name", "label04", "invasive", "native_in_sw", "label_edits"]].to_string(index=False))
    changed = set(cand.loc[cand["label_edits"] != "", "inat_name"])
    no_op = sorted((set(INVASIVE_OVERRIDES) | set(INTRODUCED_OVERRIDES)) - changed)
    if no_op:
        print("override entries that agree with the lists (no effect):", ", ".join(no_op))
    return cand


# ------------------------------------------------------------------------------------------------------------------
# 4 · Resolve every listed candidate to a species-level class
# ------------------------------------------------------------------------------------------------------------------
def resolve_listed(api, cand):
    """Map each candidate to its class key and collapse to one row per class.

    Returns (listed, complex_keys, listed_member_ids): the listed classes, the class keys that are species complexes,
    and every taxon id a listed class covers (used to keep those species out of the background pool).
    """
    work = cand[cand["excluded"] == ""].copy()
    res = [api.resolve(int(tid)) for tid in work["taxon_id"]]
    work["class_key"] = [k for k, _ in res]
    work["resolve_note"] = [n for _, n in res]
    dropped = work[work["class_key"].isna()]
    work = work[work["class_key"].notna()].copy()
    work["class_key"] = work["class_key"].astype(int)

    tx = api.fetch_taxa(sorted(set(work["class_key"])))
    complex_keys = {k for k, t in tx.items() if t["rank_level"] == 11}
    if MERGE_INTO_COMPLEX:                               # e.g. Tamarix ramosissima -> Tamarix chinensis complex
        def merge(k):
            hits = [a for a in tx[k]["ancestor_ids"] if a in complex_keys and a != k]
            return hits[-1] if hits else k
        new = work["class_key"].map(merge)
        for old, nw in zip(work["class_key"], new):
            if old != nw:
                print(f"merged {tx[old]['name']} ({old}) into {tx[nw]['name']} complex ({nw})")
        work["class_key"] = new

    def first_nonempty(s):
        s = [x for x in s if x]
        return s[0] if s else ""

    listed = (work.groupby("class_key", as_index=False)
                  .agg(member_names=("inat_name", lambda s: ";".join(sorted(set(s)))),
                       member_taxon_ids=("taxon_id", lambda s: ";".join(map(str, sorted(set(s))))),
                       label04=("label04", lambda s: ";".join(sorted(set(s)))),
                       invasive_votes=("invasive", lambda s: ";".join(sorted({str(v) for v in s}))),
                       invasive=("invasive", "any"), native_in_sw=("native_in_sw", "any"),
                       label_edits=("label_edits", lambda s: "; ".join(sorted({x for x in s if x}))),
                       ecological_sources=("ecological_sources", first_nonempty),
                       noxious_states=("noxious_states", first_nonempty),
                       source_list=("source_list", lambda s: ";".join(sorted(set(";".join(s).split(";"))))),
                       resolve_note=("resolve_note", lambda s: ";".join(sorted(set(s))))))
    listed["class_name"] = listed["class_key"].map(lambda k: tx[k]["name"])
    listed["class_rank"] = listed["class_key"].map(lambda k: tx[k]["rank"])
    listed["common_name"] = listed["class_key"].map(lambda k: tx[k].get("preferred_common_name") or "")
    listed["hrank"] = np.where(listed["class_rank"] == "complex", "complex", "species")
    listed["group"] = [group_for_listed(l, i) for l, i in zip(listed["label04"], listed["invasive"])]
    listed["origin"] = "listed"

    conflicts = listed[listed["invasive_votes"].str.contains(";")]
    assert conflicts.empty, f"merged members disagree on the invasive label:\n{conflicts[['class_name','member_names']]}"
    listed_member_ids = {int(i) for s in listed["member_taxon_ids"] for i in s.split(";")} | set(listed["class_key"])

    print(f"{len(listed)} listed classes ({listed['invasive'].sum()} invasive) from {len(work)} candidates; {len(dropped)} dropped")
    print(listed["group"].value_counts().to_string())
    if len(dropped):
        print("\nDropped:"); print(dropped[["inat_name", "taxon_id", "resolve_note"]].to_string(index=False))
    chg = work[work["class_key"] != work["taxon_id"]]
    print("\nCandidates whose class differs from the list taxon id:")
    print(chg.assign(class_name=chg["class_key"].map(lambda k: tx[k]["name"]))[["inat_name", "taxon_id", "class_key", "class_name", "resolve_note"]].to_string(index=False))
    return listed, complex_keys, listed_member_ids


# ------------------------------------------------------------------------------------------------------------------
# 5 · Background classes from /observations/species_counts
# ------------------------------------------------------------------------------------------------------------------
def background_pool(C, api, listed_member_ids):
    """Every species-rank plant in the place (per establishment flag), minus what the listed classes already cover.

    Two species-count queries per flag: over the whole upload window and from the cutoff on. Cached in ``meta/``.
    """
    bg_parts = []
    for flag in C.BACKGROUND_FLAGS:
        for tag, extra in (("window", {}), ("post", {"created_d1": C.CUTOFF})):
            path = C.OUT_DIR / "meta" / f"background_species_counts_{flag}_{tag}.csv"
            if path.exists() and not C.REFRESH:
                df = pd.read_csv(path, keep_default_na=False)
            else:
                pairs = api.species_counts(**{**C.BASE_FILTERS, **extra, "taxon_id": C.PLANTAE_ID, flag: "true"})   # no hrank: subspecies roll up to species
                df = pd.DataFrame([dict(taxon_id=t["id"], name=t["name"], rank=t["rank"], rank_level=t["rank_level"],
                                        common_name=t.get("preferred_common_name") or "",
                                        ancestor_ids=";".join(map(str, t.get("ancestor_ids", []))), count=c) for t, c in pairs])
                df.to_csv(path, index=False)
            df["flag"] = flag; df["tag"] = tag
            bg_parts.append(df)
            print(f"{flag:10s} {tag:6s} {len(df):5d} species, {int(df['count'].sum()):8d} observations")
    bg_raw = pd.concat(bg_parts, ignore_index=True)

    # one row per taxon: window and post counts summed over flags (a species can be native in one part of the polygon
    # and introduced in another; the larger window count decides its background group)
    win = bg_raw[bg_raw["tag"] == "window"]
    flag_of = win.sort_values("count", ascending=False).drop_duplicates("taxon_id").set_index("taxon_id")["flag"]
    bg = (bg_raw.pivot_table(index="taxon_id", columns="tag", values="count", aggfunc="sum", fill_value=0)
                .reindex(columns=["window", "post"], fill_value=0).reset_index()
                .rename(columns={"window": "n_window", "post": "n_post"}))
    info = bg_raw.drop_duplicates("taxon_id").set_index("taxon_id")
    bg["name"] = bg["taxon_id"].map(info["name"]); bg["rank"] = bg["taxon_id"].map(info["rank"])
    bg["rank_level"] = bg["taxon_id"].map(info["rank_level"]); bg["common_name"] = bg["taxon_id"].map(info["common_name"])
    bg["ancestor_ids"] = bg["taxon_id"].map(info["ancestor_ids"]).fillna("")
    bg["flag"] = bg["taxon_id"].map(flag_of).fillna(bg["taxon_id"].map(info["flag"]))
    bg["mixed_flags"] = bg["taxon_id"].map(win.groupby("taxon_id")["flag"].nunique()).fillna(1).gt(1)
    print(f"\n{len(bg)} distinct species in the pool; {bg['mixed_flags'].sum()} flagged both native and introduced within the polygon")

    # remove what the listed classes already cover
    anc = bg["ancestor_ids"].map(lambda s: {int(a) for a in s.split(";") if a})
    in_listed = bg["taxon_id"].isin(listed_member_ids) | anc.map(lambda a: bool(a & listed_member_ids))
    print(f"{in_listed.sum()} removed because a listed class covers them:", ", ".join(sorted(bg.loc[in_listed, "name"]))[:600])
    bg = bg[~in_listed].copy()
    odd = bg[bg["rank_level"] != 10]
    if len(odd):
        print(f"{len(odd)} species_counts taxa are not species/hybrid rank (dropped):", odd[["name", "rank"]].values.tolist()[:10])
        bg = bg[bg["rank_level"] == 10].copy()
    bg = bg.sort_values("n_window", ascending=False).reset_index(drop=True)
    print("\nbackground pool by flag:"); print(bg.groupby("flag").agg(species=("taxon_id", "size"), rows_in_window=("n_window", "sum")).to_string())
    for t in (C.MIN_IN_WINDOW, 100, 200, 500):
        print(f"  species with >= {t:3d} rows in window: {(bg['n_window'] >= t).sum():5d}  (and >= {C.MIN_POST_CUTOFF} post-cutoff: {((bg['n_window'] >= t) & (bg['n_post'] >= C.MIN_POST_CUTOFF)).sum()})")
    return bg


# ------------------------------------------------------------------------------------------------------------------
# 6 · Availability, thresholds, the row budget and the unseen hold-out
# ------------------------------------------------------------------------------------------------------------------
def select_classes(C, api, listed, bg):
    """Count photos per class, apply thresholds and the row budget, mark the unseen hold-out.

    Writes ``classes.csv`` and ``class_availability.csv``; returns (classes, sel) where ``sel`` is the selected classes.
    """
    counts_path = C.OUT_DIR / "meta" / "listed_counts.csv"
    lc = pd.read_csv(counts_path).set_index("class_key") if counts_path.exists() and not C.REFRESH else pd.DataFrame(columns=["n_window", "n_post"])
    lc.index = lc.index.astype(int)
    for i, cls in enumerate(listed.itertuples(index=False), 1):
        if cls.class_key in lc.index and not pd.isna(lc.loc[cls.class_key, "n_window"]):
            continue
        n_window = api.count(taxon_id=cls.class_key, hrank=cls.hrank, **C.BASE_FILTERS)
        n_post = api.count(taxon_id=cls.class_key, hrank=cls.hrank, **{**C.BASE_FILTERS, "created_d1": C.CUTOFF}) if n_window >= C.MIN_IN_WINDOW else 0
        lc.loc[cls.class_key, ["n_window", "n_post"]] = [n_window, n_post]
        if i % 25 == 0 or n_window >= C.MIN_IN_WINDOW:
            print(f"[{i:3d}/{len(listed)}] {cls.class_name:36s} window {n_window:5d}  post-cutoff {n_post:5d}")
        lc.rename_axis("class_key").to_csv(counts_path)
    lc = lc.astype(int)
    listed["n_window"] = listed["class_key"].map(lc["n_window"]); listed["n_post"] = listed["class_key"].map(lc["n_post"])

    background = pd.DataFrame(dict(
        class_key=bg["taxon_id"].astype(int), class_name=bg["name"], class_rank=bg["rank"], common_name=bg["common_name"],
        hrank="species", group=bg["flag"].map(BACKGROUND_GROUPS), invasive=False, native_in_sw=bg["flag"].eq("native"),
        label04="", invasive_votes="False", label_edits="", member_names=bg["name"], member_taxon_ids=bg["taxon_id"].astype(str),
        ecological_sources="", noxious_states="", source_list="species_counts", resolve_note="background",
        origin="background", n_window=bg["n_window"].astype(int), n_post=bg["n_post"].astype(int), ancestor_ids=bg["ancestor_ids"]))

    classes = pd.concat([listed, background], ignore_index=True)
    classes["label"] = [label_for(n, i) for n, i in zip(classes["class_name"], classes["invasive"])]
    classes["label_dir"] = classes["label"].map(label_dir)
    classes["drop_reason"] = ""
    classes.loc[classes["n_window"] < C.MIN_IN_WINDOW, "drop_reason"] = f"n_window < {C.MIN_IN_WINDOW}"
    classes.loc[(classes["drop_reason"] == "") & (classes["n_post"] < C.MIN_POST_CUTOFF), "drop_reason"] = f"n_post < {C.MIN_POST_CUTOFF}"
    passing = classes["drop_reason"] == ""

    # ---- family and genus for every class that passes the thresholds -----------------------------------------
    # listed classes: the full /taxa records from step 4 carry their ancestors; background classes: only ancestor ids
    # (from species_counts), so the ancestors' ranks are fetched once, 30 per call, into a small cached table
    anc_path = C.OUT_DIR / "meta" / "ancestor_taxa.csv"
    anc = pd.read_csv(anc_path).set_index("taxon_id") if anc_path.exists() else pd.DataFrame(columns=["name", "rank", "rank_level", "parent_id"]).rename_axis("taxon_id")
    need = set()
    for s in classes.loc[passing & (classes["origin"] == "background"), "ancestor_ids"].fillna(""):
        need.update(int(a) for a in s.split(";")[:-1] if a)
    need -= set(anc.index)
    if need:
        print(f"fetching rank and name for {len(need)} ancestor taxa ({math.ceil(len(need) / 30)} calls)")
        anc = pd.concat([anc, pd.DataFrame(api.fetch_ranks(sorted(need))).set_index("taxon_id")])
        anc.to_csv(anc_path)

    def lineage(row):
        if row["origin"] == "listed":
            pairs = [(a["rank"], a["name"]) for a in api.taxa_cache[str(row["class_key"])].get("ancestors", [])]
        else:
            ids = [int(a) for a in str(row["ancestor_ids"]).split(";")[:-1] if a]
            pairs = [(anc["rank"].get(i), anc["name"].get(i)) for i in ids if i in anc.index]
        ranks = dict(pairs)
        return pd.Series({"family": ranks.get("family", ""), "genus": ranks.get("genus", str(row["class_name"]).split()[0])})
    classes[["family", "genus"]] = ""
    classes.loc[passing, ["family", "genus"]] = classes[passing].apply(lineage, axis=1)
    print(f"taxonomy: {classes.loc[passing, 'family'].nunique()} families, {classes.loc[passing, 'genus'].nunique()} genera among the "
          f"{passing.sum()} classes that pass the thresholds; family unknown for {(classes.loc[passing, 'family'] == '').sum()}")

    # ---- per-class quotas and projected rows ----------------------------------------------------------------
    classes["q_train"], classes["q_test_lo"], classes["q_test_hi"] = C.N_TRAIN, C.TEST_LO, C.TEST_HI
    classes["q_val"] = [split_rules.test_size(n, C.VAL_LO, C.VAL_HI, C.MIN_IN_WINDOW, C.TEST_ANCHOR, C.TEST_SCALE) for n in classes["n_window"]]   # test curve
    is_bg = classes["origin"] == "background"
    if C.BACKGROUND_QUOTA is not None:
        classes.loc[is_bg, ["q_train", "q_val", "q_test_lo", "q_test_hi"]] = list(C.BACKGROUND_QUOTA)
    classes["test_target"] = [split_rules.test_size(n, lo, hi, C.MIN_IN_WINDOW, C.TEST_ANCHOR, C.TEST_SCALE)
                              for n, lo, hi in zip(classes["n_window"], classes["q_test_lo"], classes["q_test_hi"])]
    classes["projected_rows"] = classes["q_train"] + classes["q_val"] + classes["test_target"]

    # ---- row budget: background classes in priority order ---------------------------------------------------
    inv_genera = set(classes.loc[passing & classes["invasive"], "genus"])
    classes["lookalike"] = passing & is_bg & classes["genus"].isin(inv_genera)
    rand = pd.Series([np.random.default_rng([C.SEED, int(k)]).random() for k in classes["class_key"]], index=classes.index)
    bgp = classes[passing & is_bg]
    prio = rand[bgp.index].groupby(bgp["family"]).rank() + rand[bgp.index]    # round-robin over families, seeded order within a family
    prio = prio - 1e6 * bgp["lookalike"] - 1e6 * (bgp["group"] == "introduced_unlisted")   # look-alikes and introduced-unlisted first
    classes["priority"] = np.nan
    classes.loc[bgp.index, "priority"] = prio.rank().astype(int)
    if C.ROW_BUDGET is not None:
        listed_rows = int(classes.loc[passing & ~is_bg, "projected_rows"].sum())
        order = classes.loc[bgp.index].sort_values("priority")
        over = order.index[order["projected_rows"].cumsum() > C.ROW_BUDGET - listed_rows]
        classes.loc[over, "drop_reason"] = "beyond ROW_BUDGET"
        print(f"row budget {C.ROW_BUDGET:,}: listed classes project {listed_rows:,} rows, leaving {C.ROW_BUDGET - listed_rows:,} for background; "
              f"{len(bgp) - len(over)} of {len(bgp)} background classes fit")
    classes["selected"] = classes["drop_reason"] == ""

    # ---- unseen hold-out: a seeded sample of the selected background classes, spread over abundance tertiles --------
    classes["holdout_unseen"] = False
    bg_sel = classes[classes["selected"] & is_bg].sort_values(["n_window", "class_key"], ascending=[False, True])
    n_hold = min(C.UNSEEN_MAX, int(round(C.UNSEEN_FRAC * len(bg_sel))))
    if n_hold:
        rng = np.random.default_rng(C.SEED)
        tert = np.array_split(bg_sel.index.values, 3)
        per = [n_hold // 3 + (1 if r < n_hold % 3 else 0) for r in range(3)]
        chosen = np.concatenate([rng.choice(ix, size=min(k, len(ix)), replace=False) for ix, k in zip(tert, per)])
        classes.loc[chosen, "holdout_unseen"] = True

    # the existing split's classes must be kept, or its folds and images no longer apply
    prev_path = C.OUT_DIR / "classes.csv"
    if prev_path.exists():
        prev_cls = pd.read_csv(prev_path, usecols=["class_key", "selected", "holdout_unseen"])
        for col in ("selected", "holdout_unseen"):
            diff = set(classes.loc[classes[col], "class_key"]) ^ set(prev_cls.loc[prev_cls[col], "class_key"])
            assert not diff, f"{col} classes differ from the existing classes.csv ({len(diff)}: {sorted(diff)[:8]}); adjust ROW_BUDGET"
        print(f"same {int(classes['selected'].sum())} selected classes and {int(classes['holdout_unseen'].sum())} unseen hold-out classes as the existing classes.csv")
    classes["projected_images"] = np.where(~classes["selected"], 0,
                                    np.where(classes["holdout_unseen"], np.minimum(classes["n_post"], classes["q_test_hi"]), classes["projected_rows"]))
    classes = classes.sort_values(["selected", "origin", "priority", "n_window"], ascending=[False, True, True, False]).reset_index(drop=True)
    is_bg = classes["origin"] == "background"
    passing = ~classes["drop_reason"].isin([f"n_window < {C.MIN_IN_WINDOW}", f"n_post < {C.MIN_POST_CUTOFF}"])   # thresholds passed (budget aside)
    classes.to_csv(C.OUT_DIR / "classes.csv", index=False)
    classes[["class_key", "class_name", "origin", "group", "label", "label_edits", "family", "genus", "n_window", "n_post", "lookalike", "priority",
             "selected", "drop_reason", "holdout_unseen", "test_target", "projected_images"]].to_csv(C.OUT_DIR / "class_availability.csv", index=False)

    sel = classes[classes["selected"]]
    print(f"\n{len(sel)} classes selected of {len(classes)}; {sel['invasive'].sum()} invasive classes; "
          f"{sel['holdout_unseen'].sum()} background classes held out as test_unseen")
    print(sel.groupby(["origin", "group"]).agg(classes=("class_key", "size"), rows_in_window=("n_window", "sum"),
                                               projected_images=("projected_images", "sum")).to_string())
    print(f"\nprojected images to download: {int(sel['projected_images'].sum()):,}  "
          f"(invasive {int(sel.loc[sel['invasive'], 'projected_images'].sum()):,}, {NONE_LABEL} {int(sel.loc[~sel['invasive'], 'projected_images'].sum()):,})")
    bgk = sel[sel["origin"] == "background"]
    fam = bgk["family"].value_counts()
    print(f"background classes kept: {len(bgk)} species in {fam.size} of {classes.loc[passing & is_bg, 'family'].nunique()} families "
          f"(largest family {fam.index[0]} with {fam.iloc[0]}); look-alikes of invasive genera: {int(bgk['lookalike'].sum())} of "
          f"{int(classes.loc[passing, 'lookalike'].sum())}; introduced-unlisted: {int((bgk['group'] == 'introduced_unlisted').sum())} of "
          f"{int((classes.loc[passing & is_bg, 'group'] == 'introduced_unlisted').sum())}")
    print("in-window count of kept background species (quartiles):", bgk["n_window"].quantile([0, .25, .5, .75, 1]).astype(int).tolist(),
          " pool:", classes.loc[passing & is_bg, "n_window"].quantile([0, .25, .5, .75, 1]).astype(int).tolist())
    print("look-alike background species kept:", ", ".join(sorted(bgk.loc[bgk["lookalike"], "class_name"])))
    print(f"API calls so far: {api.n_requests}; metadata fetch will need about {2 * len(sel)} more")

    print("\nListed classes that pass the thresholds:")
    print(sel[sel["origin"] == "listed"][["class_name", "common_name", "class_rank", "group", "n_window", "n_post", "test_target"]].to_string(index=False))
    print("\nListed classes just under the thresholds (25 <= n_window < 50, or post-cutoff short):")
    near = classes[(classes["origin"] == "listed") & ~classes["selected"] & (classes["n_window"] >= 25)]
    print(near[["class_name", "group", "n_window", "n_post", "drop_reason"]].to_string(index=False) if len(near) else "  none")
    print("\nCal-IPC Moderate species kept as invasive:")
    print(classes[classes["member_names"].str.contains("|".join(CALIPC_KEEP))][["class_name", "n_window", "n_post", "selected"]].to_string(index=False))
    conf = classes[classes["member_names"].str.contains("|".join(CALIPC_CONFIRM))][["class_name", "n_window", "n_post", "selected"]]
    print("\nCal-IPC Moderate species expected to fall under the threshold:"); print(conf.to_string(index=False))
    if conf["selected"].any():
        print("  !! some of these passed the thresholds; decide whether they stay invasive")
    print("\nUnseen hold-out classes:")
    print(classes[classes["holdout_unseen"]][["class_name", "common_name", "group", "n_window", "n_post"]].to_string(index=False))
    return classes, sel


# ------------------------------------------------------------------------------------------------------------------
# 7 · One observation query per selected class
# ------------------------------------------------------------------------------------------------------------------
def fetch_observations(C, api, sel, complex_keys):
    """Fetch up to MAX_PER_CLASS newest observations per selected class into ``meta/species/<class_key>.csv``.

    Classes that already have a cache file are skipped (unless ``refresh``), so an interrupted run resumes.
    """
    species_dir = C.OUT_DIR / "meta" / "species"
    order = pd.concat([sel[sel["origin"] == "listed"].sort_values("class_name"),
                       sel[sel["origin"] == "background"].sort_values("n_window", ascending=False)])
    t0, fetched, log = time.time(), 0, []
    for i, cls in enumerate(order.itertuples(index=False), 1):
        path = species_dir / f"{cls.class_key}.csv"
        if path.exists() and not C.REFRESH:
            continue
        df, pages, skipped = inat_api.fetch_class(api, cls, C.BASE_FILTERS, per_page=C.PER_PAGE, max_rows=C.MAX_PER_CLASS,
                                                  complex_keys=complex_keys, merge_into_complex=MERGE_INTO_COMPLEX, image_size=C.IMAGE_SIZE)
        df.to_csv(path, index=False)
        fetched += 1
        log.append(dict(class_key=cls.class_key, class_name=cls.class_name, rows=len(df), pages=pages, skipped=len(skipped)))
        if skipped:
            print(f"    {cls.class_name}: {len(skipped)} observations did not roll up to the class, e.g. {skipped[:3]}")
        if cls.origin == "listed" or i % 25 == 0:
            print(f"[{i:4d}/{len(order)}] {cls.class_name:36s} {len(df):4d} rows ({pages} page(s))  {(time.time()-t0)/60:5.1f} min")
    print(f"fetched {fetched} classes in {(time.time()-t0)/60:.1f} min; {len(order) - fetched} were cached")
    if log:
        pd.DataFrame(log).to_csv(C.OUT_DIR / "meta" / "fetch_log.csv", index=False)


# ------------------------------------------------------------------------------------------------------------------
# 8 · Combine into metadata_all.csv and summarize
# ------------------------------------------------------------------------------------------------------------------
def combine(C, api, sel):
    """One table with every fetched row of every selected class; writes ``metadata_all.csv`` and ``species_summary.csv``.

    Photos attached to two observations are dropped (ambiguous for any class). Rows that fall in the excluded place
    (``not_in_place``) are flagged ``exclude`` so the split skips them.
    """
    species_dir = C.OUT_DIR / "meta" / "species"
    tables = {}
    for cls in sel.itertuples(index=False):
        p = species_dir / f"{cls.class_key}.csv"
        tables[cls.class_key] = pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()
    missing_cache = [k for k, t in tables.items() if not (species_dir / f"{k}.csv").exists()]
    assert not missing_cache, f"{len(missing_cache)} selected classes have no cache file; re-run step 7"
    meta = pd.concat([t for t in tables.values() if len(t)], ignore_index=True)
    assert meta["observation_id"].is_unique, "an observation landed in two classes"
    # label-derived columns come from classes.csv, not from the cache, so a label-policy change never needs a refetch
    by_key = sel.set_index("class_key")
    for col in ("label", "group", "invasive", "native_in_sw"):
        meta[col] = meta["class_key"].map(by_key[col])
    meta["image_path"] = "images/" + meta["class_key"].map(by_key["label_dir"]) + "/" + meta["observation_id"].astype(str) + "_" + meta["photo_id"].astype(str) + ".jpg"

    shared = meta[meta["photo_id"].duplicated(keep=False)]
    if len(shared):
        print(f"dropping {len(shared)} rows whose photo is shared across observations:")
        print(shared[["photo_id", "observation_id", "class_name", "observer"]].to_string(index=False))
        meta = meta[~meta["photo_id"].isin(shared["photo_id"])].reset_index(drop=True)
    assert meta["photo_id"].is_unique

    # cached rows may predate not_in_place in BASE_FILTERS: ask the API which of them that filter would exclude (200 ids per call, cached)
    chk_path = C.OUT_DIR / "meta" / "not_in_place_check.csv"
    checked = pd.read_csv(chk_path) if chk_path.exists() else pd.DataFrame({"observation_id": pd.Series(dtype=int), "excluded": pd.Series(dtype=bool)})
    for batch in inat_api.chunks(meta.loc[~meta["observation_id"].isin(checked["observation_id"]), "observation_id"].tolist(), 200):
        hits = {r["id"] for r in api.get("observations", id=",".join(map(str, batch)), place_id=C.BASE_FILTERS["not_in_place"], per_page=200, only_id="true")["results"]}
        checked = pd.concat([checked, pd.DataFrame(dict(observation_id=batch, excluded=[i in hits for i in batch]))], ignore_index=True)
        checked.to_csv(chk_path, index=False)
    meta["exclude"] = meta["observation_id"].isin(checked.loc[checked["excluded"].astype(bool), "observation_id"])
    print(f"rows in place {C.BASE_FILTERS['not_in_place']}, excluded from the split: {int(meta['exclude'].sum())}")

    meta["created_at"] = pd.to_datetime(meta["created_at"], utc=True, errors="coerce")
    meta["post_cutoff"] = meta["created_at"] >= pd.Timestamp(C.CUTOFF, tz="UTC")
    meta.to_csv(C.OUT_DIR / "metadata_all.csv", index=False)

    summary = sel.copy()
    g = meta.groupby("class_key")
    summary["n_fetched"] = summary["class_key"].map(g.size()).fillna(0).astype(int)
    summary["n_observers"] = summary["class_key"].map(g["observer_id"].nunique()).fillna(0).astype(int)
    summary["n_post_fetched"] = summary["class_key"].map(g["post_cutoff"].sum()).fillna(0).astype(int)
    summary["inat_introduced_share"] = summary["class_key"].map(g["inat_introduced"].apply(lambda s: s.fillna(False).astype(bool).mean())).round(2)
    summary["inat_native_share"] = summary["class_key"].map(g["inat_native"].apply(lambda s: s.fillna(False).astype(bool).mean())).round(2)
    summary["phenology_share"] = summary["class_key"].map(g["phenology"].apply(lambda s: s.fillna("").ne("").mean())).round(2)
    summary.to_csv(C.OUT_DIR / "species_summary.csv", index=False)

    print(f"\n{len(meta):,} observations across {meta['class_key'].nunique()} classes; "
          f"{meta['post_cutoff'].mean():.1%} uploaded on/after {C.CUTOFF}; uploads {meta['created_at'].min().date()} .. {meta['created_at'].max().date()}")
    print(meta.groupby("group").agg(classes=("class_key", "nunique"), rows=("observation_id", "size"), observers=("observer_id", "nunique")).to_string())
    print("\nphoto licenses:", meta["photo_license"].value_counts().to_dict())
    print("obscured coordinates:", int(meta["coordinates_obscured"].astype(bool).sum()), "| rows with phenology annotation:", f"{meta['phenology'].fillna('').ne('').mean():.0%}")

    lst = summary[summary["origin"] == "listed"]
    mismatch = lst[(lst["invasive"] & (lst["inat_native_share"] > 0.05)) | (lst["native_in_sw"] != (lst["inat_native_share"] >= 0.5))]
    print("\nListed classes where the list label and iNaturalist's flags disagree (review these):")
    print(mismatch[["class_name", "group", "invasive", "native_in_sw", "inat_introduced_share", "inat_native_share", "n_fetched"]].to_string(index=False) if len(mismatch) else "  none")
    print("\nClasses whose fetched rows fall short of the in-window count by more than the cap explains (roll-up filter or shared photos):")
    short = summary[(summary["n_fetched"] < np.minimum(summary["n_window"], C.MAX_PER_CLASS) - 5)]
    print(short[["class_name", "n_window", "n_fetched"]].to_string(index=False) if len(short) else "  none")
    return meta


# ------------------------------------------------------------------------------------------------------------------
# 9 · Train / val / test split
# ------------------------------------------------------------------------------------------------------------------
def split(C, meta, sel):
    """Apply ``split_rules.make_split`` and write the four final tables plus working tables in ``meta/``.

    If ``meta/split_assignment_all_rows.csv`` exists, its observer folds are reused so the sides stay as they are.
    """
    # observer folds of the existing split, if there is one (one fold per observer by construction)
    prev_path = C.OUT_DIR / "meta" / "split_assignment_all_rows.csv"
    prev_folds = None
    if prev_path.exists():
        prev_rows = pd.read_csv(prev_path, usecols=["observation_id", "fold"]).merge(meta[["observation_id", "observer_id"]], on="observation_id")
        assert (prev_rows.groupby("observer_id")["fold"].nunique() == 1).all(), "an observer has two folds in the existing split"
        prev_folds = prev_rows.drop_duplicates("observer_id").set_index("observer_id")["fold"]
        prev_used = pd.read_csv(C.OUT_DIR / "meta" / "split_rows.csv", usecols=["observation_id", "split"])

    split_df, per_class = split_rules.make_split(
        meta, sel, cutoff=C.CUTOFF, n_train=C.N_TRAIN, n_val=C.VAL_LO, test_lo=C.TEST_LO, test_hi=C.TEST_HI, test_anchor=C.TEST_ANCHOR,
        min_in_window=C.MIN_IN_WINDOW, seed=C.SEED, test_share=C.TEST_SHARE, val_share=C.VAL_SHARE, n_splits=C.N_SPLITS,
        drop_shortfall=C.DROP_SHORTFALL, scale=C.TEST_SCALE, eligibility=C.ELIGIBILITY, train_floor=C.TRAIN_FLOOR, val_floor=C.VAL_FLOOR,
        seed_tries=C.SPLIT_SEED_TRIES, folds=prev_folds)

    by_key = sel.set_index("class_key")
    split_df["family"] = split_df["class_key"].map(by_key["family"]); split_df["genus"] = split_df["class_key"].map(by_key["genus"])
    used = split_df[split_df["split"].isin(FINAL_SPLITS)].sort_values(["label", "scientific_name", "observation_id"])

    for name in FINAL_SPLITS:
        used.loc[used["split"] == name, FINAL_COLUMNS].to_csv(C.OUT_DIR / f"{name}.csv", index=False)

    # working tables (meta/): the combined assignment, plus per-class counts and every row's fold/part/split
    work_cols = ["image_path", "observation_id", "photo_id", "taxon_id", "scientific_name", "family", "label", "group", "invasive", "observer",
                 "observer_id", "latitude", "longitude", "observed_on", "created_at", "fold", "split", "photo_url"]
    used[work_cols].to_csv(C.OUT_DIR / "meta" / "split_rows.csv", index=False)
    per_class.to_csv(C.OUT_DIR / "meta" / "split_per_species_counts.csv", index=False)
    split_df[["observation_id", "class_key", "eligible", "fold", "part", "split"]].to_csv(C.OUT_DIR / "meta" / "split_assignment_all_rows.csv", index=False)

    print("observer folds reused from the existing split:" if prev_folds is not None else "fold draws tried:")
    print(per_class.attrs["seed_scores"].to_string(index=False))
    print(f"rows excluded (not_in_place): {int((split_df['split'] == 'excluded').sum())}")
    print(f"\nELIGIBILITY = {C.ELIGIBILITY!r}: rows from observers who may serve val/test: {split_df['eligible'].mean():.1%} of fetched rows; "
          f"post-cutoff rows: {split_df['post'].mean():.1%}; parts of those rows:",
          split_df.loc[split_df["eligible"], "part"].value_counts(normalize=True).round(3).to_dict())
    print(f"\nrows per split (total {len(used):,}; ROW_BUDGET {C.ROW_BUDGET}):"); print(used["split"].value_counts().to_string())
    if prev_folds is not None:
        same = used[["observation_id", "split"]].merge(prev_used, on="observation_id", suffixes=("", "_prev"))
        print(f"\nvs the existing split: {len(same):,} of {len(used):,} rows kept ({int((same['split'] == same['split_prev']).sum()):,} in the same split); "
              f"{len(used) - len(same):,} rows new (images to download); {int((~prev_used['observation_id'].isin(used['observation_id'])).sum()):,} rows no longer used")
        print(pd.crosstab(same["split_prev"], same["split"]).to_string())
    kept_bg = used[~used["invasive"] & (used["split"] != "test_unseen")]
    print(f"{NONE_LABEL} species in train/val/test: {kept_bg['class_key'].nunique()} in {kept_bg['family'].nunique()} families; "
          f"largest families by rows:", kept_bg.groupby("family")["class_key"].nunique().nlargest(6).to_dict())
    print("\nrows per split by label group:")
    print(used.pivot_table(index="group", columns="split", values="observation_id", aggfunc="count", fill_value=0).to_string())
    print(f"\nclasses: {len(per_class)} selected -> {(per_class['status'] == 'ok').sum()} filled, "
          f"{per_class['status'].str.startswith('short').sum()} short but kept, {per_class['status'].str.startswith('dropped').sum()} dropped")
    print(f"invasive classes in the final set: {used.loc[used['invasive'], 'class_key'].nunique()}; "
          f"{NONE_LABEL} species: {used.loc[~used['invasive'] & (used['split'] != 'test_unseen'), 'class_key'].nunique()}; "
          f"unseen species: {used.loc[used['split'] == 'test_unseen', 'class_key'].nunique()}")
    print("\ninvasive share of rows  train:", round(used.loc[used['split'] == 'train', 'invasive'].mean(), 3),
          " val:", round(used.loc[used['split'] == 'val', 'invasive'].mean(), 3),
          " test:", round(used.loc[used['split'] == 'test', 'invasive'].mean(), 3))
    print("\nper-class test rows (min / median / max):", per_class.loc[per_class["test"] > 0, "test"].describe()[["min", "50%", "max"]].to_dict())
    resc = per_class[per_class["rescue"] != ""]
    print(f"\nClasses filled by moving still-unused observers into the short part ({len(resc)}):")
    print(resc[["class_name", "group", "train", "val", "test", "status", "rescue"]].to_string(index=False) if len(resc) else "  none")
    print("\nClasses not filled:")
    nf = per_class[per_class["status"] != "ok"]
    print(nf[["class_name", "group", "n_window", "n_fetched", "n_eligible", "train", "val", "test", "test_unseen", "status"]].to_string(index=False) if len(nf) else "  none")
    print("\nInvasive classes in the final set:")
    print(per_class[per_class["invasive"] & (per_class["status"] != "dropped")].sort_values("n_window", ascending=False)
          [["class_name", "n_window", "train", "val", "test", "pool", "status"]].to_string(index=False))
    return used


# ------------------------------------------------------------------------------------------------------------------
# 11 · Checks
# ------------------------------------------------------------------------------------------------------------------
def check_outputs(C):
    """Re-read the written tables and assert the split rules hold (one side per observer, no pre-cutoff val/test rows, ...)."""
    final = {name: pd.read_csv(C.OUT_DIR / f"{name}.csv") for name in FINAL_SPLITS}   # default read: labels must survive
    for name, t in final.items():
        assert t["label"].notna().all() and t["image_path"].is_unique and not {"split", "fold", "part", "rescue"} & set(t.columns)
        print(f"{name + '.csv':16s} {len(t):6,d} rows  {t['taxon_id'].nunique():4d} classes  {t['label'].nunique():3d} labels  "
              f"invasive share {t['invasive'].mean():.3f}")
    chk = pd.read_csv(C.OUT_DIR / "meta" / "split_rows.csv", low_memory=False)
    assert len(chk) == sum(len(t) for t in final.values()) and (C.ROW_BUDGET is None or len(chk) <= C.ROW_BUDGET)
    chk["created_at"] = pd.to_datetime(chk["created_at"], utc=True)
    print("\nall rows:", len(chk), "| classes:", chk["taxon_id"].nunique(), "| labels:", chk["label"].nunique(), f"(incl. {NONE_LABEL})")
    assert chk["label"].notna().all() and chk["label"].eq(NONE_LABEL).any()
    print("duplicate photo ids:", chk["photo_id"].duplicated().sum(), "| duplicate observation ids:", chk["observation_id"].duplicated().sum())
    side = chk["split"].replace({"test_unseen": "test"})
    print("observers in more than one side:", int((chk.assign(side=side).groupby("observer_id")["side"].nunique() > 1).sum()))
    print("pre-cutoff rows in val/test:", int((chk.loc[chk["split"].isin(["val", "test", "test_unseen"]), "created_at"] < pd.Timestamp(C.CUTOFF, tz="UTC")).sum()))
    excl = pd.read_csv(C.OUT_DIR / "metadata_all.csv", usecols=["observation_id", "exclude"])
    print(f"excluded rows in the tables: {int(chk['observation_id'].isin(excl.loc[excl['exclude'], 'observation_id']).sum())} (of {int(excl['exclude'].sum())} excluded)")
    assert not chk["observation_id"].isin(excl.loc[excl["exclude"], "observation_id"]).any()
    lst = chk[chk["group"].isin(["invasive", "listed_not_invasive", "native_listed"])].groupby(["taxon_id", "split"]).size().unstack(fill_value=0)
    print(f"listed classes: val rows min / median / max {int(lst['val'].min())} / {lst['val'].median():.0f} / {int(lst['val'].max())}; "
          f"val == test for {int((lst['val'] == lst['test']).sum())} of {len(lst)} classes; "
          f"Spearman corr(val rows, test rows) = {lst[['val', 'test']].corr('spearman').iloc[0, 1]:.2f}; "
          f"val/test ratio median {(lst['val'] / lst['test']).median():.2f}")
    print("\nobservation months (all selected rows):", chk["observed_on"].astype(str).str[5:7].value_counts().sort_index().to_dict())
    print("\nrows and classes by split and invasive label:")
    print(chk.groupby(["split", "invasive"]).agg(rows=("observation_id", "size"), classes=("taxon_id", "nunique")).to_string())


# ------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--out-dir", help="override paths.out_dir")
    ap.add_argument("--refresh", action="store_true", help="re-query classes and counts that already have a cache file")
    ap.add_argument("--stop-after-availability", action="store_true", help="stop after step 6, before fetching observations")
    args = ap.parse_args(argv)

    C = load_config(args.config)
    if args.out_dir:
        C.OUT_DIR = Path(args.out_dir)
    C.REFRESH = C.REFRESH or args.refresh
    C.STOP_AFTER_AVAILABILITY = C.STOP_AFTER_AVAILABILITY or args.stop_after_availability
    assert (C.RANKED_DIR / "ranked_desert_sw.csv").exists(), f"ranked lists not found in {C.RANKED_DIR}"
    save_run_config(C)
    print("outputs to", C.OUT_DIR)

    api = inat_api.INat(C.USER_AGENT, pause=C.REQUEST_PAUSE, cache_dir=C.OUT_DIR / "meta")       # step 2
    print("place", C.PLACE_ID, "->", api.place_name(C.PLACE_ID))

    print("\n== 3 · listed candidates"); cand = listed_candidates(C)
    print("\n== 4 · resolve"); listed, complex_keys, listed_member_ids = resolve_listed(api, cand)
    print("\n== 5 · background pool"); bg = background_pool(C, api, listed_member_ids)
    print("\n== 6 · availability"); classes, sel = select_classes(C, api, listed, bg)
    if C.STOP_AFTER_AVAILABILITY:
        print("\nstop_after_availability is set: review classes.csv / class_availability.csv, then re-run without it")
        return
    print("\n== 7 · fetch observations"); fetch_observations(C, api, sel, complex_keys)
    print("\n== 8 · combine"); meta = combine(C, api, sel)
    print("\n== 9 · split"); split(C, meta, sel)
    print("\n== 11 · checks"); check_outputs(C)


if __name__ == "__main__":
    main()
