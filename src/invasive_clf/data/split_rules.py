"""Train / val / test split for the final dataset. This includes a test_unseen set consisting of observations 
of species the model has not seen at all. Thus the expected prediction is `not_invasive` for these species.
Evaluation on test_unseen is separate during the same final phase as on test. 

Rules:
  1. val, test and test_unseen contain only rows uploaded on or after CUTOFF, so no pretrained model
     can have seen them;
  2. grouped by photographer: an observer's rows all go to one side;
  3. observers supplying val/test rows must be able to have *all* selected rows on/after CUTOFF
     (2025-07-01 in configs/dataset.yaml); everyone else is train-only (relaxed by ``eligibility="row"``, below);
  4. eligible observers are assigned to train/val/test folds with StratifiedGroupKFold (stratified by class).
  5. (new for the final sample): per class exactly N_TRAIN train rows, N_VAL val rows and a log-scaled 
     number of test rows between TEST_LO and TEST_HI, subsampled from the fold assignment. Per-class quotas 
     can be overridden with columns q_train, q_val, q_test_lo, q_test_hi on `classes` (e.g. smaller quotas 
     for background classes, or a log-scaled val quota per class).
  6. (new for the final sample): `folds=` reuses the observer -> fold assignment of a previous run instead of drawing
     new folds, and rows flagged True in an optional boolean column `exclude` on `meta` are never picked but keep
     their position in the per-class draw, so excluding a few rows does not re-draw the rest of the class.

`eligibility` selects how rule 3 is applied:
  "observer": The rule as written in the previous iteration. Only observers with no pre-cutoff upload in 
              the dataset can be on the val/test side. Cheap when nearly every row is post-cutoff (the v1 
              sample), very restrictive when the window starts before the cutoff.
  "row":      every observer is grouped into one side; on the val/test side only that observer's
              post-cutoff rows are used and their pre-cutoff rows are discarded (never moved to train, so
              rule 2 holds). Rule 1 holds by construction.
Because observers are grouped globally, exact val/test counts cannot be guaranteed for every class;
classes that cannot be filled are reported and (by default) dropped. A held-out set of background
classes goes entirely to `test_unseen` (test-fold rows only), never to train or val.
"""
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def test_size(n, lo, hi, n_min, anchor, scale="log"):
    """Test rows for a class with n in-window rows: lo at n_min, hi at `anchor` rows or more."""
    if n <= n_min or anchor <= n_min:
        return lo
    if scale == "log":
        frac = math.log(n / n_min) / math.log(anchor / n_min)
    elif scale == "linear":
        frac = (n - n_min) / (anchor - n_min)
    else:
        raise ValueError(scale)
    return int(round(lo + (hi - lo) * max(0.0, min(1.0, frac))))



USED_SPLITS = ("train", "val", "test", "test_unseen")


def rescue_class(dd, idx, q_train, q_val, target, floor, vfloor, q_lo):
    """Try to fill a class that would otherwise be dropped, without breaking the one-observer-one-side rule.

    An observer none of whose rows (in any class) has been assigned yet can still be moved to another part
    as a whole. For each part the class is short of (val, then train, then test) the smallest such observers
    of this class are moved into that part until the quota is met, as long as the part they leave keeps enough
    rows for this class. Edits dd["part"] in place; returns a note or "".
    """
    notes = []
    mins = {"train": floor, "val": vfloor, "test": q_lo}
    quota = {"train": q_train, "val": q_val, "test": target}
    for side in ("val", "train", "test"):
        g = dd.loc[idx]
        def avail(part, g):
            if part == "train":
                return int(g["part"].isin(["train", "ineligible"]).sum())
            return int(((g["part"] == part) & g["post"]).sum())
        have = avail(side, g)
        if have >= mins[side]:
            continue                                   # not fatal on this side
        need = quota[side] - have
        used_obs = set(dd.loc[dd["split"].isin(USED_SPLITS), "observer_id"])
        cand = g[~g["observer_id"].isin(used_obs) & (g["part"] != side) & (g["post"] | (side == "train"))]
        moved = []
        for obs, r in cand.groupby("observer_id").size().sort_values().items():
            src = g.loc[g["observer_id"] == obs, "part"].iloc[0]
            src = "train" if src == "ineligible" else src
            if avail(src, g) - r < mins[src]:
                continue                               # would starve the part it leaves
            dd.loc[dd["observer_id"] == obs, "part"] = side
            g = dd.loc[idx]
            moved.append(obs)
            need -= r
            if need <= 0:
                break
        if moved:
            notes.append(f"{side} +{len(moved)} unused observer(s)")
    return "; ".join(notes)

def make_split(meta, classes, *, cutoff, n_train, n_val, test_lo, test_hi, test_anchor, min_in_window, seed,
               test_share=0.55, val_share=0.15, n_splits=20, drop_shortfall=True, scale="log", eligibility="observer",
               train_floor=None, val_floor=None, seed_tries=1, rescue=True, folds=None):
    """Assign every row of `meta` to train / val / test / test_unseen / pool / dropped.

    meta: one row per observation for the selected classes (class_key, observer_id, created_at, observation_id).
    classes: the selected classes (class_key, class_name, label, group, invasive, n_window, holdout_unseen).
    train_floor / val_floor: a class short of n_train train rows (n_val val rows) is still kept, with status
    "short", if it has at least this many; None means the quota itself, so any shortfall drops the class when
    drop_shortfall is set. Test rows below q_test_lo always drop the class.
    seed_tries: number of fold draws (random states seed, seed+1, ...) to try; the draw that drops the fewest
    invasive classes is kept. per_class.attrs["seed_scores"] lists every draw.
    rescue: when a class would be dropped, move observers none of whose rows are assigned yet into the short
    part (see rescue_class); the per-class table notes it in column "rescue".
    folds: optional Series observer_id -> fold from a previous run (its meta/split_assignment_all_rows.csv joined
    to observer ids). The fold draw is skipped (seed_tries is ignored, fold_seed is reported as -1); observers the
    previous run did not have get one random fold each. Use it to re-split with new quotas and keep the sides.
    meta["exclude"] (optional, boolean): rows never assigned to any split (split = "excluded"); they keep their
    position in the per-class draw, so excluding them changes only the rows that replace them.
    Returns (meta with columns post, eligible, fold, part, split; per-class table).
    """
    def q(info, col, default):
        v = info.get(col, default)
        return default if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)
    if train_floor is None:
        train_floor = n_train
    if val_floor is None:
        val_floor = n_val
    d = meta.copy()
    d["created_at"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
    d = d.dropna(subset=["created_at"]).reset_index(drop=True)
    cut = pd.Timestamp(cutoff, tz="UTC")
    d["post"] = d["created_at"] >= cut
    if eligibility == "observer":
        d["eligible"] = d["observer_id"].map(d.groupby("observer_id")["post"].all())
    elif eligibility == "row":
        d["eligible"] = True          # every observer is grouped; only post-cutoff rows may serve val/test
    else:
        raise ValueError(eligibility)

    # ---- eligible observers -> folds -> parts (test / val / train) ---------------------------------
    el = d[d["eligible"]]
    n_splits = min(n_splits, el["observer_id"].nunique())
    assert n_splits >= 3, "too few eligible observers to split"
    k_test = int(round(test_share * n_splits))
    k_val = max(1, int(round(val_share * n_splits)))
    cls_info = classes.set_index("class_key")

    def assign(fold_seed):
        """Fold the eligible observers with one random state, then subsample every class."""
        dd = d.copy()
        if folds is not None:                                      # re-split: the previous run's observer folds
            fold = el["observer_id"].map(folds)
            new = fold.isna()
            if new.any():                                          # observers the previous run did not have: one random fold each
                new_obs = el.loc[new, "observer_id"].unique()
                draw = np.random.default_rng(fold_seed).integers(0, n_splits, len(new_obs))
                fold[new] = el.loc[new, "observer_id"].map(dict(zip(new_obs, draw)))
            fold = fold.astype(int)
        else:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
            fold = pd.Series(-1, index=el.index)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # "least populated class" warnings: expected for 50-row classes
                for k, (_, idx) in enumerate(sgkf.split(el, y=el["class_key"].astype(str), groups=el["observer_id"])):
                    fold.iloc[idx] = k
        dd["fold"] = -1
        dd.loc[fold.index, "fold"] = fold.values
        dd["part"] = "ineligible"                                  # observer has a pre-cutoff upload: train only
        dd.loc[(dd["fold"] >= 0) & (dd["fold"] < k_test), "part"] = "test"
        dd.loc[(dd["fold"] >= k_test) & (dd["fold"] < k_test + k_val), "part"] = "val"
        dd.loc[dd["fold"] >= k_test + k_val, "part"] = "train"

        # ---- per class: subsample fixed counts (the per-class draw uses the base seed, so only the folds vary) ----
        # invasive classes first, so the rescue step below finds the most still-unused observers for them
        dd["split"] = "pool"
        excluded = dd["exclude"].fillna(False).astype(bool) if "exclude" in dd else pd.Series(False, index=dd.index)
        dd.loc[excluded, "split"] = "excluded"
        rows = []
        groups = dict(iter(dd.groupby("class_key", sort=True)))
        order = sorted(groups, key=lambda k: (not bool(cls_info.loc[k, "invasive"]), k))
        for key in order:
            g = dd.loc[groups[key].index]          # re-read: a rescue in an earlier class may have moved observers
            info = cls_info.loc[key]
            rng = np.random.default_rng([seed, int(key)])
            g = g.sort_values("observation_id")
            g = g.iloc[rng.permutation(len(g))]
            n_excluded = int(excluded[g.index].sum())
            g = g[~excluded[g.index]]                 # excluded rows keep their place in the draw but are never picked
            holdout = bool(info.get("holdout_unseen", False))
            n_window = int(info["n_window"])
            q_train, q_val = q(info, "q_train", n_train), q(info, "q_val", n_val)
            q_lo, q_hi = q(info, "q_test_lo", test_lo), q(info, "q_test_hi", test_hi)
            floor, vfloor = min(train_floor, q_train), min(val_floor, q_val)
            target = test_size(n_window, q_lo, q_hi, min_in_window, test_anchor, scale)
            rec = dict(class_key=key, class_name=info["class_name"], label=info["label"], group=info["group"],
                       invasive=bool(info["invasive"]), holdout_unseen=holdout, n_window=n_window, n_fetched=len(g),
                       n_post=int(g["post"].sum()), n_eligible=int(g["eligible"].sum()), n_excluded=n_excluded,
                       train=0, val=0, test=0, test_unseen=0, pool=0, test_target=target, status="ok", rescue="")
            if holdout:
                te = g[(g["part"] == "test") & g["post"]].head(q_hi)
                dd.loc[te.index, "split"] = "test_unseen"
                rec["test_unseen"] = len(te)
                if len(te) < q_lo:
                    rec["status"] = f"short: test_unseen {len(te)}/{q_lo}"
            else:
                def pick(g):
                    tr = g[g["part"].isin(["train", "ineligible"])].head(q_train)
                    va = g[(g["part"] == "val") & g["post"]].head(q_val)
                    te = g[(g["part"] == "test") & g["post"]].head(target)
                    short, fatal = [], False
                    if len(tr) < q_train: short.append(f"train {len(tr)}/{q_train}"); fatal |= len(tr) < floor
                    if len(va) < q_val:   short.append(f"val {len(va)}/{q_val}"); fatal |= len(va) < vfloor
                    if len(te) < q_lo:    short.append(f"test {len(te)}/{q_lo}"); fatal = True
                    return tr, va, te, short, fatal
                tr, va, te, short, fatal = pick(g)
                if fatal and rescue:
                    note = rescue_class(dd, g.index, q_train, q_val, target, floor, vfloor, q_lo)
                    if note:
                        rec["rescue"] = note
                        g = dd.loc[g.index]
                        tr, va, te, short, fatal = pick(g)
                if fatal and drop_shortfall:
                    dd.loc[g.index, "split"] = "dropped"
                    rec["status"] = "dropped: " + "; ".join(short)
                else:
                    dd.loc[tr.index, "split"] = "train"; dd.loc[va.index, "split"] = "val"; dd.loc[te.index, "split"] = "test"
                    rec.update(train=len(tr), val=len(va), test=len(te))
                    if short:
                        rec["status"] = "short: " + "; ".join(short)
            rec["pool"] = int((dd.loc[g.index, "split"] == "pool").sum())
            rows.append(rec)
        pc = pd.DataFrame(rows)
        pc["fold_seed"] = -1 if folds is not None else fold_seed
        return dd, pc

    # Which observers land in the val folds is a random draw; a class whose rows come from few observers can miss
    # its val quota in one draw and fill it in the next. Try `seed_tries` fold draws (seed, seed+1, ...) and keep
    # the one that drops the fewest invasive classes, then the fewest classes, then the most assigned rows.
    scores = []
    best = None
    tries = [seed] if folds is not None else [seed + t for t in range(max(1, seed_tries))]   # reused folds: nothing to draw
    for fs in tries:
        dd, pc = assign(fs)
        dropped = pc["status"].str.startswith("dropped")
        score = (int((pc["invasive"] & dropped).sum()), int(dropped.sum()),
                 -int(dd["split"].isin(["train", "val", "test", "test_unseen"]).sum()))
        scores.append(dict(fold_seed=-1 if folds is not None else fs, invasive_dropped=score[0], classes_dropped=score[1], rows_assigned=-score[2]))
        if best is None or score < best[0]:
            best = (score, dd, pc)
    _, d, per_class = best
    per_class.attrs["seed_scores"] = pd.DataFrame(scores)

    # ---- sanity checks ----------------------------------------------------------------------
    used = d[d["split"].isin(["train", "val", "test", "test_unseen"])]
    side = used["split"].replace({"test_unseen": "test"})
    assert used.assign(side=side).groupby("observer_id")["side"].nunique().max() == 1, "an observer appears in more than one split"
    assert used.loc[used["split"].isin(["val", "test", "test_unseen"]), "post"].all(), "pre-cutoff row in val/test"
    return d, per_class
