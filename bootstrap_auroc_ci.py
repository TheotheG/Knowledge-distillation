"""
bootstrap_auroc_ci.py
---------------------------------------------------------------
95% bootstrap confidence intervals στα AUROC + PAIRED bootstrap test για τη
διαφορά ΔAUROC μεταξύ μοντέλων (π.χ. logitKD − baseline, ή featureKD − baseline).

--------------------------------------------------------------------------
ΓΙΑΤΙ PAIRED (και όχι απλή σύγκριση δύο ανεξάρτητων CIs):
    Και τα δύο μοντέλα αξιολογούνται στις ΙΔΙΕΣ 234 εικόνες. Η μη-επικάλυψη δύο
    περιθωριακών CIs είναι ΙΚΑΝΗ αλλά ΟΧΙ ΑΝΑΓΚΑΙΑ συνθήκη σημαντικότητας:
        - δεν επικαλύπτονται  -> σίγουρα σημαντικό
        - επικαλύπτονται      -> ΔΕΝ συμπεραίνεις τίποτα (μπορεί να είναι σημαντικό!)
    Το σωστό/ισχυρό τεστ είναι το paired bootstrap πάνω στη ΔΙΑΦΟΡΑ: με το ΙΔΙΟ
    resample για τα δύο μοντέλα, υπολογίζεις ΔAUROC και ρωτάς αν το 95% CI της
    διαφοράς αποκλείει το 0. Έτσι ακυρώνεται η κοινή διακύμανση (ίδιες εικόνες)
    και το τεστ γίνεται πολύ πιο σφιχτό/τίμιο.
--------------------------------------------------------------------------
ΕΠΕΚΤΑΣΙΜΟΤΗΤΑ (feature-KD): φτιάχνεις preds_featureKD.npz με το dump_preds.py και
το βάζεις στη λίστα --preds. Μηδέν αλλαγή κώδικα — παίρνεις αυτόματα CI του
featureKD και paired test featureKD−baseline (και, με --all-pairs, featureKD−logitKD).

Χρήση:
    # τώρα (baseline vs logitKD)
    python bootstrap_auroc_ci.py \
        --preds baseline=preds_baseline.npz logitKD=preds_logitKD.npz \
        --reference baseline --plot

    # όταν έρθει το feature-KD
    python bootstrap_auroc_ci.py \
        --preds baseline=preds_baseline.npz logitKD=preds_logitKD.npz featureKD=preds_featureKD.npz \
        --reference baseline --all-pairs --plot

Επιλογές:
    --n-boot 10000      αριθμός resamples (default 10000)
    --seed 42
    --by-patient        cluster bootstrap στο επίπεδο ασθενή (αντί για εικόνα)
    --all-classes       τύπωσε CI και για τις 14 (default: 5 competition + macros)
    --all-pairs         paired test για ΟΛΑ τα ζεύγη (όχι μόνο έναντι reference)
    --plot              forest plot της ΔAUROC (η «απόδειξη» για το paper) -> PNG
    --out-prefix        prefix για csv/png (default: bootstrap)
"""
import os
import argparse
import itertools

import numpy as np
from scipy.stats import rankdata

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


# ============================ AUROC (γρήγορο, tie-aware) ============================
def fast_auc(y_true, y_score):
    """AUROC μέσω Mann-Whitney U. Ταυτίζεται με sklearn.roc_auc_score (και στα ties).
    Επιστρέφει np.nan αν λείπει μία από τις δύο κλάσεις (μη ορίσιμο)."""
    y_true = np.asarray(y_true)
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(y_score)                       # average ranks -> σωστό με ties
    sum_pos = r[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ============================ I/O ============================
def load_preds(spec):
    """spec: 'tag=path.npz' -> (tag, dict με probs/labels/patient_ids)."""
    if "=" not in spec:
        raise ValueError(f"Περίμενα 'tag=path.npz', πήρα: {spec}")
    tag, path = spec.split("=", 1)
    d = np.load(os.path.expanduser(path), allow_pickle=True)
    return tag, {
        "probs": d["probs"].astype(np.float64),
        "labels": d["labels"].astype(np.float64),
        "patient_ids": d["patient_ids"] if "patient_ids" in d.files else None,
    }


# ============================ BOOTSTRAP CORE ============================
def make_resample_indices(labels, patient_ids, n_boot, rng, by_patient):
    """Επιστρέφει λίστα από index-arrays (ένα ανά resample). ΚΟΙΝΑ για όλα τα μοντέλα
    -> εξασφαλίζει το pairing. by_patient=True: cluster bootstrap (resample ασθενείς)."""
    n = labels.shape[0]
    if not by_patient:
        # image-level: n δείγματα με επανάθεση, ίδιο μέγεθος
        idx = rng.integers(0, n, size=(n_boot, n))
        return [idx[b] for b in range(n_boot)]
    # patient-level cluster bootstrap
    if patient_ids is None:
        raise ValueError("--by-patient ζητήθηκε αλλά δεν υπάρχουν patient_ids στο npz.")
    uniq = np.unique(patient_ids)
    rows_by_pid = {p: np.where(patient_ids == p)[0] for p in uniq}
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([rows_by_pid[p] for p in chosen]))
    return out


def auc_matrix(probs, labels, resamples, class_idx):
    """Για ΕΝΑ μοντέλο: πίνακας [n_boot, n_classes] με AUROC ανά resample & κλάση.
    Χρησιμοποιεί τα ΚΟΙΝΑ resamples (paired)."""
    B = len(resamples)
    C = len(class_idx)
    out = np.full((B, C), np.nan)
    for b, idx in enumerate(resamples):
        yb = labels[idx]
        pb = probs[idx]
        for jc, j in enumerate(class_idx):
            out[b, jc] = fast_auc(yb[:, j], pb[:, j])
    return out


def macro(auc_arr, cols):
    """nan-aware macro πάνω σε επιλεγμένες στήλες. auc_arr: [B, C]. Επιστρέφει [B]."""
    return np.nanmean(auc_arr[:, cols], axis=1)


def ci(vals, lo=2.5, hi=97.5):
    v = vals[~np.isnan(vals)]
    if v.size == 0:
        return (np.nan, np.nan)
    return (np.percentile(v, lo), np.percentile(v, hi))


def boot_pvalue(diffs):
    """Two-sided bootstrap ASL για τη διαφορά: 2*min(P(Δ*<=0), P(Δ*>=0)).
    Επιστρέφει (p, frac_gt0, n_valid)."""
    d = diffs[~np.isnan(diffs)]
    if d.size == 0:
        return (np.nan, np.nan, 0)
    p_le = np.mean(d <= 0.0)
    p_ge = np.mean(d >= 0.0)
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return (p, float(np.mean(d > 0.0)), int(d.size))


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="ζευγάρια tag=preds.npz (>=1). π.χ. baseline=preds_baseline.npz")
    ap.add_argument("--reference", default=None,
                    help="tag αναφοράς για τα paired tests (default: το 1ο)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--by-patient", action="store_true")
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out-prefix", default="bootstrap")
    ap.add_argument("--plot-out", default=None,
                help="ακριβές όνομα PNG για το forest plot (υπερισχύει του --out-prefix)")
    args = ap.parse_args()

    # ---- φόρτωσε όλα τα μοντέλα ----
    models = {}
    order = []
    for spec in args.preds:
        tag, d = load_preds(spec)
        models[tag] = d
        order.append(tag)
    reference = args.reference or order[0]
    if reference not in models:
        raise ValueError(f"reference '{reference}' δεν είναι στα preds: {order}")

    # ---- έλεγχος: ΙΔΙΑ labels/σειρά σε όλα (απαραίτητο για paired) ----
    ref_labels = models[reference]["labels"]
    N = ref_labels.shape[0]
    for tag in order:
        L = models[tag]["labels"]
        if L.shape != ref_labels.shape or not np.array_equal(L, ref_labels):
            raise ValueError(
                f"Τα labels του '{tag}' δεν ταιριάζουν με του '{reference}'. "
                "Τα npz πρέπει να προέρχονται από το ΙΔΙΟ valid set, ίδια σειρά.")
    patient_ids = models[reference]["patient_ids"]
    labels = ref_labels

    # ---- ποιες κλάσεις τυπώνουμε per-class ----
    # macro(14): μόνο κλάσεις με δύο labels στο full sample (π.χ. Fracture=0 θετικά -> έξω)
    valid14 = [j for j in range(len(PATHOLOGIES))
               if len(np.unique(labels[:, j])) == 2]
    comp_idx = [PATHOLOGIES.index(c) for c in COMPETITION]
    per_class_idx = list(range(len(PATHOLOGIES))) if args.all_classes else comp_idx

    print(f"N εικόνες = {N} | resamples = {args.n_boot} | "
          f"{'PATIENT-level' if args.by_patient else 'image-level'} bootstrap | seed={args.seed}")
    excluded = [PATHOLOGIES[j] for j in range(len(PATHOLOGIES)) if j not in valid14]
    if excluded:
        print(f"Εκτός macro(14) (μονή κλάση στο valid): {excluded}")
    print()

    # ---- ΚΟΙΝΑ resamples (paired) + AUROC matrices ----
    rng = np.random.default_rng(args.seed)
    all_class_idx = list(range(len(PATHOLOGIES)))
    resamples = make_resample_indices(labels, patient_ids, args.n_boot, rng, args.by_patient)

    aucs_boot = {}   # tag -> [B, 14]
    aucs_point = {}  # tag -> [14] (full sample)
    for tag in order:
        probs = models[tag]["probs"]
        aucs_boot[tag] = auc_matrix(probs, labels, resamples, all_class_idx)
        aucs_point[tag] = np.array([fast_auc(labels[:, j], probs[:, j])
                                    for j in all_class_idx])

    # helper: macro columns
    def macros(tag):
        b = aucs_boot[tag]
        p = aucs_point[tag]
        return {
            "MACRO(5 competition)": (np.nanmean(p[comp_idx]), macro(b, comp_idx)),
            "MACRO(14)": (np.nanmean(p[valid14]), macro(b, valid14)),
        }

    # ======================= 1) ΠΕΡΙΘΩΡΙΑΚΑ CIs ανά μοντέλο =======================
    print("=" * 78)
    print("1) PER-MODEL AUROC με 95% bootstrap CI  (point [lo, hi])")
    print("=" * 78)
    for tag in order:
        print(f"\n[{tag}]")
        for j in per_class_idx:
            name = PATHOLOGIES[j]
            lo, hi = ci(aucs_boot[tag][:, j])
            pt = aucs_point[tag][j]
            flag = "" if not np.isnan(pt) else "  (μη ορίσιμο)"
            print(f"  {name:28s} {pt:6.4f}  [{lo:6.4f}, {hi:6.4f}]{flag}")
        for mname, (mpt, mb) in macros(tag).items():
            lo, hi = ci(mb)
            print(f"  {mname:28s} {mpt:6.4f}  [{lo:6.4f}, {hi:6.4f}]")

    # ======================= 2) PAIRED ΔAUROC tests =======================
    if args.all_pairs:
        pairs = [(a, b) for a, b in itertools.combinations(order, 2)]
    else:
        pairs = [(reference, t) for t in order if t != reference]

    print("\n" + "=" * 78)
    print("2) PAIRED ΔAUROC = (B − A), ίδιο resample -> 95% CI + bootstrap p")
    print("   Σημαντικό (α=0.05)  <=>  το 95% CI της Δ ΔΕΝ περιέχει το 0.")
    print("=" * 78)

    comparison_rows = []  # για csv/plot
    for A, B in pairs:
        print(f"\n[{B} − {A}]   (θετικό = το {B} καλύτερο)")
        # per-class Δ (paired): ίδιο resample, διαφορά AUROC
        for j in per_class_idx:
            name = PATHOLOGIES[j]
            dpt = aucs_point[B][j] - aucs_point[A][j]
            dboot = aucs_boot[B][:, j] - aucs_boot[A][:, j]
            lo, hi = ci(dboot)
            p, fgt, nval = boot_pvalue(dboot)
            sig = "✓" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "·"
            print(f"  {name:22s} Δ={dpt:+.4f}  CI[{lo:+.4f}, {hi:+.4f}]  "
                  f"p={p:.4f}  P(Δ>0)={fgt:.2f}  {sig}")
            comparison_rows.append((f"{B}-{A}", name, dpt, lo, hi, p, fgt))
        # macros
        for mname, cols in [("MACRO(5 competition)", comp_idx), ("MACRO(14)", valid14)]:
            dpt = np.nanmean(aucs_point[B][cols]) - np.nanmean(aucs_point[A][cols])
            dboot = macro(aucs_boot[B], cols) - macro(aucs_boot[A], cols)
            lo, hi = ci(dboot)
            p, fgt, nval = boot_pvalue(dboot)
            sig = "✓ ΣΗΜΑΝΤΙΚΟ" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "· μη σημαντικό"
            pstr = f"p={p:.4f}" if p >= 1.0 / max(nval, 1) else f"p<{1.0/max(nval,1):.1e}"
            print(f"  {mname:22s} Δ={dpt:+.4f}  CI[{lo:+.4f}, {hi:+.4f}]  "
                  f"{pstr}  P(Δ>0)={fgt:.2f}  -> {sig}")
            comparison_rows.append((f"{B}-{A}", mname, dpt, lo, hi, p, fgt))

    # ======================= 3) Έλεγχος επικάλυψης περιθωριακών CIs (για αναφορά) =======================
    print("\n" + "=" * 78)
    print("3) Έλεγχος ΕΠΙΚΑΛΥΨΗΣ περιθωριακών CIs στο MACRO(5) — (το ΑΔΥΝΑΜΟ τεστ)")
    print("   Θύμισε: μη-επικάλυψη => σημαντικό· επικάλυψη => αναποφάσιστο. Το κριτήριο")
    print("   της απόφασης είναι το paired test (ενότητα 2), όχι αυτό.")
    print("=" * 78)
    for A, B in pairs:
        loA, hiA = ci(macro(aucs_boot[A], comp_idx))
        loB, hiB = ci(macro(aucs_boot[B], comp_idx))
        overlap = not (loB > hiA or loA > hiB)
        print(f"  {A}: [{loA:.4f}, {hiA:.4f}]   {B}: [{loB:.4f}, {hiB:.4f}]   "
              f"-> {'ΕΠΙΚΑΛΥΠΤΟΝΤΑΙ' if overlap else 'ΔΕΝ επικαλύπτονται'}")

    # ======================= CSV export =======================
    csv_path = f"{args.out_prefix}_diffs.csv"
    with open(csv_path, "w") as f:
        f.write("comparison,metric,delta,ci_lo,ci_hi,p_value,P_delta_gt0\n")
        for row in comparison_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n[OK] Πίνακας διαφορών -> {csv_path}")

    # ======================= FOREST PLOT (η «απόδειξη») =======================
    if args.plot:
        plot_path = args.plot_out or f"{args.out_prefix}_forest.png"
        make_forest_plot(pairs, aucs_boot, aucs_point, comp_idx, plot_path)


def make_forest_plot(pairs, aucs_boot, aucs_point, comp_idx, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = COMPETITION + ["MACRO(5)"]
    metric_cols = [[PATHOLOGIES.index(c)] for c in COMPETITION] + [comp_idx]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 1)))

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(metrics) + 1.5))
    ypos = np.arange(len(metrics))[::-1]
    offset = np.linspace(-0.22, 0.22, len(pairs)) if len(pairs) > 1 else [0.0]

    for k, (A, B) in enumerate(pairs):
        pts, los, his = [], [], []
        for cols in metric_cols:
            if len(cols) == 1:
                j = cols[0]
                dpt = aucs_point[B][j] - aucs_point[A][j]
                dboot = aucs_boot[B][:, j] - aucs_boot[A][:, j]
            else:
                dpt = np.nanmean(aucs_point[B][cols]) - np.nanmean(aucs_point[A][cols])
                dboot = np.nanmean(aucs_boot[B][:, cols], axis=1) - \
                        np.nanmean(aucs_boot[A][:, cols], axis=1)
            v = dboot[~np.isnan(dboot)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            pts.append(dpt); los.append(dpt - lo); his.append(hi - dpt)
        ax.errorbar(pts, ypos + offset[k], xerr=[los, his], fmt="o",
                    color=colors[k], capsize=3, label=f"{B} − {A}", markersize=5)

    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("ΔAUROC (95% bootstrap CI)")
    ax.set_title("Paired bootstrap: κέρδος έναντι baseline\n(CI δεξιά του 0 = στατιστικά σημαντικό)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Forest plot -> {out_png}")


if __name__ == "__main__":
    main()
