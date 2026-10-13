"""
Revised analysis for: Integrating Explainable AI, Causal Inference, and Survival Modeling
for COPD Severity and Quality-of-Life Assessment (Discover Computing, major revision).

Leakage-free, nested, repeated cross-validation; SMOTE inside training folds only;
SMOTE-free sensitivity; class-wise metrics; calibration; SHAP/LIME stability;
propensity-score adjusted association (not a causal treatment effect).
Run: python copd_revision_analysis.py --data dataset.csv --out results
"""
import argparse, json, os, sys, platform, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sklearn, xgboost, shap, lime, imblearn, statsmodels
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, StratifiedKFold, KFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve, mean_squared_error,
                             mean_absolute_error, r2_score, cohen_kappa_score)
from xgboost import XGBClassifier, XGBRegressor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")
SEED = 42
N_SPLITS, N_REPEATS = 5, 10
CLASSES = ["MILD", "MODERATE", "SEVERE", "VERY SEVERE"]

# ---------------------------------------------------------------- predictor sets
CLS_PREDICTORS = ["AGE", "gender", "PackHistory", "smoking", "MWT1Best", "CAT", "HAD", "SGRQ",
                  "Diabetes", "muscular", "hypertension", "AtrialFib", "IHD"]
REG_PREDICTORS = ["AGE", "gender", "PackHistory", "smoking", "MWT1Best", "FEV1", "FEV1PRED", "FVC",
                  "FVCPRED", "CAT", "HAD", "Diabetes", "muscular", "hypertension", "AtrialFib", "IHD"]
ORIGINAL_CLS_SET = ["AGE", "PackHistory", "MWT1", "MWT2", "MWT1Best", "FEV1", "FEV1PRED", "FVC", "FVCPRED",
                    "CAT", "HAD", "SGRQ", "AGEquartiles", "copd", "gender", "smoking", "Diabetes",
                    "muscular", "hypertension", "AtrialFib", "IHD"]
CONFOUNDERS = ["AGE", "gender", "PackHistory", "smoking", "Diabetes", "muscular", "hypertension",
               "AtrialFib", "IHD"]


def load(path):
    d = pd.read_csv(path)
    d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
    # out-of-range questionnaire values set to missing (CAT range 0-40, HADS total 0-42)
    qc = {"CAT_out_of_range": int((d.CAT > 40).sum()), "HAD_out_of_range": int((d.HAD > 42).sum())}
    d.loc[d.CAT > 40, "CAT"] = np.nan
    d.loc[d.HAD > 42, "HAD"] = np.nan
    d["y4"] = d.COPDSEVERITY.map({c: i for i, c in enumerate(CLASSES)})
    d["severe"] = (d.y4 >= 2).astype(int)
    return d, qc


def ci_corrected(scores, n_train, n_test):
    """Nadeau-Bengio corrected resampled-t 95% CI for repeated k-fold scores."""
    s = np.asarray(scores); k = len(s); m = s.mean(); v = s.var(ddof=1)
    se = np.sqrt((1 / k + n_test / n_train) * v)
    t = stats.t.ppf(0.975, k - 1)
    return m, s.std(ddof=1), m - t * se, m + t * se


def corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * v)
    return float(2 * stats.t.sf(abs(t), k - 1))


# ---------------------------------------------------------------- model zoo
def cls_models(use_smote):
    base = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    smote = [("smote", SMOTE(k_neighbors=3, random_state=SEED))] if use_smote else []
    mk = lambda est: ImbPipeline(base + smote + [("m", est)])
    return {
        "KNN": (mk(KNeighborsClassifier()), {"m__n_neighbors": [3, 5, 7, 9], "m__weights": ["uniform", "distance"]}),
        "LR": (mk(LogisticRegression(max_iter=5000)), {"m__C": [0.01, 0.1, 1, 10]}),
        "RF": (mk(RandomForestClassifier(random_state=SEED, n_jobs=1)),
               {"m__n_estimators": [300], "m__max_depth": [3, None], "m__min_samples_split": [2, 5]}),
        "XGB": (mk(XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="mlogloss", subsample=0.8,
                                 reg_lambda=1.0, verbosity=0)),
                {"m__n_estimators": [100, 300], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}),
    }


def reg_models():
    base = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    mk = lambda est: Pipeline(base + [("m", est)])
    return {
        "KNN-REG": (mk(KNeighborsRegressor()), {"m__n_neighbors": [3, 5, 7, 9], "m__weights": ["uniform", "distance"]}),
        "LR-REG": (mk(LinearRegression()), {}),
        "RF-REG": (mk(RandomForestRegressor(random_state=SEED, n_jobs=1)),
                   {"m__n_estimators": [300], "m__max_depth": [3, None], "m__min_samples_leaf": [1, 3]}),
        "XGB-REG": (mk(XGBRegressor(random_state=SEED, n_jobs=1, subsample=0.8, verbosity=0)),
                    {"m__n_estimators": [100, 300], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}),
    }


def macro_specificity(cm):
    sp = []
    for k in range(cm.shape[0]):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        sp.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return np.array(sp)


def run_classification(d, predictors, use_smote, splits, tag, log):
    X = d[predictors].values.astype(float); y = d.y4.values
    res = {m: {"acc": [], "bacc": [], "prec": [], "rec": [], "f1": [], "spec": [], "auc": [], "qwk": []}
           for m in cls_models(use_smote)}
    oof = {m: np.zeros((N_REPEATS, len(y), 4)) for m in res}
    best_params = {m: [] for m in res}
    smote_log = []
    for fi, (tr, te) in enumerate(splits):
        rep = fi // N_SPLITS
        for name, (pipe, grid) in cls_models(use_smote).items():
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + fi)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="f1_macro", n_jobs=-1) if grid else pipe
            gs.fit(X[tr], y[tr])
            est = gs.best_estimator_ if grid else gs
            if grid:
                best_params[name].append(gs.best_params_)
            p = est.predict_proba(X[te]); yh = p.argmax(1)
            oof[name][rep, te] = p
            cm = confusion_matrix(y[te], yh, labels=range(4))
            pr, rc, f1, _ = precision_recall_fscore_support(y[te], yh, labels=range(4), average="macro", zero_division=0)
            r = res[name]
            r["acc"].append(accuracy_score(y[te], yh)); r["bacc"].append(balanced_accuracy_score(y[te], yh))
            r["prec"].append(pr); r["rec"].append(rc); r["f1"].append(f1)
            r["spec"].append(np.nanmean(macro_specificity(cm)))
            r["auc"].append(roc_auc_score(label_binarize(y[te], classes=range(4)), p, average="macro", multi_class="ovr")
                            if len(np.unique(y[te])) == 4 else np.nan)
            r["qwk"].append(cohen_kappa_score(y[te], yh, weights="quadratic"))
        if use_smote and rep == 0:
            tr_counts = np.bincount(y[tr], minlength=4)
            # replicate the SMOTE step on the imputed/scaled training fold to log class counts
            Xt = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[tr]))
            _, ys = SMOTE(k_neighbors=3, random_state=SEED).fit_resample(Xt, y[tr])
            smote_log.append({"fold": fi % N_SPLITS + 1, "train_before": tr_counts.tolist(),
                              "train_after": np.bincount(ys, minlength=4).tolist(),
                              "test": np.bincount(y[te], minlength=4).tolist()})
        log(f"[{tag}] fold {fi + 1}/{len(splits)} done")
    return res, oof, best_params, smote_log


def summarise_cls(res, n_train, n_test):
    out = {}
    for m, r in res.items():
        out[m] = {k: list(ci_corrected(np.array(v)[~np.isnan(v)], n_train, n_test)) for k, v in r.items()}
    return out


def run_regression(d, predictors, splits, log):
    X = d[predictors].values.astype(float); y = d.SGRQ.values
    res = {m: {"mse": [], "mae": [], "r2": []} for m in reg_models()}
    oof = {m: np.zeros((N_REPEATS, len(y))) for m in res}
    for fi, (tr, te) in enumerate(splits):
        rep = fi // N_SPLITS
        for name, (pipe, grid) in reg_models().items():
            if grid:
                gs = GridSearchCV(pipe, grid, cv=KFold(3, shuffle=True, random_state=SEED + fi),
                                  scoring="neg_mean_squared_error", n_jobs=-1).fit(X[tr], y[tr])
                est = gs.best_estimator_
            else:
                est = pipe.fit(X[tr], y[tr])
            yh = est.predict(X[te]); oof[name][rep, te] = yh
            res[name]["mse"].append(mean_squared_error(y[te], yh))
            res[name]["mae"].append(mean_absolute_error(y[te], yh))
            res[name]["r2"].append(r2_score(y[te], yh))
        log(f"[REG] fold {fi + 1}/{len(splits)} done")
    return res, oof


# ---------------------------------------------------------------- propensity score analysis
def smd(x, t, w=None):
    if w is None:
        w = np.ones_like(x, dtype=float)
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    v1 = np.var(x[t == 1]); v0 = np.var(x[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0


def ps_match(df, caliper_sd=0.2):
    Xc = df[CONFOUNDERS].values.astype(float); t = df.severe.values
    lr = LogisticRegression(max_iter=5000, C=1e6).fit(StandardScaler().fit_transform(Xc), t)
    ps = lr.predict_proba(StandardScaler().fit_transform(Xc))[:, 1]
    logit = np.log(ps / (1 - ps)); cal = caliper_sd * logit.std()
    treated = np.where(t == 1)[0]; controls = list(np.where(t == 0)[0])
    pairs = []
    # greedy 1:1 nearest-neighbour without replacement, treated in random order
    rng = np.random.default_rng(SEED)
    for i in rng.permutation(treated):
        if not controls:
            break
        dist = np.abs(logit[controls] - logit[i]); j = int(np.argmin(dist))
        if dist[j] <= cal:
            pairs.append((i, controls.pop(j)))
    return ps, pairs, cal


def bootstrap_att(df, B=1000):
    rng = np.random.default_rng(SEED); est = []
    for b in range(B):
        s = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        if s.severe.sum() < 5 or (1 - s.severe).sum() < 5:
            continue
        _, pairs, _ = ps_match(s)
        if len(pairs) < 5:
            continue
        ti = [p[0] for p in pairs]; ci = [p[1] for p in pairs]
        est.append(s.SGRQ.values[ti].mean() - s.SGRQ.values[ci].mean())
    return np.array(est)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); fig = os.path.join(a.out, "figures"); os.makedirs(fig, exist_ok=True)
    logf = open(os.path.join(a.out, "run_log.txt"), "w")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                "imbalanced-learn": imblearn.__version__, "shap": shap.__version__, "lime": lime.__version__ if hasattr(lime, "__version__") else "0.2.0.1",
                "statsmodels": statsmodels.__version__, "seed": SEED}
    log(json.dumps(versions))
    d, qc = load(a.data); R = {"versions": versions, "qc": qc, "n": len(d)}
    R["class_counts"] = d.COPDSEVERITY.value_counts().reindex(CLASSES).tolist()
    R["missing_after_qc"] = d[list(set(CLS_PREDICTORS + REG_PREDICTORS))].isna().sum().loc[lambda s: s > 0].to_dict()

    # ---------------- leakage audit
    R["copd_equals_severity"] = bool((d.copd - 1 == d.y4).all())
    gold = pd.cut(d.FEV1PRED, [-np.inf, 30, 50, 80, np.inf], right=False, labels=[3, 2, 1, 0]).astype(int)
    R["gold_rule_concordance"] = int((gold.values == d.y4.values).sum())
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(cv.split(d, d.y4))
    pd.DataFrame([{"fold_index": i, "repeat": i // N_SPLITS + 1, "fold": i % N_SPLITS + 1,
                   "test_ids": " ".join(map(str, d.ID.values[te]))} for i, (tr, te) in enumerate(splits)]
                 ).to_csv(os.path.join(a.out, "cv_splits_classification.csv"), index=False)
    n_te = len(d) / N_SPLITS; n_tr = len(d) - n_te
    leak = {}
    for nm, feats in {"Original predictor set (incl. copd, FEV1PRED)": ORIGINAL_CLS_SET, "FEV1PRED only": ["FEV1PRED"],
                      "Spirometry only (FEV1, FEV1PRED, FVC, FVCPRED)": ["FEV1", "FEV1PRED", "FVC", "FVCPRED"]}.items():
        X = d[feats].values.astype(float); accs, f1s = [], []
        for tr, te in splits:
            p = ImbPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                             ("smote", SMOTE(k_neighbors=3, random_state=SEED)),
                             ("m", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=SEED,
                                                 verbosity=0, n_jobs=1))]).fit(X[tr], d.y4.values[tr])
            yh = p.predict(X[te]); accs.append(accuracy_score(d.y4.values[te], yh))
            f1s.append(precision_recall_fscore_support(d.y4.values[te], yh, average="macro", zero_division=0)[2])
        leak[nm] = {"acc": list(ci_corrected(accs, n_tr, n_te)), "f1": list(ci_corrected(f1s, n_tr, n_te))}
        log(f"leakage audit {nm}: acc={np.mean(accs):.3f}")
    R["leakage_audit"] = leak

    # ---------------- primary classification (leakage-free) with and without SMOTE
    res_s, oof_s, bp_s, smote_log = run_classification(d, CLS_PREDICTORS, True, splits, "CLS-SMOTE", log)
    res_n, oof_n, bp_n, _ = run_classification(d, CLS_PREDICTORS, False, splits, "CLS-noSMOTE", log)
    R["cls_smote"] = summarise_cls(res_s, n_tr, n_te); R["cls_nosmote"] = summarise_cls(res_n, n_tr, n_te)
    R["smote_log"] = smote_log
    R["best_params_cls"] = {m: pd.Series([json.dumps(b, sort_keys=True) for b in v]).value_counts().head(3).to_dict()
                            for m, v in bp_s.items()}
    best = max(R["cls_smote"], key=lambda m: R["cls_smote"][m]["f1"][0]); R["best_cls"] = best
    R["pvals_cls_f1"] = {m: corrected_ttest(res_s[best]["f1"], res_s[m]["f1"], n_tr, n_te) for m in res_s if m != best}
    R["pvals_smote_vs_nosmote_f1"] = {m: corrected_ttest(res_s[m]["f1"], res_n[m]["f1"], n_tr, n_te) for m in res_s}
    # class-wise metrics + pooled confusion matrix for best model
    y = d.y4.values
    P = oof_s[best]; yh_all = P.argmax(2)
    cm = sum(confusion_matrix(y, yh_all[r], labels=range(4)) for r in range(N_REPEATS))
    R["cm_pooled"] = cm.tolist()
    cw = []
    for r in range(N_REPEATS):
        pr, rc, f1, _ = precision_recall_fscore_support(y, yh_all[r], labels=range(4), zero_division=0)
        sp = macro_specificity(confusion_matrix(y, yh_all[r], labels=range(4)))
        auc = [roc_auc_score((y == k).astype(int), P[r, :, k]) for k in range(4)]
        cw.append(np.vstack([pr, rc, f1, sp, auc]))
    cw = np.array(cw)
    R["classwise"] = {CLASSES[k]: {met: [float(cw[:, i, k].mean()), float(cw[:, i, k].std(ddof=1))]
                                   for i, met in enumerate(["precision", "recall", "f1", "specificity", "auc"])}
                      for k in range(4)}
    for m in oof_s:
        cmm = sum(confusion_matrix(y, oof_s[m].argmax(2)[r], labels=range(4)) for r in range(N_REPEATS))
        R.setdefault("cm_all_models", {})[m] = cmm.tolist()
    # calibration: P(severe or very severe)
    psev = P[:, :, 2:].sum(2).ravel(); ysev = np.tile(d.severe.values, N_REPEATS)
    brier = float(np.mean((psev - ysev) ** 2))
    lp = np.log(np.clip(psev, 1e-6, 1 - 1e-6) / (1 - np.clip(psev, 1e-6, 1 - 1e-6)))
    cal = sm.Logit(ysev, sm.add_constant(lp)).fit(disp=0)
    bins = np.quantile(psev, np.linspace(0, 1, 6)); bins[0] -= 1e-9; bins[-1] += 1e-9
    idx = np.digitize(psev, bins) - 1
    calpts = [(float(psev[idx == b].mean()), float(ysev[idx == b].mean()), int((idx == b).sum())) for b in range(5)]
    ece = float(sum(n * abs(p - o) for p, o, n in calpts) / len(psev))
    R["calibration"] = {"brier": brier, "slope": float(cal.params[1]), "intercept": float(cal.params[0]),
                        "ece": ece, "points": calpts,
                        "multiclass_brier": float(np.mean(((P - np.eye(4)[y][None]) ** 2).sum(2)))}
    for m in oof_s:
        Pm = oof_s[m]; R.setdefault("multiclass_brier_all", {})[m] = float(np.mean(((Pm - np.eye(4)[y][None]) ** 2).sum(2)))

    # figures: confusion matrix, ROC, calibration
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    f, ax = plt.subplots(figsize=(5.2, 4.4))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j] * 100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=9)
    lab = ["Mild", "Moderate", "Severe", "Very severe"]
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, rotation=20); ax.set_yticks(range(4)); ax.set_yticklabels(lab)
    ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage"); f.tight_layout(); f.savefig(f"{fig}/fig_cm.png", dpi=300); plt.close(f)
    f, ax = plt.subplots(figsize=(5.2, 4.4))
    for k in range(4):
        fpr, tpr, _ = roc_curve(np.tile((y == k).astype(int), N_REPEATS), P[:, :, k].ravel())
        ax.plot(fpr, tpr, lw=1.8, label=f"{lab[k]} (AUC {R['classwise'][CLASSES[k]]['auc'][0]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=9); f.tight_layout(); f.savefig(f"{fig}/fig_roc.png", dpi=300); plt.close(f)
    f, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot([c[0] for c in calpts], [c[1] for c in calpts], "o-", lw=1.8, label=f"{best} (quintiles)")
    ax.set_xlabel("Predicted probability of severe or very severe COPD"); ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=9); f.tight_layout()
    f.savefig(f"{fig}/fig_calibration.png", dpi=300); plt.close(f)

    # ---------------- regression
    rsplits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(d))
    rres, roof = run_regression(d, REG_PREDICTORS, rsplits, log)
    R["reg"] = {m: {k: list(ci_corrected(v, n_tr, n_te)) for k, v in r.items()} for m, r in rres.items()}
    bestr = min(R["reg"], key=lambda m: R["reg"][m]["mse"][0]); R["best_reg"] = bestr
    R["pvals_reg_mse"] = {m: corrected_ttest(rres[bestr]["mse"], rres[m]["mse"], n_tr, n_te) for m in rres if m != bestr}
    yr = d.SGRQ.values; pr_ = roof[bestr].mean(0)
    sl = sm.OLS(yr, sm.add_constant(pr_)).fit(); R["reg_calibration"] = {"slope": float(sl.params[1]), "intercept": float(sl.params[0])}
    f, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(pr_, yr, s=22, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo, hi = 0, 85; ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Predicted SGRQ ({bestr}, out-of-fold)"); ax.set_ylabel("Observed SGRQ")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); f.tight_layout(); f.savefig(f"{fig}/fig_reg_calibration.png", dpi=300); plt.close(f)

    # ---------------- XAI: SHAP stability across outer folds + LIME consistency
    def fit_best_cls(Xtr, ytr, fi):
        pipe, grid = cls_models(True)[best]
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED + fi), scoring="f1_macro", n_jobs=-1)
        return gs.fit(Xtr, ytr).best_estimator_

    def fit_best_reg(Xtr, ytr, fi):
        pipe, grid = reg_models()[bestr]
        if not grid:
            return pipe.fit(Xtr, ytr)
        return GridSearchCV(pipe, grid, cv=KFold(3, shuffle=True, random_state=SEED + fi), scoring="neg_mean_squared_error", n_jobs=-1).fit(Xtr, ytr).best_estimator_

    def xai_block(task, predictors, splits_, yv, fitter, predict_fn_name, mode):
        X = d[predictors].values.astype(float)
        imp_rows, lime_rows = [], np.zeros((len(yv), len(predictors)))
        shap_all = np.zeros((len(yv), len(predictors))); shap_vals_rep0 = np.zeros((len(yv), len(predictors)))
        models_rep0 = {}
        for fi, (tr, te) in enumerate(splits_):
            est = fitter(X[tr], yv[tr], fi)
            bg = SimpleImputer(strategy="median").fit(X[tr]).transform(X[tr])
            Xte = X[te]
            if mode == "cls":
                fn = lambda Z, e=est: e.predict_proba(Z)
            else:
                fn = lambda Z, e=est: e.predict(Z)
            expl = shap.Explainer(fn, shap.maskers.Independent(bg, max_samples=100), algorithm="permutation", seed=SEED)
            sv = expl(SimpleImputer(strategy="median").fit(X[tr]).transform(Xte), max_evals=2 * len(predictors) + 1)
            vals = sv.values
            if mode == "cls":
                pred = est.predict(Xte)
                vals_pred = np.array([vals[i, :, pred[i]] for i in range(len(te))])
                mag = np.abs(vals).sum(2)
            else:
                vals_pred = vals; mag = np.abs(vals)
            imp_rows.append(mag.mean(0))
            if fi < N_SPLITS:
                shap_vals_rep0[te] = vals_pred; models_rep0[fi] = (est, tr, te)
                lexp = LimeTabularExplainer(bg, feature_names=predictors, mode="classification" if mode == "cls" else "regression",
                                            discretize_continuous=False, random_state=SEED)
                for i, row in zip(te, SimpleImputer(strategy="median").fit(X[tr]).transform(Xte)):
                    if mode == "cls":
                        lab_ = int(est.predict(row[None])[0])
                        e = lexp.explain_instance(row, fn, labels=[lab_], num_features=len(predictors), num_samples=2000)
                        for fidx, w in e.as_map()[lab_]:
                            lime_rows[i, fidx] = w
                    else:
                        e = lexp.explain_instance(row, fn, num_features=len(predictors), num_samples=2000)
                        for fidx, w in e.as_map()[1]:
                            lime_rows[i, fidx] = w
            log(f"[XAI-{task}] fold {fi + 1}/{len(splits_)}")
        imp = np.array(imp_rows)
        ranks = np.array([stats.rankdata(-r) for r in imp])
        rho = [stats.spearmanr(imp[i], imp[j])[0] for i in range(len(imp)) for j in range(i + 1, len(imp))]
        tau = [stats.kendalltau(imp[i], imp[j])[0] for i in range(len(imp)) for j in range(i + 1, len(imp))]
        top5 = np.argsort(-imp.mean(0))[:5]
        top5_freq = {predictors[k]: float(np.mean([k in np.argsort(-r)[:5] for r in imp])) for k in top5}
        shap_glob_rep0 = np.abs(shap_vals_rep0).mean(0); lime_glob = np.abs(lime_rows).mean(0)
        sl_rho = stats.spearmanr(shap_glob_rep0, lime_glob)[0]
        # instance-level agreement (sign agreement on top-3 SHAP features)
        agree = []
        for i in range(len(yv)):
            t3 = np.argsort(-np.abs(shap_vals_rep0[i]))[:3]
            agree.append(np.mean(np.sign(shap_vals_rep0[i, t3]) == np.sign(lime_rows[i, t3])))
        order = np.argsort(-imp.mean(0))
        out = {"mean_abs_shap": dict(zip(predictors, imp.mean(0).round(4).tolist())),
               "sd_abs_shap": dict(zip(predictors, imp.std(0, ddof=1).round(4).tolist())),
               "median_rank": dict(zip(predictors, np.median(ranks, 0).tolist())),
               "rank_iqr": dict(zip(predictors, (np.percentile(ranks, 75, 0) - np.percentile(ranks, 25, 0)).tolist())),
               "spearman_between_folds_mean": float(np.nanmean(rho)), "spearman_between_folds_sd": float(np.nanstd(rho)),
               "kendall_between_folds_mean": float(np.nanmean(tau)), "top5_inclusion_freq": top5_freq,
               "shap_vs_lime_spearman": float(sl_rho), "lime_global": dict(zip(predictors, lime_glob.round(4).tolist())),
               "shap_global_rep0": dict(zip(predictors, shap_glob_rep0.round(4).tolist())),
               "top3_sign_agreement": float(np.mean(agree)), "order": [predictors[k] for k in order]}
        # global importance figure with fold variability
        f, ax = plt.subplots(figsize=(5.6, 4.6))
        o = order[::-1]
        ax.barh([predictors[k] for k in o], imp.mean(0)[o], xerr=imp.std(0, ddof=1)[o], color="#4C72B0", alpha=0.85, capsize=2)
        ax.set_xlabel("Mean |SHAP value| (mean ± SD over 50 outer folds)"); f.tight_layout()
        f.savefig(f"{fig}/fig_shap_global_{task}.png", dpi=300); plt.close(f)
        # rank stability figure
        f, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.boxplot([ranks[:, k] for k in o], vert=False, labels=[predictors[k] for k in o], widths=0.6)
        ax.set_xlabel("Importance rank across 50 outer folds (1 = most important)"); ax.invert_xaxis(); f.tight_layout()
        f.savefig(f"{fig}/fig_shap_rank_stability_{task}.png", dpi=300); plt.close(f)
        return out, shap_vals_rep0, lime_rows, models_rep0

    R["xai_cls"], shap_c, lime_c, mod_c = xai_block("cls", CLS_PREDICTORS, splits, d.y4.values, fit_best_cls, "predict_proba", "cls")
    R["xai_reg"], shap_r, lime_r, mod_r = xai_block("reg", REG_PREDICTORS, rsplits, d.SGRQ.values, fit_best_reg, "predict", "reg")
    R["corr_spirometry"] = d[["FEV1", "FVC", "FEV1PRED", "FVCPRED"]].corr().round(2).to_dict()
    grp = ["FEV1", "FEV1PRED", "FVC", "FVCPRED"]
    tot = sum(R["xai_reg"]["mean_abs_shap"].values())
    R["xai_reg_spirometry_group_share"] = float(sum(R["xai_reg"]["mean_abs_shap"][g] for g in grp) / tot)

    # representative patient-level explanations (out-of-fold, repeat 1)
    def patient_fig(vals, limev, preds, i, title_tag, fname, fnames):
        o = np.argsort(-np.abs(vals[i]))[:8][::-1]
        f, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.barh([fnames[k] for k in o], vals[i, o], color=["#C44E52" if v > 0 else "#4C72B0" for v in vals[i, o]])
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(title_tag); f.tight_layout(); f.savefig(fname, dpi=300); plt.close(f)
    pred_rep0 = oof_s[best][0].argmax(1)
    sev_ids = np.where((d.y4.values == 2) & (pred_rep0 == 2))[0]
    pat_c = int(sev_ids[np.argmax(oof_s[best][0][sev_ids, 2])]) if len(sev_ids) else int(np.where(d.y4.values == 2)[0][0])
    patient_fig(shap_c, None, None, pat_c, f"SHAP contribution to predicted class ({CLASSES[pred_rep0[pat_c]].title()})",
                f"{fig}/fig_patient_shap_cls.png", CLS_PREDICTORS)
    patient_fig(lime_c, None, None, pat_c, f"LIME weight for predicted class ({CLASSES[pred_rep0[pat_c]].title()})",
                f"{fig}/fig_patient_lime_cls.png", CLS_PREDICTORS)
    pat_r = int(np.argmax(d.SGRQ.values))  # highest symptom burden
    patient_fig(shap_r, None, None, pat_r, "SHAP contribution to predicted SGRQ (points)", f"{fig}/fig_patient_shap_reg.png", REG_PREDICTORS)
    patient_fig(lime_r, None, None, pat_r, "LIME weight for predicted SGRQ", f"{fig}/fig_patient_lime_reg.png", REG_PREDICTORS)
    R["patient_cls"] = {"ID": int(d.ID.values[pat_c]), "true": CLASSES[d.y4.values[pat_c]], "pred": CLASSES[pred_rep0[pat_c]],
                        "prob": oof_s[best][0][pat_c].round(3).tolist(),
                        "values": {CLS_PREDICTORS[k]: (float(d[CLS_PREDICTORS[k]].values[pat_c]), round(float(shap_c[pat_c, k]), 3), round(float(lime_c[pat_c, k]), 3))
                                   for k in np.argsort(-np.abs(shap_c[pat_c]))[:5]}}
    R["patient_reg"] = {"ID": int(d.ID.values[pat_r]), "obs": float(d.SGRQ.values[pat_r]), "pred": float(roof[bestr][0][pat_r]),
                        "values": {REG_PREDICTORS[k]: (float(d[REG_PREDICTORS[k]].values[pat_r]), round(float(shap_r[pat_r, k]), 2), round(float(lime_r[pat_r, k]), 2))
                                   for k in np.argsort(-np.abs(shap_r[pat_r]))[:5]}}

    # ---------------- propensity-score adjusted association (severe/very severe vs mild/moderate)
    dc = d.copy()
    ps, pairs, cal = ps_match(dc)
    ti = np.array([p[0] for p in pairs]); ci_ = np.array([p[1] for p in pairs])
    t = dc.severe.values
    bal = []
    for c in CONFOUNDERS:
        x = dc[c].values.astype(float)
        mt = np.concatenate([ti, ci_]); tt = np.concatenate([np.ones(len(ti)), np.zeros(len(ci_))])
        bal.append({"covariate": c, "smd_before": float(smd(x, t)), "smd_after": float(smd(x[mt], tt))})
    att = float(dc.SGRQ.values[ti].mean() - dc.SGRQ.values[ci_].mean())
    diff = dc.SGRQ.values[ti] - dc.SGRQ.values[ci_]
    boot = bootstrap_att(dc)
    ols = sm.OLS(dc.SGRQ.values, sm.add_constant(dc[["severe"] + CONFOUNDERS].astype(float))).fit(cov_type="HC3")
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    ipw = float(np.average(dc.SGRQ.values[t == 1], weights=w[t == 1]) - np.average(dc.SGRQ.values[t == 0], weights=w[t == 0]))
    rng = np.random.default_rng(SEED); ipw_b = []
    for b in range(1000):
        s = rng.integers(0, len(dc), len(dc)); sd_ = dc.iloc[s]; tb = sd_.severe.values
        if tb.sum() < 5: continue
        Xs = StandardScaler().fit_transform(sd_[CONFOUNDERS].values.astype(float))
        pb = LogisticRegression(max_iter=5000, C=1e6).fit(Xs, tb).predict_proba(Xs)[:, 1]
        wb = np.where(tb == 1, 1 / pb, 1 / (1 - pb))
        ipw_b.append(np.average(sd_.SGRQ.values[tb == 1], weights=wb[tb == 1]) - np.average(sd_.SGRQ.values[tb == 0], weights=wb[tb == 0]))
    R["psm"] = {"n_exposed": int(t.sum()), "n_unexposed": int((1 - t).sum()), "n_pairs": len(pairs),
                "caliper_logit": float(cal), "unmatched_exposed": int(t.sum() - len(pairs)),
                "mean_sgrq_exposed_before": float(dc.SGRQ[t == 1].mean()), "mean_sgrq_unexposed_before": float(dc.SGRQ[t == 0].mean()),
                "mean_sgrq_exposed_after": float(dc.SGRQ.values[ti].mean()), "mean_sgrq_unexposed_after": float(dc.SGRQ.values[ci_].mean()),
                "att": att, "att_boot_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "att_boot_n": int(len(boot)),
                "paired_t_p": float(stats.ttest_rel(dc.SGRQ.values[ti], dc.SGRQ.values[ci_]).pvalue),
                "ols_coef": float(ols.params["severe"]), "ols_ci": [float(x) for x in ols.conf_int().loc["severe"]], "ols_p": float(ols.pvalues["severe"]),
                "ipw_ate": ipw, "ipw_ci": [float(np.percentile(ipw_b, 2.5)), float(np.percentile(ipw_b, 97.5))],
                "ps_range_exposed": [float(ps[t == 1].min()), float(ps[t == 1].max())],
                "ps_range_unexposed": [float(ps[t == 0].min()), float(ps[t == 0].max())],
                "max_abs_smd_before": float(max(abs(b["smd_before"]) for b in bal)),
                "max_abs_smd_after": float(max(abs(b["smd_after"]) for b in bal)), "balance": bal}
    f, ax = plt.subplots(figsize=(5.2, 3.8))
    bins_ = np.linspace(0, 1, 21)
    ax.hist(ps[t == 0], bins=bins_, alpha=0.6, label="Mild/moderate (unexposed)", color="#4C72B0")
    ax.hist(ps[t == 1], bins=bins_, alpha=0.6, label="Severe/very severe (exposed)", color="#C44E52")
    ax.set_xlabel("Estimated propensity score"); ax.set_ylabel("Number of patients"); ax.legend(fontsize=9)
    f.tight_layout(); f.savefig(f"{fig}/fig_ps_overlap.png", dpi=300); plt.close(f)
    f, ax = plt.subplots(figsize=(5.2, 4.0))
    names = [b["covariate"] for b in bal][::-1]
    ax.scatter([abs(b["smd_before"]) for b in bal][::-1], names, marker="o", color="#C44E52", label="Before matching")
    ax.scatter([abs(b["smd_after"]) for b in bal][::-1], names, marker="s", color="#4C72B0", label="After matching")
    ax.axvline(0.1, ls="--", color="k", lw=1); ax.set_xlabel("Absolute standardized mean difference"); ax.legend(fontsize=9, loc="lower right")
    f.tight_layout(); f.savefig(f"{fig}/fig_love_plot.png", dpi=300); plt.close(f)

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    pd.Series({"classification_target": "COPDSEVERITY (4 ordinal GOLD grades)", "classification_predictors": ", ".join(CLS_PREDICTORS),
               "regression_target": "SGRQ total score (0-100, higher = worse QoL)", "regression_predictors": ", ".join(REG_PREDICTORS),
               "psm_exposure": "severe or very severe COPDSEVERITY vs mild or moderate", "psm_outcome": "SGRQ",
               "psm_confounders": ", ".join(CONFOUNDERS)}).to_csv(os.path.join(a.out, "targets_and_predictors.csv"))
    log("DONE")


if __name__ == "__main__":
    main()
