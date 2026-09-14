# -*- coding: utf-8 -*-
"""백본별 환자 단위 5-겹 교차검증 실행기.

사용 예:
  python backbone_exp/run.py --backbones resnet18 resnet50
  python backbone_exp/run.py --backbones dinov2_b --hidden 0        # 선형 프로브
  python backbone_exp/run.py --backbones resnet18 --label-source union
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casexp as X


def pick_threshold(y, p, recall_target):
    """학습 fold 예측에서만 임계값을 고른다.

    규칙: 재현율 >= 목표치를 만족하는 후보 중 정밀도 최대.
    만족하는 후보가 없으면 Youden's J로 되돌린다.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]                      # thr와 길이 맞춤
    ok = rec >= recall_target
    if ok.any():
        return float(thr[np.argmax(np.where(ok, prec, -1))])
    from sklearn.metrics import roc_curve
    fpr, tpr, t2 = roc_curve(y, p)
    return float(t2[np.argmax(tpr - fpr)])


def inner_oof_preds(feats, y, img_pat, tr_p, y_pat_col, dim, hidden, n_inner=4):
    """학습 fold 안에서 다시 K-겹을 돌려 '표본 외' 학습 예측을 만든다.

    헤드가 학습셋을 과적합하므로, 학습 예측으로 임계값을 고르면 값이 과도하게
    높아져 검증에서 아무것도 양성으로 잡지 못한다. 임계값은 반드시 학습 fold의
    표본 외 예측에서 골라야 한다(중첩 교차검증).
    """
    from sklearn.model_selection import StratifiedKFold
    strat = y_pat_col[tr_p]
    n_inner = min(n_inner, int(min(np.bincount(strat, minlength=2))) or 1)
    tr_mask = np.isin(img_pat, tr_p)
    out = np.full(int(tr_mask.sum()), np.nan)
    if n_inner < 2:                                   # 양성 환자가 1명 이하면 중첩 불가
        return None
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=X.SEED)
    tr_img_idx = np.where(tr_mask)[0]
    for ia, ib in skf.split(tr_p, strat):
        a_img = np.isin(img_pat, tr_p[ia])
        b_img = np.isin(img_pat, tr_p[ib])
        head = X.train_head(feats[a_img], y[a_img], dim, hidden=hidden)
        pos = np.searchsorted(tr_img_idx, np.where(b_img)[0])
        out[pos] = X.predict(head, feats[b_img])
    return out


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def run_backbone(name, paths, owners, y_img, patients, y_pat, folds,
                 hidden, recall_target):
    print(f"\n{'='*66}\n[{name}]  특징 추출")
    t0 = time.time()
    feats = X.extract_features(name, paths)
    print(f"  특징 {feats.shape}  ({time.time()-t0:.1f}s)")

    owner_idx = {p: i for i, p in enumerate(patients)}
    img_pat = np.array([owner_idx[o] for o in owners])
    dim = feats.shape[1]

    oof = np.full_like(y_img, np.nan)
    # fold마다 헤드의 출력 스케일이 다르므로, 각 fold의 임계값은
    # 그 fold의 검증 예측에만 적용하고 이진 결정만 모은다.
    oof_bin = np.full(y_img.shape, -1, dtype=int)
    thr_used = {c: [] for c in X.CAS_ITEMS}
    fold_auc = []

    for k, (tr_p, va_p) in enumerate(folds):
        tr = np.isin(img_pat, tr_p)
        va = np.isin(img_pat, va_p)
        per_item = []
        for j, cas in enumerate(X.CAS_ITEMS):
            head = X.train_head(feats[tr], y_img[tr, j], dim, hidden=hidden)
            p_va = X.predict(head, feats[va])
            oof[va, j] = p_va
            p_in = inner_oof_preds(feats, y_img[:, j], img_pat, tr_p,
                                   y_pat[:, j], dim, hidden)
            p_thr = p_in if p_in is not None else X.predict(head, feats[tr])
            thr = pick_threshold(y_img[tr, j], p_thr, recall_target)
            thr_used[cas].append(thr)
            oof_bin[va, j] = (p_va >= thr).astype(int)
            per_item.append(safe_auc(y_img[va, j], p_va))
        fold_auc.append(float(np.nanmean(per_item)))
        print(f"  fold {k}: AUROC {fold_auc[-1]:.4f}  (검증 환자 {len(va_p)}명, 영상 {va.sum()}장)")

    res = {"backbone": name, "hidden": hidden, "dim": dim,
           "fold_auc": fold_auc,
           "fold_mean": float(np.mean(fold_auc)), "fold_std": float(np.std(fold_auc)),
           "items": {}}
    aurocs, auprcs = [], []
    for j, cas in enumerate(X.CAS_ITEMS):
        y, p = y_img[:, j], oof[:, j]
        pred = oof_bin[:, j]
        tp = int(((pred == 1) & (y == 1)).sum())
        recall = tp / max(int(y.sum()), 1)
        precision = tp / max(int(pred.sum()), 1)
        a, ap = safe_auc(y, p), float(average_precision_score(y, p))
        aurocs.append(a); auprcs.append(ap)
        res["items"][cas] = {"auroc": a, "auprc": ap,
                             "thr_per_fold": [round(t, 4) for t in thr_used[cas]],
                             "recall": recall, "precision": precision,
                             "n_pos": int(y.sum()), "n": int(len(y))}
    res["oof_macro_auroc"] = float(np.nanmean(aurocs))
    res["oof_macro_auprc"] = float(np.nanmean(auprcs))
    res["macro_recall"] = float(np.mean([v["recall"] for v in res["items"].values()]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["resnet18"])
    ap.add_argument("--label-source", default="ted", choices=["ted", "union"])
    ap.add_argument("--hidden", type=int, default=256, help="0이면 선형 프로브")
    ap.add_argument("--recall-target", type=float, default=0.95)
    ap.add_argument("--out", default="backbone_exp/results.json")
    a = ap.parse_args()

    paths, owners, y_img, patients, y_pat = X.build_dataset(a.label_source)
    folds = X.patient_folds(patients, y_pat)
    print(f"라벨 출처 {a.label_source} | 환자 {len(patients)}명 | 영상 {len(paths)}장 "
          f"| 헤드 hidden={a.hidden} | 재현율 목표 {a.recall_target}")
    print("항목별 비정상: " + "  ".join(
        f"{c}({X.CAS_NAMES[c]}) 영상 {int(y_img[:,j].sum())} / 환자 {int(y_pat[:,j].sum())}"
        for j, c in enumerate(X.CAS_ITEMS)))

    out_path = Path(a.out)
    allres = json.loads(out_path.read_text("utf-8")) if out_path.exists() else []
    for name in a.backbones:
        r = run_backbone(name, paths, owners, y_img, patients, y_pat, folds,
                         a.hidden, a.recall_target)
        r["label_source"], r["recall_target"] = a.label_source, a.recall_target
        allres = [x for x in allres if not (x["backbone"] == name
                  and x["hidden"] == a.hidden and x["label_source"] == a.label_source)]
        allres.append(r)
        print(f"\n  OOF 거시 AUROC {r['oof_macro_auroc']:.4f} | "
              f"AUPRC {r['oof_macro_auprc']:.4f} | 거시 재현율 {r['macro_recall']:.3f}")
        for c, v in r["items"].items():
            print(f"    {c} {X.CAS_NAMES[c]:8s} AUROC {v['auroc']:.4f}  AUPRC {v['auprc']:.4f}  "
                  f"재현율 {v['recall']:.3f}  정밀도 {v['precision']:.3f}  (양성 {v['n_pos']}/{v['n']})")
        out_path.write_text(json.dumps(allres, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n결과 저장 → {out_path}")


if __name__ == "__main__":
    main()
