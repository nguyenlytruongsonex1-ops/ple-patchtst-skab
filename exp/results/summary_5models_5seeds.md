# SKAB – 5 mô hình × 5 fold × 5 seed (D11)

Số lượt: 125 (cần 125). Mean ± std qua 5 fold (mỗi fold = TB 5 seed). FPR đếm trực tiếp từ dự đoán test (results/preds5).

| Mô hình | Tham số | AP | AUC | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|
| DLinear | 7801 | 0.629 ± 0.075 | 0.673 ± 0.091 | 0.515 ± 0.070 | 0.471 ± 0.097 | 0.631 ± 0.224 | 0.462 ± 0.329 |
| PatchTST | 13154 | 0.741 ± 0.089 | 0.756 ± 0.103 | 0.633 ± 0.103 | 0.699 ± 0.127 | 0.598 ± 0.120 | 0.167 ± 0.112 |
| PatchTST-wide | 21409 | 0.743 ± 0.089 | 0.758 ± 0.107 | 0.636 ± 0.101 | 0.713 ± 0.132 | 0.600 ± 0.127 | 0.163 ± 0.120 |
| PatchTST-PLE-input | 13218 | 0.793 ± 0.084 | 0.804 ± 0.107 | 0.694 ± 0.095 | 0.802 ± 0.154 | 0.642 ± 0.146 | 0.114 ± 0.114 |
| TabM | 48656 | 0.782 ± 0.092 | 0.801 ± 0.091 | 0.666 ± 0.130 | 0.665 ± 0.196 | 0.688 ± 0.113 | 0.239 ± 0.182 |

Std AP qua seed (TB trên fold): DLinear 0.009, PatchTST 0.018, PatchTST-wide 0.027, PatchTST-PLE-input 0.014, TabM 0.007

## AP theo fold

| Fold | DLinear | PatchTST | PatchTST-wide | PatchTST-PLE-input | TabM |
|---|---|---|---|---|---|
| 0 | 0.585 | 0.661 | 0.652 | 0.691 | 0.681 |
| 1 | 0.534 | 0.685 | 0.692 | 0.746 | 0.688 |
| 2 | 0.725 | 0.879 | 0.872 | 0.911 | 0.872 |
| 3 | 0.624 | 0.780 | 0.793 | 0.834 | 0.817 |
| 4 | 0.676 | 0.698 | 0.706 | 0.782 | 0.854 |

## PLE-input so với từng mô hình (AP)

- vs DLinear: thắng 5/5 fold, chênh +0.164 (std của DLinear qua fold 0.075); seed-level thắng 25/25
- vs PatchTST: thắng 5/5 fold, chênh +0.052 (std của PatchTST qua fold 0.089); seed-level thắng 25/25
- vs PatchTST-wide: thắng 5/5 fold, chênh +0.050 (std của PatchTST-wide qua fold 0.089); seed-level thắng 25/25
- vs TabM: thắng 4/5 fold, chênh +0.011 (std của TabM qua fold 0.092); seed-level thắng 19/25

Nguồn mô hình: {'trained': 80, 'ckpt': 45}
Kiểm tra tái lập seed 0–2 so với CSV cũ: 75 lượt, |ΔAP| lớn nhất = 0.00e+00

## Che ngẫu nhiên test (5 seed × 3 lần che)

| Mức | PatchTST AP | PatchTST-wide AP | PatchTST-PLE-input AP | PLE-in thắng base | PLE-in thắng wide |
|---|---|---|---|---|---|
| 0% | 0.741 ± 0.089 | 0.743 ± 0.089 | 0.793 ± 0.084 | 5/5 | 5/5 |
| 10% | 0.670 ± 0.060 | 0.679 ± 0.060 | 0.729 ± 0.059 | 5/5 | 5/5 |
| 30% | 0.595 ± 0.049 | 0.599 ± 0.047 | 0.644 ± 0.040 | 5/5 | 5/5 |
| 50% | 0.535 ± 0.052 | 0.532 ± 0.054 | 0.569 ± 0.057 | 4/5 | 5/5 |

F1: 0%: 0.633/0.636/0.694; 10%: 0.559/0.556/0.609; 30%: 0.456/0.440/0.477; 50%: 0.331/0.301/0.322 (base/wide/PLE-in)
FPR: 0%: 0.167/0.163/0.114; 10%: 0.186/0.169/0.130; 30%: 0.144/0.124/0.092; 50%: 0.099/0.085/0.058 (base/wide/PLE-in)

- Chênh độ dốc AP 0→50% (PLE-in − PatchTST): -0.018 ± 0.052; PLE-in giảm ít hơn ở 2/5 fold
- Chênh độ dốc AP 0→50% (PLE-in − PatchTST-wide): -0.013 ± 0.049; PLE-in giảm ít hơn ở 2/5 fold
- ΔAP 0→50%: PatchTST -0.206, PatchTST-wide -0.211, PatchTST-PLE-input -0.224
