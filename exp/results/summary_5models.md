# 5 mô hình (DLinear, PatchTST, wide, PLE, TabM; RBF tham khảo) — SKAB phát hiện tại t, patch 8/4, 5 fold × 3 seed

| model                    |   params |   s/run | AP              | AUC             | F1              | Precision       | Recall          | FPR             |
|:-------------------------|---------:|--------:|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| DLinear                  |     7801 |       3 | 0.6289 ± 0.0774 | 0.6747 ± 0.0914 | 0.5162 ± 0.0714 | 0.4690 ± 0.0970 | 0.6334 ± 0.2217 | 0.4636 ± 0.3268 |
| PatchTST                 |    13154 |     141 | 0.7420 ± 0.0865 | 0.7599 ± 0.1030 | 0.6351 ± 0.0990 | 0.6995 ± 0.1092 | 0.5998 ± 0.1308 | 0.1583 ± 0.0901 |
| PatchTST-wide            |    21409 |     221 | 0.7483 ± 0.0931 | 0.7602 ± 0.1130 | 0.6414 ± 0.1086 | 0.7321 ± 0.1274 | 0.5867 ± 0.1261 | 0.1336 ± 0.0894 |
| PatchTST-PLE             |    21411 |     240 | 0.7834 ± 0.0898 | 0.8092 ± 0.0852 | 0.6865 ± 0.0898 | 0.7202 ± 0.1294 | 0.6694 ± 0.0766 | 0.1662 ± 0.0964 |
| TabM                     |    48656 |       8 | 0.7821 ± 0.0924 | 0.8014 ± 0.0924 | 0.6666 ± 0.1295 | 0.6682 ± 0.1979 | 0.6879 ± 0.1235 | 0.2361 ± 0.1803 |
| PatchTST-RBF (tham khảo) |    21411 |     238 | 0.7655 ± 0.0948 | 0.8043 ± 0.0775 | 0.6699 ± 0.0733 | 0.6984 ± 0.1250 | 0.6712 ± 0.0460 | 0.2009 ± 0.1354 |

## AP theo fold

|   fold |   DLinear |   PatchTST |   PatchTST-wide |   PatchTST-PLE |   TabM |   PatchTST-RBF |
|-------:|----------:|-----------:|----------------:|---------------:|-------:|---------------:|
|      0 |    0.5863 |     0.66   |          0.6523 |         0.7332 | 0.6786 |         0.744  |
|      1 |    0.5266 |     0.6881 |          0.6838 |         0.6498 | 0.6878 |         0.608  |
|      2 |    0.7262 |     0.8778 |          0.8878 |         0.8587 | 0.8725 |         0.8219 |
|      3 |    0.6294 |     0.7726 |          0.7875 |         0.824  | 0.8186 |         0.8287 |
|      4 |    0.676  |     0.7115 |          0.7302 |         0.8511 | 0.853  |         0.8247 |

- DLinear thắng PatchTST: 0/5 fold, chênh AP TB -0.1131
- DLinear thắng PatchTST-wide: 0/5 fold, chênh AP TB -0.1194
- PatchTST-PLE thắng DLinear: 5/5 fold, chênh AP TB +0.1545
- PatchTST-PLE thắng PatchTST: 3/5 fold, chênh AP TB +0.0413
- PatchTST-PLE thắng PatchTST-wide: 3/5 fold, chênh AP TB +0.0351
- PatchTST-wide thắng DLinear: 5/5 fold, chênh AP TB +0.1194
- PatchTST-wide thắng PatchTST: 3/5 fold, chênh AP TB +0.0063
- TabM thắng DLinear: 5/5 fold, chênh AP TB +0.1532
- TabM thắng PatchTST: 3/5 fold, chênh AP TB +0.0401
- TabM thắng PatchTST-wide: 4/5 fold, chênh AP TB +0.0338

## FPR thật DLinear (đếm từ dự đoán test đã lưu, ngưỡng F1 chọn trên val)
- FPR thật DLinear: TB 0.4636 (n=15 lượt); FPR suy ra trong bảng: 0.4636
- PLE alpha: f0s0=0.1612, f0s1=0.2174, f0s2=0.1328, f1s0=0.1584, f1s1=0.1545, f1s2=0.1472, f2s0=0.1908, f2s1=0.1976, f2s2=0.2015, f3s0=0.1640, f3s1=0.1691, f3s2=0.1706, f4s0=0.2025, f4s1=0.1833, f4s2=0.2251
## Ghi chú
- DLinear mới chạy (15 lượt, ~3 s/lượt); các mô hình khác dùng lại kết quả cũ, cùng split/cấu hình. Split kiểm lại giống hệt splits.json gốc; audit rò rỉ 15/15 kiểm tra = 0.
- FPR các mô hình khác là suy ra từ P, R, tỉ lệ dương (khớp với đếm thật ở DLinear). Dự đoán test chỉ lưu cho DLinear.
- DLinear (tuyến tính) kém rõ rệt: AP 0.63 (ngẫu nhiên ≈ 0.35), thua mọi mô hình ở 5/5 fold, báo nhầm cao và rất dao động giữa các fold. Cần phi tuyến/attention để phân loại bất thường SKAB.
- Nhiều lượt DLinear có best epoch 26–30 → có thể chưa hội tụ ở 30 epoch (giữ nguyên theo kế hoạch).
- FPR TB 0.46 của DLinear bị kéo lên bởi fold 4 (FPR 0.98, recall 0.996): ngưỡng F1 chọn trên val quá thấp nên gần như báo động mọi thời điểm. Bỏ fold 4, FPR DLinear ≈ 0.33 — vẫn cao nhất.
