# PatchTST-PLE-input — SKAB phát hiện tại t, cửa sổ 60 s, patch 8/4, 5 fold nhóm × 3 seed

x' = x + W_c·PLE(x) theo từng kênh (8 bin đều [-2,2], W khởi tạo 0, không bias, không chuẩn hoá) → PatchTST gốc (1 lớp, d_model 32, 4 head, d_ff 64, RevIN tắt). Chỉ thêm 8×8 = 64 tham số (13,218 vs 13,154).

| model                  |   params |   s/run | AP              | AUC             | F1              | Precision       | Recall          | FPR             |
|:-----------------------|---------:|--------:|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| DLinear                |     7801 |       3 | 0.6289 ± 0.0774 | 0.6747 ± 0.0914 | 0.5162 ± 0.0714 | 0.4690 ± 0.0970 | 0.6334 ± 0.2217 | 0.4636 ± 0.3268 |
| PatchTST               |    13154 |     141 | 0.7420 ± 0.0865 | 0.7599 ± 0.1030 | 0.6351 ± 0.0990 | 0.6995 ± 0.1092 | 0.5998 ± 0.1308 | 0.1583 ± 0.0901 |
| PatchTST-wide          |    21409 |     221 | 0.7483 ± 0.0931 | 0.7602 ± 0.1130 | 0.6414 ± 0.1086 | 0.7321 ± 0.1274 | 0.5867 ± 0.1261 | 0.1336 ± 0.0894 |
| PatchTST-PLE (PLE-FFN) |    21411 |     240 | 0.7834 ± 0.0898 | 0.8092 ± 0.0852 | 0.6865 ± 0.0898 | 0.7202 ± 0.1294 | 0.6694 ± 0.0766 | 0.1662 ± 0.0964 |
| PatchTST-PLE-input     |    13218 |     102 | 0.7909 ± 0.0894 | 0.8020 ± 0.1098 | 0.6884 ± 0.1047 | 0.8075 ± 0.1501 | 0.6296 ± 0.1535 | 0.1062 ± 0.1037 |
| TabM                   |    48656 |       8 | 0.7821 ± 0.0924 | 0.8014 ± 0.0924 | 0.6666 ± 0.1295 | 0.6682 ± 0.1979 | 0.6879 ± 0.1235 | 0.2361 ± 0.1803 |

(± = độ lệch chuẩn giữa 5 fold, trên trung bình 3 seed/fold. FPR suy ra từ P, R, tỉ lệ dương.)

## AP theo fold

|   fold |   DLinear |   PatchTST |   PatchTST-wide |   PatchTST-PLE |   PatchTST-PLE-input |   TabM |
|-------:|----------:|-----------:|----------------:|---------------:|---------------------:|-------:|
|      0 |    0.5863 |     0.66   |          0.6523 |         0.7332 |               0.6859 | 0.6786 |
|      1 |    0.5266 |     0.6881 |          0.6838 |         0.6498 |               0.7386 | 0.6878 |
|      2 |    0.7262 |     0.8778 |          0.8878 |         0.8587 |               0.9161 | 0.8725 |
|      3 |    0.6294 |     0.7726 |          0.7875 |         0.824  |               0.839  | 0.8186 |
|      4 |    0.676  |     0.7115 |          0.7302 |         0.8511 |               0.7751 | 0.853  |

## Số fold thắng (AP)
- PLE-input thắng PatchTST: 5/5, chênh AP TB +0.0489 (std PatchTST: 0.0865)
- PLE-input thắng PatchTST-wide: 5/5, chênh AP TB +0.0426 (std PatchTST-wide: 0.0931)
- PLE-input thắng PatchTST-PLE: 3/5, chênh AP TB +0.0076 (std PatchTST-PLE: 0.0898)

## FPR thật (đếm từ dự đoán test)
- PLE-input: 0.1062 ± 0.1007 (n=15); suy ra: 0.1062

## Độ lớn W đã học (khởi tạo 0)
- |W| TB 0.0571, max 0.4289 (trên 15 lượt)
- |W| TB theo kênh: Acc1=0.046, Acc2=0.049, Current=0.042, Pressure=0.075, Temp=0.073, Thermo=0.056, Voltage=0.055, Flow=0.061
## Ghi chú trung thực
- Số tham số 13,218 ≈ PatchTST gốc 13,154 (+0.5%) → đối chứng gần nhất là PatchTST gốc, không phải bản wide (21,409).
- Tiêu chí cũ (thắng ≥4/5 fold so với cả PatchTST và wide, và std không lớn hơn max(std PatchTST, std wide)): thắng 5/5 và 5/5; std 0.089 ≤ 0.093 → **ĐẠT** theo nhánh "thắng fold". Nhánh "gain ≥ 1 std" KHÔNG đạt (+0.049 < 0.087). Std hơi lớn hơn PatchTST gốc (0.089 vs 0.087).
- So với PLE-FFN: ngang nhau (thắng 3/5, +0.008) — không khẳng định được bản nào hơn.
- Precision cao nhất / FPR thấp nhất (0.106, đếm thật từ dự đoán test) nhưng recall thấp hơn PLE-FFN/TabM (0.63 vs 0.67–0.69); ROC-AUC không hơn PLE-FFN.
- Chỉ 3 seed × 5 fold; thắng 5/5 vẫn cần thêm seed để chắc chắn. |W| học được nhỏ (TB 0.057) nhưng khác 0 ở mọi kênh, lớn nhất ở Pressure/Temperature.
- best epoch 8–29 (TB 18), ít lo chưa hội tụ. Thời gian/lượt (102 s) do chạy song song 5 fold nên không so sánh trực tiếp với mô hình cũ.
- Audit rò rỉ chạy lại: 15/15 kiểm tra = 0 trùng/chồng; splits.json giống hệt bản gốc. Các mô hình khác dùng lại kết quả cũ.
