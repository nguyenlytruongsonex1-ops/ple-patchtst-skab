# D10 – Che ngẫu nhiên đầu vào test (SKAB)

Số lượt: 45 (cần 45). Mean ± std qua 5 fold (mỗi fold = trung bình 3 seed × 3 lần che). Ngưỡng giữ nguyên từ val không che.

## AP

| Mức che | PatchTST | PatchTST-wide | PatchTST-PLE-input |
|---|---|---|---|
| 0% | 0.742 ± 0.086 | 0.748 ± 0.093 | 0.791 ± 0.089 |
| 10% | 0.666 ± 0.059 | 0.679 ± 0.065 | 0.729 ± 0.062 |
| 30% | 0.589 ± 0.050 | 0.596 ± 0.050 | 0.645 ± 0.035 |
| 50% | 0.524 ± 0.053 | 0.523 ± 0.053 | 0.565 ± 0.052 |

## AUC

| Mức che | PatchTST | PatchTST-wide | PatchTST-PLE-input |
|---|---|---|---|
| 0% | 0.760 ± 0.103 | 0.760 ± 0.113 | 0.802 ± 0.110 |
| 10% | 0.711 ± 0.071 | 0.713 ± 0.084 | 0.754 ± 0.081 |
| 30% | 0.659 ± 0.043 | 0.662 ± 0.051 | 0.697 ± 0.037 |
| 50% | 0.610 ± 0.033 | 0.611 ± 0.039 | 0.641 ± 0.032 |

## F1

| Mức che | PatchTST | PatchTST-wide | PatchTST-PLE-input |
|---|---|---|---|
| 0% | 0.635 ± 0.099 | 0.641 ± 0.109 | 0.688 ± 0.105 |
| 10% | 0.555 ± 0.051 | 0.557 ± 0.067 | 0.607 ± 0.074 |
| 30% | 0.436 ± 0.074 | 0.430 ± 0.094 | 0.472 ± 0.088 |
| 50% | 0.292 ± 0.118 | 0.276 ± 0.118 | 0.303 ± 0.120 |

## FPR

| Mức che | PatchTST | PatchTST-wide | PatchTST-PLE-input |
|---|---|---|---|
| 0% | 0.158 ± 0.090 | 0.134 ± 0.089 | 0.106 ± 0.104 |
| 10% | 0.169 ± 0.089 | 0.145 ± 0.082 | 0.121 ± 0.068 |
| 30% | 0.120 ± 0.092 | 0.104 ± 0.080 | 0.083 ± 0.055 |
| 50% | 0.078 ± 0.092 | 0.073 ± 0.085 | 0.051 ± 0.047 |

## Suy giảm AP (0% → 50%) và thắng theo fold

| Mô hình | ΔAP mean | std qua fold | std qua seed (TB fold) |
|---|---|---|---|
| PatchTST | -0.218 | 0.111 | 0.031 |
| PatchTST-wide | -0.226 | 0.118 | 0.020 |
| PatchTST-PLE-input | -0.226 | 0.113 | 0.027 |

- PLE-input thắng PatchTST theo AP: 0%: 5/5, 10%: 5/5, 30%: 5/5, 50%: 4/5
  - Chênh độ dốc (ΔAP PLE-input − ΔAP PatchTST): -0.008 ± 0.046 qua fold; fold PLE-input suy giảm ít hơn: 2/5
- PLE-input thắng PatchTST-wide theo AP: 0%: 5/5, 10%: 5/5, 30%: 4/5, 50%: 4/5
  - Chênh độ dốc (ΔAP PLE-input − ΔAP PatchTST-wide): +0.000 ± 0.034 qua fold; fold PLE-input suy giảm ít hơn: 3/5

## Diễn giải (trung thực)
- Không có checkpoint cũ → huấn luyện lại 45 lượt với đúng cấu hình cố định; AP khi không che trùng khớp kết quả D7/D9 (vd f0 PatchTST s0 = 0.6717), nên mô hình tái lập được.
- Mặt nạ giống hệt nhau cho mọi mô hình/seed trong cùng fold (seed = 10000·fold + 100·mức + lần), che từng giá trị (60×8), điền 0, đặt trước PLE/patching. Chỉ che test.
- **Không có bằng chứng PLE-input bền hơn:** AP của cả 3 mô hình giảm gần như cùng tốc độ (−0.218 / −0.226 / −0.226 khi che 50%). Chênh độ dốc ≈ 0 (−0.008 ± 0.046 so với PatchTST; +0.000 ± 0.034 so với wide), nhỏ hơn nhiều so với độ dao động qua fold, và PLE-input suy giảm ít hơn chỉ ở 2/5 và 3/5 fold.
- Điều thấy được: PLE-input **giữ được khoảng cách** ban đầu (+0.04–0.06 AP) ở mọi mức che, thắng PatchTST 5/5, 5/5, 5/5, 4/5 fold. Tức là ưu thế ban đầu không biến mất khi thiếu dữ liệu, nhưng cũng không lớn thêm.
- Ngưỡng cố định từ val không che làm F1 sụp mạnh (≈0.29–0.30 ở 50%) và FPR giảm: mô hình báo động ít hơn và bỏ sót nhiều hơn khi đầu vào bị kéo về trung bình. Đây là hạn chế chung, không riêng mô hình nào.
- Hình: exp/results/masking.png. Dữ liệu từng lượt: exp/results/masking/*.csv; checkpoint: exp/ckpt/.
