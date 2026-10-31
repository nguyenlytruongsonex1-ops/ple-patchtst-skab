# PLE-input PatchTST trên SKAB

Báo cáo môn **Học sâu và Ứng dụng trong kinh doanh - IS6109.11.CH**, Trường Đại học Công nghệ Thông tin (UIT).
Giảng viên: **TS. Tạ Hoàng Thắng**. Học viên: **Nguyễn Lý Trường Sơn**.

## Ý tưởng
PLE-input PatchTST thêm một lớp Piecewise Linear Encoding (PLE) ngay trên đầu vào, trước bước chia patch:

x' = x + W · PLE(x)

- PLE dùng 8 bin đều trên [-2, 2] (sau chuẩn hoá), áp riêng cho từng kênh.
- W khởi tạo bằng 0, nên lúc đầu mô hình giống hệt PatchTST gốc.
- Chỉ thêm 64 tham số (13.218 so với 13.154).

## Thiết lập
- Dữ liệu: SKAB (8 cảm biến). Nhãn là bất thường tại thời điểm t, cửa sổ 60 giây.
- PatchTST: patch 8, stride 4 (15 patch).
- Chia 5 fold theo nhóm file (không rò rỉ dữ liệu) × 5 seed. Siêu tham số cố định; checkpoint và ngưỡng chọn trên tập val.
- Chỉ số chính: AP (Average Precision). Mô hình so sánh: DLinear, PatchTST, PatchTST-wide, TabM (gói `tabm` công khai).

## Tải dữ liệu
Dữ liệu không kèm trong repo. Tải từ https://github.com/waico/SKAB, rồi chép các thư mục `anomaly-free/`, `valve1/`, `valve2/`, `other/` trong `data/` vào `exp/data/` (xem `exp/data/README.md`).

## Cách chạy
```bash
pip install -r requirements.txt
cd exp
bash run5.sh                 # 5 mô hình × 5 fold × 5 seed -> results/
python3 summarize5s.py       # bảng tổng hợp
bash run_mask.sh             # thực nghiệm che ngẫu nhiên đầu vào test
python3 summarize_mask.py
cd eda_forecast && python3 eda.py   # EDA (effect size, lead time)
```
Chạy lẻ: `python3 eval5.py --fold 0 --model PatchTST-PLE-input --seed 0`.

## Kết quả (5 fold × 5 seed, mean ± std qua fold)
| Mô hình | Tham số | AP | FPR |
|---|---|---|---|
| DLinear | 7.801 | 0.629 ± 0.075 | 0.462 |
| PatchTST | 13.154 | 0.741 ± 0.089 | 0.167 |
| PatchTST-wide | 21.409 | 0.743 ± 0.089 | 0.163 |
| PatchTST-PLE-input | 13.218 | 0.793 ± 0.084 | 0.114 |
| TabM | 48.656 | 0.782 ± 0.092 | 0.239 |

Nhận xét trung thực:
- PLE-input hơn PatchTST +0.052 AP, thắng 25/25 lượt, nhưng mức chênh **nhỏ hơn 1 độ lệch chuẩn** giữa các fold (0.089).
- So với TabM chỉ hơn +0.011 (thắng 4/5 fold).
- Khi che ngẫu nhiên 0→50% đầu vào, AP giảm: PatchTST −0.206, PatchTST-wide −0.211, PLE-input −0.224. Tức là PLE-input **không bền hơn** khi thiếu dữ liệu.

Chi tiết: `exp/results/summary_5models_5seeds.md`, `exp/results/summary_masking.md`, `exp/eda_forecast/report.md`, nhật ký quyết định `decision_log.md`.

## Ghi chú
- Nhánh PatchTST-RBF (và PLE-FFN) có trong code/kết quả cũ nhưng đã được gác lại, không dùng trong báo cáo.
- Bike Sharing và các bộ chuỗi thời gian khác là hướng mở rộng tương lai.
- `exp/vendor/` chứa mã PatchTST gốc kèm giấy phép (`PatchTST_LICENSE`). Dữ liệu SKAB theo giấy phép của SKAB.
