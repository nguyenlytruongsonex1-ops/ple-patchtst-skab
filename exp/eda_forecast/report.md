# Khảo sát dữ liệu SKAB: hướng dự báo giá trị cảm biến tương lai (EDA, chỉ đọc)

Dữ liệu: 34 file có nhãn trong `exp/data`, dùng cách nạp và chia fold y như `run_detect.py` (`splits.json`, 5 fold theo nhóm file). Không sửa `source/`. Code: `eda.py`, `ctl.py`.

## 1. Cảm biến nào thay đổi khi có bất thường
Tính riêng từng file (Cohen's d và KS của đoạn bất thường so với đoạn bình thường), rồi lấy trung vị trên 34 file (`effect_agg.csv`, `effect_size.png`).

| Cảm biến | median abs d | median d | KS median |
|---|---|---|---|
| Volume Flow RateRMS | 2.22 | −2.20 (giảm ở 85% số file) | 0.77 |
| Thermocouple | 0.94 | −0.57 | 0.48 |
| Temperature | 0.83 | −0.41 | 0.37 |
| Accelerometer2RMS | 0.42 | +0.14 | 0.18 |
| Accelerometer1RMS | 0.26 | +0.08 | 0.14 |
| Current / Voltage / Pressure | ≤0.09 | ≈0 | ≤0.07 |

Cảm biến tách tốt nhất là **lưu lượng (Volume Flow RateRMS)**, nó gần như luôn giảm khi có lỗi. Tiếp theo là hai cảm biến nhiệt. Current, Voltage và Pressure gần như không đổi.

**Có tín hiệu báo trước không?** Lấy 60 giây trước lúc bất thường bắt đầu (34 lần bắt đầu, mỗi file một lần) và kiểm tra xem có cảm biến nào lệch quá 3σ so với 60 giây trước đó không. Kết quả: 22/34 lần (65%) có lệch, trung vị 42 giây trước. Nhưng làm y hệt trên file anomaly-free (300 điểm ngẫu nhiên) cũng ra 68%, chủ yếu do Thermocouple trôi chậm. **Vậy chưa có bằng chứng về tín hiệu báo trước**, độ lệch này chỉ là nhiệt độ trôi bình thường (`onset_profiles.png`, `lead_time.csv`).

## 2. Dự báo giá trị tương lai có khả thi không
Dữ liệu bình thường, scaler và mô hình chỉ fit trên train của từng fold, đánh giá trên các file test. Mô hình AR10 là Ridge dùng 10 giây trước, dự báo phần thay đổi. R² gộp 5 fold (`forecast_metrics.csv`):

| Cảm biến | h=1 naive / AR | h=5 naive / AR | h=10 naive / AR |
|---|---|---|---|
| Thermocouple | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 |
| Temperature | 0.95 / 0.96 | 0.94 / 0.95 | 0.91 / 0.92 |
| Volume Flow | 0.91 / 0.86 | 0.79 / 0.69 | 0.76 / 0.62 |
| Accel1 | 0.77 / 0.62 | 0.15 / −0.77 | 0.05 / −1.03 |
| Accel2 | 0.58 / 0.44 | −0.08 / −0.87 | −0.13 / −0.96 |
| Current, Voltage, Pressure | ≤ −0.6 / ≈0 | ≈ −1 / ≈0 | ≈ −1 / ≈0 |

ACF trên dữ liệu bình thường (`acf.png`, `acf.csv`) cho thấy cùng điều đó. Nhiệt độ dự báo được rất tốt nhưng chỉ vì nó trôi chậm, nên dự báo dễ mà lại không có ích. Lưu lượng dự báo được vừa phải. Gia tốc thì sau 5 giây gần như là nhiễu. Current, Voltage và Pressure là nhiễu trắng, nên không dự báo được (R² ≈ 0 là mức trung bình). AR10 còn thua naive ở nhiều cảm biến, có thể do mức nền khác nhau giữa các file. Đây chỉ là baseline thô.

## 3. Sai số dự báo có tách được bất thường không
Điểm = tổng bình phương sai số AR của 8 cảm biến, làm mượt 5 giây, đánh giá trên test của từng fold (`residual_detect.csv`):

| h | AP (TB ± sd) | ROC-AUC | tỉ lệ dương |
|---|---|---|---|
| 1 | 0.40 ± 0.06 | 0.54 ± 0.05 | 0.35 |
| 5 | 0.39 ± 0.04 | 0.53 ± 0.04 | 0.35 |
| 10 | 0.43 ± 0.09 | 0.55 ± 0.06 | 0.35 |

Kết quả này chỉ nhỉnh hơn đoán ngẫu nhiên (AP ngẫu nhiên ≈ 0.35, ROC 0.5) và kém xa các mô hình có giám sát trên cùng split: PatchTST AP 0.74, PLE 0.78, TabM 0.78. Lý do là bất thường trong SKAB phần lớn là **mức thay đổi kéo dài** (lưu lượng tụt), chứ không phải cú sốc khó dự báo. Sau vài giây, mô hình dự báo đã "bắt kịp" mức mới nên sai số lại nhỏ.

## 4. File trùng hoặc chồng nhau
Giữ nguyên quy tắc trong `splits.json`: các file chồng khoảng thời gian hoặc trùng dữ liệu được gom chung một nhóm (ví dụ other/10–11, other/12–13, other/2–3–4), nên không lọt giữa train và test. Effect size và lead time tính theo từng file, nên file trùng được đếm hai lần trong trung vị. Ảnh hưởng nhỏ nhưng có. File anomaly-free chỉ dùng làm đối chứng, không có trong fold nào.

## Kết luận
- Cảm biến có ích nhất là Volume Flow RateRMS, rồi tới Thermocouple và Temperature. Current, Voltage và Pressure gần như vô dụng.
- Dự báo giá trị thì làm được cho nhiệt và lưu lượng ở 1–10 giây, nhưng **dùng sai số dự báo để phát hiện lỗi thì không khả thi** ở dạng đơn giản này (AP ≈ 0.40 so với 0.74–0.78 của mô hình có giám sát).
- Không thấy tín hiệu báo trước rõ ràng trong 60 giây trước lỗi.
- Nếu vẫn muốn đi hướng dự báo, cần mô hình mạnh hơn (ví dụ dự báo nhiều bước hoặc so mức hiện tại với "mức bình thường" học trên train) và kỳ vọng không cao. Hướng phân loại có giám sát hiện tại hợp với dữ liệu này hơn.
