# Pilot quantile PLE — SKAB và Bike Sharing

Giai đoạn này chỉ có mã, giao thức và kiểm thử đúng đắn. **Chưa huấn luyện pilot.** Không có điểm AP/MAE của PatchTST, PLE quantile hay TabM cho thí nghiệm mới. Bảng cũ trong `README.md` giữ nguyên.

Số đo xử lý bên dưới được sinh bởi `verify_processing.py` trên file đã khớp hash, ghi ở `measured_processing.json` (`kind = processing_and_sanity_not_model_pilot`). Đó không phải điểm mô hình.

## Những gì đã có và được giữ

Nhánh `main` lúc bắt đầu chỉ có `7352437`. Không có nhánh khác, không có bản sửa quantile+RevIN nào đã commit để chép nguyên xi.

`run_detect.py` không đổi:

- PLE đều 8 bin trên `[-2, 2]` sau `StandardScaler`, `W` khởi tạo 0, cộng thặng dư `x' = x + W·PLE(x)`.
- PatchTST gọi với `revin=False`.
- Cửa sổ SKAB, scaler chỉ fit trên file train, 5 fold trong `splits.json`, checkpoint theo AP validation.
- `TabM.make(n_num_features=480, d_out=1, n_blocks=2, d_block=64, dropout=0.1, k=16)` từ gói `tabm==0.0.3`. Đếm tham số 48.656.

Pilot **không** thêm mô hình equal-bin vào 18 lượt. Mã đó chỉ để đối chiếu lịch sử.

## Ba phương pháp

Cùng cửa sổ, cùng đặc trưng, cùng đích, cùng seed, cùng epoch tối đa, patience, batch, AdamW `lr=1e-3`, `weight_decay=1e-4`, clip gradient 1. Không chỉnh riêng PLE.

| | PatchTST | PatchTST-PLE-quantile | TabM |
|---|---|---|---|
| Thân | PatchTST vendor, patch 8 / stride 4, `d_model=32`, 1 lớp, 4 head, `d_ff=64` | giống hệt, cộng PLE | `tabm.TabM`, `k=16`, 2 block, `d_block=64` |
| RevIN | bật, pre-affine rồi affine | cùng chỗ; PLE nằm **trước** affine | không. Đây là kiến trúc TabM, không gắn RevIN |
| Đầu | flatten mọi kênh rồi `Linear` ra đích | cùng đầu | làm phẳng `[thời gian, kênh]` (C-contiguous) thành một vector |
| Tham số SKAB | 13.170 | 13.234 (+64) | 48.656 |
| Tham số Bike | 429.555 | 429.659 (+104) | 208.896 |

13.170 = 13.154 của PatchTST cũ + 16 tham số affine RevIN. Phần +64 / +104 là ma trận `W` (8 hoặc 13 kênh × 8 bin), không phải lớp mới.

Đầu flatten trộn kênh là đầu đã dùng trong `run_detect.py`. Đầu tách từng biến của bài PatchTST không trộn thời tiết vào dự báo `cnt`, trong khi TabM thấy toàn bộ vector. Pilot không dùng đầu tách biến. Hệ quả: với Bike (`H=24`, 13 kênh, 42 patch) lớp `Linear` chiếm gần hết 429.555 tham số. Không nới `d_block` của TabM cho bằng số tham số — ngân sách bằng nhau là epoch, patience, batch, lr, seed.

`PatchTST_backbone` nhận `norm='LayerNorm'` nhưng **không truyền** sang encoder. Encoder vẫn là `BatchNorm1d` mặc định, đúng lời gọi cũ. RevIN không có running stats. `predict` gọi `eval()`: BatchNorm dùng running mean, dropout tắt, hai lần eval trùng nhau, biên bin không đổi. Kiểm thử khóa hành vi này.

Gói TabM cảnh báo khi không truyền `start_scaling_init_chunks`. Pilot không truyền, giống `run_detect.py`, để khỏi thành siêu tham số mới.

## PLE quantile và RevIN

RevIN (mã vendor `layers/RevIN.py`): trên tensor `[batch, thời gian, kênh]`, mean/std theo **trục thời gian của từng cửa sổ, từng kênh** (`unbiased=False`, `sqrt(var+1e-5)`), không theo batch và không trộn kênh. Affine học được đặt **sau** bước này. `subtract_last=False`.

PLE đọc đúng tọa độ pre-affine đó. Biên bin là `torch.quantile` trên mọi bước thời gian của **mọi cửa sổ train**, mỗi kênh một bộ biên, 8 mức `linspace(0,1,9)`. Không fit trên raw, không fit trên dữ liệu chỉ mới qua `StandardScaler`, không dùng nhãn, val hay test. Affine đổi khi train nên không nằm trong hệ tọa độ của biên. Lúc khởi tạo affine là 1 và 0.

Công thức khớp `PLEInput` cũ và phần trong của `rtdl_num_embeddings`: bin đã đi qua = 1, bin đang chứa = nội suy, bin phía sau = 0. **Ngoài hai biên ngoài cùng thì kẹp về 0 hoặc 1, không ngoại suy tuyến tính.** Bản nhanh của rtdl để thành phần đầu < 0 và thành phần cuối > 1; pilot không dùng kiểu đó. Kiểm thử khóa cả hai phía.

Biên trùng (độ rộng ≤ `1e-8` trong đơn vị z-score) thành bước: 0 nếu nhỏ hơn mốc, 1 nếu ≥ mốc. Cảm biến hằng sau RevIN ra toàn 0, mọi biên trùng, forward/backward hữu hạn. Không bỏ bin nên số tham số không phụ thuộc fold. `rtdl.compute_bins` từ chối cột hằng và xóa biên trùng; pilot không gọi nó để fit, chỉ dùng làm mốc đối chiếu khi biên tăng nghiêm ngặt.

Padding `ReplicationPad1d` của PatchTST nằm sau PLE, như bản gốc. Biên không fit trên bước pad.

Phân loại trả logit. **Không** denorm. Nhân-cộng một kênh đầu vào không đổi logit khi RevIN đã chuẩn hóa kênh đó. Dự báo denorm kênh `cnt` bằng đúng công thức vendor (trừ bias, chia `weight+eps²`, nhân std cửa sổ, cộng mean cửa sổ) ngay trong forward vừa tính stats, rồi đưa về đơn vị `cnt` gốc bằng scaler train. Đích dự báo không bị denorm như thể nó là một cảm biến phân loại.

`W` khởi tạo 0 nên khi copy trọng số chung, PLE và PatchTST trùng nhau. Seed chỉ ảnh hưởng khởi tạo mạng và thứ tự batch, sau khi biên và scaler đã fit.

## SKAB — phân loại bất thường

- Nguồn: [SKAB @ `b2c0d46`](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2) (2024-08-11). 35 file CSV khớp `exp/data_hashes.json` (0 lệch). DOI upstream `10.34740/KAGGLE/DSV/1693952`.
- README upstream ghi cột `RateRMS`; file CSV (đúng hash của repo) dùng `Volume Flow RateRMS`. Pilot dùng tên trong file.
- Nhãn là `anomaly` (0/1) **tại thời điểm t**, mép phải cửa sổ. Không dùng `changepoint`. Không `pos_weight`.
- `anomaly-free/anomaly-free.csv` **không có** cột `anomaly` hay `changepoint`. Không nằm trong fold nào. Không dùng để huấn luyện. EDA cũ chỉ dùng file này làm đối chứng.
- Thời gian không đều: 34 file có nhãn có 1.993 khoảng `dt>1` giây, `dt≤0` = 0, trung vị mỗi file là 1 giây. Quy ước cũ, giữ cho cả ba phương pháp: lưới 1 giây trong từng file, `ffill(limit=1)` chỉ trên cảm biến (lấp đúng một giây từ giá trị ngay trước; không `bfill`, không đụng nhãn). Ô lỗ: 21.336; lấp được 15.944; còn NaN 5.392. Cửa sổ là 60 mốc lưới, bước 4. Ứng viên 9.532, giữ 8.675. Đây **không** phải 60 giây vật lý đều.
- Cửa sổ không vượt file. Nhóm trong `splits.json` (chồng thời gian, hoặc trùng timestamp+8 cảm biến, hoặc trùng 8 cảm biến) nằm trọn một phía train/val/test ở mọi fold. Không chia ngẫu nhiên theo dòng.
- Trùng đo được trên file đã hash: 718 bộ 8 cảm biến xuất hiện ở hơn một file (1.436 dòng; cũng là 718 khóa timestamp+cảm biến). 235 khóa trong số đó có nhãn `anomaly` mâu thuẫn (470 dòng). Gợi ý EDA đính kèm nói 1.290 và 235: **235 khóa mâu thuẫn khớp**; 1.290 **không** khớp định nghĩa tuple chính xác này. Không xóa trùng — xóa sẽ lệch protocol cũ. Trùng đã bị nhóm nên không lọt qua fold.
- Scaler: `StandardScaler` fit trên dòng raw của file train, rồi áp lên cửa sổ. Không có imputer học tham số.
- Fold pilot, chọn trước, cùng cho mọi mô hình: **0 và 1** (hai bucket test đầu của 5 fold). Không chọn theo test.

| Fold | train | val | test | tỉ lệ dương test |
|---|---:|---:|---:|---:|
| 0 | 5.067 | 1.837 | 1.771 | 0,3834 |
| 1 | 5.186 | 1.652 | 1.837 | 0,3674 |

Đúng số mẫu fold 0 của pipeline cũ. Checkpoint: **AP validation** (càng cao càng tốt), patience 3, tối đa 10 epoch. Ngưỡng F1 trên val chỉ để tính F1/FPR phụ, sau khi đã chốt checkpoint. Test không tham gia chọn mô hình hay ngưỡng. Chỉ số chính: **AP**.

## Bike — dự báo `cnt`

- Nguồn: [UCI Bike Sharing, dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), zip `sha256 b70182d0…5401`, `hour.csv` `sha256 e03de4ee…504f`, md5 `d50bd5a6f55131e72a7bedc334e2fce1` (trùng gợi ý đính kèm sau khi tải zip UCI). 17.379 dòng, `casual+registered=cnt` trên mọi dòng, `cnt` từ 1 đến 977.
- Đồng hồ = `dteday + hr`, từ `2011-01-01 00:00` đến `2012-12-31 23:00`. Lưới đủ 17.544 giờ, **thiếu 165 giờ** (76 ngày có lỗ). Không điền 0, không nội suy, không dùng giá trị tương lai. `cnt` không bao giờ bằng 0 trong file; giờ thiếu là giờ không có dòng.
- `windspeed = 0` có **2.180** dòng và `hum = 0` có **22** dòng. Đó là giá trị ghi trong file, **không** coi là missing và không sửa.
- Ngày nhiều giờ thiếu nhất (đếm trên lưới, không gán nguyên nhân): 2012-10-29 (23), 2011-01-27 (16), 2012-10-30 (13), 2011-01-18 (12), 2011-01-26 (8).
- Bỏ `instant`, `casual`, `registered`. Đặc trưng quá khứ: `cnt, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed` (mã số như file UCI, không one-hot).
- Lookback **168 giờ đồng hồ**, horizon **H=24**. Repo không có protocol dự báo Bike trước đó nên không có mốc để giữ thay cho cặp này. Cửa sổ phải đủ giờ trên đồng hồ; không lấy 168 dòng quan sát nếu chúng nhảy giờ.
- Chia theo **lưới giờ đầy đủ**, 70% / 15% / 15%, không theo số dòng quan sát (lỗ sẽ kéo mốc). Mốc đo được: train kết thúc `2012-05-26 16:00`, val kết thúc `2012-09-13 08:00` (mốc là giờ đầu của kỳ sau). Một mẫu chỉ thuộc một kỳ nếu **cả 24 giờ đích** nằm trong kỳ. Cửa sổ cắt mốc thì bỏ. Context được bắt đầu ở kỳ trước (quá khứ đã quan sát); không chứa giờ đích hay giờ sau đó. Gợi ý EDA chia 70/15/15 trên dòng quan sát (khoảng `2012-05-27 01:00` và `2012-09-12 16:00`) **không** phải protocol này.
- Lịch tương lai biết được về nguyên tắc nhưng **không** đưa vào, để cả ba mô hình cùng nhìn một tensor quá khứ. Thời tiết tương lai không có và không dùng.
- Scaler fit trên các giờ hữu hạn **trước** mốc train. Sửa val/test không đổi mean. Không imputer.

Ứng viên 17.353. Bỏ vì đích cắt mốc: phần còn lại chia như sau.

| Kỳ | giữ | bỏ vì lỗ | context lấn kỳ trước |
|---|---:|---:|---:|
| train | 6.984 | 5.105 | 0 |
| val | 2.609 | 0 | 168 |
| test | 1.810 | 799 | 168 |

Bỏ cửa sổ chạm lỗ là hệ quả lớn, nhất là train (nhiều lỗ đầu 2011) và test (lỗ cuối 2012). Val trên đoạn này không mất cửa sổ vì lỗ. Không điền để khỏi thêm một mô hình mask. Ba phương pháp cùng tập cửa sổ này.

Biên quantile Bike: khoảng `[-12,9, 12,9]`, **24,0%** độ rộng bin bị coi là trùng — chủ yếu kênh lịch gần như hằng trong 168 giờ (`yr`, thường cả `season`/`mnth`) sau RevIN thành 0. SKAB fold 0 và 1: 0% bin trùng, biên khoảng ±7,6 (đuôi z-score trong cửa sổ, không phải lưới `[-2,2]` của PLE đều).

Chỉ số chính: **MAE** trên `cnt` gốc, gộp mọi giờ đích. RMSE là phụ. Checkpoint: MAE val, patience 3, tối đa 10 epoch. Test không dùng để chọn.

Naive mùa vụ (không train), cùng 1.810 cửa sổ test: trễ 24 giờ MAE 83,25 / RMSE 137,16; trễ 168 giờ MAE 59,04 / RMSE 103,15. Mốc tham khảo, không phải một phương pháp thứ tư.

## Siêu tham số pilot

- Seed `0` rồi `1`. Vòng seed nằm ngoài cùng.
- Tối đa 10 epoch, patience 3, batch 256, 4 luồng CPU, tuần tự.
- 8 bin, chọn vì PLE đều cũ dùng 8, không dò trên test.
- Fold SKAB 0 và 1. Bike một split thời gian.
- 3×2×(2+1) = **18** lượt. Script từ chối kế hoạch dài hơn trừ khi `--allow-large` (không dùng cho pilot). 5 fold × 5 seed không nằm trong giai đoạn này.

## Lệnh

Từ gốc repo, sau khi mã đã ở trên GitHub và được duyệt:

```bash
PYTHONPATH=exp python3 exp/quantile_pilot/download_data.py
PYTHONPATH=exp python3 exp/quantile_pilot/run_pilot.py
```

`download_data.py` tải đúng commit SKAB và zip UCI, kiểm sha256, không ghi đè nếu lệch hash. CSV nằm ở `exp/data/` và bị gitignore.

Chạy sàng seed 0 cho **cả ba** mô hình trước, rồi seed còn lại (lượt đã `ok` trong registry được bỏ qua):

```bash
PYTHONPATH=exp python3 exp/quantile_pilot/run_pilot.py --seeds 0
PYTHONPATH=exp python3 exp/quantile_pilot/run_pilot.py --seeds 1
```

Xem kế hoạch, không train: `python3 exp/quantile_pilot/run_pilot.py --dry-run`.

Naive, cũng không train: `PYTHONPATH=exp python3 exp/quantile_pilot/seasonal_naive.py`.

## Ngân sách

Chưa bấm giờ epoch của mã mới — cố ý chưa train.

Mốc cũ, máy của lần chạy trước, 1 thread, PatchTST `revin=False`: `exp/results/f0_PatchTST_s0.csv` ghi 151,7 giây cho 30 epoch. Pilot trần 10 epoch và có patience, cùng cỡ đầu vào SKAB. Mười hai lượt SKAB vì vậy có thể nằm trong khoảng vài chục phút nếu máy tương đương, nhưng đây là suy từ mốc cũ chứ không phải đo mới.

Bike nặng hơn rõ: chuỗi 168 so với 60, 42 patch so với 15, đầu PatchTST khoảng 430 nghìn tham số. Chưa có số giây. Đề xuất: chạy `--seeds 0` (9 lượt) trên 4 luồng, một job một lúc. Nếu một lượt Bike vượt khoảng 15 phút, dừng và báo trước khi mở seed 1. Trần tường để đủ 18 lượt: khoảng 3 giờ. Không kết luận ý nghĩa thống kê từ 2 fold và 2 seed.

## Đăng ký lần chạy

`exp/quantile_pilot/results/registry.jsonl` (gitignore) và checkpoint trong `exp/quantile_pilot/ckpt/`. Mỗi dòng: dataset, fold, model, seed, status `ok` hoặc `error`, traceback nếu lỗi, số tham số, cỡ split, epoch tốt nhất, số epoch đã chạy, history val, chỉ số val/test, `preprocess_seconds`, `train_seconds`, `eval_seconds`, `seconds` (cộng cả tiền xử lý). Lỗi được ghi rồi dừng. Chưa có dòng nào vì chưa train.

## Kiểm thử đã chạy

```bash
PYTHONPATH=exp python3 -m unittest quantile_pilot.tests.test_correctness
```

Kết quả trên sandbox này: **18 tests, OK, 1,495 giây.** Không có lượt train pilot. Kiểm thử khóa: giá trị PLE và gradient, kẹp ngoài biên, khác ngoại suy rtdl, biên trùng và kênh hằng hữu hạn, biên đúng quantile của RevIN chứ không phải raw, dịch vị trí không đổi biên còn đổi dạng trong cửa sổ thì đổi, val không được truyền vào hàm fit, serialize, RevIN khớp vendor theo trục thời gian, denorm đúng kênh, phân loại không denorm, dự báo biến đổi đúng khi co giãn kênh `cnt`, affine không nằm trong tọa độ PLE, `eval` đóng băng BatchNorm, `W=0` trùng PatchTST, TabM đúng lớp `tabm.TabM` và 48.656 tham số, nhóm file SKAB, `anomaly-free` không vào fold, ffill một giây và không lấy `changepoint`, Bike bỏ casual/registered/instant, lỗ không thành 0, context được lấn biên, đích không cắt split, scaler không thấy test, batch cuối kích thước 1 được gộp, fold 0 đủ hash thì đúng 5.067/1.837/1.771 mẫu.

## Không làm trong giai đoạn này

Không train, không sweep, không gộp `main`, không sửa kết quả cũ, không thêm Conv hay mô hình khác.
