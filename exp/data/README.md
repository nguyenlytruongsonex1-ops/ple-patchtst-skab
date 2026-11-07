Đặt dữ liệu SKAB vào đây (không kèm trong repo): exp/data/anomaly-free/, exp/data/valve1/, exp/data/valve2/, exp/data/other/ (các file *.csv, sep=';').
Nguồn: https://github.com/waico/SKAB (thư mục data/), commit ghim trong exp/quantile_pilot/data_pins.json. Đối chiếu checksum: exp/data_hashes.json.

Bike Sharing (UCI 275), chỉ `hour.csv`: exp/data/bike/hour.csv. Tải và kiểm hash bằng `python3 exp/quantile_pilot/download_data.py`. Không commit CSV.
