# Speech-Processing

1. Sinh viên sử dụng dữ liệu đã thu (theo nhóm), trích xuất đặc trưng MFCC (39 đặc trưng, gồm cả MFCC, delta, deltadelta) của từng khẩu lệnh / con số
- folder /data chứa dữ liệu đã thu của nhóm
- File mfccs.py chứa chương trình trích xuất dặc trưng mfccs.
- `python mfccs.py` để chạy chương trình
2. Viết chương trình sử dụng DTW để nhận dạng khẩu lệnh đơn lẻ (mỗi từ/khẩu lệnh dùng khoảng 2-3 mẫu)
- File mfccs.py chứa chương trình sử dụng DTW để nhận dạng khẩu lệnh đơn lẻ.
- `python dtw.py` để chạy chương trình
3. Viết chương trình sử dụng HMM (segmental K-means) để nhận dạng khẩu lệnh đơn lẻ (sử dụng HMM với Mixture of Gaussians, sử dụng toàn bộ bộ dữ liệu đã thu).
- File mfccs.py chứa chương trình nhận dạng khẩu lệnh đơn lẻ (sử dụng HMM với Mixture of Gaussians, sử dụng toàn bộ bộ dữ liệu đã thu).
- `python hmm.py` để chạy chương trình

4. Link video