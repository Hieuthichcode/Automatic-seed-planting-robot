# Nhận diện cây bằng YOLOv8 TFLite

Tài liệu này mô tả pipeline nhận diện đang được `app.py` sử dụng. Trạng thái
được kiểm tra ngày **23/10/2026**.

## 1. Model đang sử dụng

Model mặc định:

```text
model/new_float32.tflite
```

Metadata đã kiểm tra:

| Thuộc tính | Giá trị |
|---|---|
| Framework xuất model | Ultralytics YOLOv8 |
| Task | Detect |
| Input | `[1, 640, 640, 3]`, NHWC, `float32` |
| Output | `[1, 6, 8400]`, `float32` |
| Class 0 | `Disease` / `Benh` |
| Class 1 | `Normal` / `Binh_thuong` |
| Objectness riêng | Không |

Với hai lớp, sáu giá trị của mỗi dự đoán YOLOv8 là:

```text
x, y, width, height, confidence_Disease, confidence_Normal
```

`8400` là số vị trí dự đoán trước khi lọc confidence và NMS, không phải số cây
được phát hiện.

`detect_stream.py` cũng tương thích với output YOLOv5 hai lớp dạng
`[1, N, 7]`:

```text
x, y, width, height, objectness, confidence_Disease, confidence_Normal
```

YOLOv5 tính confidence cuối bằng `objectness × confidence lớp`; YOLOv8 dùng
trực tiếp confidence lớp.

## 2. Cấu trúc liên quan

| File/thư mục | Vai trò |
|---|---|
| `app.py` | Flask server và API detection |
| `detect_stream.py` | Nạp TFLite, camera, tiền/hậu xử lý và vẽ box |
| `detect_web.html` | Giao diện camera và ảnh 2D |
| `test.py` | Công cụ Tkinter thử TFLite/YOLO bằng RealSense hoặc ảnh |
| `classes.txt` | Tên và thứ tự hai lớp |
| `model/` | Thư mục chứa các model TFLite |
| `settings.json` | Cấu hình được lưu vĩnh viễn từ trang Config |
| `tests/test_detection_geometry.py` | Kiểm thử resize và quy đổi tọa độ box |

`detect.py` là tiện ích standalone cũ, không được `app.py` import. Pipeline
chính nằm trong `detect_stream.py`.

## 3. Pipeline xử lý

Ảnh tải lên và camera realtime dùng chung quy trình:

```text
Ảnh BGR gốc
  -> letterbox theo kích thước input đọc từ model
  -> BGR sang RGB
  -> chuẩn hóa/lượng tử hóa theo dtype model
  -> TFLite inference
  -> nhận dạng layout YOLOv8 hoặc YOLOv5
  -> lọc confidence
  -> NMS theo IoU
  -> bỏ padding và scale box về ảnh gốc
  -> vẽ box lên đúng frame nguồn
```

`letterbox` giữ nguyên tỷ lệ ảnh và thêm viền, không kéo méo ảnh. Ảnh kết quả
giữ nguyên chiều rộng/chiều cao của ảnh gốc.

Luồng camera ghép mỗi kết quả inference với chính frame đã đưa vào model. Không
vẽ kết quả của frame cũ lên frame mới, tránh box bị lệch khi camera hoặc cây
chuyển động.

Màu box:

- Đỏ `(0, 0, 255)`: class 0, cây bệnh.
- Xanh `(0, 255, 0)`: class 1, cây bình thường.

## 4. Chạy ứng dụng

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

Chạy từ thư mục `CODE/GPS_RASPI`:

```powershell
python app.py
```

Server dùng cổng HTTP 80:

- Trang chính: `http://control.local/`
- Detection: `http://control.local/detection`
- Cấu hình: `http://control.local/config`

Không cần thêm `:5000`. `control.local` dùng mDNS/Zeroconf; thiết bị truy cập
phải ở cùng mạng LAN. Nếu mDNS không hoạt động, dùng địa chỉ IP máy chạy server.

Sau khi thay model hoặc sửa code Python, phải dừng và chạy lại `app.py`.

## 5. Camera và ảnh 2D

Camera ưu tiên Intel RealSense D435 ở `640×480 @ 30 FPS`. Nếu không có
RealSense, code dùng `cv2.VideoCapture(0)`.

Chế độ ảnh 2D được bật/tắt tại trang Config. Khi bật, trang Detection hiển thị
nút chọn ảnh từ máy tính. Giới hạn upload là 10 MiB; hỗ trợ các định dạng OpenCV
đọc được như JPG, PNG, BMP và WEBP.

## 6. Cấu hình detection trên web

Trang Config yêu cầu mật khẩu:

```text
0123
```

Hai ngưỡng có thể chỉnh trực tiếp:

- Confidence: giữ các dự đoán đủ tin cậy.
- IoU/NMS: loại các box trùng nhau.

Giá trị đang lưu trong `settings.json`:

```text
Confidence = 0.40
IoU/NMS    = 0.45
Ảnh 2D     = bật
```

`Lưu tạm thời` áp dụng trong bộ nhớ và mất sau khi khởi động lại. `Lưu vĩnh
viễn` ghi vào `settings.json` và được nạp lại khi chạy lần sau.

## 7. API detection

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/detection` | Trang detection |
| GET | `/video_feed` | MJPEG stream |
| POST | `/camera_start` | Khởi động camera |
| POST | `/camera_stop` | Dừng camera |
| POST | `/start_detection` | Bật inference realtime |
| POST | `/stop_detection` | Tắt inference realtime |
| GET | `/detection_stats` | FPS, số cây, lớp và inference time |
| GET | `/detection_options` | Trạng thái chức năng ảnh 2D |
| POST | `/process_2d_image` | Xử lý một ảnh tải lên |

## 8. Công cụ kiểm thử Tkinter

Chạy:

```powershell
python test.py
```

Chức năng:

- Chọn backend TensorFlow Lite hoặc YOLO Ultralytics.
- Chọn file model.
- Chọn ảnh từ máy tính hoặc dùng RealSense.
- Chỉnh confidence và IoU.
- Capture, inference và giao diện chạy trên các luồng riêng để tránh lag.
- Tự đọc input size, dtype/quantization và layout output của model TFLite.

Đuôi model YOLO hỗ trợ trong công cụ test: `.pt`, `.onnx`, `.engine`. Các
checkpoint YOLOv5 cũ có thể không mở trực tiếp bằng thư viện Ultralytics mới;
trong trường hợp đó nên export sang TFLite/ONNX hoặc dùng đúng repository
YOLOv5 đã tạo checkpoint.

## 9. Thay model TFLite

1. Chép model vào thư mục `model/`.
2. Đổi `MODEL_NAME` trong `detect_stream.py`.
3. Kiểm tra thứ tự lớp khớp `classes.txt`.
4. Chạy `python test.py` để thử ảnh và camera.
5. Chạy test tự động rồi khởi động lại `app.py`.

Pipeline hiện hỗ trợ:

- Input NHWC, ba kênh màu.
- Kích thước input tĩnh như 480, 512 hoặc 640; không ép cố định 480.
- Input/output số thực hoặc lượng tử hóa integer có scale/zero-point.
- Hai lớp theo `classes.txt`.
- Output YOLOv5 có objectness hoặc YOLOv8 không có objectness.
- Layout `[1, N, C]` hoặc `[1, C, N]`.

## 10. Kiểm thử và hiệu năng

Lệnh kiểm tra:

```powershell
python -m py_compile app.py detect_stream.py test.py
python -m pytest -q
```

Tại lần rà soát ngày 23/10/2026:

- Model YOLOv8 mới nạp thành công.
- Input được nhận đúng là 640×640.
- Output được nhận đúng là `[1, 6, 8400]`, không có objectness.
- Ảnh kết quả giữ nguyên kích thước nguồn.
- Suy luận thử và hậu xử lý hoàn thành không lỗi.
- `30/30` bài kiểm thử đạt.

Thời gian inference đo trên máy Windows hiện tại khoảng 125–185 ms tùy ảnh.
Hiệu năng trên Raspberry Pi phải đo trực tiếp; không nên dùng con số Windows
làm cam kết FPS cho Pi.

## 11. Xử lý lỗi

### Không tìm thấy model

- Kiểm tra file nằm trong `CODE/GPS_RASPI/model/`.
- Kiểm tra `MODEL_NAME` đúng cả tên và đuôi `.tflite`.
- Khởi động lại `app.py` sau khi đổi model.

### Output model không được hỗ trợ

- Xác nhận model là task detect, không phải classify/segment/pose.
- Xác nhận model có đúng hai lớp và thứ tự lớp khớp `classes.txt`.
- Kiểm tra metadata/output bằng `test.py`.

### Box lệch khỏi vật thể

- Không resize ảnh thủ công trước khi gọi pipeline.
- Giữ cơ chế `letterbox` và `scale_coords` hiện tại.
- Xác nhận camera trả frame BGR; pipeline tự đổi sang RGB cho model.
- Nếu vừa sửa code, phải restart server để tiến trình dùng bản mới.

### Camera không mở được

- Đóng chương trình khác đang giữ camera.
- Với RealSense, kiểm tra `pyrealsense2` và cáp USB 3.x.
- Với camera OpenCV, kiểm tra `cv2.VideoCapture(0)`.

### FPS thấp

- Model 640 giữ nhiều chi tiết hơn nhưng chậm hơn model 480.
- Không đổi `INPUT_SIZE` để ép model; kích thước được đọc từ model.
- Dùng model nhỏ hơn hoặc model lượng tử hóa nếu cần tăng tốc.
