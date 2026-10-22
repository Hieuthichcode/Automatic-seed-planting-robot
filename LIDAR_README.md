# Lidar trong hệ thống giám sát và điều khiển robot

## 📋 Mô tả
Hệ thống hiển thị dữ liệu LIDAR từ ESP32 lên web browser theo thời gian thực.

## 🗂️ Cấu trúc files

### 1. **Read_lidar.py**
- Đọc dữ liệu từ ESP32 LIDAR qua Serial
- Parse format firmware hiện tại: `khoảng_cách góc chất_lượng`
  (ví dụ: `450.5 90.0 123`); vẫn nhận format hai trường cũ.
- Lưu trữ điểm với timestamp
- Tự reconnect khi mất cổng; dữ liệu stale làm trạng thái điều khiển fail-safe

### 2. **lidar_web.html**
- Giao diện web hiển thị LIDAR
- Canvas vẽ radar 2D với lưới
- Hiển thị điểm theo góc và khoảng cách
- Hiệu ứng mờ dần cho điểm cũ
- Thống kê real-time

### 3. **app.py** (đã cập nhật)
- Thêm routes cho LIDAR:
  - `/lidar` - Trang hiển thị LIDAR
  - `/startLidar` - Bắt đầu đọc dữ liệu
  - `/stopLidar` - Dừng đọc dữ liệu
  - `/getLidarData` - Lấy dữ liệu hiện tại
  - `/clearLidarData` - Xóa dữ liệu

### 4. **map.html** (đã cập nhật)
- Thêm nút "📡 LIDAR" trong Control Panel
- Chuyển đến trang LIDAR khi nhấn

## 🚀 Cách sử dụng

### Bước 1: Kết nối phần cứng
1. Kết nối ESP32 + LIDAR với máy tính qua USB
2. Kiểm tra cổng COM (cấu hình đang lưu hiện tại là COM44)

### Bước 2: Cấu hình code ESP32
Đảm bảo ESP32 gửi dữ liệu theo format:
```
khoảng_cách góc chất_lượng
```
Ví dụ:
```
450.5 90.0 123
523.2 91.0 98
498.7 92.0 105
```

### Bước 3: Chạy server
```bash
python app.py
```

### Bước 4: Mở web browser
1. Truy cập: `http://control.local`
2. Nhấn nút "📡 LIDAR" trong Control Panel
3. Nhấn "🚀 Bắt đầu LIDAR"

### Bước 5: Điều chỉnh (nếu cần)
Giá trị đang lưu:
- **Port**: COM44
- **Baudrate**: 115200
- **Lifetime**: 1 giây

Để thay đổi, sửa trong `Read_lidar.py`:
```python
# Dòng 9-10
self.max_points = 1000  # Số điểm tối đa
self.dot_lifetime = 1.0  # Thời gian hiển thị (giây)
```

Để đổi cổng, mở `http://control.local/config`, đăng nhập bằng mật khẩu `0123`,
chọn cổng Lidar rồi nhấn kiểm tra. Chỉ lưu khi hệ thống nhận đúng dữ liệu Lidar
trên cổng đó. Lưu vĩnh viễn ghi lựa chọn vào `settings.json`; thay đổi cổng có
hiệu lực sau khi khởi động lại ứng dụng.

Baudrate mặc định nằm trong `config.py`:

```python
LIDAR_BAUDRATE = 115200
```

## 🎨 Giao diện

### Canvas LIDAR
- **Màu nền**: Đen (#0f0f1e)
- **Lưới**: Xanh lá mờ (#00ff0040)
- **Điểm mới** (< 0.5s): Xanh sáng (lime)
- **Điểm trung bình** (0.5-1.0s): Xanh lá (green)
- **Điểm cũ** (> 1.0s): Xanh tối (darkgreen)

### Hệ tọa độ
- **N (North)**: Phía trên (0°)
- **E (East)**: Phía phải (90°)
- **S (South)**: Phía dưới (180°)
- **W (West)**: Phía trái (270°)

### Thống kê
- Số điểm hiện tại
- Khoảng cách gần nhất (mm)
- Khoảng cách xa nhất (mm)

## 🔧 Tùy chỉnh nâng cao

### Thay đổi tốc độ cập nhật
Trong `lidar_web.html`, dòng 232:
```javascript
updateInterval = setInterval(fetchLidarData, 100); // 100ms
```

### Thay đổi kích thước canvas
Trong `lidar_web.html`, dòng 151:
```html
<canvas id="lidarCanvas" width="800" height="800"></canvas>
```

### Thay đổi khoảng cách max
Trong `lidar_web.html`, dòng 192:
```javascript
const maxDistance = 4000; // 4000mm = 4m
```

## 📊 Format dữ liệu

### Từ ESP32 → Python (Serial)
```
khoảng_cách góc chất_lượng
```
- **khoảng_cách**: mm (float)
- **góc**: độ (float), 0-360°

### Từ Python → Web (JSON)
```json
{
  "points": [
    {
      "angle": 90.0,
      "distance": 450.5,
      "quality": 123.0,
      "age": 0.234
    },
    ...
  ],
  "connected": true
}
```

## 🐛 Troubleshooting

### Không kết nối được Serial
- Kiểm tra cổng COM đúng chưa
- Kiểm tra baudrate khớp với ESP32
- Đảm bảo không có ứng dụng khác đang dùng cổng COM
- Dùng nút kiểm tra cổng trong trang Config trước khi lưu

### Không hiển thị điểm
- Kiểm tra format dữ liệu từ ESP32
- Mở console browser (F12) xem lỗi
- Kiểm tra log trong terminal Python

### Điểm hiển thị sai vị trí
- Kiểm tra góc từ ESP32 (0-360°)
- Đảm bảo góc 0° tương ứng với hướng mong muốn

## 📝 Notes
- LIDAR chạy độc lập với GPS tracking
- Có thể chạy đồng thời Detection và LIDAR
- Dữ liệu tự động xóa sau `dot_lifetime` giây
- Không lưu vào database, chỉ hiển thị real-time
- Trạng thái vật cản do monitor phía server quyết định; trình duyệt chỉ hiển thị
