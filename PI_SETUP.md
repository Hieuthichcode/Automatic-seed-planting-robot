# Raspberry Pi 4 deployment

Trạng thái được kiểm tra ngày 23/10/2026 trên Raspberry Pi 4 Model B, Raspberry
Pi OS Bookworm 32-bit (`armhf`), Python 3.11.

## Vị trí và truy cập

- Mã nguồn: `/home/pi/Desktop/Automatic-seed-planting-robot`
- Môi trường Python: `.venv` (được tạo với `--system-site-packages`)
- Web: `http://control.local/` hoặc `http://192.168.1.77/`
- Service: `cothu-robot.service`

```bash
sudo systemctl status cothu-robot.service
sudo systemctl restart cothu-robot.service
journalctl -u cothu-robot.service -f
```

Service được bật bằng `systemctl enable`, chạy qua Waitress và tự khởi động lại
nếu tiến trình Python bị lỗi. File mẫu nằm tại `deploy/cothu-robot.service`.

## Thiết bị hiện có

- GPS/MPU ESP32-C3: `/dev/robot-gps-imu`.
- Intel RealSense D435 RGB: `/dev/robot-camera-color` (udev liên kết đến
  video node màu). Rule mẫu: `deploy/99-cothu-robot.rules`.
- LIDAR LDS02RR qua ESP32-C3: `/dev/robot-lidar`.
- Controller CH340 có symlink chẩn đoán `/dev/robot-controller`, nhưng service
  đang trỏ `CONTROL_PORT` đến `/dev/robot-controller-disabled`. Thiết bị có dấu
  hiệu lặp reset do brownout/watchdog và ghi serial bị timeout; không để service
  liên tục mở lại cổng hoặc mở khóa chuyển động trước khi xử lý nguồn/firmware.

## An toàn cơ khí

Service đang đặt `ROBOT_ARM_ENABLED=0`. Đây là chủ ý: lúc Python khởi động,
code cánh tay có thể đưa bốn servo về 90 độ; chưa được phép làm vậy trước khi
kiểm tra cơ khí và nguồn servo. Ở chế độ này API cánh tay vẫn chạy mô phỏng
nhưng không phát xung GPIO.

Chỉ sau khi đã kiểm tra cơ khí, nguồn và vùng chuyển động, đổi biến này thành
`1` trong `/etc/systemd/system/cothu-robot.service`, rồi chạy:

```bash
sudo systemctl daemon-reload
sudo systemctl restart cothu-robot.service
```

AUTO vẫn bị khóa bởi `AUTO_TODO`; không coi đây là lỗi triển khai.

## Kết quả kiểm tra

- 31 bài test trên Pi: đạt.
- Web/API và mDNS: đạt.
- Toàn bộ nút trên giao diện chính có trạng thái hover/active đồng nhất; nút
  disabled giảm opacity và không đổi màu khi rê chuột.
- GPS/MPU: serial 115200 hoạt động; khi ở trong nhà GPS có thể chưa có fix và
  web dùng tọa độ mặc định.
- LIDAR: tự kết nối sau boot, dữ liệu tươi và API vật cản hoạt động.
- Camera RGB: khoảng 27 FPS khi không bật nhận diện.
- Stream khi bật nhận diện vẫn liên tục, nhưng một lần suy luận model 640x640
  mất khoảng 3,5 giây trên Pi hiện tại; kết quả nhận diện vì vậy không phải
  thời gian thực.
- Cơ chế `Restart=always`: đã thử kill tiến trình và service tự phục hồi.
- Đã reboot với cấu hình cuối; các symlink thiết bị và service tự khởi động.
- Nhiệt độ khi kiểm tra khoảng 43–47°C, không bị throttling.

## Giới hạn còn lại

`pyrealsense2` không có wheel phù hợp với userland `armhf` đang dùng. RGB hoạt
động qua OpenCV/V4L2, nhưng trang depth chưa hoạt động. Muốn dùng depth nên
chuyển sang Raspberry Pi OS 64-bit rồi cài binding phù hợp, hoặc tự biên dịch
librealsense. LIDAR đã kết nối và cho dữ liệu tươi. Controller và chuyển động
servo phải được kiểm tra lại sau khi xử lý nguồn/firmware của controller.

Vì web điều khiển đang phục vụ trong mạng LAN và cấu hình đăng nhập còn đơn
giản, không mở cổng 80 trực tiếp ra Internet. Nên đổi mật khẩu tài khoản Pi và
dùng SSH key.
