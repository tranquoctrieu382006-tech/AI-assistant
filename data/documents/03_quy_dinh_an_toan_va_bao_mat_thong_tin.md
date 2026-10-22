# QUY ĐỊNH AN TOÀN VÀ BẢO MẬT THÔNG TIN

*Mã tài liệu: QD-IT-03*
*Mức độ bảo mật: NỘI BỘ (INTERNAL)*
*Áp dụng: Toàn thể Cán bộ nhân viên, Nhà thầu, Thực tập sinh*

---

## 1. Nguyên tắc quản lý tài khoản và mật khẩu

### 1.1. Yêu cầu độ phức tạp mật khẩu
Mọi tài khoản truy cập hệ thống công ty (Email, GitLab, VPN, Jira, HRM, AWS/Cloud) phải tuân thủ:
- Độ dài tối thiểu: **12 ký tự**.
- Phải bao gồm ít nhất: Chữ in hoa (A-Z), chữ thường (a-z), chữ số (0-9) và ký tự đặc biệt (`!@#$%^&*`).
- Không chứa tên người dùng, tên công ty, ngày tháng năm sinh hoặc từ ngữ phổ biến.
- Thời hạn hiệu lực: Tối đa **90 ngày** phải đổi mật khẩu một lần.

### 1.2. Xác thực đa yếu tố (MFA / 2FA)
- Bắt buộc kích hoạt MFA trên toàn bộ các tài khoản: Google Workspace, GitLab, AWS, VPN.
- Ứng dụng xác thực được khuyến nghị: **Google Authenticator** hoặc **1Password**. Nghiêm cấm nhận mã OTP qua SMS đối với các tài khoản hạ tầng quan trọng.

---

## 2. Bảo mật mã nguồn và Dữ liệu dự án

### 2.1. Quy tắc làm việc với Git & Mã nguồn
- **Tuyệt đối không commit nhạy cảm**: Không bao giờ commit file `.env`, API Key (OpenAI, Claude, AWS, Langfuse), mật khẩu CSDL hoặc private certificate lên repository.
- Mọi repository phải có file `.gitignore` chuẩn trước khi tạo commit đầu tiên.
- Khi phát hiện rò rỉ secret lên Git:
  1. Xoay (revoke và tạo mới) ngay lập tức các secret bị lộ.
  2. Báo cáo cho Trưởng nhóm và Bộ phận IT Security trong vòng **30 phút**.
  3. Dùng lệnh `git commit --amend` hoặc `git-filter-repo` để xóa triệt để khỏi lịch sử commit.

### 2.2. Quy định về dữ liệu khách hàng
- Nghiêm cấm sử dụng dữ liệu thật của khách hàng, dữ liệu sản xuất (Production Data) vào môi trường phát triển (Dev) hoặc môi trường kiểm thử (Testing).
- Dữ liệu phục vụ demo RAG và Text-to-SQL phải được tổng hợp giả lập hoặc làm mờ thông tin định danh (Anonymization/Masking).

---

## 3. Quản lý thiết bị làm việc

| Quy định | Chi tiết hướng dẫn |
|---|---|
| Khóa màn hình (Lock Screen) | Nhân viên phải nhấn tổ hợp phím `Win + L` (Windows) hoặc `Cmd + Ctrl + Q` (Mac) mỗi khi rời khỏi bàn làm việc. |
| Kết nối mạng | Thiết bị cá nhân chỉ được kết nối vào mạng `IOC-Guest`. Chỉ máy tính công ty mới được truy cập `IOC-Staff` và VPN nội bộ. |
| Sử dụng USB / Ổ cứng ngoài | Nghiêm cấm sao chép dữ liệu công ty ra thiết bị lưu trữ ngoài khi chưa có văn bản phê duyệt của Trưởng bộ phận. |
| Cài đặt phần mềm | Nghiêm cấm cài đặt các phần mềm bẻ khóa (crack), công cụ chia sẻ file ngang hàng (BitTorrent). |
