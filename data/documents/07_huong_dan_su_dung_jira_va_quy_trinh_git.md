# HƯỚNG DẪN QUY TRÌNH QUẢN LÝ CÔNG VIỆC TRÊN JIRA VÀ GITFLOW

*Mã tài liệu: HD-TECH-07*  
*Áp dụng: Toàn bộ Kỹ sư phần mềm, AI Engineer, QA và Intern*

---

## 1. Vòng đời Ticket trên Jira

Quy trình quản lý trạng thái ticket gồm 6 bước chính:

```
[Open] ──(Start Progress)──> [In Progress] ──(Tạo MR)──> [Ready for Test] ──> [IN TESTING] ──> [Done]
                                  │
                              (Chờ bên khác)
                                  ▼
                              [PENDING]
```

- **Open**: Ticket mới được phân công, có mô tả bài toán và tiêu chí nghiệm thu rõ ràng.
- **In Progress**: Dev bắt buộc bấm chuyển sang trạng thái này **trước khi viết dòng code đầu tiên**.
- **Ready for Test**: Dev code xong, đã tự kiểm thử đạt yêu cầu, đã tạo Merge Request (MR) và viết hướng dẫn test trong phần comment. Yêu cầu kéo trạng thái trước **16:00** để QA kịp kiểm thử trong ngày.
- **IN TESTING**: Trạng thái do QA kiểm thử kéo sang. Dev không tự ý kéo.
- **Done**: QA đã xác nhận đạt tiêu chí và MR đã được merge vào nhánh chính.
- **Re Open**: QA phát hiện lỗi hoặc không đạt nghiệm thu, chuyển ngược về để Dev sửa lại.

---

## 2. Quy chuẩn đặt tên nhánh Git và Commit

### 2.1. Quy ước đặt tên nhánh (Branching convention)
- Tính năng mới: `feature/<ma-task>-<ten-ngan-gon>` (Ví dụ: `feature/TASK-AI02-llm-abstraction`)
- Sửa lỗi: `fix/<ma-task>-<mo-ta-loi>` (Ví dụ: `fix/TASK-AI03-langfuse-context-leak`)
- Đánh giá / Thử nghiệm: `experiment/<ten-thu-nghiem>` (Ví dụ: `experiment/hybrid-search-bm25`)

### 2.2. Quy ước thông điệp commit (Conventional Commits)
- `feat`: Tính năng mới đưa vào hệ thống.
- `fix`: Sửa lỗi phát sinh trong mã nguồn.
- `docs`: Bổ sung hoặc chỉnh sửa tài liệu, README.
- `test`: Viết thêm unit test hoặc integration test.
- `refactor`: Tối ưu hóa cấu trúc code mà không thay đổi hành vi nghiệp vụ.
- `chore`: Thay đổi cấu hình build, dependencies, tooling.

*Ví dụ commit chuẩn*: `feat(llm): implement multi-provider abstraction with fallback`

---

## 3. Quy định về Log Work hàng ngày
- **Thời điểm**: Log ngay sau khi kết thúc một mạch việc trong ngày, tuyệt đối không dồn toàn bộ vào cuối tuần.
- **Đơn vị thời gian**: Tối thiểu 30 phút (`30m`). Một lần log không quá 4 giờ (`4h`).
- **Nội dung mô tả**: Bắt buộc ghi rõ ràng hành động cụ thể đã làm.
  - ❌ Viết chung chung: *"Fix bug"*, *"Code tiếp"*, *"Đọc tài liệu"*.
  - ✅ Viết cụ thể: *"Sửa lỗi validate SQL injection cho node kiểm tra NLQ, bổ sung 5 test case pytest"*.
