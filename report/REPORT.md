# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh, vì trong record k có object_id hay các thông số label của 1 object và task cũng ghi rõ là image_classification.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list) đi kèm với taxonomy và lưu trữ trong checkpoint:
Taxonomy có nghĩa ngắn : "Danh sách tên lớp có sẵn kèm model"
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Việc phân rõ cả class_id, class_name và taxonomy_name nhằm đảm bảo tính độc nhất cũng như dễ quy chuẩn dữ liệu. 
class_id là định danh dạng số, class_name là định danh tên vật thể
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần nói rõ về chủ đề bài toán hoặc mức độ tiêu chí ưu tiên của các object trong ảnh như độ nổi bật hoặc mức độ bao quát trong ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì "score" chỉ là phỏng đoán chủ quan của model còn ground truth là quy tắc phải tuân theo, là kết quả chính xác cuối cùng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "bowl" ; `score`= 0.701283 ; `bbox_xyxy`=[155.02, 168.67, 182.38, 184.18] ; `bbox_width`= 27.36 ; `bbox_height`= 15.51
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Box "bowl": Bắt đầu từ góc trên-trái ở tọa độ pixel (155.02, 168.67) và kéo dài đến góc dưới-phải ở tọa độ (182.38, 184.18). Khung hình chữ nhật này có chiều rộng là 27.36 pixel và chiều cao là 15.51 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Dựa vào điểm tự tin (score) của cái bát (0.701) và lò nướng (0.686), giả sử ta chọn 2 ngưỡng là 0.60 và 0.70:
- Ngưỡng 0.60: Giữ lại 2 prediction. Vì cả bát (0.701) và lò nướng (0.686) đều có điểm số vượt qua mức 0.60.
- Ngưỡng 0.70: Chỉ giữ lại 1 prediction. Lúc này mô hình chỉ hiển thị cái bát (0.701 > 0.70), còn chiếc lò nướng sẽ bị ẩn đi vì điểm tự tin chưa đạt mức 0.70.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Xóa dương tính giả, sửa phân loại sai, sửa kích thước box.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
"Khung giới hạn phải được vẽ sao cho 4 cạnh của box chạm sát nhất vào các điểm cực ngoài cùng (trên, dưới, trái, phải) của vật thể. Không để thừa quá nhiều khoảng trống (background) bên trong viền, nhưng tuyệt đối không được cắt lẹm vào thân vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể: Lò nướng (oven) bị cắt mép trái ảnh (tọa độ x bắt đầu từ 0.11, dính sát viền).
- Cách xử lý: Viết thành Quy tắc trong Guideline, tuyệt đối không hỏi Lab Coach.
- Vì hiện tượng vật bị cắt mép hoặc che khuất 1 phần là tình huống cực kỳ phổ biến trong mọi tập dữ liệu. Cần quy định sẵn một lần trong Guideline (ví dụ: "Chỉ vẽ box ôm phần có thể nhìn thấy, kéo sát viền") để người gán nhãn tự xử lý hàng loạt, tránh hỏi lắt nhắt gây mất thời gian cho các vấn đề cơ bản. Chỉ escalate những trường hợp quá dị, khó có thể nhận diện bằng mắt thường.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-003" ; `class_name`= "bowl" ; `score`= 0.698698 ; số điểm polygon=`polygon_point_count`= 70 ; 3–5 điểm đầu của `polygon_xy`= [84.0, 289.0], [83.0, 290.0], [81.0, 290.0], [80.0, 291.0], [77.0, 291.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp đường viền thực tế (contour), hình dáng chính xác, độ cong, góc nghiêng và diện tích pixel thật của riêng vật thể, loại bỏ hoàn toàn phần nền dư thừa.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để đếm và phân tách các cá thể độc lập trong cùng một bức ảnh (ví dụ: cái bát số 1, cái bát số 2).
Không phải Class ID: Vì nó không phân loại vật thể là gì. Cả hai vật có cùng Class ID (đều là "bát") nhưng sẽ có instance_id khác nhau (bát số 1, bát số 2).
Không phải Tracking ID: Vì nó chỉ có giá trị cục bộ trong một bức ảnh tĩnh duy nhất. Nó không dùng để theo dõi sự di chuyển của cái bát đó qua các khung hình liên tiếp trong một đoạn video (đó là nhiệm vụ của tracking ID).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
"Các điểm chấm polygon (vertices) phải nằm chính xác trên ranh giới điểm ảnh (pixel) phân chia giữa vật thể và cảnh nền. Không lấn ra ngoài background (nhỏ hơn 1-2 pixel sai số) và tuyệt đối không cắt lẹm vào cấu trúc bên trong của vật."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Đáy của cái bát (bowl) chạm vào mặt bàn hoặc thân lò nướng (oven), khiến ranh giới 2 vật thể bị mờ, khó phân biệt pixel nào thuộc về vật nào.
Cách xử lý: Dùng Guideline:
- Vì hiện tượng che khuất (occlusion), dính mép hoặc viền mờ do ánh sáng là tình trạng hiển nhiên trong 99% các dự án Segmentation. Guideline phải quy định sẵn một quy tắc xử lý chung (Ví dụ: "Chỉ vẽ mask cho những pixel nhìn thấy rõ, không tự tưởng tượng vẽ bù phần bị che") để người gán nhãn tự xử lý hàng loạt. Chỉ escalate khi hình dạng quá kỳ dị không thể nhận diện đó là vật thể gì.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 text/ID nhãn cho toàn ảnh (VD JSON: `{"label": "dog"}`) | Ảnh PNG bị mờ/ngược sáng khó nhìn; hoặc JSON yêu cầu 1 nhãn nhưng PNG có 2 vật thể to ngang nhau. | Đánh giá toàn cảnh và chọn nhãn (category) mô tả chính xác nhất nội dung chính của ảnh. | Kiểm tra nhãn được chọn có đúng với hướng dẫn và bao quát đúng đối tượng trọng tâm không. |
| Phát hiện vật thể | Tọa độ Bounding box kèm nhãn (VD JSON: `bbox: [x, y, w, h]`, `"label": "car"`) | Box bao quá rộng chứa cả nền trên PNG; hoặc JSON bị sót box của các vật thể nhỏ, bị che khuất (occlusion). | Vẽ hộp chữ nhật (bounding box) bám sát nhất (tightest fit) các mép ngoài cùng của vật thể. | Check mép bounding box có bị lỏng/cắt lẹm vật thể không, có gán sai class hay bỏ sót vật không. |
| Instance segmentation | Tọa độ đa giác (polygon) hoặc pixel mask (VD JSON: `segmentation: [[x1,y1, x2,y2...]]`) | Vẽ lẹm viền (pixel tràn sang nền trên PNG); hoặc file JSON gộp nhầm (merge) mask của 2 cá thể đứng sát nhau. | Chấm đa giác/tô mask chi tiết dọc theo đường viền mép của *từng cá thể* vật thể riêng biệt. | Zoom kỹ các viền (độ chính xác pixel) trên PNG xem có lẹm/hở không, check xem các cá thể sát nhau có bị dính mask không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tuyệt đối không chụp màn hình, không tải xuống và không chia sẻ bất kỳ hình ảnh hay dữ liệu nào của bài lab ra bên ngoài hệ thống dưới mọi hình thức.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng ngay việc gán nhãn trên dữ liệu đó và báo cáo (escalate) cho Lab Coach (hoặc Quản lý dự án / Project Manager) để được hướng dẫn xử lý.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [X] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [X] Đã nộp **link repository** trên LMS
