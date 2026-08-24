# Báo cáo thống kê bộ dữ liệu Bronchoscopy VQA derived_v4.3

## Phạm vi và tính toàn vẹn

- Cohort phân tích: **27.570 QA sạch** trong `train.json`, `val.json`, `test.json`.
- Đối chiếu Master–clean: **PASS**; không có QA trùng giữa các split.
- Giao nhau theo ảnh, bệnh nhân và quy trình giữa mọi cặp split: **0**.
- Không đưa 455 record `REVIEW_REQUIRED` vào các thống kê huấn luyện/phát hành.

## Quy mô dữ liệu

- **12,861** ảnh độc nhất; **1,606** bệnh nhân; **1,606** quy trình.
- **27,570** cặp QA và **28,853** bbox instances.
- Sau khi khử việc cùng bbox được dùng lại cho nhiều QA: **12,513** unique image–box tuples.
- **25,435 QA (92.26%)** có ít nhất một bbox; **2,135 QA (7.74%)** không có bbox. Toàn bộ nhóm không có bbox là câu hỏi `image_normality` cấp toàn ảnh.
- Số bbox có tọa độ không hợp lệ: **0**.
- Phân bố QA Train/Validation/Test: **70.65% / 14.73% / 14.62%**.

## Định dạng và nội dung câu hỏi

- `closed_ended_questions`: **13,364 (48.47%)**.
- `open_ended_questions`: **12,529 (45.44%)**.
- `single_choice_questions`: **1,677 (6.08%)**.
- Không xuất hiện trong bản sạch: `multi_choice_questions`.

Năm nhóm nội dung lớn nhất:

- `abnormality`: **9,640 (34.97%)**.
- `image_normality`: **6,374 (23.12%)**.
- `abnormality_presence`: **5,426 (19.68%)**.
- `stenosis_degree`: **1,677 (6.08%)**.
- `secretion_type`: **1,253 (4.54%)**.


## Bình thường và bất thường

Trong **6,374** QA thuộc `image_normality`:

- `normal=true`: **1,062 (16.66%)**.
- `normal=false`: **5,312 (83.34%)**.

Đây là phân bố **QA/image-normality**, không phải prevalence ở cấp bệnh nhân. Tỷ lệ này tự nó không chứng minh “cân bằng công bằng”; cần đánh giá thêm sensitivity/specificity và calibration theo bệnh nhân.

## Độ dài ngôn ngữ

| Thành phần | Mean | Median | P25–P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| `question_stem_words` | 11.97 | 12.00 | 10.00–13.00 | 16.00 | 28 |
| `reason_words` | 14.68 | 14.00 | 11.00–17.00 | 22.00 | 39 |
| `visual_evidence_words` | 15.14 | 15.00 | 13.00–17.00 | 20.00 | 30 |
| `explanation_words` | 29.82 | 29.00 | 26.00–34.00 | 40.00 | 61 |


## Vùng giải phẫu

- **9,408/27,570 QA (34.12%)** có vùng giải phẫu xác định được bằng provenance canonical.
- **18,162 QA** có bbox/evidence nhưng canonical region chưa mang nhãn giải phẫu; các QA này không bị ép gán vùng bằng dò từ khóa.
- Phân bố trong Figure 3(c) là số liên kết QA–anatomy; một QA có thể liên kết nhiều vùng nên tổng cột có thể lớn hơn số QA grounded.
- `question_text_mentions` được xuất riêng trong CSV như một phân tích ngôn ngữ thứ cấp, không thay thế provenance.

## Lưu ý diễn giải khoa học

1. `BBox instances` đếm bbox theo từng QA và có thể đếm lại cùng một bbox. Khi so sánh với dataset khác, nên báo cáo song song `unique image–box tuples`.
2. Word cloud mang tính mô tả khám phá; không phải bằng chứng độc lập về độ sâu lâm sàng hoặc chất lượng reasoning.
3. Độ dài không đồng nghĩa với chất lượng. Chất lượng explainability cần được kiểm tra thêm bằng fact consistency, visual grounding và đánh giá chuyên gia.
4. Các tỷ lệ của bộ dữ liệu này phải được trình bày theo số liệu thực tế, không nên khẳng định tuân theo 8:1:1.

## Tệp đầu ra

- `table_1_dataset_split_statistics.csv/.md/.tex`
- `figure_3_dataset_composition.png/.pdf`
- `figure_4_language_analysis.png/.pdf`
- Các bảng phân bố CSV, word-frequency CSV, word clouds riêng và `statistics.json`.
