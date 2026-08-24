# VQA nội soi phế quản: tạo dữ liệu, huấn luyện và đánh giá

> **Hướng dẫn hiện hành:** đọc [`PROJECT_HANDOFF_STAGE_0_4.md`](PROJECT_HANDOFF_STAGE_0_4.md) trước khi làm việc với dữ liệu. Tài liệu đó mô tả pipeline versioned `raw_v1 → canonical_v2.2 → derived_v2 → derived_v3 → derived_v4/v4.1`, các bất biến, protocol và artifact hiện tại. Phần README bên dưới ghi lại pipeline GEMeX/report–LLM cũ để tái hiện baseline; không dùng nó làm nguồn hướng dẫn sinh dữ liệu mới.

Repository này xây dựng bộ dữ liệu Visual Question Answering (VQA) tiếng Việt cho ảnh nội soi phế quản và fine-tune mô hình `BoKelvin/GEMeX-VQA-Model-Simple`.

Với một ảnh và câu hỏi, mô hình được huấn luyện để trả về:

```text
<answer> câu trả lời
<reason> lý do dựa trên hình ảnh
<location> [[x1, y1, x2, y2]]
```

README trình bày toàn bộ quy trình theo thứ tự:

1. Chuẩn bị môi trường.
2. Sinh báo cáo nguyên tử từ ảnh và annotation.
3. Sinh câu hỏi–trả lời VQA.
4. Chuyển đổi và chia dữ liệu theo bệnh nhân.
5. Smoke test và train bằng QLoRA.
6. Validation.
7. Test và đọc kết quả.

## 1. Tổng quan pipeline

```text
Ảnh + annotation
        │
        ▼
llm_input*_with_colors.json
        │  generate_bronchoscopy_reports_gpt.py
        ▼
*_reports.json
        │  generate_bronchoscopy_vqa_gpt.py
        ▼
*_vqa.json
        │  gộp và làm phẳng
        ▼
all_vqa.json
        │  convert_bronchoscopy_vqa_to_gemex.py
        ▼
bronchoscopy_train_data.json
        │  split_bronchoscopy_by_patient.py
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
train 70%       validation 15%      test 15%
        │              │               │
        ▼              ▼               ▼
QLoRA train       chọn mô hình       báo cáo cuối
```

Số liệu của phiên bản hiện tại:

| Thành phần | Số lượng |
|---|---:|
| Cặp hỏi–đáp | 54.684 |
| Ảnh | 6.781 |
| Bệnh nhân/ca | 445 |
| Train | 38.287 câu |
| Validation | 8.196 câu |
| Test | 8.201 câu |

Dữ liệu được chia theo bệnh nhân, không chia ngẫu nhiên từng câu hỏi. Manifest hiện tại xác nhận không có bệnh nhân hoặc ảnh trùng giữa train, validation và test.

## 2. Cấu trúc thư mục quan trọng

```text
.
├── Cabenh/                         # ảnh, annotation và llm_input
├── Cabenh_reports/                 # báo cáo nguyên tử và VQA theo ca
├── all_vqa.json                    # dữ liệu VQA đã gộp
├── generate_bronchoscopy_reports_gpt.py
├── generate_bronchoscopy_vqa_gpt.py
├── GEMeX-Colab/
│   ├── data/
│   │   ├── bronchoscopy_train_data.json
│   │   ├── bronchoscopy_train_70.json
│   │   ├── bronchoscopy_validation_15.json
│   │   ├── bronchoscopy_test_15.json
│   │   └── bronchoscopy_split_manifest.json
│   ├── images/                     # ảnh dùng cho train/evaluation
│   ├── llava-med/                  # mã nguồn LLaVA-Med/GEMeX
│   ├── convert_bronchoscopy_vqa_to_gemex.py
│   ├── split_bronchoscopy_by_patient.py
│   ├── evaluate_bronchoscopy.py
│   └── install_colab.sh
└── gemex_outputs/
    ├── patient_split_local_v1/     # adapter đã train
    └── evaluation/                 # prediction và metric
```

Các lệnh bên dưới giả sử terminal đang đứng ở thư mục gốc:

```bash
cd /home/ailab/Documents/KC4.0_Final_annots_data_png
```

## 3. Yêu cầu hệ thống

Khuyến nghị:

- Ubuntu/Linux.
- Python 3.10–3.12.
- NVIDIA GPU có ít nhất khoảng 12 GB VRAM.
- CUDA driver hoạt động.
- Khoảng 20 GB trống cho model gốc, cache và checkpoint.
- Internet để tải model từ Hugging Face.
- API tương thích OpenAI nếu muốn sinh lại report/VQA bằng LLM.

Kiểm tra GPU:

```bash
nvidia-smi
```

Kiểm tra Python:

```bash
python --version
```

Không đưa API key, token Hugging Face hoặc thông tin định danh bệnh nhân vào Git.

## 4. Cài đặt

### 4.1. Cài trên máy Linux

Tạo môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Cài các phiên bản thư viện đã dùng trong dự án:

```bash
bash GEMeX-Colab/install_colab.sh
python -m pip install requests pillow
```

Script cài đặt chính gồm:

- `transformers==4.44.0`
- `accelerate==0.33.0`
- `peft==0.12.0`
- `bitsandbytes==0.49.2`
- `sentencepiece==0.2.0`
- `open_clip_torch==2.26.1`
- `bert-score==0.3.13`
- package LLaVA-Med nằm trong `GEMeX-Colab/llava-med`

Nếu model yêu cầu đăng nhập Hugging Face:

```bash
huggingface-cli login
```

Kiểm tra import:

```bash
python - <<'PY'
import torch
import transformers
import peft
import bitsandbytes

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 4.2. Cài trên Google Colab

Giải nén bundle vào `/content/gemex`, sau đó:

```bash
bash /content/gemex/GEMeX-Colab/install_colab.sh
```

Sau khi cài, restart runtime nếu Colab vẫn dùng phiên bản thư viện cũ.

## 5. Tạo dữ liệu VQA

Nếu chỉ muốn train lại trên dữ liệu hiện có, có thể bỏ qua mục này và bắt đầu từ [Mục 6](#6-chuyển-dữ-liệu-sang-format-gemex).

### 5.1. Dữ liệu đầu vào

Mỗi ca trong `Cabenh/` cần có:

- Ảnh `.png`.
- Annotation đã được chuẩn hóa.
- Một file `llm_input*.json`, ưu tiên bản `_with_colors.json`.

Ví dụ:

```text
Cabenh/BH/HHNBM-112/
├── imgs/
│   └── <object_id>.png
└── llm_input_HHNBM-112_ver1_with_colors.json
```

Trước khi gọi API, chạy dry-run để xem script tìm thấy bao nhiêu file:

```bash
python generate_bronchoscopy_reports_gpt.py \
  Cabenh \
  --output-root Cabenh_reports \
  --dry-run
```

### 5.2. Cấu hình API sinh dữ liệu

Hai script sinh dữ liệu gọi endpoint tương thích OpenAI:

```text
POST <BASE_URL>/chat/completions
```

Ví dụ với 9Router chạy local:

```bash
export NINEROUTER_BASE_URL="http://127.0.0.1:20128/v1"
export NINEROUTER_API_KEY="YOUR_API_KEY"
```

Có thể thay `--base-url`, `--model` và tên biến key nếu dùng nhà cung cấp khác.

### 5.3. Sinh báo cáo nguyên tử

Chạy thử một file trước:

```bash
python generate_bronchoscopy_reports_gpt.py \
  Cabenh \
  --output-root Cabenh_reports \
  --model gen-VQA \
  --limit 1 \
  --image-detail low
```

Nếu output hợp lệ, chạy toàn bộ:

```bash
python generate_bronchoscopy_reports_gpt.py \
  Cabenh \
  --output-root Cabenh_reports \
  --model gen-VQA \
  --image-detail low \
  --max-retries 5 \
  --base-sleep 5
```

Output có dạng:

```text
Cabenh_reports/<cohort>/<patient>/..._reports.json
```

Script tự bỏ qua `object_id` đã có trong output và lưu sau mỗi ảnh. Nếu tiến trình bị ngắt, chạy lại đúng lệnh để tiếp tục.

> Không thêm `--overwrite` khi muốn resume. Tùy chọn này tạo lại output từ đầu.

Nếu chỉ muốn sinh report từ annotation, không gửi ảnh lên API:

```bash
python generate_bronchoscopy_reports_gpt.py \
  Cabenh \
  --output-root Cabenh_reports \
  --model gen-VQA \
  --no-images
```

### 5.4. Sinh câu hỏi–trả lời VQA

Smoke test trên một file report:

```bash
REPORT_FILE="$(find Cabenh_reports -type f -name '*_reports.json' | head -n 1)"

python generate_bronchoscopy_vqa_gpt.py \
  "$REPORT_FILE" \
  --model gen-VQA \
  --max-retries 5
```

Chạy toàn bộ thư mục:

```bash
python generate_bronchoscopy_vqa_gpt.py \
  Cabenh_reports \
  --model gen-VQA \
  --max-retries 5 \
  --base-sleep 5
```

Mỗi ảnh có thể sinh tối đa:

- 3 câu open-ended.
- 2 câu closed-ended.
- 3 câu single-choice.
- 3 câu multi-choice.

Script không ép đủ số câu nếu finding không đủ tin cậy. Output được lưu cạnh report với hậu tố `_vqa.json`. Cơ chế resume dựa trên `object_id`, vì vậy có thể chạy lại lệnh sau khi lỗi mạng hoặc dừng máy.

### 5.5. Kiểm tra nhanh output VQA

Đếm file:

```bash
find Cabenh_reports -type f -name '*_reports.json' | wc -l
find Cabenh_reports -type f -name '*_vqa.json' | wc -l
```

Kiểm tra JSON có đọc được:

```bash
python - <<'PY'
import json
from pathlib import Path

files = list(Path("Cabenh_reports").rglob("*_vqa.json"))
errors = []
objects = 0

for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError("top-level value must be a list")
        objects += len(data)
    except Exception as exc:
        errors.append((str(path), str(exc)))

print("VQA files:", len(files))
print("Image objects:", objects)
print("Invalid files:", len(errors))
for item in errors[:20]:
    print(item)
PY
```

### 5.6. File VQA tổng hợp

Artifact đã gộp của dự án là:

```text
all_vqa.json
```

Nếu sinh lại dữ liệu từ đầu, có thể làm phẳng toàn bộ `*_vqa.json` thành một
record cho mỗi câu hỏi bằng lệnh sau. Lệnh ghi ra
`all_vqa_rebuilt.json` để không vô tình đè artifact hiện có:

```bash
python - <<'PY'
import json
from pathlib import Path

groups = (
    "open_ended_questions",
    "closed_ended_questions",
    "single_choice_questions",
    "multi_choice_questions",
)
rows = []

for source in sorted(Path("Cabenh_reports").rglob("*_vqa.json")):
    objects = json.loads(source.read_text(encoding="utf-8"))
    for obj in objects:
        for group in groups:
            for index, question in enumerate(obj.get(group, []), start=1):
                clues = question.get("visual_clue") or []
                rows.append(
                    {
                        "case_id": obj.get("case_id", ""),
                        "object_id": obj.get("object_id", ""),
                        "image_path": obj.get("image_path", ""),
                        "source_file": str(source.resolve()),
                        "question_group": group,
                        "question_index_in_group": index,
                        "question": question.get("question", ""),
                        "question_format": question.get("question_format", ""),
                        "type": question.get("question_type", ""),
                        "question_type": question.get("question_type", ""),
                        "answer": question.get("answer", ""),
                        "reason": question.get("reason", ""),
                        "options": question.get("options", {}),
                        "visual_clue": clues,
                        "visual_regions": [
                            clue.get("loc_token")
                            for clue in clues
                            if clue.get("loc_token")
                        ],
                        "visual_locations": [
                            clue.get("anatomical_location")
                            for clue in clues
                            if clue.get("anatomical_location")
                        ],
                    }
                )

output = Path("all_vqa_rebuilt.json")
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", output)
print("VQA records:", len(rows))
PY
```

Sau khi kiểm tra số lượng, schema và đường dẫn ảnh, mới đổi bản rebuilt thành
`all_vqa.json`. Nên sao lưu artifact cũ trước khi thay thế.

Mỗi dòng logic tương ứng một câu hỏi, với các trường quan trọng:

```json
{
  "case_id": "...",
  "object_id": "...",
  "image_path": "...",
  "question_group": "open_ended_questions",
  "question": "...",
  "question_format": "open-ended",
  "question_type": "mucosa_status",
  "answer": "...",
  "reason": "...",
  "options": {},
  "visual_clue": []
}
```

Kiểm tra số lượng và đường dẫn ảnh:

```bash
python - <<'PY'
import json
from pathlib import Path

rows = json.loads(Path("all_vqa.json").read_text(encoding="utf-8"))
missing = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]

print("VQA records:", len(rows))
print("Missing image paths:", len(missing))
print("First missing paths:", missing[:10])
PY
```

## 6. Chuyển dữ liệu sang format GEMeX

### 6.1. Chuẩn hóa đường dẫn ảnh

Training bundle dùng đường dẫn tương đối:

```text
images/Cabenh/<cohort>/<patient>/imgs/<image>.png
```

Trong khi `all_vqa.json` có thể chứa đường dẫn tuyệt đối. Tạo một bản trung gian đã chuẩn hóa:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("all_vqa.json")
target = Path("GEMeX-Colab/data/all_vqa_training_paths.json")
rows = json.loads(source.read_text(encoding="utf-8"))

for row in rows:
    parts = Path(row["image_path"]).parts
    try:
        start = parts.index("Cabenh")
    except ValueError as exc:
        raise ValueError(f"Path does not contain Cabenh: {row['image_path']}") from exc
    row["image_path"] = str(Path("images").joinpath(*parts[start:]))

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", target)
print("Records:", len(rows))
PY
```

Đảm bảo ảnh tương ứng nằm trong `GEMeX-Colab/images/`. Workspace hiện tại đã có 6.781 ảnh tại đây.

### 6.2. Chuyển sang conversation format

```bash
(
  cd GEMeX-Colab
  python convert_bronchoscopy_vqa_to_gemex.py \
    --input data/all_vqa_training_paths.json \
    --output data/bronchoscopy_train_data.json
)
```

Một record sau chuyển đổi có dạng:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nCâu hỏi..."},
    {
      "from": "gpt",
      "value": "<answer> ... <reason> ... <location> [[x1,y1,x2,y2]]"
    }
  ],
  "row_id": 0,
  "case_id": "...",
  "object_id": "...",
  "q_type": "open_ended_questions",
  "question_type": "mucosa_status",
  "type_prompt": "...",
  "image": "images/Cabenh/..."
}
```

Nếu script báo `Missing image paths > 0`, không bắt đầu train. Kiểm tra lại bước chuẩn hóa và thư mục ảnh.

## 7. Chia train, validation và test

Chia dữ liệu theo bệnh nhân với seed cố định:

```bash
python GEMeX-Colab/split_bronchoscopy_by_patient.py \
  --input GEMeX-Colab/data/bronchoscopy_train_data.json \
  --output-dir GEMeX-Colab/data \
  --seed 42
```

Output:

```text
GEMeX-Colab/data/bronchoscopy_train_70.json
GEMeX-Colab/data/bronchoscopy_validation_15.json
GEMeX-Colab/data/bronchoscopy_test_15.json
GEMeX-Colab/data/bronchoscopy_split_manifest.json
```

Kiểm tra dòng cuối phải là:

```text
Leakage checks passed: True
```

Có thể kiểm tra lại manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("GEMeX-Colab/data/bronchoscopy_split_manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))

for name, stats in manifest["splits"].items():
    print(
        name,
        "records=", stats["records"],
        "patients=", stats["patients"],
        "images=", stats["unique_images"],
    )

print("Leakage passed:", manifest["leakage_checks"]["passed"])
PY
```

> Không chia ngẫu nhiên từng câu hỏi hoặc từng ảnh. Nhiều ảnh của cùng một bệnh nhân phải nằm trong cùng một split.

## 8. Train mô hình

### 8.1. Cấu hình mặc định

Script `fine_tune_bronchoscopy.sh` dùng:

- Base model: `BoKelvin/GEMeX-VQA-Model-Simple`.
- Vision encoder: `openai/clip-vit-large-patch14`.
- 4-bit QLoRA.
- FP16.
- 3 epoch.
- Batch size 1.
- Gradient accumulation 16.
- Context length 1.024.
- Learning rate `2e-5`, cosine scheduler.

### 8.2. Luôn chạy smoke test trước

```bash
MAX_STEPS=1 \
OUTPUT_DIR="$PWD/gemex_outputs/smoke_test" \
bash GEMeX-Colab/llava-med/fine_tune_bronchoscopy.sh
```

Smoke test đạt yêu cầu khi:

- Kết thúc bước `1/1`.
- `train_loss` hữu hạn và khác 0.
- `grad_norm` hữu hạn.
- Không có lỗi CUDA out of memory.

Nếu smoke test lỗi, không chạy full training.

### 8.3. Chạy full training

```bash
OUTPUT_DIR="$PWD/gemex_outputs/patient_split_local_v1" \
bash GEMeX-Colab/llava-med/fine_tune_bronchoscopy.sh \
  2>&1 | tee "$PWD/gemex_outputs/train_console.log"
```

Không dùng lại một output directory cũ cho thí nghiệm có cấu hình khác. Nên đặt tên theo phiên bản:

```text
gemex_outputs/patient_split_seed42_v1
gemex_outputs/patient_split_seed42_lr1e-5_v2
```

### 8.4. Tùy chỉnh khi thiếu VRAM

Giảm context length:

```bash
MODEL_MAX_LENGTH=768 \
OUTPUT_DIR="$PWD/gemex_outputs/patient_split_ctx768_v1" \
bash GEMeX-Colab/llava-med/fine_tune_bronchoscopy.sh
```

Chọn GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR="$PWD/gemex_outputs/patient_split_local_v1" \
bash GEMeX-Colab/llava-med/fine_tune_bronchoscopy.sh
```

### 8.5. File cần giữ sau train

Trong output directory, tối thiểu cần giữ:

```text
adapter_model.safetensors
adapter_config.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
added_tokens.json
trainer_state.json
README.md
```

Các `checkpoint-*` chứa optimizer và scheduler cần thiết nếu muốn resume training. Chỉ xóa chúng sau khi chắc chắn không cần tiếp tục train và adapter cuối đã được lưu ở thư mục cha.

## 9. Validation

Đặt các biến dùng chung:

```bash
PROJECT_ROOT="$PWD"
ADAPTER_DIR="$PROJECT_ROOT/gemex_outputs/patient_split_local_v1"
EVAL_DIR="$PROJECT_ROOT/gemex_outputs/evaluation"
mkdir -p "$EVAL_DIR"
```

### 9.1. Smoke validation

Chạy thử 10 mẫu:

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_validation_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/validation_predictions.jsonl" \
  --max-samples 10
```

### 9.2. Full validation

Chạy lại mà không có `--max-samples`:

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_validation_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/validation_predictions.jsonl" \
  2>&1 | tee -a "$EVAL_DIR/validation_console.log"
```

Evaluator ghi từng prediction vào JSONL và flush ngay. Nếu bị ngắt, chạy lại đúng lệnh; các `index` đã hoàn thành sẽ được bỏ qua.

### 9.3. Tính lại metric, không load model

Sau khi prediction hoàn tất:

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_validation_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/validation_predictions.jsonl" \
  --metrics-only \
  --bertscore \
  --bertscore-device cuda
```

Metric được lưu tại:

```text
gemex_outputs/evaluation/validation_predictions.jsonl.metrics.json
```

Chỉ dùng validation để chọn checkpoint, prompt, parser và hyperparameter. Không điều chỉnh mô hình dựa trên test.

## 10. Test

Chỉ chạy test sau khi đã khóa:

- Adapter/checkpoint.
- Prompt.
- `max_new_tokens`.
- Cách parse output.
- Bộ metric.

### 10.1. Smoke test

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_test_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/test_predictions.jsonl" \
  --max-samples 10
```

### 10.2. Full test

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_test_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/test_predictions.jsonl" \
  2>&1 | tee -a "$EVAL_DIR/test_console.log"
```

Không chạy validation và test đồng thời trên cùng GPU.

Theo dõi tiến độ:

```bash
wc -l "$EVAL_DIR/test_predictions.jsonl"
tail -f "$EVAL_DIR/test_console.log"
```

Tổng test hiện tại có 8.201 mẫu. Khi `wc -l` bằng 8.201, prediction đã đủ.

### 10.3. Metric test cuối

```bash
PYTHONPATH="$PROJECT_ROOT/GEMeX-Colab/llava-med" \
python "$PROJECT_ROOT/GEMeX-Colab/evaluate_bronchoscopy.py" \
  --adapter "$ADAPTER_DIR" \
  --data "$PROJECT_ROOT/GEMeX-Colab/data/bronchoscopy_test_15.json" \
  --image-folder "$PROJECT_ROOT/GEMeX-Colab" \
  --output "$EVAL_DIR/test_predictions.jsonl" \
  --metrics-only \
  --bertscore \
  --bertscore-device cuda
```

Output:

```text
gemex_outputs/evaluation/test_predictions.jsonl.metrics.json
```

## 11. Ý nghĩa các metric

| Metric | Ý nghĩa |
|---|---|
| Answer Exact Match | Câu trả lời sau chuẩn hóa phải trùng hoàn toàn |
| Answer Token F1 | Mức chồng lấp token giữa dự đoán và đáp án |
| Closed-ended precision/recall/F1 | Chất lượng phân loại yes/no |
| Specificity | Khả năng nhận đúng mẫu âm |
| Single-choice accuracy | Tỷ lệ chọn đúng một đáp án |
| Multi-choice exact-set | Tập lựa chọn phải trùng hoàn toàn |
| Multi-choice micro/macro F1 | Đúng theo từng lựa chọn |
| Jaccard | Mức giao nhau của hai tập lựa chọn |
| ROUGE-L | Độ giống chuỗi cho câu hỏi mở |
| CIDEr | Độ tương đồng nội dung với đáp án tham chiếu |
| BERTScore | Độ gần nhau về ngữ nghĩa |
| Location IoU | Mức chồng lấp bounding box |
| IoU@0.5 | Tỷ lệ mẫu có IoU từ 0,5 trở lên |

Không nên chỉ nhìn Exact Match cho câu hỏi mở. Cần đọc đồng thời Token F1, ROUGE-L, CIDEr, BERTScore và một mẫu lỗi do chuyên gia kiểm tra.

## 12. Đánh giá đối chứng bằng GPT‑4o mini

Để so sánh công bằng, GPT và mô hình fine-tuned phải chạy trên đúng cùng test split. Tạo raw test set từ `row_id`:

```bash
python - <<'PY'
import json
from pathlib import Path

raw = json.loads(Path("all_vqa.json").read_text(encoding="utf-8"))
test = json.loads(
    Path("GEMeX-Colab/data/bronchoscopy_test_15.json").read_text(encoding="utf-8")
)
selected = [raw[item["row_id"]] for item in test]

output = Path("vqa_llm_eval/test_patient_split_raw.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", output, "records:", len(selected))
PY
```

Sau đó gọi một endpoint tương thích OpenAI:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

python vqa_llm_eval/run_openai_compatible_vqa.py \
  --input vqa_llm_eval/test_patient_split_raw.json \
  --output vqa_llm_eval/results/gpt4o_mini_test.jsonl \
  --model gpt-4o-mini \
  --image-detail low \
  --sleep 0.5
```

Chạy thử `--limit 20` trước khi chạy toàn bộ để kiểm tra quyền truy cập, prompt, parser và chi phí:

```bash
python vqa_llm_eval/run_openai_compatible_vqa.py \
  --input vqa_llm_eval/test_patient_split_raw.json \
  --output vqa_llm_eval/results/gpt4o_mini_smoke.jsonl \
  --model gpt-4o-mini \
  --image-detail low \
  --limit 20
```

Khi báo cáo, cần lưu model ID thực tế, provider, model snapshot, temperature, prompt version, token và chi phí. Không so sánh GPT trên một phần `all_vqa.json` với BoKelvin trên patient-level test split.

## 13. Lỗi thường gặp

### CUDA out of memory

- Đảm bảo không có hai tiến trình model cùng dùng GPU.
- Giảm `MODEL_MAX_LENGTH` từ 1.024 xuống 768 hoặc 512.
- Dừng notebook/process cũ còn giữ VRAM.
- Không tăng batch size trên GPU 12 GB.

Kiểm tra process GPU:

```bash
nvidia-smi
```

### Thiếu ảnh

Nếu convert hoặc evaluation báo thiếu ảnh:

```bash
find GEMeX-Colab/images -type f -name '*.png' | wc -l
```

Kiểm tra trường `image` phải có dạng tương đối bắt đầu bằng `images/`.

### API bị 429 hoặc 5xx

- Tăng `--base-sleep`.
- Giảm tốc độ gọi.
- Chạy lại cùng lệnh để resume.
- Không dùng `--overwrite`.

### Output LLM không phải JSON

- Smoke test trên một file.
- Kiểm tra model có tuân theo JSON mode/prompt hay không.
- Tăng `--max-tokens` nếu output bị cắt.
- Không gộp file lỗi vào `all_vqa.json` trước khi kiểm tra.

### Evaluation bị ngắt

Không xóa file prediction JSONL. Chạy lại đúng lệnh; evaluator tự tiếp tục từ các index chưa có.

### Metric JSON chưa phản ánh đủ test

File metric chỉ phản ánh các prediction có tại thời điểm tính. Chỉ tạo metric cuối khi số dòng prediction bằng số record của split.

## 14. Quy tắc tái lập thí nghiệm

Mỗi lần chạy nên ghi lại:

- Git commit hoặc phiên bản source.
- Seed chia dữ liệu.
- SHA256 của manifest và các split.
- Base model và adapter.
- Cấu hình QLoRA.
- GPU, CUDA và phiên bản thư viện.
- Prompt/model ID dùng để sinh VQA.
- Prediction JSONL gốc.
- Metric JSON.

Tính checksum:

```bash
sha256sum \
  GEMeX-Colab/data/bronchoscopy_split_manifest.json \
  GEMeX-Colab/data/bronchoscopy_train_70.json \
  GEMeX-Colab/data/bronchoscopy_validation_15.json \
  GEMeX-Colab/data/bronchoscopy_test_15.json \
  gemex_outputs/patient_split_local_v1/adapter_model.safetensors
```

## 15. Checklist ngắn

Trước khi train:

- [ ] Tất cả JSON đọc được.
- [ ] Không thiếu ảnh.
- [ ] `Leakage checks passed: True`.
- [ ] Smoke train có loss và grad norm hữu hạn.

Trước khi validation:

- [ ] Adapter và tokenizer đã lưu.
- [ ] Smoke validation sinh đúng `<answer>`, `<reason>`, `<location>`.
- [ ] Không có process khác chiếm GPU.

Trước khi test:

- [ ] Đã khóa checkpoint và evaluator.
- [ ] Không dùng test để chỉnh hyperparameter.
- [ ] Smoke test thành công.
- [ ] Output JSONL đặt tên riêng, không dùng chung với validation.

Sau test:

- [ ] Số dòng prediction bằng 8.201.
- [ ] Đã tính metric cuối.
- [ ] Đã lưu log, metadata và checksum.
- [ ] Đã phân tích lỗi theo dạng câu hỏi và nhờ chuyên gia kiểm tra mẫu.

## 16. Báo cáo

Báo cáo đánh giá hiện tại nằm tại:

```text
reports/BAO_CAO_DANH_GIA_VQA_NOI_SOI_PHE_QUAN.md
```
# Gen_VQA
