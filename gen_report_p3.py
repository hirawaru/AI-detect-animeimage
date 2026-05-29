# -*- coding: utf-8 -*-
REPORT_PATH = r"e:\aidetect\SEMINAR_REPORT.md"

lines = []
lines.append("## Chuong 3. Trien khai cac chuc nang\n")
lines.append("\n")
lines.append("### 3.1 Chuan bi du lieu\n")
lines.append("\n")
lines.append("#### 3.1.1 Nguon du lieu\n")
lines.append("\n")
lines.append("Dataset duoc lay tu Kaggle voi ten: **[TEN DATASET - sinh vien dien]**\n")
lines.append("\n")
lines.append("- **Link Kaggle:** [...]\n")
lines.append("- **Mo ta:** Tap du lieu gom anh anime/illustration duoc phan thanh hai lop: anh do con nguoi ve (Natural) va anh do AI tao ra (Synthetic).\n")
lines.append("- **Tong so anh:** ~6,543 anh\n")
lines.append("  - Natural (anh thuc): ~3,275 anh\n")
lines.append("  - Synthetic (anh AI): ~3,268 anh\n")
lines.append("- **Dinh dang:** JPG, PNG, BMP, GIF\n")
lines.append("\n")

with open(REPORT_PATH, "a", encoding="utf-8") as f:
    f.writelines(lines)
print("Written OK")
