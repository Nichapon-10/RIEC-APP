# หมอข้าว — ตรวจโรคใบข้าวด้วย AI

แอป Streamlit ที่ใช้ EfficientNetB0 (transfer learning) ตรวจโรคใบข้าว 4 กลุ่ม:
โรคขอบใบแห้ง, โรคใบจุดสีน้ำตาล, โรคไหม้, ใบปกติ

- ผล 5-fold CV: accuracy 0.9432 ± 0.0051, F1-macro 0.9429
- รันในเครื่อง: `pip install -r requirements.txt` แล้ว `streamlit run app.py`
- `landing.html` คือหน้าต้อนรับ (โฮสต์แยกได้ เช่น GitHub Pages)

เป็นเครื่องมือคัดกรองเบื้องต้นเพื่อการศึกษา ไม่แทนการวินิจฉัยของผู้เชี่ยวชาญ
