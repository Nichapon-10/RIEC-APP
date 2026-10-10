"""ตรรกะทำนาย แยกจากหน้าเว็บ เพื่อทดสอบง่าย

ต้องเหมือนตอนเทรน (fold_trainer.py ใน Colab):
  - ภาพ RGB ขนาด 224x224
  - ย่อภาพแบบ nearest (ค่าเริ่มต้นของ flow_from_dataframe)
  - ส่งค่าพิกเซล 0-255 เข้าโมเดลตรง ๆ  (efficientnet.preprocess_input ใน Keras 3
    ไม่ทำอะไร เพราะ EfficientNetB0 มี Rescaling/Normalization อยู่ในตัวโมเดลแล้ว)
  - ลำดับคลาสต้องตรงกับ class_names.json
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).parent
IMG_SIZE = (224, 224)
CLASS_NAMES = json.loads((HERE / "class_names.json").read_text(encoding="utf-8"))


def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(HERE / "rice_model.keras", compile=False)


def prepare(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img)      # รูปจากมือถือมักมีแท็กหมุนภาพ
    img = img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    return np.asarray(img, dtype="float32")[None, ...]


def predict(model, img: Image.Image) -> np.ndarray:
    """คืนความน่าจะเป็น 4 ค่า ตามลำดับ CLASS_NAMES"""
    return model.predict(prepare(img), verbose=0)[0]
