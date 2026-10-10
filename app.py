import streamlit as st
from PIL import Image

import predict as P

st.set_page_config(page_title="หมอข้าว — ตรวจโรคใบข้าวด้วย AI", page_icon="🌾", layout="centered")

TH = {
    "Bacterial Leaf Blight": ("โรคขอบใบแห้ง", "แผลเป็นทางยาวสีเหลืองถึงขาวเริ่มจากขอบใบ แล้วลามเข้าด้านใน"),
    "Brown Spot":            ("โรคใบจุดสีน้ำตาล", "จุดรูปวงรีหรือกลมสีน้ำตาลกระจายตามใบ"),
    "Healthy Rice Leaf":     ("ใบข้าวปกติ", "ไม่พบอาการของโรคทั้ง 3 ชนิดที่ระบบรู้จัก"),
    "Leaf Blast":            ("โรคไหม้", "แผลรูปตาหรือรูปกระสวย กลางแผลสีเทาขาว ขอบสีน้ำตาล"),
}
THRESHOLD = 0.70

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:640px}
h1,h2,h3,p,li,label,button{font-family:'Noto Sans Thai','Sarabun',system-ui,sans-serif!important}
.result{border-radius:14px;padding:18px 20px;margin:12px 0;background:#eef6e6;border:1px solid #c5dca8}
.result.warn{background:#fff6e0;border-color:#ecd28b}
.result h2{margin:0 0 4px 0;font-size:1.6rem;color:#1d5e2a}
.result.warn h2{color:#7a5200}
.small{color:#555;font-size:.92rem}
div[data-testid="stCameraInput"] button, div[data-testid="stFileUploader"] button{min-height:48px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def get_model():
    return P.load_model()


st.title("🌾 หมอข้าว")
st.caption("ตรวจโรคใบข้าวเบื้องต้นด้วย AI — ถ่ายรูปใบข้าวหรืออัปโหลดรูป")

with st.expander("วิธีถ่ายรูปให้แม่นยำ"):
    st.markdown("- ถ่ายในที่มีแสงสว่างเพียงพอ\n- ให้ใบอยู่เต็มกรอบและชัด\n- ใช้พื้นหลังเรียบ ไม่รก\n- ถ่ายใบเดียวต่อหนึ่งรูป")

tab_cam, tab_up = st.tabs(["📷 ถ่ายรูป", "🖼️ อัปโหลดรูป"])
with tab_cam:
    shot = st.camera_input("ถ่ายรูปใบข้าว", label_visibility="collapsed")
with tab_up:
    up = st.file_uploader("เลือกรูปใบข้าว", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

file = shot or up
if file is not None:
    img = Image.open(file)
    st.image(img, caption="รูปที่ใช้ตรวจ", use_container_width=True)
    with st.spinner("กำลังวิเคราะห์…"):
        probs = P.predict(get_model(), img)
    best = int(probs.argmax())
    conf = float(probs[best])
    name_th, desc = TH[P.CLASS_NAMES[best]]
    low = conf < THRESHOLD

    cls = "result warn" if low else "result"
    st.markdown(f'<div class="{cls}"><h2>{name_th}</h2>'
                f'<div>ความมั่นใจ <b>{conf*100:.1f}%</b></div>'
                f'<div class="small">{desc}</div></div>', unsafe_allow_html=True)
    if low:
        st.warning("ความมั่นใจต่ำกว่า 70% ผลนี้อาจไม่น่าเชื่อถือ ลองถ่ายใหม่ให้ชัดขึ้น "
                   "หรืออาจเป็นโรคอื่นที่ระบบไม่รู้จัก ควรปรึกษาเจ้าหน้าที่เกษตร")

    st.subheader("ความน่าจะเป็นแต่ละกลุ่ม")
    for i in sorted(range(len(probs)), key=lambda k: -probs[k]):
        st.write(TH[P.CLASS_NAMES[i]][0])
        st.progress(float(probs[i]), text=f"{probs[i]*100:.1f}%")

st.divider()
st.caption("⚠️ เป็นเครื่องมือคัดกรองเบื้องต้นเพื่อการศึกษา ตรวจได้เฉพาะ 4 กลุ่ม "
           "(โรคขอบใบแห้ง, โรคใบจุดสีน้ำตาล, โรคไหม้, ใบปกติ) ไม่แทนการวินิจฉัยของผู้เชี่ยวชาญ "
           "ความแม่นยำราว 94% จากการทดสอบแบบ 5-fold cross validation")
