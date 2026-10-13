import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from gtts import gTTS
import sqlite3
import os
import io
import datetime

# ========================================================
# 1. การตั้งค่าหน้าเว็บ
# ========================================================
st.set_page_config(
    page_title="Thai Dessert Cultural AI",
    page_icon="🍡",
    layout="wide"
)

# ========================================================
# 2. ฐานข้อมูล SQLite สำหรับประวัติการทำนาย
# ========================================================
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            predicted_class TEXT,
            thai_name TEXT,
            eng_name TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_history(model_name, pred_class, th_name, en_name, conf):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, model_name, predicted_class, thai_name, eng_name, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (now, model_name, pred_class, th_name, en_name, conf))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

def generate_speech(text, lang='th'):
    tts = gTTS(text=text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ========================================================
# 3. ฐานข้อมูลความรู้ วัฒนธรรม ชื่อท้องถิ่นรายภาค และโภชนาการ (10 คลาส)
# ========================================================
DESSERT_INFO = {
    "chao_kuai": {
        "th_name": "เฉาก๊วย",
        "en_name": "Grass Jelly (Chao Kuai)",
        "calories": "90 - 120 kcal",
        "regional": {
            "central": "เฉาก๊วย",
            "north": "วุ้นดำ / เฉาก๊วย",
            "isan": "วุ้นดำ / เฉาก๊วย",
            "south": "เฉาก๊วย / วุ้นดำ"
        },
        "simple_th": "วุ้นสมุนไพรสีดำเนื้อหนึบ ทำจากต้นเฉาก๊วย นิยมกินคู่กับน้ำแข็งและน้ำตาลทรายแดงหรือน้ำเชื่อม",
        "simple_en": "Chewy black herbal jelly made from Chinese mesona, served cold with ice and brown sugar syrup.",
        "detail_th": "ทำจากการต้มต้นเฉาก๊วยแห้ง กรองน้ำแล้วผสมแป้งมันหรือแป้งท้าวยายม่อมเพื่อสร้างความหนึบ เนื้อสัมผัสเด้งนุ่ม ชุ่มคอ สรรพคุณแก้ร้อนใน",
        "detail_en": "Crafted by boiling dried Platostoma palustre stems and setting with starch. Famous for its cooling medicinal herbal properties.",
        "history_th": "มีต้นกำเนิดจากประเทศจีน เข้ามาในไทยผ่านชาวจีนโพ้นทะเล กลายเป็นของหวานดับร้อนยอดนิยมในวัฒนธรรมสตรีทฟู้ดไทย",
        "history_en": "Originated in southern China and brought to Thailand by Chinese settlers, becoming a beloved heat-relief treat."
    },
    "kaoniaomamuang": {
        "th_name": "ข้าวเหนียวมะม่วง",
        "en_name": "Mango Sticky Rice",
        "calories": "350 - 450 kcal",
        "regional": {
            "central": "ข้าวเหนียวมะม่วง",
            "north": "ข้าวนึ่งมะม่วง / ข้าวเหนียวมะม่วง",
            "isan": "ข้าวเหนียวหมากม่วง",
            "south": "เหนียวมะม่วง / ข้าวเหนียวมะม่วง"
        },
        "simple_th": "ข้าวเหนียวมูนกะทิรสหวานมัน ทานคู่กับมะม่วงน้ำดอกไม้สุกสีทองและราดหัวกะทิข้น",
        "simple_en": "Sweet glutinous rice steamed in coconut milk, served with fresh ripe mango and salted coconut drizzle.",
        "detail_th": "ใช้ข้าวเหนียวเขี้ยวงูมูนกับหัวกะทิสด น้ำตาล และเกลือ โรยหน้าด้วยถั่วทองคั่วกรอบ ทานคู่กับมะม่วงสุกรสหวานฉ่ำ",
        "detail_en": "Features glutinous rice steamed and steeped in sweetened rich coconut milk, garnished with crispy split mung beans.",
        "history_th": "ขนมหวานระดับโลกที่มีบันทึกตั้งแต่ปลายสมัยอยุธยา สะท้อนภูมิปัญญาการถนอมและประยุกต์ผลไม้ตามฤดูกาลของไทย",
        "history_en": "A globally acclaimed national dessert dating back to the late Ayutthaya era, representing Thai fruit pairing ingenuity."
    },
    "khanom_chan": {
        "th_name": "ขนมชั้น",
        "en_name": "Layer Sweet Cake (Khanom Chan)",
        "calories": "200 - 250 kcal (2-3 ชิ้น)",
        "regional": {
            "central": "ขนมชั้น",
            "north": "ขนมชั้น",
            "isan": "ขนมชั้น",
            "south": "ขนมชั้น"
        },
        "simple_th": "ขนมไทยโบราณเนื้อเหนียวนุ่ม เรียงซ้อนเป็นชั้นๆ หอมกลิ่นใบเตยและกะทิสด",
        "simple_en": "Steamed multi-layered jelly cake with a chewy texture, scented with pandan and coconut milk.",
        "detail_th": "ผสมแป้งมัน แป้งข้าวเจ้า และแป้งท้าว นวดกับน้ำกะทิและน้ำตาล นึ่งหยอดทีละชั้นสลับสีเขียวใบเตย นิยมทำ 9 ชั้นตามความเชื่อ",
        "detail_en": "Prepared from tapioca, rice, and arrowroot flours, steamed meticulously layer by layer. Traditionally crafted with 9 auspicious layers.",
        "history_th": "ขนมมงคลโบราณ นิยมใช้ในงานแต่งงานหรืองานเลื่อนตำแหน่ง สื่อถึงความเจริญก้าวหน้าและการเลื่อนขั้นที่เป็นลำดับขั้น",
        "history_en": "An auspicious ceremony dessert symbolizing career advancement and successive steps toward prosperity."
    },
    "khanomfakbua": {
        "th_name": "ขนมฝักบัว",
        "en_name": "Lotus Leaf Pancake (Khanom Fak Bua)",
        "calories": "180 - 220 kcal (ต่อชิ้น)",
        "regional": {
            "central": "ขนมฝักบัว / จูจุน",
            "north": "ขนมฝักบัว / จู่จุน",
            "isan": "ขนมฝักบัว / ขนมจี่จุน",
            "south": "ขนมจูจุน / ขนมจี่จุน"
        },
        "simple_th": "ขนมทอดทรงกลม ขอบบางกรอบหยัก ตรงกลางพองฟูนุ่ม หอมหวานน้ำตาลโตนด",
        "simple_en": "Deep-fried round sweet cake with crispy lace ruffled edges and a soft, airy domed center.",
        "detail_th": "ใช้แป้งข้าวเจ้าผสมแป้งข้าวเหนียว น้ำตาลโตนดและน้ำใบเตย ทอดในกระทะก้นลึกจนตรงกลางปูดฟูเป็นกระเปาะคล้ายฝักบัว",
        "detail_en": "Made from palm sugar and rice flour batter, deep-fried in hot oil to create a distinctive volcanic dome shape.",
        "history_th": "ขนมโบราณที่สื่อถึงความเบ่งบาน รุ่งเรือง ทางภาคใต้นิยมเรียกว่า 'ขนมจูจุน' ใช้ในงานบุญและเทศกาลสารทเดือนสิบ",
        "history_en": "Symbolizes blossoming good fortune. Commonly known as 'Joo Joon' in Southern Thailand for festive ceremonies."
    },
    "khanomkrok": {
        "th_name": "ขนมครก",
        "en_name": "Coconut Rice Pancakes (Khanom Krok)",
        "calories": "180 - 230 kcal (ต่อ 1 คู่)",
        "regional": {
            "central": "ขนมครก",
            "north": "ขนมครก (มักทานแบบต้นตำรับ)",
            "isan": "ขนมครก (นิยมโรยต้นหอม/ข้าวโพด)",
            "south": "ขนมครก (บางพื้นที่จิ้มน้ำตาลทราย)"
        },
        "simple_th": "ขนมกะทิหยอดหลุมกระทะเหล็ก ผิวด้านนอกกรอบ ด้านในนุ่มเยิ้มรสชาติหวานมันเค็ม",
        "simple_en": "Cast-iron grilled two-layer coconut pudding cups, crispy on the outer shell and silky inside.",
        "detail_th": "ประกอบด้วยแป้งข้าวเจ้าชั้นล่างที่กรอบ และหน้ากะทิสดหวานมัน โรยหน้าด้วยต้นหอม ข้าวโพดหวาน เผือก หรือฟักทอง",
        "detail_en": "Consists of two layers: crispy rice-flour shell and sweet coconut custard, topped with scallions or sweet corn.",
        "history_th": "ขนมคู่บ้านคู่เมืองตั้งแต่สมัยอยุธยา ทำในกระทะหลุมดินเผา เป็นสัญลักษณ์ของขนมคู่แฝดที่ต้องประกบกันก่อนรับประทาน",
        "history_en": "An iconic Ayutthaya-era street treat cooked in dimpled pans, paired face-to-face before being served."
    },
    "kluaithot": {
        "th_name": "กล้วยทอด (กล้วยแขก)",
        "en_name": "Deep-Fried Banana (Kluai Thot)",
        "calories": "250 - 300 kcal (3-4 ชิ้น)",
        "regional": {
            "central": "กล้วยแขก / กล้วยทอด",
            "north": "กล้วยทอด / กล้วยแขก",
            "isan": "กล้วยทอด / กล้วยแขก",
            "south": "กล้วยทอด / กล้วยเจี่ยน"
        },
        "simple_th": "กล้วยน้ำว้าฝานชิ้นชุบแป้งข้าวเจ้าผสมมะพร้าวขูดและงา ทอดจนสีเหลืองทองกรอบกรุบ",
        "simple_en": "Crisp golden banana slices coated in sesame, shredded coconut, and rice flour batter.",
        "detail_th": "ใช้กล้วยน้ำว้าห่ามชุบแป้งผสมมะพร้าวขูด หัวกะทิ และงาขาว ทอดในน้ำมันร้อนปานกลางจนแป้งกรอบนาน ไม่อมน้ำมัน",
        "detail_en": "Prepared using semi-ripe Namwa bananas sliced and dipped in coconut-sesame batter, deep-fried to perfection.",
        "history_th": "ได้รับอิทธิพลการทอดชุบแป้งจากพ่อค้าชาวอินเดียในอดีต จึงถูกเรียกว่า 'กล้วยแขก' ก่อนปรับเป็นสูตรมะพร้าวสไตล์ไทย",
        "history_en": "Originating from Indian batter-frying influences, later infused with native Thai grated coconut and sesame."
    },
    "lodchongnamkathi": {
        "th_name": "ลอดช่องน้ำกะทิ",
        "en_name": "Lod Chong in Sweet Coconut Milk",
        "calories": "220 - 280 kcal",
        "regional": {
            "central": "ลอดช่องน้ำกะทิ / ลอดช่องไทย",
            "north": "ลอดช่องน้ำกะทิ",
            "isan": "ลอดช่อง / คันหลาว",
            "south": "ลอดช่องน้ำกะทิ / เชงซิม"
        },
        "simple_th": "ตัวแป้งสีเขียวใบเตยเหนียวนุ่ม ลอยในน้ำกะทิน้ำตาลมะพร้าวเคี่ยวหอมควันเทียน ทานกับน้ำแข็งไส",
        "simple_en": "Fragrant green pandan droplet noodles served in iced smoked palm sugar coconut milk.",
        "detail_th": "กวนแป้งข้าวเจ้าผสมน้ำปูนใสและใบเตยจนเหนียว กดผ่านพิมพ์รูลงในน้ำเย็นจัด เสิร์ฟกับน้ำกะทิเคี่ยวน้ำตาลโตนด",
        "detail_en": "Rice flour cooked with pandan and limestone water, pressed through a perforated mold into ice water, served in coconut syrup.",
        "history_th": "ขนมไทยโบราณตั้งแต่สมัยอยุธยา คำว่า 'ลอดช่อง' สื่อถึงกรรมวิธีการกดแป้งให้ลอดผ่านรูกระป๋องพิมพ์",
        "history_en": "A classic Ayutthaya dessert named after the unique process of pressing dough through perforated sieve holes."
    },
    "pathongko": {
        "th_name": "ปาท่องโก๋",
        "en_name": "Thai Deep-Fried Dough Stick",
        "calories": "240 - 290 kcal (ต่อ 1 คู่)",
        "regional": {
            "central": "ปาท่องโก๋",
            "north": "ปาท่องโก๋",
            "isan": "ปาท่องโก๋",
            "south": "จาก๋วย / อิ่วจาก้วย / ปาท่องโก๋"
        },
        "simple_th": "แป้งสาลีตัดท่อนประกบคู่ทอดกรอบฟู นิยมทานคู่กับกาแฟโบราณ นมข้นหวาน หรือสังขยาใบเตย",
        "simple_en": "Golden crispy twin fried dough sticks, light and airy inside, served with pandan custard or sweet condensed milk.",
        "detail_th": "นวดแป้งสาลีหมักยีสต์และแอมโมเนีย ตัดเป็นชิ้นคู่ประกบกัน ทอดในน้ำมันร้อนจนพองตัวกรอบนอกเหนียวนุ่มใน",
        "detail_en": "Yeast-leavened wheat flour dough shaped into twin strips and deep-fried until puffed and crisp.",
        "history_th": "ดัดแปลงมาจาก 'อิ่วจาก้วย' ของชาวจีน ในภาคใต้ยังคงเรียกตามสำเนียงเดิมว่า 'จาก๋วย' นิยมกินเป็นอาหารเช้า",
        "history_en": "Adapted from Chinese 'Youtiao'. In Southern Thailand, it retains its traditional Hokkien name 'Cha Kway'."
    },
    "ruamnit": {
        "th_name": "รวมมิตร",
        "en_name": "Ruam Mit (Mixed Sweet Toppings)",
        "calories": "250 - 320 kcal",
        "regional": {
            "central": "รวมมิตร / รวมมิตรน้ำกะทิ",
            "north": "รวมมิตร / หวานเย็นรวมมิตร",
            "isan": "รวมมิตร",
            "south": "รวมมิตร / น้ำแข็งไสรวมมิตร"
        },
        "simple_th": "ของหวานรวมเครื่องหลากสีสัน เช่น ลอดช่อง ทับทิมกรอบ ขนุน วุ้นมะพร้าว ข้าวโพด ในน้ำกะทิสดใส่น้ำแข็ง",
        "simple_en": "A refreshing bowl uniting colorful jellies, water chestnuts, sweet corn, and jackfruit in chilled coconut syrup.",
        "detail_th": "รวบรวมทับทิมกรอบ แป้งกรอบสามสี ขนุนฉีก และเม็ดชิด ราดด้วยน้ำเชื่อมกะทิอบควันเทียนและน้ำแข็งไสละเอียด",
        "detail_en": "Assembled with mock pomegranate seeds, colorful starch jellies, and jackfruit, smothered with crushed ice and sweetened coconut milk.",
        "history_th": "สะท้อนวิถีวัฒนธรรมการแบ่งปันและความหลากหลาย โดยนำเครื่องขนมหวานที่มีในท้องถิ่นมารวมในชามเดียว",
        "history_en": "Embodies communal harmony by combining diverse local dessert textures and flavors into a celebratory single bowl."
    },
    "thongmuan": {
        "th_name": "ทองม้วน",
        "en_name": "Crispy Coconut Rolls (Thong Muan)",
        "calories": "150 - 180 kcal (3-4 ชิ้น)",
        "regional": {
            "central": "ทองม้วน",
            "north": "ทองม้วน / ขนมทองม้วน",
            "isan": "ทองม้วน",
            "south": "ทองม้วน / ทองพับ / ขนมพับ"
        },
        "simple_th": "แผ่นแป้งกะทิปิ้งบนพิมพ์ทองเหลืองจนกรอบ ม้วนเป็นแท่งทรงกระบอก หอมหวานกลิ่นกะทิและงาดำ",
        "simple_en": "Crisp rolled wafer cylinders made with rich coconut milk, palm sugar, and aromatic toasted black sesame.",
        "detail_th": "ทำจากแป้งสาลีผสมกะทิสด ไข่ไก่ และงาดำ หยอดลงเตาพิมพ์ปิ้งไฟอ่อนให้สุกหอม แล้วรีบม้วนเป็นทรงกระบอกทันทีขณะยังร้อน",
        "detail_en": "Made from rich coconut cream batter grilled on patterned brass waffle tongs, swiftly rolled into rolls while hot.",
        "history_th": "ได้รับอิทธิพลจากขนมอบโปรตุเกสสมัยอยุธยา คำว่า 'ทอง' สื่อถึงความร่ำรวยมั่งคั่ง ในภาคใต้นิยมทำแบบพับเรียกว่า 'ทองพับ'",
        "history_en": "Influenced by Portuguese pastries in the 17th century. The word 'Thong' (Gold) symbolizes continuous prosperity."
    }
}

CLASS_NAMES = [
    'chao_kuai', 'kaoniaomamuang', 'khanom_chan', 'khanomfakbua', 'khanomkrok',
    'kluaithot', 'lodchongnamkathi', 'pathongko', 'ruamnit', 'thongmuan'
]

# ========================================================
# 4. โมเดลและการ Preprocess ทั้ง 7 โมเดล
# ========================================================
MODELS_DIR = "/content/drive/MyDrive/INET/Applied Machine Learning/Kanomthai-Group/models"

MODEL_OPTIONS = {
    "EfficientNetB0 (ความแม่นยำ 97.2%)": {
        "file": "kanomthai_efficientnetb0.keras",
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    "DenseNet201 (96.53%)": {
        "file": "kanomthai_densenet201.keras",
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
    "MobileNetV2 (96.27%)": {
        "file": "kanomthai_mobilenetv2.keras",
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    "MobileNetV3Large (96.13%)": {
        "file": "kanomthai_mobilenetv3large.keras",
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input
    },
    "ResNet50 (95.60%)": {
        "file": "kanomthai_resnet50.keras",
        "preprocess": tf.keras.applications.resnet50.preprocess_input
    },
    "EfficientNetV2M (94.80%)": {
        "file": "kanomthai_efficientnetv2m.keras",
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input
    },
    "ResNet50V2 (93.07%)": {
        "file": "kanomthai_resnet50v2.keras",
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input
    }
}

@st.cache_resource
def load_prediction_model(model_filename):
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        model_path = os.path.join("models", model_filename)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=False)

def predict_image(image: Image.Image, model, preprocess_fn):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_fn(img_array)
    preds = model.predict(preprocessed_img, verbose=0)[0]
    return preds

# ========================================================
# 5. พจนานุกรมคำแปล UI (ระบบ 2 ภาษา ไทย-อังกฤษ)
# ========================================================
UI_TEXT = {
    "th": {
        "title": "🍡 ระบบจำแนกประเภทและเผยแพร่วัฒนธรรมขนมไทย",
        "subtitle": "Thai Dessert Cultural Image Classification AI with Deep Learning",
        "settings": "⚙️ การตั้งค่า",
        "model_select": "เลือกโมเดลทำนาย (Model):",
        "about_title": "📊 ข้อมูลระบบ",
        "about_desc": "- **จำนวนคลาส:** 10 คลาส\n- **ขนาดภาพ:** 224 x 224 pixels\n- **อ้างอิงงานวิจัย:** KediNet, AUCC & ThaiFood-50",
        "tab1": "📷 ถ่ายภาพและทำนาย",
        "tab2": "📜 ประวัติการใช้งาน",
        "tab3": "📚 คลังความรู้ขนมไทย 10 คลาส",
        "input_header": "1. นำเข้าภาพขนมไทย",
        "mode_select": "วิธีนำเข้าภาพ:",
        "mode_cam": "ถ่ายภาพผ่านกล้อง (Camera)",
        "mode_up": "อัปโหลดไฟล์ภาพ (Upload)",
        "cam_label": "ถ่ายภาพขนมไทยที่ต้องการตรวจสอบ",
        "up_label": "เลือกไฟล์ภาพ (JPG, PNG, WEBP):",
        "result_header": "2. ผลการวิเคราะห์",
        "waiting": "👈 กรุณาถ่ายภาพหรืออัปโหลดรูปภาพเพื่อเริ่มต้นการทำนาย",
        "pred_label": "🎉 ผลการทำนาย:",
        "confidence_label": "ความมั่นใจ (Confidence Score)",
        "warn_low_conf": "⚠️ คำเตือน: ความมั่นใจต่ำกว่า 60% ภาพอาจไม่ชัดเจน แสงไม่พอ หรืออาจไม่ใช่ขนมไทยใน 10 คลาสที่รองรับ",
        "tts_audio": "🔊 ฟังเสียงอ่านชื่อขนม:",
        "tts_th": "สำเนียงไทย",
        "tts_en": "สำเนียงอังกฤษ",
        "regional_title": "🗺️ ชื่อเรียกตามภูมิภาคต่างๆ ในประเทศไทย:",
        "reg_c": "ภาคกลาง:",
        "reg_n": "ภาคเหนือ:",
        "reg_i": "ภาคอีสาน:",
        "reg_s": "ภาคใต้:",
        "calories_label": "🔥 พลังงานโดยประมาณ:",
        "find_maps": "📍 ค้นหาร้านขายขนมนี้บน Google Maps",
        "detail_header": "📖 รายละเอียดเมนูและประวัติวัฒนธรรม",
        "detail_simple": "คำอธิบายเบื้องต้น:",
        "detail_proc": "วัตถุดิบและวิธีทำ:",
        "detail_hist": "🏛️ ประวัติความเป็นมา:",
        "top3_title": "🏆 ความน่าจะเป็นสูงสุด 3 อันดับแรก:",
        "col_rank": "อันดับ",
        "col_name_th": "ชื่อขนม (TH)",
        "col_name_en": "ชื่อขนม (EN)",
        "col_prob": "ความน่าจะเป็น (%)",
        "hist_total": "จำนวนการทำนายทั้งหมด",
        "hist_avg_conf": "ความมั่นใจเฉลี่ย",
        "hist_clear": "🗑️ ล้างประวัติทั้งหมด",
        "hist_empty": "ยังไม่มีประวัติการทำนายในระบบ",
        "hist_export": "📥 ดาวน์โหลดประวัติ (CSV)"
    },
    "en": {
        "title": "🍡 Thai Dessert Cultural AI & Classification",
        "subtitle": "AI-Powered Thai Dessert Recognition & Cultural Knowledge Platform",
        "settings": "⚙️ Settings",
        "model_select": "Select Model Architecture:",
        "about_title": "📊 System Info",
        "about_desc": "- **Classes:** 10 Categories\n- **Input Shape:** 224 x 224 pixels\n- **References:** KediNet, AUCC & ThaiFood-50",
        "tab1": "📷 Capture & Predict",
        "tab2": "📜 Prediction History",
        "tab3": "📚 Thai Dessert Encyclopedia",
        "input_header": "1. Provide Dessert Image",
        "mode_select": "Input Mode:",
        "mode_cam": "Camera Capture",
        "mode_up": "Upload File",
        "cam_label": "Take a photo of Thai dessert",
        "up_label": "Upload dessert image (JPG, PNG, WEBP):",
        "result_header": "2. Analysis Result",
        "waiting": "👈 Please capture or upload an image to start classification.",
        "pred_label": "🎉 Prediction:",
        "confidence_label": "Confidence Score",
        "warn_low_conf": "⚠️ Warning: Confidence is below 60%. The image might be blurry, poorly lit, or not among the 10 supported desserts.",
        "tts_audio": "🔊 Pronunciation Audio:",
        "tts_th": "Thai Accent",
        "tts_en": "English Accent",
        "regional_title": "🗺️ Regional Names across Thailand:",
        "reg_c": "Central Thailand:",
        "reg_n": "Northern Thailand:",
        "reg_i": "Northeastern (Isan):",
        "reg_s": "Southern Thailand:",
        "calories_label": "🔥 Estimated Energy:",
        "find_maps": "📍 Find this dessert on Google Maps",
        "detail_header": "📖 Menu Details & Cultural Heritage",
        "detail_simple": "Quick Overview:",
        "detail_proc": "Ingredients & Method:",
        "detail_hist": "🏛️ Cultural Heritage:",
        "top3_title": "🏆 Top-3 Probabilities:",
        "col_rank": "Rank",
        "col_name_th": "Thai Name",
        "col_name_en": "English Name",
        "col_prob": "Probability (%)",
        "hist_total": "Total Predictions",
        "hist_avg_conf": "Average Confidence",
        "hist_clear": "🗑️ Clear All History",
        "hist_empty": "No prediction history recorded yet.",
        "hist_export": "📥 Download History (CSV)"
    }
}

# ========================================================
# 6. ส่วนติดต่อผู้ใช้ (UI Layout)
# ========================================================
with st.sidebar:
    lang_choice = st.radio("🌐 ภาษา / Language", ["ภาษาไทย (TH)", "English (EN)"], horizontal=True)
    lang = "th" if "ไทย" in lang_choice else "en"
    t = UI_TEXT[lang]

    st.header(t["settings"])
    selected_model_name = st.selectbox(t["model_select"], list(MODEL_OPTIONS.keys()))
    current_model_cfg = MODEL_OPTIONS[selected_model_name]
    
    st.divider()
    st.subheader(t["about_title"])
    st.markdown(t["about_desc"])

st.title(t["title"])
st.caption(t["subtitle"])

tab1, tab2, tab3 = st.tabs([t["tab1"], t["tab2"], t["tab3"]])

# --------------------------------------------------------
# แท็บที่ 1: ถ่ายภาพและทำนาย
# --------------------------------------------------------
with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader(t["input_header"])
        input_mode = st.radio(t["mode_select"], [t["mode_cam"], t["mode_up"]], horizontal=True)
        
        image = None
        if input_mode == t["mode_cam"]:
            camera_img = st.camera_input(t["cam_label"])
            if camera_img:
                image = Image.open(camera_img)
        else:
            upload_img = st.file_uploader(t["up_label"], type=["jpg", "jpeg", "png", "webp"])
            if upload_img:
                image = Image.open(upload_img)

        if image:
            st.image(image, caption="Input Image", use_container_width=True)

    with col_result:
        st.subheader(t["result_header"])
        if image is not None:
            with st.spinner("Analyzing image..."):
                model = load_prediction_model(current_model_cfg["file"])
                preds = predict_image(image, model, current_model_cfg["preprocess"])
                
                top_idx = int(np.argmax(preds))
                confidence = float(preds[top_idx]) * 100
                pred_key = CLASS_NAMES[top_idx]
                info = DESSERT_INFO[pred_key]
                
                # บันทึกประวัติ
                save_history(
                    selected_model_name.split()[0],
                    pred_key,
                    info["th_name"],
                    info["en_name"],
                    confidence
                )

            # แสดงผลการทำนาย
            st.success(f"### {t['pred_label']} **{info['th_name'] if lang=='th' else info['en_name']}**")
            st.markdown(f"**{'English' if lang=='th' else 'ชื่อไทย'}:** {info['en_name'] if lang=='th' else info['th_name']}")
            st.metric(label=t["confidence_label"], value=f"{confidence:.2f}%")
            st.progress(min(confidence / 100.0, 1.0))

            # แจ้งเตือนความมั่นใจต่ำกว่า 60%
            if confidence < 60.0:
                st.warning(t["warn_low_conf"])

            # พลังงาน และปุ่มค้นหาร้านใน Google Maps
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"{t['calories_label']} **{info['calories']}**")
            with col_info2:
                maps_url = f"https://www.google.com/maps/search/{info['th_name']}"
                st.link_button(t["find_maps"], maps_url, use_container_width=True)

            # เสียงอ่าน Text-to-Speech (gTTS)
            st.markdown(f"#### {t['tts_audio']}")
            c_audio1, c_audio2 = st.columns(2)
            with c_audio1:
                st.caption(t["tts_th"])
                audio_th = generate_speech(info["th_name"], lang="th")
                st.audio(audio_th, format="audio/mp3")
            with c_audio2:
                st.caption(t["tts_en"])
                audio_en = generate_speech(info["en_name"], lang="en")
                st.audio(audio_en, format="audio/mp3")

            # ชื่อเรียกแต่ละภูมิภาค
            st.divider()
            st.markdown(f"#### {t['regional_title']}")
            reg_col1, reg_col2 = st.columns(2)
            with reg_col1:
                st.markdown(f"- **{t['reg_c']}** {info['regional']['central']}")
                st.markdown(f"- **{t['reg_n']}** {info['regional']['north']}")
            with reg_col2:
                st.markdown(f"- **{t['reg_i']}** {info['regional']['isan']}")
                st.markdown(f"- **{t['reg_s']}** {info['regional']['south']}")

            # ข้อมูลเชิงลึกและประวัติวัฒนธรรม
            st.divider()
            with st.expander(t["detail_header"], expanded=True):
                st.write(f"**{t['detail_simple']}** {info['simple_th'] if lang=='th' else info['simple_en']}")
                st.write(f"**{t['detail_proc']}** {info['detail_th'] if lang=='th' else info['detail_en']}")
                st.write(f"**{t['detail_hist']}** {info['history_th'] if lang=='th' else info['history_en']}")

            # ตาราง Top-3 Probabilities
            st.divider()
            st.markdown(f"#### {t['top3_title']}")
            top_3_indices = np.argsort(preds)[::-1][:3]
            top_3_data = []
            for rank, idx in enumerate(top_3_indices, 1):
                key = CLASS_NAMES[idx]
                top_3_data.append({
                    t["col_rank"]: rank,
                    t["col_name_th"]: DESSERT_INFO[key]["th_name"],
                    t["col_name_en"]: DESSERT_INFO[key]["en_name"],
                    t["col_prob"]: f"{preds[idx]*100:.2f}%"
                })
            st.table(pd.DataFrame(top_3_data))

        else:
            st.info(t["waiting"])

# --------------------------------------------------------
# แท็บที่ 2: ประวัติการใช้งานย้อนหลัง
# --------------------------------------------------------
with tab2:
    st.subheader(t["tab2"])
    history_df = get_history()
    
    if not history_df.empty:
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
            st.metric(t["hist_total"], f"{len(history_df)}")
        with col_m2:
            st.metric(t["hist_avg_conf"], f"{history_df['confidence'].mean():.2f}%")
        with col_m3:
            csv_data = history_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(t["hist_export"], data=csv_data, file_name="dessert_predictions_history.csv", mime="text/csv")
            
        st.dataframe(
            history_df[["id", "timestamp", "model_name", "thai_name", "eng_name", "confidence"]].rename(columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "model_name": "Model",
                "thai_name": "Thai Name",
                "eng_name": "English Name",
                "confidence": "Confidence (%)"
            }),
            use_container_width=True
        )
        
        if st.button(t["hist_clear"], type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.rerun()
    else:
        st.info(t["hist_empty"])

# --------------------------------------------------------
# แท็บที่ 3: คลังความรู้ขนมไทย 10 คลาส
# --------------------------------------------------------
with tab3:
    st.subheader(t["tab3"])
    for key, item in DESSERT_INFO.items():
        with st.expander(f"🍡 {item['th_name']} ({item['en_name']}) - 🔥 {item['calories']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🇹🇭 ข้อมูลภาษาไทย")
                st.write(f"**คำอธิบาย:** {item['simple_th']}")
                st.write(f"**กรรมวิธี:** {item['detail_th']}")
                st.write(f"**ประวัติ:** {item['history_th']}")
                st.markdown("**ชื่อเรียกตามภูมิภาค:**")
                st.write(f"- กลาง: {item['regional']['central']} | เหนือ: {item['regional']['north']}")
                st.write(f"- อีสาน: {item['regional']['isan']} | ใต้: {item['regional']['south']}")
            with c2:
                st.markdown("#### 🇬🇧 English Information")
                st.write(f"**Description:** {item['simple_en']}")
                st.write(f"**Preparation:** {item['detail_en']}")
                st.write(f"**History:** {item['history_en']}")
                st.markdown("**Regional Names:**")
                st.write(f"- Central: {item['regional']['central']} | North: {item['regional']['north']}")
                st.write(f"- Isan: {item['regional']['isan']} | South: {item['regional']['south']}")
