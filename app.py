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
# 2. ฐานข้อมูลประวัติการทำนาย (SQLite)
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
# 3. ฐานข้อมูลวัฒนธรรมและภาษาท้องถิ่น 10 คลาส (แยก TH / EN สมบูรณ์)
# ========================================================
DESSERT_INFO = {
    "chao_kuai": {
        "name": {"th": "เฉาก๊วย", "en": "Grass Jelly (Chao Kuai)"},
        "simple": {
            "th": "วุ้นสมุนไพรสีดำเนื้อหนึบ ทำจากต้นเฉาก๊วย นิยมกินคู่กับน้ำแข็งและน้ำตาลทรายแดงหรือน้ำเชื่อม",
            "en": "Chewy black herbal jelly made from Chinese mesona plants, served cold with ice and brown sugar syrup."
        },
        "detail": {
            "th": "ทำจากการต้มต้นเฉาก๊วยแห้ง กรองน้ำแล้วผสมแป้งมันสำปะหลังหรือแป้งท้าวยายม่อมเพื่อสร้างความหนึบ เนื้อสัมผัสเด้งนุ่ม ชุ่มคอ สรรพคุณแก้ร้อนใน กระหายน้ำ",
            "en": "Prepared by boiling dried Platostoma palustre stems and setting with starch. Renowned for its refreshing aroma and traditional cooling medicinal properties."
        },
        "history": {
            "th": "มีต้นกำเนิดจากประเทศจีนตอนใต้ เข้ามาในไทยผ่านชาวจีนโพ้นทะเลและกลายเป็นของหวานดับร้อนยอดนิยมในวัฒนธรรมสตรีทฟู้ดไทย",
            "en": "Originated in southern China and brought to Thailand by Chinese immigrants, quickly becoming an indispensable heat-relief treat in Thai street food culture."
        },
        "regional": {
            "th": {"central": "เฉาก๊วย", "north": "วุ้นดำ / เฉาก๊วย", "isan": "วุ้นดำ / เฉาก๊วย", "south": "เฉาก๊วย / วุ้นดำ"},
            "en": {"central": "Chao Kuai", "north": "Wun Dam / Chao Kuai", "isan": "Wun Dam / Chao Kuai", "south": "Chao Kuai / Wun Dam"}
        }
    },
    "kaoniaomamuang": {
        "name": {"th": "ข้าวเหนียวมะม่วง", "en": "Mango Sticky Rice"},
        "simple": {
            "th": "ข้าวเหนียวมูนกะทิรสหวานมัน ทานคู่กับมะม่วงน้ำดอกไม้สุกสีทองและราดหัวกะทิสด",
            "en": "Sweet glutinous rice steeped in rich coconut milk, paired with ripe golden mango and salted coconut cream drizzle."
        },
        "detail": {
            "th": "ใช้ข้าวเหนียวเขี้ยวงูอย่างดีมูนกับหัวกะทิคั้นสด น้ำตาลทราย และเกลือ โรยหน้าด้วยถั่วทองคั่วกรอบ ทานเคียงกับมะม่วงสุกรสหวานอมเปรี้ยว",
            "en": "Crafted from long-grain sticky rice steeped in warm coconut cream, palm sugar, and sea salt, garnished with toasted crispy split mung beans."
        },
        "history": {
            "th": "ของหวานระดับตำนานของไทยที่มีประวัติบันทึกตั้งแต่สมัยอยุธยาตอนปลาย สะท้อนภูมิปัญญาการนำผลไม้ตามฤดูกาลมาจับคู่กับข้าวเหนียวมูนอย่างลงตัว",
            "en": "A globally celebrated Thai dessert dating back to the late Ayutthaya era, demonstrating centuries of culinary finesse in pairing seasonal fruits."
        },
        "regional": {
            "th": {"central": "ข้าวเหนียวมะม่วง", "north": "ข้าวนึ่งมะม่วง / ข้าวเหนียวมะม่วง", "isan": "ข้าวเหนียวหมากม่วง", "south": "เหนียวมะม่วง / ข้าวเหนียวมะม่วง"},
            "en": {"central": "Khao Niao Mamuang", "north": "Khao Nueng Mamuang", "isan": "Khao Niao Mak Muang", "south": "Niao Mamuang"}
        }
    },
    "khanom_chan": {
        "name": {"th": "ขนมชั้น", "en": "Layer Sweet Cake (Khanom Chan)"},
        "simple": {
            "th": "ขนมไทยโบราณเนื้อเหนียวนุ่ม เรียงซ้อนเป็นชั้นๆ หอมกลิ่นใบเตยและกะทิสด",
            "en": "Traditional steamed multi-layered cake featuring a chewy, delicate texture scented with natural pandan and coconut milk."
        },
        "detail": {
            "th": "ผสมแป้งมันสำปะหลัง แป้งข้าวเจ้า และแป้งท้าวยายม่อม นวดกับน้ำกะทิและน้ำตาล นึ่งหยอดทีละชั้นสลับสีเขียวใบเตย นิยมทำ 9 ชั้นตามความเชื่อมงคล",
            "en": "Steamed meticulously layer by layer using tapioca, rice, and arrowroot starch blended with fragrant coconut milk. Traditionally made in 9 auspicious layers."
        },
        "history": {
            "th": "หนึ่งในขนมมงคลโบราณที่นิยมใช้ในพิธีมงคลสมรสหรืองานเลื่อนตำแหน่ง สื่อถึงความเจริญก้าวหน้า การได้เลื่อนยศเลื่อนขั้นที่เป็นลำดับชั้นอย่างมั่นคง",
            "en": "An ancient royal dessert symbolizing career advancement and prosperous milestones, often presented at weddings and promotion ceremonies."
        },
        "regional": {
            "th": {"central": "ขนมชั้น", "north": "ขนมชั้น", "isan": "ขนมชั้น", "south": "ขนมชั้น"},
            "en": {"central": "Khanom Chan", "north": "Khanom Chan", "isan": "Khanom Chan", "south": "Khanom Chan"}
        }
    },
    "khanomfakbua": {
        "name": {"th": "ขนมฝักบัว", "en": "Lotus Leaf Pancake (Khanom Fak Bua)"},
        "simple": {
            "th": "ขนมทอดทรงกลม ขอบบางกรอบหยัก ตรงกลางพองฟูนุ่ม หอมหวานน้ำตาลโตนด",
            "en": "Deep-fried round pastry with crispy ruffled edges and a fluffy, puffy domed center flavored with natural palm sugar."
        },
        "detail": {
            "th": "ใช้แป้งข้าวเจ้าผสมแป้งข้าวเหนียว น้ำตาลโตนดหรือน้ำตาลมะพร้าว และน้ำใบเตย ทอดในกระทะก้นลึกจนตรงกลางปูดฟูเป็นรูปกระเปาะคล้ายฝักบัว",
            "en": "Prepared from a batter of rice and sticky rice flours sweetened with palm sugar, deep-fried in hot oil to create its iconic volcano-like raised center."
        },
        "history": {
            "th": "ขนมโบราณที่สื่อถึงความเจริญรุ่งเรืองและการเบ่งบานของชีวิต ทางภาคใต้นิยมเรียกว่า 'ขนมจูจุน' ใช้ในงานบุญและเทศกาลสำคัญ",
            "en": "Symbolizes blossoming fortune and life growth. Widely known as 'Joo Joon' in Southern Thailand and served during sacred merit-making events."
        },
        "regional": {
            "th": {"central": "ขนมฝักบัว / จูจุน", "north": "ขนมฝักบัว / จู่จุน", "isan": "ขนมฝักบัว / จี่จุน", "south": "ขนมจูจุน / ขนมจี่จุน"},
            "en": {"central": "Khanom Fak Bua / Joo Joon", "north": "Khanom Fak Bua / Joo Joon", "isan": "Khanom Fak Bua / Jee Joon", "south": "Khanom Joo Joon / Jee Joon"}
        }
    },
    "khanomkrok": {
        "name": {"th": "ขนมครก", "en": "Coconut Rice Pancakes (Khanom Krok)"},
        "simple": {
            "th": "ขนมกะทิหยอดหลุมกระทะเหล็ก ผิวด้านนอกกรอบ ด้านในนุ่มเยิ้มรสหวานมันเค็ม",
            "en": "Traditional grilled two-layer coconut pudding cups, crispy on the outside shell and luxuriously creamy on the inside."
        },
        "detail": {
            "th": "ประกอบด้วยแป้งข้าวเจ้าชั้นล่างที่กรอบเกรียม และหน้ากะทิสดหวานมัน แต่งหน้าด้วยต้นหอมซอย ข้าวโพดหวาน เผือก หรือฟักทอง",
            "en": "Consists of two layers: a crunchy rice-flour base and rich sweet coconut cream, commonly topped with sweet corn, scallions, or taro."
        },
        "history": {
            "th": "ขนมคู่บ้านคู่เมืองตั้งแต่สมัยอยุธยา ทำในกระทะหลุมดินเผา เป็นสัญลักษณ์ของขนมคู่แฝดที่ต้องนำสองฝามาประกบกันก่อนรับประทาน",
            "en": "An iconic Ayutthaya-era street delicacy cooked in dimpled cast-iron pans, traditionally served face-to-face as joined twin halves."
        },
        "regional": {
            "th": {"central": "ขนมครก", "north": "ขนมครก (ต้นตำรับ)", "isan": "ขนมครก (โรยข้าวโพด/ต้นหอม)", "south": "ขนมครก (ทานคู่กับน้ำตาลทราย)"},
            "en": {"central": "Khanom Krok", "north": "Khanom Krok (Traditional)", "isan": "Khanom Krok (Corn/Scallion)", "south": "Khanom Krok (Dip with sugar)"}
        }
    },
    "kluaithot": {
        "name": {"th": "กล้วยทอด (กล้วยแขก)", "en": "Deep-Fried Banana (Kluai Thot)"},
        "simple": {
            "th": "กล้วยน้ำว้าหั่นชิ้นชุบแป้งผสมมะพร้าวขูดและงา ทอดจนสีเหลืองทองกรอบกรุบ",
            "en": "Crisp golden banana fritters coated in sesame seeds, fresh grated coconut, and aromatic rice flour batter."
        },
        "detail": {
            "th": "ใช้กล้วยน้ำว้าห่ามชุบแป้งผสมมะพร้าวขูด หัวกะทิ น้ำตาลทราย และงาขาว ทอดในน้ำมันร้อนปานกลางจนแป้งกรอบทนนาน ไม่อมน้ำมัน",
            "en": "Uses firm Namwa bananas sliced and dipped in coconut-sesame batter, fried to golden-brown crunchiness."
        },
        "history": {
            "th": "ได้รับอิทธิพลการทอดชุบแป้งจากพ่อค้าชาวอินเดียในอดีต จึงถูกเรียกว่า 'กล้วยแขก' ก่อนปรับรสชาติด้วยการใส่มะพร้าวขูดและงาสไตล์ไทย",
            "en": "Influenced by Indian batter-frying culinary trades centuries ago, subsequently customized with native Thai coconut and sesame."
        },
        "regional": {
            "th": {"central": "กล้วยแขก / กล้วยทอด", "north": "กล้วยทอด / กล้วยแขก", "isan": "กล้วยทอด / กล้วยแขก", "south": "กล้วยทอด / กล้วยเจี่ยน"},
            "en": {"central": "Kluai Khaek / Kluai Thot", "north": "Kluai Thot / Kluai Khaek", "isan": "Kluai Thot / Kluai Khaek", "south": "Kluai Thot / Kluai Chian"}
        }
    },
    "lodchongnamkathi": {
        "name": {"th": "ลอดช่องน้ำกะทิ", "en": "Lod Chong in Sweet Coconut Milk"},
        "simple": {
            "th": "ตัวแป้งสีเขียวใบเตยเหนียวนุ่ม ลอยในน้ำกะทิน้ำตาลมะพร้าวเคี่ยวหอมควันเทียน ทานกับน้ำแข็งไส",
            "en": "Fragrant green pandan droplet noodles served submerged in iced palm sugar coconut milk smoked with scented candles."
        },
        "detail": {
            "th": "กวนแป้งข้าวเจ้าผสมน้ำปูนใสและใบเตยจนเหนียวใส กดผ่านพิมพ์รูลงในน้ำเย็นจัด เสิร์ฟพร้อมน้ำกะทิเคี่ยวน้ำตาลโตนดและน้ำแข็งไส",
            "en": "Pandan rice flour cooked to a smooth paste, pressed through perforated brass molds into ice water, served in rich coconut nectar syrup."
        },
        "history": {
            "th": "ขนมไทยโบราณที่มีมาตั้งแต่สมัยอยุธยา คำว่า 'ลอดช่อง' มาจากกรรมวิธีการกดแป้งให้ลอดผ่านรูกระป๋องพิมพ์",
            "en": "An ancient Thai classic dating back to Ayutthaya, directly named after the unique action of pressing batter through sieve holes."
        },
        "regional": {
            "th": {"central": "ลอดช่องน้ำกะทิ / ลอดช่องไทย", "north": "ลอดช่องน้ำกะทิ", "isan": "ลอดช่อง / ขนมคันหลาว", "south": "ลอดช่องน้ำกะทิ / เชงซิม"},
            "en": {"central": "Lod Chong Thai", "north": "Lod Chong Nam Kathi", "isan": "Lod Chong / Khanom Khan Lao", "south": "Lod Chong Nam Kathi / Cheng Sim"}
        }
    },
    "pathongko": {
        "name": {"th": "ปาท่องโก๋", "en": "Thai Deep-Fried Dough Stick"},
        "simple": {
            "th": "แป้งสาลีตัดท่อนประกบคู่ทอดกรอบฟู นิยมทานคู่กับกาแฟโบราณ นมข้นหวาน หรือสังขยาใบเตย",
            "en": "Crispy golden twin fried dough batons, airy and soft inside, traditionally paired with morning coffee or pandan custard."
        },
        "detail": {
            "th": "นวดแป้งสาลีหมักยีสต์ ตัดเป็นชิ้นประกบคู่แล้วยืดลงทอดในน้ำมันร้อน พองตัวกรอบนอกนุ่มเหนียวด้านใน",
            "en": "Yeast-leavened wheat flour dough shaped into twin strips and flash-fried until puffy, crisp, and slightly savory."
        },
        "history": {
            "th": "ดัดแปลงมาจาก 'อิ่วจาก้วย' ของชาวจีน ในภาคใต้ยังคงเรียกตามสำเนียงเดิมว่า 'จาก๋วย' นิยมกินเป็นอาหารเช้าคู่กับเครื่องดื่มร้อน",
            "en": "Adapted from Chinese 'Youtiao'. In Southern Thailand, it faithfully preserves its Hokkien name 'Cha Kway'."
        },
        "regional": {
            "th": {"central": "ปาท่องโก๋", "north": "ปาท่องโก๋", "isan": "ปาท่องโก๋", "south": "จาก๋วย / อิ่วจาก้วย / ปาท่องโก๋"},
            "en": {"central": "Pathongko", "north": "Pathongko", "isan": "Pathongko", "south": "Cha Kway / Iu Cha Kway / Pathongko"}
        }
    },
    "ruamnit": {
        "name": {"th": "รวมมิตร", "en": "Ruam Mit (Mixed Sweet Toppings)"},
        "simple": {
            "th": "ของหวานรวมเครื่องหลากสีสัน เช่น ลอดช่อง ทับทิมกรอบ ขนุน วุ้นมะพร้าว ข้าวโพด ในน้ำกะทิสดใส่น้ำแข็ง",
            "en": "A vibrant icy dessert combining colorful jellies, crunchy water chestnuts, jackfruit, and sweet corn in cold coconut milk."
        },
        "detail": {
            "th": "รวบรวมทับทิมกรอบ แป้งกรอบสามสี ลอดช่อง ขนุนฉีก และเม็ดชิด ราดด้วยน้ำกะทิเคี่ยวและน้ำแข็งไสละเอียด ดับกระหายคลายร้อน",
            "en": "An assorted bowl of mock pomegranate pearls, jackfruit slices, and pandan noodles submerged in chilled sweet coconut nectar."
        },
        "history": {
            "th": "สะท้อนวัฒนธรรมการแบ่งปันและความหลากหลาย โดยนำวัตถุดิบหวานที่มีในท้องถิ่นมารวมไว้ในชามเดียวอย่างลงตัว",
            "en": "Symbolizes harmony and community generosity by bringing diverse regional sweet toppings together in a single bowl."
        },
        "regional": {
            "th": {"central": "รวมมิตร / รวมมิตรน้ำกะทิ", "north": "รวมมิตร / หวานเย็นรวมมิตร", "isan": "รวมมิตร", "south": "รวมมิตร / น้ำแข็งไสรวมมิตร"},
            "en": {"central": "Ruam Mit", "north": "Ruam Mit / Wan Yen", "isan": "Ruam Mit", "south": "Ruam Mit / Nam Kheng Sai"}
        }
    },
    "thongmuan": {
        "name": {"th": "ทองม้วน", "en": "Crispy Coconut Rolls (Thong Muan)"},
        "simple": {
            "th": "แผ่นแป้งกะทิปิ้งบนพิมพ์ทองเหลืองจนกรอบ ม้วนเป็นแท่งทรงกระบอก หอมหวานกลิ่นกะทิและงาดำ",
            "en": "Crisp rolled wafer cylinders made from rich coconut cream, palm sugar, and aromatic toasted black sesame."
        },
        "detail": {
            "th": "ทำจากแป้งสาลีผสมกะทิสด ไข่ไก่ และงาดำ หยอดลงเตาพิมพ์ปิ้งไฟอ่อนให้สุกหอม แล้วรีบม้วนเป็นทรงกระบอกทันทีขณะยังร้อน",
            "en": "Prepared by grilling rich coconut-egg batter on patterned brass tongs, then immediately rolling into cylinders while warm and pliable."
        },
        "history": {
            "th": "ได้รับอิทธิพลจากขนมอบโปรตุเกสตั้งแต่สมัยอยุธยา คำว่า 'ทอง' สื่อถึงความร่ำรวยมั่งคั่ง ในภาคใต้นิยมทำแบบพับเรียกว่า 'ทองพับ'",
            "en": "Influenced by Portuguese rolled pastries during the 17th century. The word 'Thong' (Gold) conveys financial luck and prosperity."
        },
        "regional": {
            "th": {"central": "ทองม้วน", "north": "ทองม้วน / ขนมทองม้วน", "isan": "ทองม้วน", "south": "ทองม้วน / ขนมทองพับ"},
            "en": {"central": "Thong Muan", "north": "Thong Muan", "isan": "Thong Muan", "south": "Thong Muan / Thong Phap"}
        }
    }
}

CLASS_NAMES = [
    'chao_kuai', 'kaoniaomamuang', 'khanom_chan', 'khanomfakbua', 'khanomkrok',
    'kluaithot', 'lodchongnamkathi', 'pathongko', 'ruamnit', 'thongmuan'
]

# ========================================================
# 4. โมเดลที่มีอยู่จริงบน GitHub (3 โมเดลยอดเยี่ยม)
# ========================================================
MODELS_DIR = "models"

AVAILABLE_MODELS = {
    "EfficientNetB0": {
        "file": "kanomthai_efficientnetb0.keras",
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "name_th": "EfficientNetB0 (ความแม่นยำ 97.20% - แนะนำ)",
        "name_en": "EfficientNetB0 (Accuracy 97.20% - Recommended)"
    },
    "MobileNetV2": {
        "file": "kanomthai_mobilenetv2.keras",
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "name_th": "MobileNetV2 (ความแม่นยำ 96.27%)",
        "name_en": "MobileNetV2 (Accuracy 96.27%)"
    },
    "MobileNetV3Large": {
        "file": "kanomthai_mobilenetv3large.keras",
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
        "name_th": "MobileNetV3Large (ความแม่นยำ 96.13%)",
        "name_en": "MobileNetV3Large (Accuracy 96.13%)"
    }
}

@st.cache_resource
def load_prediction_model(model_filename):
    model_path = os.path.join(MODELS_DIR, model_filename)
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
# 5. พจนานุกรมคำแปล UI ครบ 100% (True Localization)
# ========================================================
UI_TEXT = {
    "th": {
        "title": "🍡 ระบบจำแนกประเภทและเผยแพร่วัฒนธรรมขนมไทย",
        "subtitle": "ปัญญาประดิษฐ์จำแนกรูปภาพขนมไทยและสารานุกรมเชิงวัฒนธรรม",
        "settings": "⚙️ การตั้งค่าระบบ",
        "model_select": "เลือกโมเดลทำนาย:",
        "about_title": "📊 รายละเอียดโมเดล",
        "about_desc": "- **จำนวนคลาส:** 10 คลาสขนมไทย\n- **ความละเอียดภาพ:** 224 x 224 พิกเซล\n- **โมเดลบน GitHub:** ผ่านการคัดเลือกเฉพาะโมเดลประสิทธิภาพสูง",
        "tab1": "📷 ถ่ายภาพและทำนาย",
        "tab2": "📜 ประวัติการใช้งาน",
        "tab3": "📚 สารานุกรมขนมไทย 10 เมนู",
        "input_header": "1. นำเข้าภาพขนมไทย",
        "mode_select": "เลือกวิธีนำเข้าภาพ:",
        "mode_cam": "ถ่ายภาพจากกล้อง",
        "mode_up": "อัปโหลดไฟล์ภาพ",
        "cam_label": "กดปุ่มด้านล่างเพื่อถ่ายภาพ",
        "up_label": "เลือกไฟล์รูปภาพ (JPG, PNG, WEBP):",
        "result_header": "2. ผลการวิเคราะห์",
        "waiting": "👈 กรุณาถ่ายภาพหรืออัปโหลดรูปภาพเพื่อเริ่มต้นการทำนาย",
        "pred_label": "ผลการทำนาย:",
        "confidence_label": "ความมั่นใจ",
        "warn_low_conf": "⚠️ คำเตือน: ความมั่นใจต่ำกว่า 60% ภาพอาจไม่ชัดเจน แสงไม่พอ หรืออาจไม่ใช่ขนมไทยใน 10 เมนูที่รองรับ",
        "tts_audio": "🔊 ฟังเสียงอ่านชื่อขนม:",
        "tts_btn_th": "สำเนียงไทย",
        "tts_btn_en": "สำเนียงอังกฤษ",
        "regional_title": "🗺️ ชื่อเรียกตามแต่ละภูมิภาคในไทย:",
        "reg_c": "ภาคกลาง:",
        "reg_n": "ภาคเหนือ:",
        "reg_i": "ภาคอีสาน:",
        "reg_s": "ภาคใต้:",
        "find_maps": "📍 ค้นหาร้านขายขนมนี้บน Google Maps",
        "detail_header": "📖 ข้อมูลรายละเอียดและวัฒนธรรม",
        "detail_simple": "คำอธิบายเบื้องต้น:",
        "detail_proc": "วัตถุดิบและวิธีทำ:",
        "detail_hist": "ประวัติความเป็นมา:",
        "top3_title": "🏆 อันดับความน่าจะเป็นสูงสุด 3 อันดับแรก:",
        "col_rank": "อันดับ",
        "col_name": "ชื่อขนม",
        "col_prob": "ความน่าจะเป็น",
        "hist_total": "จำนวนการทำนายทั้งหมด",
        "hist_avg_conf": "ความมั่นใจเฉลี่ย",
        "hist_clear": "🗑️ ล้างประวัติทั้งหมด",
        "hist_empty": "ยังไม่มีประวัติการทำนายในระบบ",
        "hist_export": "📥 ดาวน์โหลดประวัติ (CSV)",
        "hist_col_id": "ลำดับ",
        "hist_col_time": "วัน-เวลา",
        "hist_col_model": "โมเดล",
        "hist_col_name": "ชื่อที่ทำนายได้",
        "hist_col_conf": "ความมั่นใจ (%)"
    },
    "en": {
        "title": "🍡 Thai Dessert Cultural AI & Classification",
        "subtitle": "Deep Learning Thai Dessert Image Recognition & Cultural Heritage Encyclopedia",
        "settings": "⚙️ System Settings",
        "model_select": "Select Model Architecture:",
        "about_title": "📊 Model Details",
        "about_desc": "- **Classes:** 10 Thai Dessert Classes\n- **Input Resolution:** 224 x 224 pixels\n- **GitHub Deployment:** Filtered for high-performing, lightweight models",
        "tab1": "📷 Capture & Predict",
        "tab2": "📜 Prediction History",
        "tab3": "📚 Thai Dessert Encyclopedia",
        "input_header": "1. Provide Dessert Image",
        "mode_select": "Input Method:",
        "mode_cam": "Camera Capture",
        "mode_up": "Upload File",
        "cam_label": "Take a photo using your camera",
        "up_label": "Select dessert image (JPG, PNG, WEBP):",
        "result_header": "2. Analysis Result",
        "waiting": "👈 Please capture or upload an image to start classification.",
        "pred_label": "Predicted Dessert:",
        "confidence_label": "Confidence Score",
        "warn_low_conf": "⚠️ Warning: Confidence is below 60%. The image might be blurry, poorly lit, or not among the 10 supported desserts.",
        "tts_audio": "🔊 Pronunciation Audio:",
        "tts_btn_th": "Thai Accent",
        "tts_btn_en": "English Accent",
        "regional_title": "🗺️ Regional Names across Thailand:",
        "reg_c": "Central Thailand:",
        "reg_n": "Northern Thailand:",
        "reg_i": "Northeastern (Isan):",
        "reg_s": "Southern Thailand:",
        "find_maps": "📍 Find this dessert on Google Maps",
        "detail_header": "📖 Menu Details & Cultural Heritage",
        "detail_simple": "Quick Overview:",
        "detail_proc": "Ingredients & Method:",
        "detail_hist": "Cultural Heritage:",
        "top3_title": "🏆 Top-3 Probabilities:",
        "col_rank": "Rank",
        "col_name": "Dessert Name",
        "col_prob": "Probability",
        "hist_total": "Total Predictions",
        "hist_avg_conf": "Average Confidence",
        "hist_clear": "🗑️ Clear All History",
        "hist_empty": "No prediction history recorded yet.",
        "hist_export": "📥 Download History (CSV)",
        "hist_col_id": "ID",
        "hist_col_time": "Timestamp",
        "hist_col_model": "Model",
        "hist_col_name": "Predicted Dessert",
        "hist_col_conf": "Confidence (%)"
    }
}

# ========================================================
# 6. ส่วนติดต่อผู้ใช้ (Streamlit UI Layout)
# ========================================================
with st.sidebar:
    lang_choice = st.radio("🌐 ภาษา / Language", ["ภาษาไทย (TH)", "English (EN)"], horizontal=True)
    lang = "th" if "ไทย" in lang_choice else "en"
    t = UI_TEXT[lang]

    st.header(t["settings"])
    
    # ดึงชื่อโมเดลตามภาษาที่เลือก
    model_keys = list(AVAILABLE_MODELS.keys())
    model_labels = [AVAILABLE_MODELS[k]["name_th" if lang == "th" else "name_en"] for k in model_keys]
    
    selected_idx = st.selectbox(
        t["model_select"],
        range(len(model_keys)),
        format_func=lambda i: model_labels[i]
    )
    
    selected_model_key = model_keys[selected_idx]
    current_model_cfg = AVAILABLE_MODELS[selected_model_key]
    
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
                    selected_model_key,
                    pred_key,
                    info["name"]["th"],
                    info["name"]["en"],
                    confidence
                )

            # แสดงผลการทำนายหลักตามภาษาที่เลือก
            st.success(f"### {t['pred_label']} **{info['name'][lang]}**")
            st.metric(label=t["confidence_label"], value=f"{confidence:.2f}%")
            st.progress(min(confidence / 100.0, 1.0))

            # แจ้งเตือนเมื่อความมั่นใจต่ำ
            if confidence < 60.0:
                st.warning(t["warn_low_conf"])

            # ปุ่มค้นหาร้านบน Google Maps
            search_query = info["name"]["th"]
            maps_url = f"https://www.google.com/maps/search/{search_query}"
            st.link_button(t["find_maps"], maps_url, use_container_width=True)

            # เสียงอ่าน Text-to-Speech
            st.markdown(f"#### {t['tts_audio']}")
            c_audio1, c_audio2 = st.columns(2)
            with c_audio1:
                st.caption(t["tts_btn_th"])
                audio_th = generate_speech(info["name"]["th"], lang="th")
                st.audio(audio_th, format="audio/mp3")
            with c_audio2:
                st.caption(t["tts_btn_en"])
                audio_en = generate_speech(info["name"]["en"], lang="en")
                st.audio(audio_en, format="audio/mp3")

            # ชื่อเรียกแต่ละภูมิภาค (เปลี่ยนภาษาครบถ้วน)
            st.divider()
            st.markdown(f"#### {t['regional_title']}")
            reg_col1, reg_col2 = st.columns(2)
            with reg_col1:
                st.markdown(f"- **{t['reg_c']}** {info['regional'][lang]['central']}")
                st.markdown(f"- **{t['reg_n']}** {info['regional'][lang]['north']}")
            with reg_col2:
                st.markdown(f"- **{t['reg_i']}** {info['regional'][lang]['isan']}")
                st.markdown(f"- **{t['reg_s']}** {info['regional'][lang]['south']}")

            # ข้อมูลเชิงลึกและประวัติวัฒนธรรม (แสดงเฉพาะภาษาที่เลือก)
            st.divider()
            with st.expander(t["detail_header"], expanded=True):
                st.write(f"**{t['detail_simple']}** {info['simple'][lang]}")
                st.write(f"**{t['detail_proc']}** {info['detail'][lang]}")
                st.write(f"**{t['detail_hist']}** {info['history'][lang]}")

            # ตาราง Top-3
            st.divider()
            st.markdown(f"#### {t['top3_title']}")
            top_3_indices = np.argsort(preds)[::-1][:3]
            top_3_data = []
            for rank, idx in enumerate(top_3_indices, 1):
                key = CLASS_NAMES[idx]
                top_3_data.append({
                    t["col_rank"]: rank,
                    t["col_name"]: DESSERT_INFO[key]["name"][lang],
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
            
        display_name_col = "thai_name" if lang == "th" else "eng_name"
        view_df = history_df[["id", "timestamp", "model_name", display_name_col, "confidence"]].copy()
        view_df.columns = [t["hist_col_id"], t["hist_col_time"], t["hist_col_model"], t["hist_col_name"], t["hist_col_conf"]]
        st.dataframe(view_df, use_container_width=True)
        
        if st.button(t["hist_clear"], type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.rerun()
    else:
        st.info(t["hist_empty"])

# --------------------------------------------------------
# แท็บที่ 3: สารานุกรมขนมไทย 10 เมนู
# --------------------------------------------------------
with tab3:
    st.subheader(t["tab3"])
    for key, item in DESSERT_INFO.items():
        with st.expander(f"🍡 {item['name'][lang]}"):
            st.write(f"**{t['detail_simple']}** {item['simple'][lang]}")
            st.write(f"**{t['detail_proc']}** {item['detail'][lang]}")
            st.write(f"**{t['detail_hist']}** {item['history'][lang]}")
            st.markdown(f"**{t['regional_title']}**")
            st.write(f"- {t['reg_c']} {item['regional'][lang]['central']} | {t['reg_n']} {item['regional'][lang]['north']}")
            st.write(f"- {t['reg_i']} {item['regional'][lang]['isan']} | {t['reg_s']} {item['regional'][lang]['south']}")
