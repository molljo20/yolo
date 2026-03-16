import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import io
from supabase import create_client, Client
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
import cv2

# Seitenkonfiguration mit Dark Mode
st.set_page_config(
    page_title="Fundbüro - YOLO-Erkennung",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode CSS (gleich wie vorher)
st.markdown("""
<style>
    /* Dark Mode Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .admin-panel {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    .admin-header {
        color: #ff6b6b;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-buttons {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .stButton button {
        background-color: #2e2e2e;
        color: #fafafa;
        border: 1px solid #444;
    }
    
    .stButton button:hover {
        background-color: #3e3e3e;
        border-color: #666;
    }
    
    .delete-btn button {
        background-color: #ff4444;
        color: white;
    }
    
    .edit-btn button {
        background-color: #4444ff;
        color: white;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #2e2e2e;
        color: #fafafa;
        border-color: #444;
    }
</style>
""", unsafe_allow_html=True)

# Supabase-Konfiguration (deine Daten)
SUPABASE_URL = "https://imntylvenimvnmocbtzy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltbnR5bHZlbmltdm5tb2NidHp5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNTk4NzcsImV4cCI6MjA4ODYzNTg3N30.48pIBqUdlqXTooorJXHm71icVSj1wdTwW4tg5m2ovns"

# Admin Passwörter
DELETE_PASSWORD = "6767"
EDIT_PASSWORD = "timgioh"

# Email-Konfiguration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "ohtimgi@gmail.com"
SMTP_PASSWORD = "ftqz vujw skbl bblu"

# COCO Klassen (die 80 Objekte, die YOLO erkennen kann)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialisiere Supabase Client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Lade YOLO Modell
@st.cache_resource
def load_yolo_model():
    try:
        # Lade YOLOv8 Modell (klein, schnell, gut für Web)
        model = YOLO('yolov8n.pt')  # 'n' für nano, 's' für small, 'm' für medium
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des YOLO Modells: {e}")
        return None

# Bild mit YOLO analysieren
def detect_objects(image, model):
    try:
        # Bild in RGB konvertieren
        if isinstance(image, Image.Image):
            # Konvertiere PIL zu numpy array
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image
        
        # YOLO Vorhersage
        results = model(image_np)
        
        # Extrahiere erkannte Objekte
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Klassen-ID und Konfidenz
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Nur Objekte mit guter Konfidenz (> 0.5)
                    if confidence > 0.5:
                        class_name = COCO_CLASSES[class_id]
                        detected_objects.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        return detected_objects
    except Exception as e:
        st.error(f"Fehler bei der Objekterkennung: {e}")
        return []

# Email senden (gleich wie vorher)
def send_email(to_email, found_item, description, location, image_url):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        msg['Subject'] = f"🔍 Mögliches Fundstück gefunden: {found_item}"
        
        body = f"""
        <h2>Ein mögliches Fundstück wurde gemeldet!</h2>
        
        <p><strong>Gegenstand:</strong> {found_item}</p>
        <p><strong>Beschreibung:</strong> {description}</p>
        <p><strong>Fundort:</strong> {location}</p>
        <p><strong>Bild:</strong> <a href="{image_url}">Hier klicken zum Ansehen</a></p>
        
        <p>Bitte überprüfe, ob dies dein verlorener Gegenstand sein könnte.</p>
        
        <p>Viele Grüße,<br>Dein KI-Fundbüro</p>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Fehler beim Senden der Email: {e}")
        return False

# Prüfe auf Übereinstimmungen (angepasst für mehrere Objekte)
def check_for_matches(supabase, detected_objects, new_item_description, new_item_image_url):
    try:
        query = supabase.table("gesuchte_gegenstaende").select("*").execute()
        matches = []
        
        for searched in query.data:
            # Prüfe jedes erkannte Objekt
            for obj in detected_objects:
                if obj['class_name'].lower() in searched['class_name'].lower() or searched['class_name'].lower() in obj['class_name'].lower():
                    # Prüfe Schlüsselwörter
                    keywords = searched['description'].lower().split()
                    new_desc_lower = new_item_description.lower()
                    
                    match_score = 0
                    for keyword in keywords:
                        if len(keyword) > 3 and keyword in new_desc_lower:
                            match_score += 1
                    
                    if match_score > 0 or searched['class_name'].lower() == obj['class_name'].lower():
                        matches.append({
                            'email': searched['email'],
                            'item': obj['class_name'],
                            'description': searched['description'],
                            'match_score': match_score
                        })
                        break  # Einmal pro Suchanfrage
        
        return matches
    except Exception as e:
        st.error(f"Fehler beim Prüfen auf Übereinstimmungen: {e}")
        return []

# In Supabase speichern (angepasst für mehrere Objekte)
def save_to_supabase(supabase, image, detected_objects, description, location, finder_name):
    try:
        # Bild in Bytes konvertieren
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Für jedes erkannte Objekt einen Eintrag erstellen
        saved_items = []
        for obj in detected_objects:
            file_name = f"fundstuecke/{timestamp}_{obj['class_name']}.png"
            
            # Bild hochladen (gleiches Bild für alle Objekte)
            supabase.storage.from_("fundbuero-bilder").upload(
                file_name, 
                img_byte_arr
            )
            
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{file_name}"
            
            data = {
                "class_name": obj['class_name'],
                "class_id": obj['class_id'],
                "confidence_score": obj['confidence'],
                "description": description,
                "location": location,
                "finder_name": finder_name,
                "image_url": image_url,
                "created_at": datetime.datetime.now().isoformat(),
                "status": "gemeldet"
            }
            
            result = supabase.table("fundstuecke").insert(data).execute()
            saved_items.append(obj['class_name'])
            
            # Prüfe auf Übereinstimmungen für jedes Objekt
            matches = check_for_matches(supabase, [obj], description, image_url)
            
            for match in matches:
                send_email(
                    match['email'],
                    obj['class_name'],
                    description,
                    location,
                    image_url
                )
        
        return True, saved_items
    except Exception as e:
        st.error(f"Fehler beim Speichern in Supabase: {e}")
        return False, None

# Gesuchten Gegenstand speichern (unverändert)
def save_searched_item(supabase, class_name, description, email):
    try:
        data = {
            "class_name": class_name,
            "description": description,
            "email": email,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        result = supabase.table("gesuchte_gegenstaende").insert(data).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Speichern des gesuchten Gegenstands: {e}")
        return False, None

# Fundstück löschen (unverändert)
def delete_fundstueck(supabase, item_id):
    try:
        result = supabase.table("fundstuecke").delete().eq("id", item_id).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Löschen: {e}")
        return False, None

# Fundstück bearbeiten (unverändert)
def update_fundstueck(supabase, item_id, updated_data):
    try:
        result = supabase.table("fundstuecke").update(updated_data).eq("id", item_id).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Bearbeiten: {e}")
        return False, None

# Fundstücke abrufen (unverändert)
def get_fundstuecke(supabase, filter_class=None, search_term=None):
    try:
        query = supabase.table("fundstuecke").select("*").order("created_at", desc=True)
        
        if filter_class and filter_class != "Alle":
            query = query.eq("class_name", filter_class)
            
        if search_term:
            query = query.ilike("description", f"%{search_term}%")
            
        result = query.execute()
        return result.data
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Fundstücke: {e}")
        return []

# Gesuchte Gegenstände abrufen (unverändert)
def get_searched_items(supabase):
    try:
        result = supabase.table("gesuchte_gegenstaende").select("*").order("created_at", desc=True).execute()
        return result.data
    except Exception as e:
        st.error(f"Fehler beim Abrufen der gesuchten Gegenstände: {e}")
        return []

# Admin Panel (unverändert)
def show_admin_panel(supabase):
    with st.expander("👨‍🏫 Admin-Panel (Lehrer)", expanded=False):
        st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
        st.markdown('<div class="admin-header">🔐 Admin-Bereich</div>', unsafe_allow_html=True)
        
        admin_password = st.text_input("Admin-Passwort", type="password", key="admin_password")
        
        if admin_password == DELETE_PASSWORD:
            st.success("✅ Lösch-Modus aktiviert")
            st.session_state['admin_mode'] = 'delete'
        elif admin_password == EDIT_PASSWORD:
            st.success("✅ Bearbeiten-Modus aktiviert (inkl. Löschen)")
            st.session_state['admin_mode'] = 'edit'
        elif admin_password:
            st.error("❌ Falsches Passwort")
            st.session_state['admin_mode'] = None
        
        if 'admin_mode' in st.session_state and st.session_state['admin_mode']:
            st.info(f"Aktiver Modus: **{st.session_state['admin_mode']}**")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Haupt-App
def main():
    st.title("🔍 YOLO-Fundbüro")
    
    supabase = init_supabase()
    model = load_yolo_model()
    
    show_admin_panel(supabase)
    
    if model is None:
        st.error("Das YOLO-Modell konnte nicht geladen werden.")
        return
    
    # Verfügbare Klassen (alle COCO Klassen)
    available_classes = ["Alle"] + sorted(COCO_CLASSES)
    
    # Haupt-Buttons
    st.markdown('<div class="main-buttons">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 FUNDSTÜCK MELDEN", key="btn_melden", use_container_width=True):
            st.session_state['app_mode'] = 'melden'
    
    with col2:
        if st.button("🔎 NACH VERLORENEM SUCHEN", key="btn_suchen", use_container_width=True):
            st.session_state['app_mode'] = 'suchen'
    
    with col3:
        if st.button("📋 GESUCHTE GEGENSTÄNDE", key="btn_gesucht", use_container_width=True):
            st.session_state['app_mode'] = 'gesucht'
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'melden'
    
    if st.session_state['app_mode'] == 'melden':
        show_report_tab(supabase, model)
    elif st.session_state['app_mode'] == 'suchen':
        show_search_tab(supabase, available_classes)
    elif st.session_state['app_mode'] == 'gesucht':
        show_wanted_tab(supabase, COCO_CLASSES)

# Fundstück melden Tab (angepasst für YOLO)
def show_report_tab(supabase, model):
    st.header("📤 Neues Fundstück melden")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Wähle ein Bild des gefundenen Gegenstands aus",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    with col2:
        if uploaded_file is not None and st.button("🔍 Objekte erkennen", type="primary", use_container_width=True):
            with st.spinner("YOLO analysiert das Bild..."):
                image = Image.open(uploaded_file)
                detected_objects = detect_objects(image, model)
                
                if detected_objects:
                    st.session_state['detected_objects'] = detected_objects
                    st.session_state['detected_image'] = image
                    
                    st.success(f"✅ {len(detected_objects)} Objekt(e) erkannt:")
                    for obj in detected_objects:
                        st.info(f"• **{obj['class_name']}** (Konfidenz: {obj['confidence']:.2%})")
                else:
                    st.warning("Keine Objekte mit ausreichender Konfidenz erkannt.")
        
        if 'detected_objects' in st.session_state:
            with st.form("fund_form"):
                st.subheader("Details zum Fundstück")
                
                # Zeige erkannte Objekte
                st.markdown("**Erkannte Objekte:**")
                detected_text = ", ".join([obj['class_name'] for obj in st.session_state['detected_objects']])
                st.text(detected_text)
                
                description = st.text_area("Beschreibung", 
                                          placeholder="z.B. Farbe, Marke, besondere Merkmale...")
                
                location = st.text_input("Fundort", 
                                        placeholder="Wo wurde der Gegenstand gefunden?")
                
                finder_name = st.text_input("Name des Finders (optional)")
                
                submitted = st.form_submit_button("📦 Fundstück speichern", use_container_width=True)
                
                if submitted:
                    if description and location:
                        with st.spinner("Speichere in Datenbank..."):
                            success, saved_items = save_to_supabase(
                                supabase,
                                st.session_state['detected_image'],
                                st.session_state['detected_objects'],
                                description,
                                location,
                                finder_name or "Anonym"
                            )
                            
                            if success:
                                st.success(f"✅ {len(saved_items)} Fundstück/Einträge erfolgreich gespeichert!")
                                del st.session_state['detected_objects']
                                del st.session_state['detected_image']
                                st.rerun()
                    else:
                        st.warning("Bitte fülle alle Pflichtfelder aus (Beschreibung und Fundort).")

# Suche Tab (angepasst)
def show_search_tab(supabase, available_classes):
    st.header("🔎 Nach verlorenen Gegenständen suchen")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Suchbegriff", placeholder="z.B. 'rote Flasche'...")
    
    with col2:
        filter_class = st.selectbox("Kategorie filtern", available_classes)
    
    with col3:
        search_button = st.button("Suchen", type="primary", use_container_width=True)
    
    fundstuecke = get_fundstuecke(supabase, filter_class if filter_class != "Alle" else None, search_term)
    
    if fundstuecke:
        st.success(f"📊 {len(fundstuecke)} Fundstück(e) gefunden")
        
        cols = st.columns(3)
        for idx, fund in enumerate(fundstuecke):
            with cols[idx % 3]:
                with st.container(border=True):
                    if fund.get('image_url'):
                        st.image(fund['image_url'], use_column_width=True)
                    
                    st.markdown(f"### {fund['class_name']}")
                    st.markdown(f"**Beschreibung:** {fund['description']}")
                    st.markdown(f"**Fundort:** {fund['location']}")
                    st.markdown(f"**Gemeldet von:** {fund['finder_name']}")
                    st.markdown(f"**Datum:** {fund['created_at'][:10]}")
                    
                    confidence = fund.get('confidence_score', 0)
                    st.progress(confidence, text=f"KI-Konfidenz: {confidence:.1%}")
                    
                    status = fund.get('status', 'gemeldet')
                    if status == 'gemeldet':
                        st.caption("🟡 Noch nicht abgeholt")
                    else:
                        st.caption("✅ Bereits abgeholt")
                    
                    if 'admin_mode' in st.session_state:
                        col_del, col_edit = st.columns(2)
                        
                        with col_del:
                            if st.session_state['admin_mode'] in ['delete', 'edit']:
                                if st.button(f"🗑️ Löschen", key=f"del_{fund['id']}", use_container_width=True):
                                    success, _ = delete_fundstueck(supabase, fund['id'])
                                    if success:
                                        st.success("✅ Gelöscht!")
                                        st.rerun()
                        
                        with col_edit:
                            if st.session_state['admin_mode'] == 'edit':
                                if st.button(f"✏️ Bearbeiten", key=f"edit_{fund['id']}", use_container_width=True):
                                    st.session_state['editing_item'] = fund
                                    st.rerun()
                    
                    if 'editing_item' in st.session_state and st.session_state['editing_item']['id'] == fund['id']:
                        with st.form(key=f"edit_form_{fund['id']}"):
                            st.markdown("### ✏️ Eintrag bearbeiten")
                            
                            new_description = st.text_area("Beschreibung", value=fund['description'])
                            new_location = st.text_input("Fundort", value=fund['location'])
                            new_finder = st.text_input("Finder", value=fund['finder_name'])
                            new_status = st.selectbox("Status", ["gemeldet", "abgeholt"], 
                                                     index=0 if fund['status'] == 'gemeldet' else 1)
                            
                            col_save, col_cancel = st.columns(2)
                            
                            with col_save:
                                if st.form_submit_button("💾 Speichern", use_container_width=True):
                                    updated_data = {
                                        "description": new_description,
                                        "location": new_location,
                                        "finder_name": new_finder,
                                        "status": new_status
                                    }
                                    success, _ = update_fundstueck(supabase, fund['id'], updated_data)
                                    if success:
                                        st.success("✅ Aktualisiert!")
                                        del st.session_state['editing_item']
                                        st.rerun()
                            
                            with col_cancel:
                                if st.form_submit_button("❌ Abbrechen", use_container_width=True):
                                    del st.session_state['editing_item']
                                    st.rerun()
    else:
        st.info("😕 Keine Fundstücke gefunden.")

# Gesuchte Gegenstände Tab (angepasst)
def show_wanted_tab(supabase, class_names):
    st.header("📋 Gesuchte Gegenstände")
    
    st.markdown("""
    <div style="background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
        <h4>🔔 So funktioniert's:</h4>
        <p>Trage hier ein, wonach du suchst. YOLO erkennt über 80 verschiedene Objekte!
        Wenn jemand einen passenden Gegenstand findet, bekommst du automatisch eine Email.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'admin_mode' in st.session_state and st.session_state['admin_mode'] == 'edit':
        with st.form("wanted_form"):
            st.subheader("🔍 Neuen Gegenstand suchen")
            
            selected_class = st.selectbox("Kategorie", sorted(class_names))
            description = st.text_area("Beschreibung des gesuchten Gegenstands", 
                                      placeholder="z.B. 'Rote Trinkflasche mit Aufkleber'")
            email = st.text_input("Email-Adresse für Benachrichtigungen")
            
            submitted = st.form_submit_button("📌 Gegenstand suchen", use_container_width=True)
            
            if submitted:
                if description and email:
                    success, _ = save_searched_item(supabase, selected_class, description, email)
                    if success:
                        st.success("✅ Gesuchter Gegenstand wurde registriert!")
                        st.rerun()
                else:
                    st.warning("Bitte fülle alle Felder aus!")
    
    st.subheader("Aktuelle Suchanfragen")
    searched_items = get_searched_items(supabase)
    
    if searched_items:
        for item in searched_items:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{item['class_name']}**")
                    st.markdown(f"*{item['description']}*")
                    st.caption(f"Gesucht seit: {item['created_at'][:10]}")
                
                with col2:
                    if 'admin_mode' in st.session_state and st.session_state['admin_mode'] == 'edit':
                        st.markdown(f"📧 {item['email']}")
                        
                        if st.button(f"🗑️", key=f"del_wanted_{item['id']}"):
                            try:
                                supabase.table("gesuchte_gegenstaende").delete().eq("id", item['id']).execute()
                                st.rerun()
                            except:
                                pass
                    else:
                        st.markdown("🔒 Email nur für Admins sichtbar")
    else:
        st.info("📭 Noch keine gesuchten Gegenstände eingetragen.")

if __name__ == "__main__":
    main()
