import cv2
import mediapipe as mp
import os
from pymol import cmd
import pymol
import time
from molecule_analyzer import analyze_molecule
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import sounddevice as sd
import threading
from pymol_renderer import highlight_chiral_centers

# --- Clap-to-exit setup ---
clap_exit = False
def listen_for_clap(threshold=1.0, duration=0.1):
    global clap_exit
    def audio_callback(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold:
            clap_exit = True
    with sd.InputStream(callback=audio_callback):
        while not clap_exit:
            sd.sleep(int(duration * 1000))
threading.Thread(target=listen_for_clap, daemon=True).start()

# --- PyMOL setup ---
pymol.finish_launching(['pymol', '-A1'])

# --- Molecule setup ---
molecule_dir = 'molecules'
molecule_files = [f for f in os.listdir(molecule_dir) if f.endswith('.mol') or f.endswith('.pdb')]
VISIBLE_WINDOW_SIZE = 6
visible_start_index = 0

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
selected_index = -1
molecule_loaded = False
prev_x, prev_y = None, None
zoom_mode = None
zoom_last_time = time.time()
last_peace_scroll_time = 0
peace_scroll_triggered = False
reset_triggered = False  # ✅ 
mode = 'rotate'
molecule_info = None

# --- Load emoji-capable font ---
try:
    font = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", 22)
except IOError:
    font = ImageFont.load_default()

# --- RDKit image generator ---
def generate_molecule_thumb(file_path, size=(40, 40)):
    try:
        mol = Chem.MolFromMolFile(file_path)
        if not mol:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img.convert("RGB")
    except:
        return None

# --- RDKit info popup window renderer ---
def render_rdkit_info_image(info, width=400, height=200):
    img = Image.new("RGB", (width, len(info) * 40 + 20), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("C:\\Windows\\Fonts\\seguiemj.ttf", 20)
    except:
        fnt = ImageFont.load_default()
    y = 10
    for k, v in info.items():
        label = f"{k}:"
        val = f"{v}" if not isinstance(v, list) else f"{len(v)} center(s)"
        draw.text((10, y), label, font=fnt, fill=(144, 238, 144))
        draw.text((180, y), val, font=fnt, fill=(255, 255, 255))
        y += 30
    return np.array(img)

# --- peace detection ---
def is_peace(landmarks):
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]
    extended = 0
    for tip, pip in fingers:
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            extended += 1
    return extended == 2
# --- Fist detection ---

def is_fist(landmarks):
    extended = 0
    for tip, pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]:
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            extended += 1
    return extended == 0

def detect_vertical_swipe(prev_y, curr_y):
    if prev_y is None:
        return None
    if abs(curr_y - prev_y) > 40:
        return 'up' if curr_y < prev_y else 'down'
    return None

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks
        hand1 = hands_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand1, mp_hands.HAND_CONNECTIONS)

        x = int(hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
        y = int(hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        if len(hands_landmarks) == 2:
            if is_fist(hands_landmarks[0]) and is_fist(hands_landmarks[1]):
                if not reset_triggered:
                    cmd.reinitialize()
                    molecule_loaded = False
                    selected_index = -1
                    molecule_info = None
                    visible_start_index = 0  # 👈 Jump to first molecule group
                    reset_triggered = True
                    # 👇 Close RDKit info popup if open
                    try: cv2.destroyWindow("Molecule Info")
                    except: pass
                    
                    print("🔁 Double Fist: Reset view + list to top.")
                    mode = 'idle'
                    continue
        else:
            reset_triggered = False # reset flag when fist released

        swipe = detect_vertical_swipe(prev_y, y)
        peace_y = hand1.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

        if not molecule_loaded:
            if is_peace(hand1):
                dy = peace_y - prev_y if prev_y else 0
                jump_size = 20
                now = time.time()

                if not peace_scroll_triggered and dy < -0.08:
                    visible_start_index = min(
                        len(molecule_files) - VISIBLE_WINDOW_SIZE,
                        visible_start_index + jump_size
                    )
                    peace_scroll_triggered = True
                    print("✌️ Peace UP — Jumped forward 20")

                prev_y = peace_y
            else:
                peace_scroll_triggered = False  # Reset when peace sign is broken
                prev_y = None


        

        if not molecule_loaded:
            if swipe == 'up' and visible_start_index > 0:
                visible_start_index -= 1
            elif swipe == 'down' and visible_start_index + VISIBLE_WINDOW_SIZE < len(molecule_files):
                visible_start_index += 1

            for idx in range(len(molecule_files[visible_start_index:visible_start_index + VISIBLE_WINDOW_SIZE])):
                box_y = 60 + idx * 55
                if 10 < x < 310 and box_y < y < box_y + 50:
                    selected_index = visible_start_index + idx

        thumb_tip = hand1.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

        if pinch_distance < 0.03 and selected_index != -1 and not molecule_loaded:
            file_path = os.path.join(molecule_dir, molecule_files[selected_index])
            cmd.reinitialize()
            cmd.load(file_path, 'mol')
            cmd.show("sticks", "mol")
            cmd.show("spheres", "mol")
            cmd.set("sphere_scale", 0.25, "mol")  # 🔧 adjust sphere size for better view
            cmd.set("stick_radius", 0.2, "mol")
            cmd.set("stick_color", "gray80")
            cmd.zoom("mol")
            highlight_chiral_centers(file_path)
            molecule_loaded = True
            molecule_info = analyze_molecule(file_path)
            zoom_mode = None

            # 👇 Show popup RDKit info
            info_img = render_rdkit_info_image(molecule_info)
            cv2.namedWindow("Molecule Info", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Molecule Info", info_img)

        if len(hands_landmarks) == 2:
            mode = 'zoom'
        elif len(hands_landmarks) == 1:
            if is_fist(hand1) and molecule_loaded:
                cmd.reset()
                zoom_mode = None
                mode = 'idle'
            else:
                mode = 'rotate'

        if molecule_loaded:
            if mode == 'rotate':
                current_x = x
                if prev_x is not None:
                    dx = current_x - prev_x
                    if abs(dx) > 5:
                        cmd.rotate('y', dx / 3, 'mol')
                prev_x = current_x

            elif mode == 'zoom' and len(hands_landmarks) == 2:
                hand2 = hands_landmarks[1]
                hand1_x = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                hand2_x = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                distance = abs(hand1_x - hand2_x)

                if zoom_mode is None:
                    zoom_mode = distance
                else:
                    dz = distance - zoom_mode
                    now = time.time()
                    if abs(dz) > 0.03 and (now - zoom_last_time) > 0.3:
                        scale = 1 + abs(dz) * 16
                        cmd.zoom("mol", scale if dz > 0 else 1 / scale)
                        zoom_mode = distance
                        zoom_last_time = now
    else:
        prev_x = None
        zoom_mode = None

    # --- Draw UI ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if int(time.time() * 2) % 2 == 0:
        draw.text((10, 10), f"🧭 Mode: {mode.upper()}", font=font, fill=(255, 255, 0))

    visible_molecules = molecule_files[visible_start_index:visible_start_index + VISIBLE_WINDOW_SIZE]
    for idx, name in enumerate(visible_molecules):
        y = 60 + idx * 55
        actual_index = visible_start_index + idx
        is_selected = (actual_index == selected_index)

        box_rect = (10, y, 330, y + 50)
        shadow_rect = (12, y + 2, 332, y + 52)
        bg_color = (245, 245, 245) if not is_selected else (153, 255, 153)
        draw.rectangle(shadow_rect, fill=(180, 180, 180))
        draw.rectangle(box_rect, fill=bg_color, outline=(0, 0, 0), width=2)

        file_path = os.path.join(molecule_dir, name)
        thumb_img = generate_molecule_thumb(file_path)
        if thumb_img:
            pil_img.paste(thumb_img.resize((40, 40)), (20, y + 5))
        draw.text((70, y + 12), name, font=font, fill=(0, 0, 0))

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Gesture Molecule Viewer", frame)

    if clap_exit:
        print("Clap detected — exiting...")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
