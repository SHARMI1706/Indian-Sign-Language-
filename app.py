from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import base64
from collections import Counter

# Translation import
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    print("✅ Translation module loaded successfully")
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ Translation module not available. Install with: pip install deep-translator")

app = Flask(__name__)
CORS(app)

# ---------------------------------
# LOAD TRAINED SVM MODEL
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_digit_svm_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ SVM model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("⚠️ Using rule-based algorithm only")
    model = None

# ---------------------------------
# LOAD REVERSE MODEL (TEXT TO ISL)
# ---------------------------------
REVERSE_MODEL_PATH = os.path.join(BASE_DIR, "isl_number_reverse_model.pkl")

try:
    reverse_model = joblib.load(REVERSE_MODEL_PATH)
    print("✅ Reverse model (Text to ISL) loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading reverse model: {e}")
    reverse_model = None

# ---------------------------------
# LANGUAGE MAPPING (ONLY 5 LANGUAGES)
# ---------------------------------
LANGUAGE_CODES = {
    'english': 'en',
    'tamil': 'ta',
    'hindi': 'hi',
    'malayalam': 'ml',
    'telugu': 'te'
}

# ---------------------------------
# PURE TRANSLATION FUNCTION - HANDLES ANY TEXT
# ---------------------------------
def translate_text(text, source_lang='en', target_lang='en'):
    """
    Translate any text using deep-translator
    Converts digit numbers to words before translation
    """
    if not TRANSLATOR_AVAILABLE:
        return text  # Return original if translator not available
    
    try:
        # Convert single digits to English words first
        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # If text is a single digit, convert to word
        if text.strip() in digit_words:
            text = digit_words[text.strip()]
        
        # If same language, return as-is
        if source_lang == target_lang:
            return text
        
        # Get language codes
        source_code = LANGUAGE_CODES.get(source_lang.lower(), source_lang)
        target_code = LANGUAGE_CODES.get(target_lang.lower(), target_lang)
        
        # If target is English, return as-is
        if target_code == 'en':
            return text
        
        # Translate using GoogleTranslator
        translator = GoogleTranslator(source=source_code, target=target_code)
        translated = translator.translate(text)
        
        return translated if translated else text
        
    except Exception as e:
        print(f"Translation error for '{text}' to {target_lang}: {e}")
        return text  # Return original on error

# ---------------------------------
# MEDIAPIPE SETUP
# ---------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Prediction history for stability
prediction_buffer = []

# ---------------------------------
# ENHANCED FINGER STATE DETECTION
# ---------------------------------
def get_finger_states(lm):
    """
    Enhanced finger state detection.
    Returns: [thumb, index, middle, ring, pinky]
    1 = extended, 0 = folded
    """
    fingers = []
    
    # Calculate palm center
    palm_center = np.array([lm[9].x, lm[9].y])
    
    # Get fingertip distances from palm
    thumb_dist = np.linalg.norm(np.array([lm[4].x, lm[4].y]) - palm_center)
    index_dist = np.linalg.norm(np.array([lm[8].x, lm[8].y]) - palm_center)
    middle_dist = np.linalg.norm(np.array([lm[12].x, lm[12].y]) - palm_center)
    ring_dist = np.linalg.norm(np.array([lm[16].x, lm[16].y]) - palm_center)
    pinky_dist = np.linalg.norm(np.array([lm[20].x, lm[20].y]) - palm_center)
    
    # THUMB: x-distance check
    thumb_extended = abs(lm[4].x - lm[2].x) > 0.04
    fingers.append(1 if thumb_extended else 0)
    
    # INDEX: tip above knuckle + distance check
    index_up = (lm[8].y < lm[6].y - 0.03) and (index_dist > 0.12)
    fingers.append(1 if index_up else 0)
    
    # MIDDLE: tip above knuckle + distance check
    middle_up = (lm[12].y < lm[10].y - 0.03) and (middle_dist > 0.12)
    fingers.append(1 if middle_up else 0)
    
    # RING: tip above knuckle + distance check
    ring_up = (lm[16].y < lm[14].y - 0.03) and (ring_dist > 0.10)
    fingers.append(1 if ring_up else 0)
    
    # PINKY: tip above knuckle + distance check
    pinky_up = (lm[20].y < lm[18].y - 0.03) and (pinky_dist > 0.10)
    fingers.append(1 if pinky_up else 0)
    
    return fingers

# ---------------------------------
# ISL DIGIT RECOGNITION
# ---------------------------------
def recognize_isl_digit(lm, states, features):
    """
    Recognize ISL digits 0-9 using hybrid approach.
    Based on standard ISL gestures.
    """
    open_count = sum(states)
    
    # Get fingertip distances
    wrist = np.array([lm[0].x, lm[0].y])
    distances = {
        'thumb': np.linalg.norm(np.array([lm[4].x, lm[4].y]) - wrist),
        'index': np.linalg.norm(np.array([lm[8].x, lm[8].y]) - wrist),
        'middle': np.linalg.norm(np.array([lm[12].x, lm[12].y]) - wrist),
        'ring': np.linalg.norm(np.array([lm[16].x, lm[16].y]) - wrist),
        'pinky': np.linalg.norm(np.array([lm[20].x, lm[20].y]) - wrist)
    }
    
    # DIGIT 0: Closed fist
    if open_count == 0:
        return 0, "Rule-Based", 0.98
    
    # DIGIT 1: Index only
    if open_count == 1 and states[1] == 1 and distances['index'] > 0.25:
        return 1, "Rule-Based", 0.96
    
    # DIGIT 2: Index + Middle (V sign)
    if open_count == 2 and states[1] == 1 and states[2] == 1:
        if distances['index'] > 0.25 and distances['middle'] > 0.25:
            return 2, "Rule-Based", 0.95
    
    # DIGIT 3: Index + Middle + Ring
    if open_count == 3 and states[1] == 1 and states[2] == 1 and states[3] == 1:
        if states[0] == 0 and states[4] == 0:
            return 3, "Rule-Based", 0.94
    
    # DIGIT 4: Four fingers (no thumb)
    if open_count == 4 and states[0] == 0:
        if states[1] == 1 and states[2] == 1 and states[3] == 1 and states[4] == 1:
            return 4, "Rule-Based", 0.93
    
    # DIGIT 5: All open
    if open_count == 5 and all(d > 0.20 for d in distances.values()):
        return 5, "Rule-Based", 0.97
    
    # DIGIT 6: Thumb + Pinky (Shaka)
    if states[0] == 1 and states[4] == 1:
        if states[1] == 0 and states[2] == 0 and states[3] == 0:
            if distances['thumb'] > 0.15 and distances['pinky'] > 0.20:
                return 6, "Rule-Based", 0.94
    
    # Alternative: Just pinky
    if open_count == 1 and states[4] == 1 and distances['pinky'] > 0.20:
        return 6, "Rule-Based", 0.90
    
    # DIGIT 7: Middle + Ring + Pinky
    if open_count == 3 and states[2] == 1 and states[3] == 1 and states[4] == 1:
        if states[0] == 0 and states[1] == 0:
            return 7, "Rule-Based", 0.92
    
    # DIGIT 8: Index + Middle (parallel orientation)
    if open_count == 2 and states[1] == 1 and states[2] == 1:
        index_pos = np.array([lm[8].x, lm[8].y])
        middle_pos = np.array([lm[12].x, lm[12].y])
        wrist_pos = np.array([lm[0].x, lm[0].y])
        
        to_index = index_pos - wrist_pos
        to_middle = middle_pos - wrist_pos
        
        finger_angle = np.arccos(np.dot(to_index, to_middle) / 
                                 (np.linalg.norm(to_index) * np.linalg.norm(to_middle) + 1e-6))
        
        if finger_angle < 0.3:  # Parallel
            return 8, "Rule-Based", 0.88
    
    # DIGIT 9: All except pinky
    if open_count == 4:
        if states[0] == 1 and states[1] == 1 and states[2] == 1 and states[3] == 1:
            if states[4] == 0 and distances['pinky'] < 0.15:
                return 9, "Rule-Based", 0.95
    
    # Fallback to SVM
    if model is not None:
        pred = model.predict(features)[0]
        return int(pred), "SVM Model", 0.82
    
    return None, "Unknown", 0.0

# ---------------------------------
# ROUTES
# ---------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict ISL digit from webcam frame with translation support
    """
    global prediction_buffer
    
    try:
        if not model:
            print("⚠️ Warning: Model not loaded, using rules only")
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get target language from request (default to English)
        target_language = data.get('language', 'english')
        
        # Decode image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        response = {
            'detected': False,
            'digit': None,
            'confidence': 0,
            'method': None
        }
        
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            
            # Get finger states
            states = get_finger_states(lm)
            open_count = sum(states)
            
            # Prepare SVM features
            base_x, base_y, base_z = lm[0].x, lm[0].y, lm[0].z
            features = []
            for p in lm:
                features.extend([p.x - base_x, p.y - base_y, p.z - base_z])
            features = np.array(features).reshape(1, -1)
            
            # Recognize digit
            pred, method, conf = recognize_isl_digit(lm, states, features)
            
            if pred is not None:
                # Add to buffer for stability
                prediction_buffer.append(pred)
                if len(prediction_buffer) > 5:
                    prediction_buffer.pop(0)
                
                # Use most common prediction
                if len(prediction_buffer) >= 3:
                    stable_pred = Counter(prediction_buffer).most_common(1)[0][0]
                    pred = stable_pred
                
                # Get the digit as text and translate it
                digit_text = str(pred)
                translated_word = translate_text(digit_text, 'en', target_language)
                
                # Get landmarks
                landmarks_list = []
                for landmark in lm:
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                response = {
                    'detected': True,
                    'digit': int(pred),
                    'digit_text': digit_text,
                    'translated_word': translated_word,
                    'language': target_language,
                    'confidence': conf,
                    'method': method,
                    'landmarks': landmarks_list,
                    'finger_states': states,
                    'open_fingers': open_count
                }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """
    Translate ANY text to target language using deep-translator
    """
    try:
        data = request.get_json()
        
        if 'text' not in data or 'target_language' not in data:
            return jsonify({'error': 'Missing text or target_language'}), 400
        
        text = data['text']
        source_language = data.get('source_language', 'english')
        target_language = data['target_language']
        
        # Translate the entire text
        translated = translate_text(text, source_language, target_language)
        
        return jsonify({
            'success': True,
            'original': text,
            'translated': translated,
            'source_language': source_language,
            'target_language': target_language,
            'translator_available': TRANSLATOR_AVAILABLE
        })
        
    except Exception as e:
        print(f"Error in translation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-isl', methods=['POST'])
def text_to_isl():
    """
    Convert text to ISL sign images with translation
    Works with ANY input text, translates it to target language
    """
    try:
        if reverse_model is None:
            return jsonify({'error': 'Reverse model not loaded'}), 500
        
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        target_language = data.get('language', 'english')
        
        # Translate the entire input text first
        translated_text = translate_text(text, 'english', target_language)
        
        # Process each character in the text
        results = []
        for i, char in enumerate(text):
            if char.isdigit():
                img_path = reverse_model.get(char)
                
                # Get individual character translation
                char_translation = translate_text(char, 'english', target_language)
                
                if img_path and os.path.exists(img_path):
                    # Read image and convert to base64
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    results.append({
                        'digit': char,
                        'translated_word': char_translation,
                        'language': target_language,
                        'found': True,
                        'image': f'data:image/png;base64,{img_data}'
                    })
                else:
                    results.append({
                        'digit': char,
                        'translated_word': char_translation,
                        'language': target_language,
                        'found': False,
                        'image': None
                    })
            else:
                # For non-digit characters, just translate
                char_translation = translate_text(char, 'english', target_language)
                results.append({
                    'digit': char,
                    'translated_word': char_translation,
                    'found': False,
                    'image': None,
                    'message': 'Only digits 0-9 have ISL signs'
                })
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translated_text': translated_text,
            'results': results,
            'translator_available': TRANSLATOR_AVAILABLE
        })
    
    except Exception as e:
        print(f"Error in text-to-isl: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """
    Get available languages (only 5 languages)
    """
    languages = [
        {'code': 'english', 'name': 'English', 'native': 'English'},
        {'code': 'tamil', 'name': 'Tamil', 'native': 'தமிழ்'},
        {'code': 'hindi', 'name': 'Hindi', 'native': 'हिन्दी'},
        {'code': 'malayalam', 'name': 'Malayalam', 'native': 'മലയാളം'},
        {'code': 'telugu', 'name': 'Telugu', 'native': 'తెలుగు'}
    ]
    
    return jsonify({
        'success': True,
        'languages': languages,
        'translator_available': TRANSLATOR_AVAILABLE
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'reverse_model_loaded': reverse_model is not None,
        'translator_available': TRANSLATOR_AVAILABLE,
        'algorithm': 'Enhanced Hybrid (Rule-Based + SVM)',
        'digits_supported': '0-9',
        'languages_supported': 5,
        'translation_mode': 'Pure deep-translator (any text)'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ISL TRANSLATOR - FLEXIBLE TRANSLATION MODE")
    print("="*60)
    print("✅ Algorithm: Enhanced Hybrid (Rule-Based + SVM)")
    print("✅ Optimized for ISL digits 0-9")
    if TRANSLATOR_AVAILABLE:
        print("✅ Pure deep-translator for ANY text input")
        print("   Languages: English, Tamil, Hindi, Malayalam, Telugu")
        print("   Input: Accepts any text (not limited to digits)")
    else:
        print("⚠️  Translation module not installed (pip install deep-translator)")
    print("✅ Server running at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)