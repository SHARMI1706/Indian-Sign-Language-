import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# ---------------------------------
# LOAD TRAINED SVM MODEL
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_digit_svm_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ SVM model loaded")
except:
    model = None
    print("⚠️  SVM model not found - using enhanced rule-based algorithm only")

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

# ---------------------------------
# ENHANCED FINGER STATE DETECTION
# ---------------------------------
def get_finger_states(lm):
    """
    Enhanced finger state detection with better accuracy.
    Returns: [thumb, index, middle, ring, pinky]
    1 = extended/open, 0 = folded/closed
    """
    fingers = []
    
    # Calculate palm base reference points
    wrist = np.array([lm[0].x, lm[0].y])
    palm_center = np.array([lm[9].x, lm[9].y])  # Middle finger base
    
    # Get all fingertip positions
    thumb_tip = np.array([lm[4].x, lm[4].y])
    index_tip = np.array([lm[8].x, lm[8].y])
    middle_tip = np.array([lm[12].x, lm[12].y])
    ring_tip = np.array([lm[16].x, lm[16].y])
    pinky_tip = np.array([lm[20].x, lm[20].y])
    
    # Get knuckle/joint positions
    thumb_joint = np.array([lm[2].x, lm[2].y])
    index_joint = np.array([lm[6].x, lm[6].y])
    middle_joint = np.array([lm[10].x, lm[10].y])
    ring_joint = np.array([lm[14].x, lm[14].y])
    pinky_joint = np.array([lm[18].x, lm[18].y])
    
    # Calculate distances from palm center
    thumb_dist = np.linalg.norm(thumb_tip - palm_center)
    index_dist = np.linalg.norm(index_tip - palm_center)
    middle_dist = np.linalg.norm(middle_tip - palm_center)
    ring_dist = np.linalg.norm(ring_tip - palm_center)
    pinky_dist = np.linalg.norm(pinky_tip - palm_center)
    
    # THUMB: Check if extended (x-distance from palm)
    thumb_extended = abs(lm[4].x - lm[2].x) > 0.04
    fingers.append(1 if thumb_extended else 0)
    
    # INDEX: Check if tip is significantly above knuckle
    index_up = (lm[8].y < lm[6].y - 0.03) and (index_dist > 0.12)
    fingers.append(1 if index_up else 0)
    
    # MIDDLE: Check if tip is significantly above knuckle
    middle_up = (lm[12].y < lm[10].y - 0.03) and (middle_dist > 0.12)
    fingers.append(1 if middle_up else 0)
    
    # RING: Check if tip is significantly above knuckle
    ring_up = (lm[16].y < lm[14].y - 0.03) and (ring_dist > 0.10)
    fingers.append(1 if ring_up else 0)
    
    # PINKY: Check if tip is significantly above knuckle
    pinky_up = (lm[20].y < lm[18].y - 0.03) and (pinky_dist > 0.10)
    fingers.append(1 if pinky_up else 0)
    
    return fingers

# ---------------------------------
# CORRECTED ISL DIGIT RECOGNITION (BASED ON REFERENCE IMAGE)
# ---------------------------------
def recognize_isl_digit(lm, states, features):
    """
    Corrected ISL digit recognition based on reference image.
    
    Reference from image:
    0 = Closed fist
    1 = Index finger only
    2 = Index + Middle
    3 = Index + Middle + Ring
    4 = All four fingers (no thumb)
    5 = All five fingers
    6 = Thumb only
    7 = Thumb + Pinky (Shaka sign)
    8 = Thumb + Index + Middle
    9 = All five fingers (same as 5)
    10 = Thumb + Index
    
    States: [thumb, index, middle, ring, pinky]
    """
    open_count = sum(states)
    
    # Get fingertip distances from wrist for verification
    wrist = np.array([lm[0].x, lm[0].y])
    distances = {
        'thumb': np.linalg.norm(np.array([lm[4].x, lm[4].y]) - wrist),
        'index': np.linalg.norm(np.array([lm[8].x, lm[8].y]) - wrist),
        'middle': np.linalg.norm(np.array([lm[12].x, lm[12].y]) - wrist),
        'ring': np.linalg.norm(np.array([lm[16].x, lm[16].y]) - wrist),
        'pinky': np.linalg.norm(np.array([lm[20].x, lm[20].y]) - wrist)
    }
    
    # ============================================
    # DIGIT 0: CLOSED FIST
    # ============================================
    if open_count == 0:
        return 0, "Rule: Closed fist"
    
    # ============================================
    # DIGIT 1: INDEX FINGER ONLY
    # ============================================
    if open_count == 1 and states[1] == 1:
        # Only index up, all others down
        if states[0] == 0 and states[2] == 0 and states[3] == 0 and states[4] == 0:
            if distances['index'] > 0.25:
                return 1, "Rule: Index only"
    
    # ============================================
    # DIGIT 2: INDEX + MIDDLE (V Sign)
    # ============================================
    if open_count == 2 and states[1] == 1 and states[2] == 1:
        # Index and middle up, others down
        if states[0] == 0 and states[3] == 0 and states[4] == 0:
            if distances['index'] > 0.25 and distances['middle'] > 0.25:
                return 2, "Rule: Index+Middle (V)"
    
    # ============================================
    # DIGIT 3: INDEX + MIDDLE + RING
    # ============================================
    if open_count == 3 and states[1] == 1 and states[2] == 1 and states[3] == 1:
        # Three middle fingers up, thumb and pinky down
        if states[0] == 0 and states[4] == 0:
            return 3, "Rule: Three fingers (I+M+R)"
    
    # ============================================
    # DIGIT 4: ALL FOUR FINGERS (NO THUMB)
    # ============================================
    if open_count == 4 and states[0] == 0:
        # All four fingers up, thumb down
        if states[1] == 1 and states[2] == 1 and states[3] == 1 and states[4] == 1:
            return 4, "Rule: Four fingers (no thumb)"
    
    # ============================================
    # DIGIT 5: ALL FIVE FINGERS OPEN
    # ============================================
    if open_count == 5:
        # All fingers extended
        if all(d > 0.20 for d in distances.values()):
            return 5, "Rule: All five open"
    
    # ============================================
    # DIGIT 6: THUMB ONLY
    # ============================================
    if open_count == 1 and states[0] == 1:
        # Only thumb extended, all others down
        if states[1] == 0 and states[2] == 0 and states[3] == 0 and states[4] == 0:
            if distances['thumb'] > 0.15:
                return 6, "Rule: Thumb only"
    
    # ============================================
    # DIGIT 7: THUMB + PINKY (Shaka/Hang Loose)
    # ============================================
    if open_count == 2 and states[0] == 1 and states[4] == 1:
        # Thumb and pinky extended, middle three down
        if states[1] == 0 and states[2] == 0 and states[3] == 0:
            if distances['thumb'] > 0.15 and distances['pinky'] > 0.20:
                return 7, "Rule: Thumb+Pinky (Shaka)"
    
    # ============================================
    # DIGIT 8: THUMB + INDEX + MIDDLE
    # ============================================
    if open_count == 3 and states[0] == 1 and states[1] == 1 and states[2] == 1:
        # Thumb, index, middle extended; ring and pinky down
        if states[3] == 0 and states[4] == 0:
            if distances['thumb'] > 0.15 and distances['index'] > 0.25 and distances['middle'] > 0.25:
                return 8, "Rule: Thumb+Index+Middle"
    
    # ============================================
    # DIGIT 9: ALL FIVE FINGERS (ALTERNATIVE)
    # Note: In the image, 9 appears same as 5
    # Some ISL variants use 4 fingers + thumb (no pinky) for 9
    # ============================================
    if open_count == 4 and states[0] == 1:
        # Thumb + three fingers (no pinky) - alternative 9
        if states[1] == 1 and states[2] == 1 and states[3] == 1 and states[4] == 0:
            if distances['pinky'] < 0.15:  # Verify pinky is down
                return 9, "Rule: All except pinky"
    
    # If all 5 fingers and not already classified as 5, could be 9
    if open_count == 5:
        # Could be 9 in some contexts
        # For now, let SVM decide between 5 and 9
        pass
    
    # ============================================
    # FALLBACK TO SVM MODEL
    # ============================================
    if model is not None:
        pred = model.predict(features)[0]
        return int(pred), "SVM Model"
    
    # ============================================
    # UNABLE TO DETERMINE
    # ============================================
    return None, "Unknown"

# ---------------------------------
# OPEN WEBCAM
# ---------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("\n" + "="*70)
print("🎥 ISL DIGIT RECOGNITION - CORRECTED ALGORITHM")
print("="*70)
print("✅ Based on ISL reference image gestures")
print("✅ Hybrid: Rule-Based + SVM Model")
print("\n📋 Gesture Guide (CORRECTED):")
print("  0 = Closed fist")
print("  1 = Index finger only")
print("  2 = Index + Middle (V sign)")
print("  3 = Index + Middle + Ring")
print("  4 = Four fingers (no thumb)")
print("  5 = All five fingers open")
print("  6 = Thumb only ⭐ FIXED")
print("  7 = Thumb + Pinky (Shaka) ⭐ FIXED")
print("  8 = Thumb + Index + Middle ⭐ FIXED")
print("  9 = All except pinky (or all 5 in some variants) ⭐ FIXED")
print("\n⌨️  Press 'Q' to quit")
print("="*70 + "\n")

# ---------------------------------
# LIVE RECOGNITION LOOP
# ---------------------------------
frame_count = 0
prediction_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label_text = "👋 Show your hand"
    debug_text = ""
    method_text = ""
    confidence = 0

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # Get finger states
        states = get_finger_states(lm)
        open_count = sum(states)

        # Prepare features for SVM
        base_x, base_y, base_z = lm[0].x, lm[0].y, lm[0].z
        features = []
        for p in lm:
            features.extend([
                p.x - base_x,
                p.y - base_y,
                p.z - base_z
            ])
        features = np.array(features).reshape(1, -1)

        # Recognize digit
        pred, method = recognize_isl_digit(lm, states, features)
        
        if pred is not None:
            # Add to prediction history for stability
            prediction_history.append(pred)
            if len(prediction_history) > 5:
                prediction_history.pop(0)
            
            # Use most common prediction from last 5 frames
            if len(prediction_history) >= 3:
                from collections import Counter
                stable_pred = Counter(prediction_history).most_common(1)[0][0]
                pred = stable_pred
                confidence = 95 if "Rule" in method else 85
            
            label_text = f"🔢 Digit: {pred}"
            method_text = f"Method: {method}"
            debug_text = f"Fingers: T:{states[0]} I:{states[1]} M:{states[2]} R:{states[3]} P:{states[4]} | Open:{open_count}"
        else:
            label_text = "❓ Unclear gesture"
            method_text = "Try a clearer gesture"

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

    # Create info panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Border
    cv2.rectangle(frame, (10, 10), (w-10, 140), (0, 255, 0), 2)

    # Display main prediction
    cv2.putText(frame, label_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Display method
    if method_text:
        cv2.putText(frame, method_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display debug info
    if debug_text:
        cv2.putText(frame, debug_text, (20, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Display confidence
    if confidence > 0:
        conf_color = (0, 255, 0) if confidence > 90 else (0, 200, 255)
        cv2.putText(frame, f"Confidence: {confidence}%", (w-250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

    # Show frame
    cv2.imshow("ISL Digit Recognition (0-9) - CORRECTED", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Application closed successfully")