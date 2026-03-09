from flask import Flask, render_template, request
import cv2
import pytesseract
import pandas as pd
import imutils
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

app = Flask(__name__)

def detect_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    from collections import Counter
    results = []
    
    # Try multiple image sizes
    for width in [300, 400, 500, 600, 800]:
        img_resized = imutils.resize(img, width=width)
        original = img_resized.copy()
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Method 1: MSER-based detection (most reliable)
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                aspect = w / float(h) if h > 0 else 0
                
                if 2 <= aspect <= 6 and 40 < w < width * 0.8 and h > 10:
                    pad = 5
                    y1, y2 = max(0, y-pad), min(gray.shape[0], y+h+pad)
                    x1, x2 = max(0, x-pad), min(gray.shape[1], x+w+pad)
                    
                    plate_region = gray[y1:y2, x1:x2]
                    if plate_region.size > 0:
                        plate_region = cv2.resize(plate_region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        _, thresh = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        text = ''.join(c for c in text if c.isalnum()).upper()
                        
                        if len(text) >= 8 and len(text) <= 12:
                            has_letter = any(c.isalpha() for c in text)
                            has_digit = any(c.isdigit() for c in text)
                            if has_letter and has_digit:
                                results.append(text)
        except:
            pass
        
        # Method 2: Contour-based with mask (backup)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 170, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [approx], 0, 255, -1)
                new_img = cv2.bitwise_and(original, original, mask=mask)
                
                config = '-l eng --oem 1 --psm 3'
                text = pytesseract.image_to_string(new_img, config=config)
                text = ''.join(c for c in text if c.isalnum()).upper()
                
                if len(text) >= 8 and len(text) <= 12:
                    has_letter = any(c.isalpha() for c in text)
                    has_digit = any(c.isdigit() for c in text)
                    if has_letter and has_digit:
                        results.append(text)
                break
    
    # Return most common result
    if results:
        counts = Counter(results)
        return counts.most_common(1)[0][0]
    
    return ""

def similarity_score(s1, s2):
    """Calculate similarity between two strings (0-1)"""
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # Use longest common subsequence approach
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    matches = sum(1 for c in shorter if c in longer)
    return matches / max(len(longer), 1)

def find_best_match(detected, plates):
    """Find the best matching plate from the database, allowing for OCR errors"""
    if not detected:
        return detected
    if detected in plates:
        return detected
    
    # Common OCR character confusions (bidirectional)
    ocr_confusions = {
        '0': ['O', 'D', 'Q'],
        'O': ['0', 'D', 'Q'],
        'D': ['0', 'O', 'Q'],
        'Q': ['0', 'O', 'D'],
        '1': ['I', 'L', '7'],
        'I': ['1', 'L', '7'],
        'L': ['1', 'I'],
        '2': ['Z', '7'],
        'Z': ['2'],
        '3': ['8', 'B', 'E'],
        '8': ['3', 'B'],
        'B': ['8', '3'],
        '4': ['A', 'H'],
        'A': ['4', 'H', 'R'],
        'H': ['4', 'A', 'N', 'R'],
        '5': ['S', '6'],
        'S': ['5', '8'],
        '6': ['G', '5', 'E', 'B'],
        'G': ['6', 'C'],
        '7': ['T', '1', 'Y'],
        'T': ['7', '1', 'Y'],
        '9': ['P', 'Q'],
        'E': ['6', 'F', 'R', 'B'],
        'R': ['A', 'K', 'H'],
        'U': ['V', 'W', 'D', 'O', 'Q'],
        'V': ['U', 'W', 'Y'],
        'N': ['H', 'M'],
        'M': ['N', 'W'],
    }
    
    def chars_similar(c1, c2):
        """Check if two characters are commonly confused by OCR"""
        if c1 == c2:
            return 1.0
        if c1 in ocr_confusions and c2 in ocr_confusions.get(c1, []):
            return 0.85
        if c2 in ocr_confusions and c1 in ocr_confusions.get(c2, []):
            return 0.85
        return 0
    
    best_plate = None
    best_score = 0.6  # Lower threshold for tolerance
    
    for plate in plates:
        if len(detected) == len(plate):
            # Same length - character by character comparison
            score = sum(chars_similar(d, p) for d, p in zip(detected, plate)) / len(plate)
        elif abs(len(detected) - len(plate)) <= 2:
            # Length difference of 1-2 - try alignment
            scores = []
            for offset in range(abs(len(detected) - len(plate)) + 1):
                shorter, longer = (detected, plate) if len(detected) < len(plate) else (plate, detected)
                aligned = longer[offset:offset + len(shorter)]
                if len(aligned) == len(shorter):
                    s = sum(chars_similar(a, b) for a, b in zip(shorter, aligned)) / len(longer)
                    scores.append(s)
            score = max(scores) if scores else 0
        else:
            # Too different in length
            score = similarity_score(detected, plate) * 0.5
        
        if score > best_score:
            best_score = score
            best_plate = plate
    
    return best_plate if best_plate else detected

@app.route("/", methods=["GET","POST"])
def index():

    plate = None
    owner = None
    vehicle = None
    city = None

    if request.method == "POST":

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        plate = detect_plate(path)

        csv_path = os.path.join(BASE_DIR, "vehicle_data.csv")
        data = pd.read_csv(csv_path)

        # Try fuzzy matching against known plates
        known_plates = data["number"].tolist()
        matched_plate = find_best_match(plate, known_plates)
        
        match = data[data["number"] == matched_plate]

        if not match.empty:
            plate = matched_plate  # Use the matched plate for display
            owner = match.iloc[0]["owner"]
            vehicle = match.iloc[0]["vehicle"]
            city = match.iloc[0]["city"]

    return render_template("index.html", plate=plate, owner=owner, vehicle=vehicle, city=city)

if __name__ == "__main__":
    app.run(debug=True)