"""
BottleGuard - Flask Backend
Gemini Vision (HTTP) + ORB + Histogram Combo
Memory optimized - no heavy AI libraries
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
import urllib.request
import urllib.error
import cloudinary
import cloudinary.uploader
import cloudinary.api
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import urllib.parse

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
SIMILARITY_THRESHOLD = 12
ROOT_FOLDER = 'bottleguard'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def url_to_bytes(url):
    resp = urllib.request.urlopen(url, timeout=10)
    return resp.read()


def bytes_to_cv2(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def url_to_base64(url):
    data = url_to_bytes(url)
    return f'data:image/jpeg;base64,{base64.b64encode(data).decode()}'


def img_to_base64_str(img_bytes):
    return base64.b64encode(img_bytes).decode('utf-8')


# ─── Gemini Vision via HTTP ───────────────────────────────────────────────────
def analyze_with_gemini(img_bytes, defect_library):
    try:
        # Build defect context
        defect_context = ""
        if defect_library:
            defect_context = "Known defect categories:\n"
            for d in defect_library:
                defect_context += f"- {d['name']}"
                if d.get('description'):
                    defect_context += f": {d['description']}"
                if d.get('keywords'):
                    defect_context += f" (keywords: {d['keywords']})"
                if d.get('affected_area'):
                    defect_context += f" [Area: {d['affected_area']}]"
                defect_context += "\n"

        prompt = f"""You are a quality control expert inspecting plastic bottles for manufacturing defects.

{defect_context}

Analyze this bottle image and respond ONLY with this JSON format:
{{
  "has_defect": true or false,
  "defect_name": "name from known categories or null",
  "confidence": 0-100,
  "affected_area": "Body/Cap/Bottom/Label/Neck/None",
  "description": "brief description",
  "keywords": "comma separated keywords",
  "reasoning": "brief reason"
}}

Rules:
- Only flag CLEAR visible defects
- Match defect_name exactly to known categories if possible
- JSON only, no other text"""

        # Convert image to base64
        img_b64 = img_to_base64_str(img_bytes)

        # Detect mime type
        if img_bytes[:3] == b'\xff\xd8\xff':
            mime = 'image/jpeg'
        elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = 'image/png'
        else:
            mime = 'image/jpeg'

        # Build request payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": img_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 512
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            GEMINI_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        text = result['candidates'][0]['content']['parts'][0]['text'].strip()

        # Clean JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        return json.loads(text)

    except Exception as e:
        print(f"Gemini error: {e}")
        return None


# ─── ORB Score ────────────────────────────────────────────────────────────────
def compute_orb_similarity(img1, img2):
    g1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (300, 300))
    g2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (300, 300))
    orb = cv2.ORB_create(nfeatures=500)
    kp1, d1 = orb.detectAndCompute(g1, None)
    kp2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 50]
    return round(min((len(good) / min(len(kp1), len(kp2))) * 100, 100.0), 2)


# ─── Histogram Score ──────────────────────────────────────────────────────────
def compute_histogram_similarity(img1, img2):
    i1 = cv2.resize(img1, (256, 256))
    i2 = cv2.resize(img2, (256, 256))
    scores = []
    for ch in range(3):
        h1 = cv2.calcHist([i1], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([i2], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        scores.append(max(0, score))
    return round(np.mean(scores) * 100, 2)


def visual_combined_score(img1, img2):
    orb = compute_orb_similarity(img1, img2)
    hist = compute_histogram_similarity(img1, img2)
    if hist > 70:
        final = round(orb * 0.35 + hist * 0.65, 2)
    else:
        final = round(orb * 0.55 + hist * 0.45, 2)
    return final, orb, hist


# ─── Cloudinary Helpers ───────────────────────────────────────────────────────
def get_all_resources():
    try:
        result = cloudinary.api.resources(
            type='upload', prefix=ROOT_FOLDER + '/',
            max_results=500, context=True
        )
        return result.get('resources', [])
    except Exception:
        return []


def group_by_defect(resources):
    groups = {}
    for r in resources:
        parts = r['public_id'].split('/')
        if len(parts) >= 2:
            groups.setdefault(parts[1], []).append(r)
    return groups


def get_meta(items):
    for r in items:
        c = r.get('context', {}).get('custom', {})
        if c:
            return {
                'description': c.get('description', ''),
                'keywords': c.get('keywords', ''),
                'affected_area': c.get('affected_area', '')
            }
    return {'description': '', 'keywords': '', 'affected_area': ''}


def get_defect_library():
    resources = get_all_resources()
    if not resources:
        return []
    groups = group_by_defect(resources)
    library = []
    for defect_name, items in sorted(groups.items()):
        meta = get_meta(items)
        library.append({'name': defect_name, **meta})
    return library


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/upload_defect', methods=['POST'])
def upload_defect():
    defect_name = request.form.get('defect_name', '').strip()
    description = request.form.get('description', '').strip()
    keywords = request.form.get('keywords', '').strip()
    affected_area = request.form.get('affected_area', '').strip()

    if not defect_name:
        return jsonify({'error': 'defect_name is required'}), 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No image files provided'}), 400

    saved = []
    folder_name = f'{ROOT_FOLDER}/{secure_filename(defect_name)}'
    context_str = f'description={description}|keywords={keywords}|affected_area={affected_area}'

    for f in files:
        if f and allowed_file(f.filename):
            try:
                f_bytes = f.read()
                result = cloudinary.uploader.upload(
                    f_bytes, folder=folder_name,
                    unique_filename=True, resource_type='image',
                    context=context_str
                )
                saved.append(result['public_id'])

                img = bytes_to_cv2(f_bytes)
                if img is not None:
                    for flip_code in [1, 0]:
                        flipped = cv2.flip(img, flip_code)
                        _, buf = cv2.imencode('.jpg', flipped)
                        cloudinary.uploader.upload(
                            buf.tobytes(), folder=folder_name,
                            unique_filename=True, resource_type='image',
                            context=context_str
                        )
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    if not saved:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    return jsonify({
        'message': f'Saved {len(saved)} image(s) + flipped versions for "{defect_name}"',
        'files': saved
    })


@app.route('/get_defects', methods=['GET'])
def get_defects():
    resources = get_all_resources()
    if not resources:
        return jsonify([])
    groups = group_by_defect(resources)
    defects = []
    for defect_name, items in sorted(groups.items()):
        try:
            preview_b64 = url_to_base64(items[0]['secure_url'])
            meta = get_meta(items)
            defects.append({
                'name': defect_name,
                'image_count': len(items),
                'preview': preview_b64,
                **meta
            })
        except Exception:
            continue
    return jsonify(defects)


@app.route('/check_image', methods=['POST'])
def check_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    raw = file.read()
    scan_img = bytes_to_cv2(raw)
    if scan_img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Step 1: Gemini Vision
    defect_library = get_defect_library()
    gemini_result = analyze_with_gemini(raw, defect_library)

    # Step 2: Visual Match
    resources = get_all_resources()
    groups = group_by_defect(resources)

    best_visual_score = 0.0
    best_defect_visual = None
    best_url = None
    best_meta = {}

    def compare_one(item, defect_name, meta):
        try:
            data = url_to_bytes(item['secure_url'])
            sample = bytes_to_cv2(data)
            if sample is None:
                return (0.0, defect_name, None, {})
            score, _, _ = visual_combined_score(scan_img, sample)
            del sample, data
            return (score, defect_name, item['secure_url'], meta)
        except Exception:
            return (0.0, defect_name, None, {})

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for defect_name, items in groups.items():
            meta = get_meta(items)
            for item in items:
                futures.append(executor.submit(compare_one, item, defect_name, meta))
        for future in as_completed(futures):
            score, defect_name, url, meta = future.result()
            if score > best_visual_score:
                best_visual_score = score
                best_defect_visual = defect_name
                best_url = url
                best_meta = meta

    # Step 3: Smart Decision
    gemini_says_defect = gemini_result and gemini_result.get('has_defect', False)
    gemini_confidence = gemini_result.get('confidence', 0) if gemini_result else 0
    visual_says_defect = best_visual_score >= SIMILARITY_THRESHOLD

    if gemini_says_defect and gemini_confidence >= 70:
        final_result = 'DEFECT'
        final_defect = gemini_result.get('defect_name') or best_defect_visual
        detection_method = 'AI Vision'
        final_description = gemini_result.get('description', best_meta.get('description', ''))
        final_keywords = gemini_result.get('keywords', best_meta.get('keywords', ''))
        final_area = gemini_result.get('affected_area', best_meta.get('affected_area', ''))
        final_reasoning = gemini_result.get('reasoning', '')
    elif gemini_says_defect and visual_says_defect:
        final_result = 'DEFECT'
        final_defect = gemini_result.get('defect_name') or best_defect_visual
        detection_method = 'AI Vision + Visual Match'
        final_description = gemini_result.get('description', best_meta.get('description', ''))
        final_keywords = gemini_result.get('keywords', best_meta.get('keywords', ''))
        final_area = gemini_result.get('affected_area', best_meta.get('affected_area', ''))
        final_reasoning = gemini_result.get('reasoning', '')
    elif visual_says_defect:
        final_result = 'DEFECT'
        final_defect = best_defect_visual
        detection_method = 'Visual Match'
        final_description = best_meta.get('description', '')
        final_keywords = best_meta.get('keywords', '')
        final_area = best_meta.get('affected_area', '')
        final_reasoning = f'Visual similarity: {best_visual_score:.1f}%'
    else:
        final_result = 'PASS'
        final_defect = None
        detection_method = 'AI Vision + Visual Match'
        final_description = ''
        final_keywords = ''
        final_area = ''
        final_reasoning = gemini_result.get('reasoning', 'No defect detected') if gemini_result else 'No match found'

    matched_b64 = url_to_base64(best_url) if best_url and final_result == 'DEFECT' else None

    return jsonify({
        'result': final_result,
        'defect_name': final_defect,
        'score': round(max(gemini_confidence, best_visual_score), 1),
        'gemini_confidence': gemini_confidence,
        'visual_score': best_visual_score,
        'detection_method': detection_method,
        'matched_image': matched_b64,
        'description': final_description,
        'keywords': final_keywords,
        'affected_area': final_area,
        'reasoning': final_reasoning
    })


@app.route('/delete_defect/<defect_name>', methods=['DELETE'])
def delete_defect(defect_name):
    folder_path = f'{ROOT_FOLDER}/{secure_filename(defect_name)}'
    try:
        resources = cloudinary.api.resources(
            type='upload', prefix=folder_path + '/', max_results=500
        )
        public_ids = [r['public_id'] for r in resources.get('resources', [])]
        if public_ids:
            cloudinary.api.delete_resources(public_ids)
        return jsonify({'message': f'Deleted "{defect_name}"'})
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
