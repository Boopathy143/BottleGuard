"""
Bottle Defect Detection System - Flask Backend
Uses OpenCV ORB feature matching to compare bottle images against defect templates.
"""

import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Folder where defect sample images are stored
DEFECTS_DIR = os.path.join(os.path.dirname(__file__), 'defects')
os.makedirs(DEFECTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Similarity threshold: scores above this are flagged as defects (0–100 scale)
SIMILARITY_THRESHOLD = 15  # tunable — lower = more sensitive


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_defect_dir(defect_name):
    """Return (and create if needed) the folder for a specific defect."""
    path = os.path.join(DEFECTS_DIR, secure_filename(defect_name))
    os.makedirs(path, exist_ok=True)
    return path


def compute_orb_similarity(img_array1, img_array2):
    """
    Compare two images using ORB keypoint matching.
    Returns a similarity score (0–100). Higher = more similar.
    """
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)

    # Resize both to the same size for fair comparison
    target_size = (300, 300)
    gray1 = cv2.resize(gray1, target_size)
    gray2 = cv2.resize(gray2, target_size)

    # ORB detector — fast and works without GPU
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0

    # BFMatcher with Hamming distance (suited for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0.0

    # Sort by distance; lower distance = better match
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only "good" matches (distance < 50)
    good_matches = [m for m in matches if m.distance < 50]

    # Score = percentage of keypoints that matched well
    score = (len(good_matches) / min(len(kp1), len(kp2))) * 100
    return round(min(score, 100.0), 2)


def image_to_base64(filepath):
    """Encode an image file as a base64 data URI for JSON transport."""
    ext = filepath.rsplit('.', 1)[-1].lower()
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
    with open(filepath, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:{mime};base64,{data}'


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the single-page application."""
    return send_from_directory('.', 'index.html')


@app.route('/defects/<path:filename>')
def serve_defect_image(filename):
    """Serve stored defect sample images."""
    return send_from_directory(DEFECTS_DIR, filename)


# ── 1. Upload a defect sample image ──────────────────────────────────────────
@app.route('/upload_defect', methods=['POST'])
def upload_defect():
    """
    Expects multipart form data:
      - defect_name: string
      - images:      one or more image files
    """
    defect_name = request.form.get('defect_name', '').strip()
    if not defect_name:
        return jsonify({'error': 'defect_name is required'}), 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No image files provided'}), 400

    saved = []
    folder = get_defect_dir(defect_name)

    for f in files:
        if f and allowed_file(f.filename):
            # Give each file a unique name to avoid collisions
            existing = os.listdir(folder)
            idx = len(existing) + 1
            ext = f.filename.rsplit('.', 1)[1].lower()
            filename = f'image{idx}.{ext}'
            filepath = os.path.join(folder, filename)
            f.save(filepath)
            saved.append(filename)

    if not saved:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    return jsonify({'message': f'Saved {len(saved)} image(s) for "{defect_name}"', 'files': saved})


# ── 2. Get all defect categories with preview images ─────────────────────────
@app.route('/get_defects', methods=['GET'])
def get_defects():
    """
    Returns a list of defect categories, each with:
      - name
      - image_count
      - preview (base64 of first image)
    """
    defects = []
    if not os.path.exists(DEFECTS_DIR):
        return jsonify([])

    for defect_name in sorted(os.listdir(DEFECTS_DIR)):
        folder = os.path.join(DEFECTS_DIR, defect_name)
        if not os.path.isdir(folder):
            continue

        images = [f for f in os.listdir(folder) if allowed_file(f)]
        if not images:
            continue

        preview = image_to_base64(os.path.join(folder, images[0]))
        defects.append({
            'name': defect_name,
            'image_count': len(images),
            'preview': preview
        })

    return jsonify(defects)


# ── 3. Check an uploaded bottle image against all defects ────────────────────
@app.route('/check_image', methods=['POST'])
def check_image():
    """
    Expects multipart form data:
      - image: the bottle image to inspect

    Returns:
      - result: 'PASS' or 'DEFECT'
      - defect_name: name of matched defect (if any)
      - score: similarity score
      - matched_image: base64 of best matching sample (if any)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Decode the uploaded image into a numpy array (no disk write needed)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    scan_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if scan_img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    best_score = 0.0
    best_defect = None
    best_image_path = None

    # Compare against every sample image in every defect folder
    for defect_name in sorted(os.listdir(DEFECTS_DIR)):
        folder = os.path.join(DEFECTS_DIR, defect_name)
        if not os.path.isdir(folder):
            continue

        for sample_file in os.listdir(folder):
            if not allowed_file(sample_file):
                continue

            sample_path = os.path.join(folder, sample_file)
            sample_img = cv2.imread(sample_path)
            if sample_img is None:
                continue

            score = compute_orb_similarity(scan_img, sample_img)

            if score > best_score:
                best_score = score
                best_defect = defect_name
                best_image_path = sample_path

    # Decide pass/fail based on threshold
    if best_score >= SIMILARITY_THRESHOLD and best_defect:
        matched_b64 = image_to_base64(best_image_path) if best_image_path else None
        return jsonify({
            'result': 'DEFECT',
            'defect_name': best_defect,
            'score': best_score,
            'matched_image': matched_b64
        })
    else:
        return jsonify({
            'result': 'PASS',
            'defect_name': None,
            'score': best_score,
            'matched_image': None
        })


# ── 4. Delete a defect category ──────────────────────────────────────────────
@app.route('/delete_defect/<defect_name>', methods=['DELETE'])
def delete_defect(defect_name):
    """Remove a defect category and all its sample images."""
    import shutil
    folder = os.path.join(DEFECTS_DIR, secure_filename(defect_name))
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return jsonify({'message': f'Deleted "{defect_name}"'})
    return jsonify({'error': 'Defect not found'}), 404


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Bottle Defect Detection System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8080, debug=False)
