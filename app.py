"""
Bottle Defect Detection System - Flask Backend
Uses OpenCV ORB feature matching + Cloudinary for persistent image storage.
"""

import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
import urllib.request
import cloudinary
import cloudinary.uploader
import cloudinary.api

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# ─── Cloudinary Config ────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
SIMILARITY_THRESHOLD = 15


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_orb_similarity(img_array1, img_array2):
    gray1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)
    target_size = (300, 300)
    gray1 = cv2.resize(gray1, target_size)
    gray2 = cv2.resize(gray2, target_size)
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return 0.0
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]
    score = (len(good_matches) / min(len(kp1), len(kp2))) * 100
    return round(min(score, 100.0), 2)


def url_to_cv2(url):
    """Download image from URL and convert to OpenCV format."""
    resp = urllib.request.urlopen(url, timeout=10)
    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def url_to_base64(url):
    """Convert image URL to base64 data URI."""
    resp = urllib.request.urlopen(url, timeout=10)
    data = base64.b64encode(resp.read()).decode('utf-8')
    return f'data:image/jpeg;base64,{data}'


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# ── 1. Upload defect sample images to Cloudinary ─────────────────────────────
@app.route('/upload_defect', methods=['POST'])
def upload_defect():
    defect_name = request.form.get('defect_name', '').strip()
    if not defect_name:
        return jsonify({'error': 'defect_name is required'}), 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No image files provided'}), 400

    saved = []
    folder_name = f'bottleguard/{secure_filename(defect_name)}'

    for f in files:
        if f and allowed_file(f.filename):
            result = cloudinary.uploader.upload(
                f,
                folder=folder_name,
                resource_type='image'
            )
            saved.append(result['public_id'])

    if not saved:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    return jsonify({
        'message': f'Saved {len(saved)} image(s) for "{defect_name}"',
        'files': saved
    })


# ── 2. Get all defect categories from Cloudinary ─────────────────────────────
@app.route('/get_defects', methods=['GET'])
def get_defects():
    try:
        # List all folders under bottleguard/
        result = cloudinary.api.subfolders('bottleguard')
        folders = result.get('folders', [])
    except Exception:
        return jsonify([])

    defects = []
    for folder in folders:
        folder_path = folder['path']
        defect_name = folder['name']

        try:
            resources = cloudinary.api.resources(
                type='upload',
                prefix=folder_path + '/',
                max_results=1
            )
            items = resources.get('resources', [])
            if not items:
                continue

            preview_url = items[0]['secure_url']
            # Get total count
            all_res = cloudinary.api.resources(
                type='upload',
                prefix=folder_path + '/',
                max_results=500
            )
            count = len(all_res.get('resources', []))

            preview_b64 = url_to_base64(preview_url)
            defects.append({
                'name': defect_name,
                'image_count': count,
                'preview': preview_b64
            })
        except Exception:
            continue

    return jsonify(defects)


# ── 3. Check image against all defects ───────────────────────────────────────
@app.route('/check_image', methods=['POST'])
def check_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    scan_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if scan_img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    best_score = 0.0
    best_defect = None
    best_image_url = None

    try:
        result = cloudinary.api.subfolders('bottleguard')
        folders = result.get('folders', [])
    except Exception:
        folders = []

    for folder in folders:
        folder_path = folder['path']
        defect_name = folder['name']

        try:
            resources = cloudinary.api.resources(
                type='upload',
                prefix=folder_path + '/',
                max_results=500
            )
            items = resources.get('resources', [])
        except Exception:
            continue

        for item in items:
            try:
                sample_img = url_to_cv2(item['secure_url'])
                if sample_img is None:
                    continue
                score = compute_orb_similarity(scan_img, sample_img)
                if score > best_score:
                    best_score = score
                    best_defect = defect_name
                    best_image_url = item['secure_url']
            except Exception:
                continue

    if best_score >= SIMILARITY_THRESHOLD and best_defect:
        matched_b64 = url_to_base64(best_image_url) if best_image_url else None
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
    folder_path = f'bottleguard/{secure_filename(defect_name)}'
    try:
        # Delete all images in folder
        resources = cloudinary.api.resources(
            type='upload',
            prefix=folder_path + '/',
            max_results=500
        )
        public_ids = [r['public_id'] for r in resources.get('resources', [])]
        if public_ids:
            cloudinary.api.delete_resources(public_ids)
        # Delete folder
        cloudinary.api.delete_folder(folder_path)
        return jsonify({'message': f'Deleted "{defect_name}"'})
    except Exception as e:
        return jsonify({'error': str(e)}), 404


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Bottle Defect Detection System (Cloudinary)")
    print("  Running at: http://127.0.0.1:8080")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8080, debug=False)
