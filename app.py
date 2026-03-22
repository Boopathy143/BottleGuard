"""
BottleGuard - Flask Backend
Memory optimized for Render free tier (512MB)
No RAM cache - direct Cloudinary compare
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
import urllib.request
import cloudinary
import cloudinary.uploader
import cloudinary.api
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
SIMILARITY_THRESHOLD = 15
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


def get_all_resources():
    try:
        result = cloudinary.api.resources(
            type='upload',
            prefix=ROOT_FOLDER + '/',
            max_results=500,
            context=True
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
                result = cloudinary.uploader.upload(
                    f, folder=folder_name,
                    use_filename=True, unique_filename=True,
                    resource_type='image', context=context_str
                )
                saved.append(result['public_id'])
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    if not saved:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    return jsonify({
        'message': f'Saved {len(saved)} image(s) for "{defect_name}"',
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
                'description': meta['description'],
                'keywords': meta['keywords'],
                'affected_area': meta['affected_area']
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

    scan_img = bytes_to_cv2(file.read())
    if scan_img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    resources = get_all_resources()
    groups = group_by_defect(resources)

    best_score = 0.0
    best_defect = None
    best_url = None
    best_meta = {}

    def compare_one(item, defect_name, meta):
        try:
            data = url_to_bytes(item['secure_url'])
            sample = bytes_to_cv2(data)
            if sample is None:
                return (0.0, defect_name, None, {})
            score = compute_orb_similarity(scan_img, sample)
            # Free memory immediately
            del sample, data
            return (score, defect_name, item['secure_url'], meta)
        except Exception:
            return (0.0, defect_name, None, {})

    # Use only 3 workers to stay within memory limits
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for defect_name, items in groups.items():
            meta = get_meta(items)
            for item in items:
                futures.append(executor.submit(compare_one, item, defect_name, meta))

        for future in as_completed(futures):
            score, defect_name, url, meta = future.result()
            if score > best_score:
                best_score = score
                best_defect = defect_name
                best_url = url
                best_meta = meta

    if best_score >= SIMILARITY_THRESHOLD and best_defect:
        matched_b64 = url_to_base64(best_url) if best_url else None
        return jsonify({
            'result': 'DEFECT',
            'defect_name': best_defect,
            'score': best_score,
            'matched_image': matched_b64,
            'description': best_meta.get('description', ''),
            'keywords': best_meta.get('keywords', ''),
            'affected_area': best_meta.get('affected_area', '')
        })

    return jsonify({
        'result': 'PASS',
        'defect_name': None,
        'score': best_score,
        'matched_image': None,
        'description': '',
        'keywords': '',
        'affected_area': ''
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
