# app.py
import os
import sys
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PATH CONFIGURATION FOR YOUR STRUCTURE ---
# BASE_DIR is ML Project/lostfound/
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# PROJECT_ROOT is ML Project/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Create necessary directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app with correct template path
app = Flask(__name__,
          # Templates are in lostfound/templates
          template_folder=os.path.join(BASE_DIR, 'templates'),
          # Static files are in lostfound/static
          static_folder=os.path.join(BASE_DIR, 'static'))

# Add template context processor for current year
@app.context_processor
def inject_year():
    from datetime import datetime
    return {'year': datetime.utcnow().year}

# Configure Flask app
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,  # 8MB
    TEMPLATES_AUTO_RELOAD=True  # Enable template auto-reload
)

try:
    from models import Pipeline
    import db
    from PIL import Image
    import cv2
    import pytesseract
    
    # Set the path to Tesseract executable - Updated for v5.5.0
    if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        raise RuntimeError("Tesseract not found in expected location. Please verify installation.")
    
    # Initialize DB and pipeline
    db.init_db()
    pipe = Pipeline(device='cpu')
    pipe.load_faiss()  # load existing index if present
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    # If ML initialization fails, it's better to log the error and let Flask start if possible, 
    # but since the core functionality relies on it, we re-raise.
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lost/submit', methods=['GET', 'POST'])
def submit_lost():
    if request.method == 'GET':
        return render_template('submit_lost.html')
    # POST: accept optional image & description
    desc = request.form.get('description','')
    category = request.form.get('category','')
    f = request.files.get('image', None)
    image_path = None
    report_id = None
    if f and f.filename != '':
        fname = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(path)
        image_path = path
        # run OCR and color detection for metadata
        img_cv = cv2.imread(path)
        ocr_text = pipe.ocr_extract(img_cv)
        color = pipe.dominant_color(img_cv)
        report_id = db.save_report('lost', path, desc, category, color, ocr_text)
        # save text embedding for description if present
        if desc.strip():
            t_emb = pipe.get_text_embedding(desc)
            db.save_embedding(report_id, 'text', t_emb)
        # if image, also compute image embedding
        # detect objects and extract embeddings per object
        dets = pipe.detect_objects(path)
        pil = Image.open(path).convert('RGB')
        for (x1,y1,x2,y2) in dets:
            crop = pil.crop((x1,y1,x2,y2))
            emb = pipe.extract_image_embedding(crop)
            db.save_embedding(report_id, 'image', emb)
            # For lost items, we might want to add image embeddings to faiss as well so they can be matched to found items.
            pipe.add_to_faiss(emb, report_id)
    else:
        # text-only lost report
        report_id = db.save_report('lost', None, desc, category, None, None)
        if desc.strip():
            t_emb = pipe.get_text_embedding(desc)
            db.save_embedding(report_id, 'text', t_emb)
    pipe.save_faiss()
    return redirect(url_for('index'))

@app.route('/found/submit', methods=['GET','POST'])
def submit_found():
    if request.method == 'GET':
        return render_template('submit_found.html')
    desc = request.form.get('description','')
    category = request.form.get('category','')
    f = request.files.get('image', None)
    if not f or f.filename == '':
        return "Found report requires an image", 400
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(path)
    img_cv = cv2.imread(path)
    ocr_text = pipe.ocr_extract(img_cv)
    color = pipe.dominant_color(img_cv)
    report_id = db.save_report('found', path, desc, category, color, ocr_text)
    # text embeddings for user description and OCR
    if desc.strip():
        t_emb = pipe.get_text_embedding(desc)
        db.save_embedding(report_id, 'text', t_emb)
    if ocr_text:
        o_emb = pipe.get_text_embedding(ocr_text)
        db.save_embedding(report_id, 'ocr', o_emb)
    
    # detect objects, extract embeddings, save, add to FAISS
    dets = pipe.detect_objects(path)
    pil = Image.open(path).convert('RGB')
    
    all_query_embeddings = []
    
    for (x1,y1,x2,y2) in dets:
        crop = pil.crop((x1,y1,x2,y2))
        emb = pipe.extract_image_embedding(crop)
        db.save_embedding(report_id, 'image', emb)
        pipe.add_to_faiss(emb, report_id) # Add this found item's objects to the index
        all_query_embeddings.append(emb)
    
    pipe.save_faiss()

    # after adding to index, attempt match search using all object embeddings
    matches = {} # {lost_id: {'lost_id', 'image', 'desc', 'score'}}
    
    if all_query_embeddings:
        query_matrix = np.stack(all_query_embeddings)
        
        # Search FAISS using all object embeddings simultaneously
        results = pipe.search_faiss_multi_query(query_matrix, topk=5)
        
        for rid, dist in results:
            # fetch report; only check for lost items
            rep = db.get_report(rid)
            if rep and rep['kind']=='lost':
                current_best_score = matches.get(rep['id'], {}).get('score', float('inf'))
                
                # We want the lowest distance (best match score)
                if float(dist) < current_best_score:
                    matches[rep['id']] = {'lost_id': rep['id'], 'image': rep['image_path'], 'desc': rep['description'], 'score': float(dist)}

    # Convert the dictionary values back to a list
    final_matches = list(matches.values())
    
    # redirect to matches page showing suggestions for this found report
    return render_template('matches.html', found_id=report_id, suggestions=final_matches)

@app.route('/matches')
def view_matches():
    # show all boxes where found vs lost suggested pairs exist — for prototype just show all lost and found with simple nearest neighbors
    found = db.fetch_reports('found')
    pairs = []
    # for each found, search using *all* its image embeddings
    import db as dbmod
    emb_rows = dbmod.fetch_embeddings('image')
    
    # 1. Group embeddings by report ID
    report_to_embeddings = {}
    for eid, rid, emb in emb_rows:
        if rid not in report_to_embeddings:
            report_to_embeddings[rid] = []
        report_to_embeddings[rid].append(emb)

    # 2. Search for matches for each found report
    for f in found:
        if f['id'] in report_to_embeddings:
            query_matrix = np.stack(report_to_embeddings[f['id']])
            results = pipe.search_faiss_multi_query(query_matrix, topk=5)
            
            # Keep track of the best match for this found report
            best_match = {} # {lost_id: best_score}

            for rid, dist in results:
                rep = db.get_report(rid)
                if rep and rep['kind']=='lost':
                    current_best_score = best_match.get(rep['id'], float('inf'))
                    if float(dist) < current_best_score:
                        best_match[rep['id']] = float(dist)

            # Convert best matches into pair structure
            for lost_id, score in best_match.items():
                lost_rep = db.get_report(lost_id)
                # Only include pairs where the item ID is not the same (no self-matches)
                if lost_rep and lost_id != f['id']:
                     pairs.append({'found': f['id'], 'found_img': f['image_path'], 
                                   'lost': lost_rep['id'], 'lost_img': lost_rep['image_path'], 
                                   'score': score})

    return render_template('matches.html', pairs=pairs, suggestions=[])

@app.route('/admin', methods=['GET'])
def admin_ui():
    # show unconfirmed suggested pairs (for prototype: show last 20 reports)
    found = db.fetch_reports('found')
    lost = db.fetch_reports('lost')
    return render_template('admin.html', found=found, lost=lost)

@app.route('/admin/confirm', methods=['POST'])
def admin_confirm():
    data = request.json
    # The current prototype UI doesn't pass a pair, just one report ID for confirmation status
    report_id = data.get('report_id')
    val = data.get('val', 1)
    db.set_admin_confirm(report_id, val)
    return jsonify({'ok': True})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # This assumes the browser asks for the filename only (e.g., 'image.jpg')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        logger.info(f"Starting Flask app with templates from: {app.template_folder}")
        logger.info(f"Static files from: {app.static_folder}")
        # Get port from environment variable or default to 10000
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        raise