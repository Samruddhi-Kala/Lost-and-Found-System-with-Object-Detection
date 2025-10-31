# models.py
import os
import numpy as np
import faiss
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer
import pytesseract
import sys

# Configure Tesseract
import logging
logger = logging.getLogger(__name__)

def setup_tesseract():
    """Setup Tesseract OCR based on the operating system."""
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
    else:  # Linux/Unix
        tesseract_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]

    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract found at: {path}")
            return True

    logger.warning("Tesseract not found. OCR features will be disabled.")
    return False

# Try to setup Tesseract
OCR_AVAILABLE = setup_tesseract()

# YOLO via ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Assuming these files are created in the lostfound/ directory alongside models.py
INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.bin")
MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_map.npy") 

class Pipeline:
    def __init__(self, device='cpu'):
        self.device = device
        # ResNet feature extractor (remove final classification layer)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = torch.nn.Identity()
        self.cnn.eval().to(self.device)
        # transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        # SBERT
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        # YOLO
        self.yolo = YOLO('yolov8n.pt') if YOLO_AVAILABLE else None
        # FAISS placeholders
        self.faiss_index = None
        self.index_to_report = []  # map faiss idx -> report_id
        self.dim = 512  # resnet18 outputs 512-d
        self._init_faiss()

    def _init_faiss(self):
        # using IndexFlatL2 for prototype
        self.faiss_index = faiss.IndexFlatL2(self.dim)

    def save_faiss(self):
        faiss.write_index(self.faiss_index, INDEX_FILE)
        np.save(MAPPING_FILE, np.array(self.index_to_report, dtype=np.int64))

    def load_faiss(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(MAPPING_FILE):
            self.faiss_index = faiss.read_index(INDEX_FILE)
            self.index_to_report = np.load(MAPPING_FILE).tolist()

    def detect_objects(self, image_path):
        """Runs YOLO detection and returns bounding boxes and labels.
           Fallback: if YOLO not available -> return entire image as one bbox."""
        img = Image.open(image_path).convert('RGB')
        w,h = img.size
        if self.yolo:
            # Check if model weights are downloaded; YOLO will often download them on first run
            res = self.yolo(image_path)[0]
            dets = []
            if len(res.boxes) > 0:
                for box in res.boxes:
                    xyxy = box.xyxy[0].tolist()  # Get the first (and only) set of coordinates
                    if len(xyxy) == 4:  # Ensure we have all 4 coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        dets.append((x1, y1, x2, y2))
            if dets:
                return dets
        # fallback full image
        return [(0,0,w,h)]

    def extract_image_embedding(self, image: Image.Image):
        """Given a PIL image (cropped object), return feature vector (numpy)."""
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.cnn(x).cpu().numpy().reshape(-1)
        # normalize
        norm = np.linalg.norm(feat)
        if norm>0:
            feat = feat / norm
        return feat.astype('float32')

    def get_text_embedding(self, text):
        v = self.sbert.encode([text], convert_to_numpy=True)[0]
        # normalize
        if np.linalg.norm(v)>0:
            v = v / np.linalg.norm(v)
        return v.astype('float32')

    def ocr_extract(self, image: np.ndarray):
        """Extract text from image with detailed error handling."""
        if not OCR_AVAILABLE:
            logger.warning("OCR requested but Tesseract is not available")
            return ""  # Return empty string when OCR is not available
        if not OCR_AVAILABLE:
            print("WARNING: OCR is disabled - Tesseract not found")
            return ""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            print("Please verify Tesseract is installed correctly")
            return ""

    def dominant_color(self, image: np.ndarray, k=3):
        """Return approximate dominant color as hex or name; output hex string."""
        # image: BGR
        data = image.reshape((-1,3)).astype(np.float32)
        # kmeans
        _, labels, centers = cv2.kmeans(data, k, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        # convert BGR -> RGB
        r,g,b = int(dominant[2]), int(dominant[1]), int(dominant[0])
        return '#{:02x}{:02x}{:02x}'.format(r,g,b)

    def add_to_faiss(self, emb: np.ndarray, report_id:int):
        """Add embedding to FAISS and record mapping."""
        if emb.ndim==1:
            emb = emb.reshape(1, -1).astype('float32')
        self.faiss_index.add(emb)
        self.index_to_report.append(int(report_id))

    def search_faiss(self, query_emb: np.ndarray, topk=5):
        """Search with a single query embedding."""
        if query_emb.ndim==1:
            query_emb = query_emb.reshape(1,-1).astype('float32')
        return self.search_faiss_multi_query(query_emb, topk)

    def search_faiss_multi_query(self, query_matrix: np.ndarray, topk=5):
        """Search with a matrix of query embeddings (one query per row)."""
        if query_matrix.ndim==1:
            query_matrix = query_matrix.reshape(1,-1).astype('float32')
        
        if self.faiss_index.ntotal==0:
            return []
            
        D, I = self.faiss_index.search(query_matrix, topk)
        
        results = []
        # D is (num_queries, topk), I is (num_queries, topk)
        for dist_list, idx_list in zip(D, I):
            for d, idx in zip(dist_list, idx_list):
                if idx<0: continue
                rid = self.index_to_report[idx]
                results.append((rid, float(d)))
        return results