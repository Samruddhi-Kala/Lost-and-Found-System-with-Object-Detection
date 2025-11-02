# db.py
import sqlite3
import numpy as np
import io
import os

# DB_PATH is now relative to the db.py file's location
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT,                 -- 'lost' or 'found'
        image_path TEXT,
        description TEXT,
        category TEXT,
        color TEXT,
        ocr_text TEXT,
        user_id INTEGER DEFAULT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        admin_confirmed INTEGER DEFAULT 0
    );
    """)
    # Create a users table for authentication
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id INTEGER,
        modality TEXT,            -- 'image' or 'text' or 'ocr'
        dim INTEGER,
        vec BLOB,
        FOREIGN KEY(report_id) REFERENCES reports(id)
    );
    """)
    # Ensure reports table has user_id column (for older DBs)
    cur.execute("PRAGMA table_info(reports)")
    cols = [r[1] for r in cur.fetchall()]
    if 'user_id' not in cols:
        try:
            cur.execute("ALTER TABLE reports ADD COLUMN user_id INTEGER DEFAULT NULL")
        except Exception:
            # if ALTER fails, ignore; new setups already have user_id
            pass
    conn.commit()
    conn.close()

def save_report(kind, image_path, description='', category=None, color=None, ocr_text=None, user_id=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports (kind, image_path, description, category, color, ocr_text, user_id)
        VALUES (?,?,?,?,?,?,?)
    """, (kind, image_path, description, category, color, ocr_text, user_id))
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def create_user(username, password_hash):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (username, password_hash)
        VALUES (?,?)
    """, (username, password_hash))
    uid = cur.lastrowid
    conn.commit()
    conn.close()
    return uid


def get_user_by_username(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    r = cur.fetchone()
    conn.close()
    return r


def get_user(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    r = cur.fetchone()
    conn.close()
    return r

def save_embedding(report_id, modality, vec: np.ndarray):
    # store float32 bytes + dim in separate column
    conn = get_conn()
    cur = conn.cursor()
    b = vec.astype('float32').tobytes()
    cur.execute("""
        INSERT INTO embeddings (report_id, modality, dim, vec)
        VALUES (?, ?, ?, ?)
    """, (report_id, modality, vec.shape[0], b))
    conn.commit()
    conn.close()

def fetch_reports(kind=None):
    conn = get_conn()
    cur = conn.cursor()
    if kind:
        cur.execute("SELECT * FROM reports WHERE kind=? ORDER BY timestamp DESC", (kind,))
    else:
        cur.execute("SELECT * FROM reports ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_report(rid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM reports WHERE id=?", (rid,))
    r = cur.fetchone()
    conn.close()
    return r

def fetch_embeddings(modality='image'):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id,report_id,dim,vec FROM embeddings WHERE modality=?", (modality,))
    rows = cur.fetchall()
    conn.close()
    # return as list of (emb_id, report_id, np.ndarray)
    out = []
    for row in rows:
        arr = np.frombuffer(row['vec'], dtype='float32').reshape((row['dim'],))
        out.append((row['id'], row['report_id'], arr))
    return out

def set_admin_confirm(report_id, val=1):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE reports SET admin_confirmed=? WHERE id=?", (int(val), report_id))
    conn.commit()
    conn.close()