"""
QuantEye Client AI Node - Dedicated Live AI Attendance Edge Application
- High-Speed Multithreaded RTSP / Webcam Grabber & AI Processor
- Cloud-Controlled Dynamic Camera Grid
- Real-Time Face Detection & Recognition
- Instant Push to Cloud Main Dashboard upon Detection
"""

import sys
import os
import io
import time
import json
import base64
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3
import urllib.request
import urllib.error

# Fix Windows console stream
class DummyStream:
    def write(self, text): pass
    def flush(self): pass
    @property
    def encoding(self): return 'utf-8'

if sys.stdout is None: sys.stdout = DummyStream()
if sys.stderr is None: sys.stderr = DummyStream()

from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np

# Ensure CUDA DLLs on Windows
try:
    add_dll_fn = getattr(os, 'add_dll_directory', None)
    if callable(add_dll_fn):
        torch_dll_dir = os.path.join(os.environ.get('CONDA_PREFIX') or sys.prefix, 'Library', 'bin')
        if os.path.isdir(torch_dll_dir): add_dll_fn(torch_dll_dir)
        cuda_lib_dir = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
        if os.path.isdir(cuda_lib_dir): add_dll_fn(cuda_lib_dir)
except Exception:
    pass

import database as db
import face_processor as fp
from logger_config import get_logger

logger = get_logger(__name__)

# Resolve persistent working directory vs bundle directory
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, '_MEIPASS', EXE_DIR)
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = EXE_DIR

templates_dir = os.path.join(BUNDLE_DIR, 'templates')
static_dir = os.path.join(BUNDLE_DIR, 'static')

app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
app.secret_key = "quanteye-client-node-secret-key"

# Persistent config must reside next to EXE, NOT inside temporary _MEIPASS
CONFIG_PATH = os.path.join(EXE_DIR, 'config_node.json')

def load_config() -> Dict[str, Any]:
    default_cfg = {
        "linked": False,
        "organization_id": "",
        "organization_name": "Unconfigured Node",
        "branch_id": "",
        "branch_name": "Setup Required",
        "branch_key": "",
        "username": "",
        "cloud_url": "http://127.0.0.1:5100",
        "node_port": 5005,
        "face_match_threshold": 0.45,
        "cameras": []
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k, v in default_cfg.items():
                    if k not in data: data[k] = v
                return data
        except Exception as e:
            logger.error(f"Error reading config_node.json: {e}")
    return default_cfg

CONFIG = load_config()

# Global Embeddings Vector Matrix
EMBEDDING_CACHE = {}
ENROLLED_MATRIX = None
ENROLLED_USER_IDS = []
ENROLLED_USER_METADATA = []
cache_lock = threading.Lock()

ACTIVE_CAMERA_READERS: Dict[str, Any] = {}
readers_lock = threading.Lock()

LATEST_DETECTIONS_FEED: List[Dict[str, Any]] = []
detections_lock = threading.Lock()


# ==============================================================================
# REAL-TIME ASYNC EVENT DISPATCHER TO CLOUD
# ==============================================================================

cloud_event_queue = queue.Queue(maxsize=1000)

def push_event_to_cloud_worker():
    """Background worker continuously flushing recognized attendance events to Cloud Portal & Main Dashboard."""
    while True:
        try:
            event = cloud_event_queue.get()
            cloud_url = f"{CONFIG.get('cloud_url', 'http://127.0.0.1:5100').rstrip('/')}/api/sync/attendance-event"
            portal_url = f"{CONFIG.get('client_portal_url', 'https://quanteye-ai-attendence-production.up.railway.app').rstrip('/')}/api/sync/attendance-event"
            req_data = json.dumps(event).encode('utf-8')

            # 1. Deliver to Cloud Client Portal
            try:
                preq = urllib.request.Request(
                    portal_url,
                    data=req_data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "QuantEye-Node",
                        "X-Client-Auth": "true",
                        "X-Organization-Id": str(event.get("organization_id", "")),
                        "X-Branch-Id": str(event.get("branch_id", ""))
                    },
                    method="POST"
                )
                with urllib.request.urlopen(preq, timeout=3.5) as presp:
                    if presp.status in (200, 201):
                        logger.info(f"[ATTENDANCE SYNC OK] Live attendance for '{event.get('name')}' delivered to Cloud Portal.")
            except Exception as pe:
                logger.debug(f"[PORTAL ATTENDANCE NOTE] {pe}")

            # 2. Deliver to Cloud Main Dashboard
            try:
                req = urllib.request.Request(
                    cloud_url,
                    data=req_data,
                    headers={"Content-Type": "application/json", "User-Agent": "QuantEye-Node"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"[CLOUD SYNC OK] Event for '{event.get('name')}' delivered to Main Cloud.")
            except Exception as ce:
                logger.debug(f"[MAIN CLOUD NOTE] {ce}")

        except Exception as e:
            logger.error(f"Error in push_event_to_cloud_worker: {e}")
        finally:
            cloud_event_queue.task_done()

threading.Thread(target=push_event_to_cloud_worker, daemon=True).start()


def dispatch_attendance_event_to_cloud(user_id: int, name: str, confidence: float, camera_id: str, camera_name: str, face_crop_b64: Optional[str] = None):
    """Enqueue immediate event for Cloud dispatch."""
    event_payload = {
        "organization_id": CONFIG.get("organization_id", "ORG-001"),
        "branch_id": CONFIG.get("branch_id", "BR-FORK-01"),
        "camera_id": camera_id,
        "camera_name": camera_name,
        "user_id": user_id,
        "name": name,
        "confidence": round(float(confidence), 3),
        "status": "Present",
        "source": f"node_cam_{camera_id}",
        "face_crop_base64": face_crop_b64,
        "timestamp": datetime.now().isoformat()
    }
    try:
        cloud_event_queue.put_nowait(event_payload)
    except queue.Full:
        logger.warning("Cloud sync queue full, dropping oldest event")


# ==============================================================================
# FAST VECTORIZED FACE MATCHING
# ==============================================================================

def sync_client_profiles_from_cloud(org_id: str, branch_id: str, cloud_url: str) -> int:
    """
    Sync client employees from Cloud Portal and Main Cloud for this organization & branch.
    Ensures newly registered staff from Cloud Staff Directory immediately appear locally.
    Crucially: If staff are deleted from Cloud (or count is 0), immediately deletes local
    user records and cached face embeddings so local matches Cloud 100%.
    """
    if not org_id:
        return 0

    cloud_clean = cloud_url.rstrip('/') if cloud_url else "https://quanteye-main-dashboard-production.up.railway.app"
    portal_url = CONFIG.get("client_portal_url", "https://quanteye-ai-attendence-production.up.railway.app").rstrip('/')
    imported_count = 0
    portal_queried_successfully = False

    # 1. Fetch staff members directly from Cloud Client Portal (/api/users)
    try:
        req_url = f"{portal_url}/api/users"
        req = urllib.request.Request(req_url, headers={
            "User-Agent": "QuantEye-Node",
            "X-Client-Auth": "true",
            "X-Organization-Id": org_id,
            "X-Branch-Id": branch_id or "DEFAULT_BRANCH"
        }, method="GET")
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                users = data.get("users", [])
                portal_queried_successfully = True
                
                with sqlite3.connect(db.DB_PATH) as conn:
                    cur = conn.cursor()
                    try: cur.execute("ALTER TABLE users ADD COLUMN organization_id TEXT")
                    except Exception: pass
                    try: cur.execute("ALTER TABLE users ADD COLUMN branch_id TEXT")
                    except Exception: pass
                    try: cur.execute("ALTER TABLE users ADD COLUMN phone TEXT")
                    except Exception: pass
                    try: cur.execute("ALTER TABLE users ADD COLUMN position TEXT")
                    except Exception: pass

                    cloud_ids = [int(u.get("id")) for u in users if u.get("id") is not None]

                    if not cloud_ids:
                        # Cloud has 0 staff for this organization -> purge all local staff and embeddings
                        cur.execute("""
                            DELETE FROM embeddings 
                            WHERE user_id IN (
                                SELECT id FROM users 
                                WHERE organization_id = ? OR organization_id IS NULL OR organization_id = '' OR organization_id = 'DEFAULT_ORG'
                            )
                        """, (org_id,))
                        cur.execute("""
                            DELETE FROM users 
                            WHERE organization_id = ? OR organization_id IS NULL OR organization_id = '' OR organization_id = 'DEFAULT_ORG'
                        """, (org_id,))
                        conn.commit()
                        logger.info(f"[PORTAL SYNC] Cloud has 0 staff profiles. Cleared all local users & embeddings for {org_id}.")
                        reload_face_cache()
                        return 0

                    # Cloud has active staff -> delete any local staff who no longer exist in Cloud
                    placeholders = ','.join('?' for _ in cloud_ids)
                    cur.execute(f"""
                        DELETE FROM embeddings 
                        WHERE user_id IN (
                            SELECT id FROM users 
                            WHERE (organization_id = ? OR organization_id IS NULL OR organization_id = '' OR organization_id = 'DEFAULT_ORG')
                              AND id NOT IN ({placeholders})
                        )
                    """, [org_id] + cloud_ids)
                    cur.execute(f"""
                        DELETE FROM users 
                        WHERE (organization_id = ? OR organization_id IS NULL OR organization_id = '' OR organization_id = 'DEFAULT_ORG')
                          AND id NOT IN ({placeholders})
                    """, [org_id] + cloud_ids)

                    # Upsert active staff from Cloud
                    for u in users:
                        u_id = u.get("id")
                        u_name = (u.get("name") or f"User-{u_id}").strip()
                        u_email = (u.get("email") or "").strip()
                        u_phone = (u.get("phone") or "").strip()
                        u_dept = (u.get("department") or "General").strip()
                        u_pos = (u.get("position") or "Staff Member").strip()
                        u_code = (u.get("notes") or f"EMP-{u_id}").strip()
                        u_org = u.get("organization_id") or org_id
                        u_branch = u.get("branch_id") or branch_id or "DEFAULT_BRANCH"
                        u_active = int(u.get("active", 1))

                        cur.execute("SELECT id FROM users WHERE id = ? OR name = ?", (u_id, u_name))
                        row = cur.fetchone()
                        if row:
                            cur.execute("""
                                UPDATE users 
                                SET name = ?, email = ?, phone = ?, department = ?, position = ?, notes = ?, active = ?, organization_id = ?, branch_id = ?
                                WHERE id = ?
                            """, (u_name, u_email, u_phone, u_dept, u_pos, u_code, u_active, u_org, u_branch, row[0]))
                        else:
                            cur.execute("""
                                INSERT INTO users (id, name, email, phone, department, position, notes, active, organization_id, branch_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (u_id, u_name, u_email, u_phone, u_dept, u_pos, u_code, u_active, u_org, u_branch))
                        imported_count += 1
                    conn.commit()
                logger.info(f"[PORTAL SYNC] Synced {imported_count} staff profiles from Cloud Portal for {org_id}")
    except Exception as pe:
        logger.warning(f"[PORTAL SYNC] Note querying portal users: {pe}")

    # If Cloud Portal query succeeded, do not run secondary or local fallback!
    if portal_queried_successfully:
        return imported_count

    # 2. Secondary fallback: Fetch branch employees from Main Cloud API
    if branch_id:
        try:
            req_url = f"{cloud_clean}/api/branches/{branch_id}/employees?organization_id={org_id}"
            req = urllib.request.Request(req_url, headers={"User-Agent": "QuantEye-Node"}, method="GET")
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    employees = data.get("employees", [])
                    with sqlite3.connect(db.DB_PATH) as conn:
                        cur = conn.cursor()
                        for emp in employees:
                            emp_name = (emp.get("name") or "Employee").strip()
                            emp_code = emp.get("employee_code") or f"EMP-{emp.get('id')}"
                            dept = emp.get("department") or "General"
                            pos = emp.get("position") or "Staff Member"

                            cur.execute("""
                                SELECT id FROM users 
                                WHERE (notes = ? OR name = ?) AND (organization_id = ? OR organization_id IS NULL OR organization_id = '')
                            """, (emp_code, emp_name, org_id))
                            row = cur.fetchone()
                            if row:
                                cur.execute("""
                                    UPDATE users SET name = ?, department = ?, position = ?, organization_id = ?, branch_id = ?, active = 1
                                    WHERE id = ?
                                """, (emp_name, dept, pos, org_id, branch_id, row[0]))
                            else:
                                cur.execute("""
                                    INSERT INTO users (name, department, position, notes, active, organization_id, branch_id)
                                    VALUES (?, ?, ?, ?, 1, ?, ?)
                                """, (emp_name, dept, pos, emp_code, org_id, branch_id))
                            imported_count += 1
                        conn.commit()
        except Exception as ce:
            logger.debug(f"[CLOUD SYNC] Note querying cloud employees: {ce}")

    return imported_count


def reload_face_cache():
    global EMBEDDING_CACHE, ENROLLED_MATRIX, ENROLLED_USER_IDS, ENROLLED_USER_METADATA
    logger.info("[*] Refreshing in-memory facial embedding cache from database...")
    new_cache = {}
    matrix_rows = []
    user_ids = []
    user_metadata = []

    target_org = CONFIG.get("organization_id")
    target_branch = CONFIG.get("branch_id")

    # Scope users strictly to authenticated tenant
    all_users = db.get_all_users()
    for user in all_users:
        u_org = user.get('organization_id')
        u_branch = user.get('branch_id')
        
        # Enforce tenant filtering with support for tenant branch schedules
        if target_org:
            if u_org and str(u_org).strip() not in (str(target_org).strip(), 'DEFAULT_ORG', 'all', 'ALL'):
                continue
        if target_branch:
            if u_branch and str(u_branch).strip() not in (str(target_branch).strip(), 'DEFAULT_BRANCH', 'all', 'ALL', ''):
                continue

        user_embeddings = db.get_embeddings_for_user(user['id'])
        if len(user_embeddings) == 0:
            continue
        user_embs = [np.array(emb['embedding'], dtype=np.float32) for emb in user_embeddings]
        aggregate_emb = fp.compute_aggregate_embedding(user_embs)
        if aggregate_emb is None:
            continue

        norm = float(np.linalg.norm(aggregate_emb))
        if norm > 1e-6:
            norm_emb = np.divide(aggregate_emb, norm, dtype=np.float32)
        else:
            norm_emb = aggregate_emb.astype(np.float32)

        meta = {
            'id': user['id'],
            'name': user['name'],
            'department': user.get('department', '')
        }

        new_cache[user['id']] = {
            'name': user['name'],
            'department': user.get('department', ''),
            'aggregate_embedding': norm_emb
        }
        matrix_rows.append(norm_emb)
        user_ids.append(user['id'])
        user_metadata.append(meta)

    with cache_lock:
        EMBEDDING_CACHE = new_cache
        if matrix_rows:
            ENROLLED_MATRIX = np.vstack(matrix_rows).astype(np.float32)
            ENROLLED_USER_IDS = user_ids
            ENROLLED_USER_METADATA = user_metadata
            logger.info(f"[OK] Cached {len(user_ids)} profiles in vectorized matrix for instant live recognition ({target_org or 'ALL'}).")
        else:
            ENROLLED_MATRIX = None
            ENROLLED_USER_IDS = []
            ENROLLED_USER_METADATA = []
            logger.info(f"[INFO] 0 profiles found for tenant {target_org} / {target_branch}.")


# ==============================================================================
# ==============================================================================
# HIGH PERFORMANCE DUAL-THREADED CAMERA STREAM READER & AI ENGINE
# ==============================================================================

# Global lock to serialize InsightFace / ONNX Runtime C++ execution across cameras
# Completely prevents multi-threaded ONNX Runtime memory corruption and C++ abort crashes
AI_INFERENCE_LOCK = threading.Lock()

class NodeCameraReader:
    def __init__(self, camera_id: str, camera_name: str, rtsp_source: Any):
        self.camera_id = str(camera_id)
        self.camera_name = camera_name
        self.rtsp_source = rtsp_source
        self.is_webcam = isinstance(rtsp_source, int) or (isinstance(rtsp_source, str) and rtsp_source.isdigit())
        self.cap = None
        self.cap_lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.tracked_faces = {}
        self.next_track_id = 1
        self.last_logged = {}  # {user_id: timestamp}
        self.running = True
        self.lock = threading.Lock()

        if self.is_webcam:
            self.rtsp_url = int(rtsp_source)
        else:
            self.rtsp_url = str(rtsp_source)

        # Thread 1: Grabber (handles opening capture in background without blocking Flask)
        threading.Thread(target=self._grabber_loop, daemon=True).start()
        # Thread 2: AI Processor
        threading.Thread(target=self._ai_loop, daemon=True).start()

    def _open_capture(self):
        try:
            if self.is_webcam:
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.rtsp_url)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;5000000|timeout;5000000"
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"[CAMERA CONNECTED] {self.camera_name} ({self.camera_id}) opened successfully.")
                return True
        except Exception as e:
            logger.debug(f"Capture open failed for {self.camera_id}: {e}")
            self.cap = None
        return False

    def _grabber_loop(self):
        last_reconnect_attempt = 0
        consecutive_failures = 0
        while self.running:
            try:
                need_open = False
                with self.cap_lock:
                    if not self.cap or not self.cap.isOpened():
                        need_open = True

                if need_open:
                    now = time.time()
                    if now - last_reconnect_attempt > 2.0:
                        last_reconnect_attempt = now
                        with self.cap_lock:
                            if self.running:
                                self._open_capture()
                    time.sleep(0.3)
                    continue

                frame = None
                ret = False
                with self.cap_lock:
                    if not self.running:
                        break
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()

                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    with self.lock:
                        self.latest_frame = frame
                    time.sleep(0.01)
                else:
                    consecutive_failures += 1
                    # Only release and trigger reconnect if continuous failure persists (15+ frames / ~1.5s)
                    if consecutive_failures > 15:
                        with self.cap_lock:
                            if self.cap is not None:
                                try:
                                    self.cap.release()
                                except Exception:
                                    pass
                                self.cap = None
                        consecutive_failures = 0
                        last_reconnect_attempt = time.time()
                    time.sleep(0.04)
            except Exception as e:
                logger.debug(f"Grabber loop exception in {self.camera_id}: {e}")
                time.sleep(0.2)

    def _ai_loop(self):
        last_ai_time = 0
        while self.running:
            now = time.time()
            if (now - last_ai_time) < 0.08:  # ~12 FPS AI check is optimal for real-time attendance
                time.sleep(0.015)
                continue

            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None or frame.size == 0:
                time.sleep(0.03)
                continue

            last_ai_time = now

            # Non-blocking lock: if another camera is currently executing InsightFace,
            # this camera skips this single frame rather than piling up threads
            # in ONNX Runtime/CUDA C++ engine. This completely prevents multi-camera C++ crashes!
            if not AI_INFERENCE_LOCK.acquire(blocking=False):
                time.sleep(0.01)
                continue

            try:
                face_results = fp.detect_and_extract_insightface(frame)
                current_detections = []

                for face in face_results:
                    x1, y1, x2, y2 = face['bbox']
                    emb = face.get('embedding')
                    name = "Unknown"
                    conf = 0.0
                    user_id = None

                    if emb is not None and ENROLLED_MATRIX is not None:
                        sims = np.dot(ENROLLED_MATRIX, emb)
                        best_idx = np.argmax(sims)
                        best_sim = float(sims[best_idx])
                        if best_sim >= CONFIG.get("face_match_threshold", 0.45):
                            user_id = ENROLLED_USER_IDS[best_idx]
                            meta = ENROLLED_USER_METADATA[best_idx]
                            name = meta['name']
                            conf = best_sim

                            # Throttle check: Log & Push to Cloud once per 5 seconds per user
                            last_t = self.last_logged.get(user_id, 0)
                            if (now - last_t) >= 5.0:
                                self.last_logged[user_id] = now
                                
                                # Crop face thumbnail
                                face_crop_b64 = None
                                try:
                                    h, w = frame.shape[:2]
                                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                                    if crop.size > 0:
                                        crop_resized = cv2.resize(crop, (80, 80))
                                        _, buf = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                        face_crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
                                except Exception:
                                    pass

                                # Push immediately to Cloud!
                                dispatch_attendance_event_to_cloud(
                                    user_id=user_id,
                                    name=name,
                                    confidence=conf,
                                    camera_id=self.camera_id,
                                    camera_name=self.camera_name,
                                    face_crop_b64=face_crop_b64
                                )

                                # Log attendance locally with shift & grace validation
                                try:
                                    db.log_attendance(
                                        user_id=user_id,
                                        detected_name=name,
                                        confidence=conf,
                                        source='camera',
                                        location=self.camera_name
                                    )
                                except Exception as le:
                                    logger.debug(f"Local attendance note: {le}")

                                # Update Node's live detection ticker
                                with detections_lock:
                                    LATEST_DETECTIONS_FEED.insert(0, {
                                        "user_id": user_id,
                                        "name": name,
                                        "confidence": int(conf * 100),
                                        "camera_name": self.camera_name,
                                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                                        "face_crop": face_crop_b64
                                    })
                                    if len(LATEST_DETECTIONS_FEED) > 50:
                                        LATEST_DETECTIONS_FEED.pop()

                    current_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "name": name,
                        "confidence": conf
                    })

                with self.lock:
                    self.latest_detections = current_detections

            except Exception as e:
                logger.error(f"AI inference error in camera {self.camera_id}: {e}")
            finally:
                AI_INFERENCE_LOCK.release()

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)

        # Draw real-time bounding boxes and smooth badges
        for d in detections:
            bbox = d.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            name = d.get('name', 'Unknown')
            conf = d.get('confidence', 0.0)
            is_match = name != "Unknown"
            # Emerald green for recognized staff, vibrant orange/cyan for verifying
            color = (0, 220, 110) if is_match else (0, 140, 255)

            # Draw rounded box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({int(conf * 100)}%)" if is_match else "Verifying..."
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + tw + 12, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return True, frame

    def release(self):
        self.running = False
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None


def fetch_all_cloud_cameras(org_id: str, branch_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all camera streams strictly configured for this organization & branch.
    If cameras were deleted on Cloud, returns empty list [] to clear local streams immediately.
    """
    cloud_url = CONFIG.get('cloud_url', 'https://quanteye-main-dashboard-production.up.railway.app').rstrip('/')
    portal_url = CONFIG.get("client_portal_url", "https://quanteye-ai-attendence-production.up.railway.app").rstrip('/')

    collected_cams: Dict[str, Dict[str, Any]] = {}
    query_successful = False

    def _add_cam(c: Any):
        if not c or not isinstance(c, dict):
            return
        cam_id = str(c.get("camera_id") or "").strip()
        raw_id = str(c.get("id") or "").strip()
        url = str(c.get("rtsp_url") or "").strip()
        if not url:
            return

        primary_id = cam_id or raw_id or f"cam_{len(collected_cams)+1}"
        if primary_id in collected_cams:
            return

        c_copy = dict(c)
        c_copy["id"] = primary_id
        c_copy["camera_id"] = primary_id
        c_copy["name"] = c.get("name") or c.get("camera_name") or f"Camera {primary_id}"
        c_copy["camera_name"] = c_copy["name"]
        c_copy["rtsp_url"] = url
        collected_cams[primary_id] = c_copy

    # 1. Fetch from Client Portal Cloud (branch & org filtered)
    if org_id and branch_id:
        try:
            req_url = f"{portal_url}/api/cameras?active_only=true&organization_id={urllib.parse.quote(org_id)}&branch_id={urllib.parse.quote(branch_id)}"
            req = urllib.request.Request(req_url, headers={
                "User-Agent": "QuantEye-Node",
                "X-Client-Auth": "true",
                "X-Organization-Id": org_id,
                "X-Branch-Id": branch_id
            }, method="GET")
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    query_successful = True
                    p_cams = json.loads(resp.read().decode('utf-8'))
                    if isinstance(p_cams, list):
                        for c in p_cams:
                            _add_cam(c)
        except Exception as pe:
            logger.debug(f"[PORTAL SYNC] Client portal camera query note: {pe}")

    # 2. Fetch from Cloud Main Dashboard
    if org_id and branch_id:
        try:
            cloud_cams_url = f"{cloud_url}/api/branches/{branch_id}/cameras?organization_id={urllib.parse.quote(org_id)}"
            req = urllib.request.Request(cloud_cams_url, headers={"User-Agent": "QuantEye-Node"}, method="GET")
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    query_successful = True
                    data = json.loads(resp.read().decode('utf-8'))
                    for c in data.get("cameras", []):
                        _add_cam(c)
        except Exception as ce:
            logger.debug(f"[CLOUD SYNC] Main cloud camera query note: {ce}")

    # If remote queries succeeded, return the exact live camera list (even if empty [])
    if query_successful:
        return list(collected_cams.values())

    # Only if network was completely offline, fallback to existing local config
    for c in CONFIG.get("cameras", []):
        _add_cam(c)

    return list(collected_cams.values())


def init_camera_streams():
    """Fetch camera list in real time from Cloud or Client Portal and update streams."""
    global ACTIVE_CAMERA_READERS
    org_id = CONFIG.get("organization_id")
    branch_id = CONFIG.get("branch_id")

    cloud_cams = fetch_all_cloud_cameras(org_id, branch_id)
    CONFIG["cameras"] = cloud_cams
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=2)
    except Exception:
        pass

    cameras_to_use = cloud_cams

    with readers_lock:
        valid_cams = {}
        for cam in cameras_to_use:
            cid = str(cam.get("id") or cam.get("camera_id"))
            cname = cam.get("name") or cam.get("camera_name", f"Camera {cid}")
            source = cam.get("rtsp_url", "")
            cam_type = str(cam.get("camera_type", "rtsp")).lower()
            if not source or (str(source).strip() in ("0", "") and cam_type != "webcam"):
                continue
            valid_cams[cid] = (cname, source)

        # 1. Stop and remove stale / deleted cameras
        stale_cids = [cid for cid in list(ACTIVE_CAMERA_READERS.keys()) if cid not in valid_cams]
        for cid in stale_cids:
            reader = ACTIVE_CAMERA_READERS.pop(cid, None)
            if reader:
                try:
                    reader.release()
                    logger.info(f"[*] Safely released deleted camera stream: {cid}")
                except Exception as e:
                    logger.debug(f"Error releasing camera {cid}: {e}")

        # 2. Add only new cameras or update changed sources
        for cid, (cname, source) in valid_cams.items():
            existing = ACTIVE_CAMERA_READERS.get(cid)
            if existing is None:
                try:
                    ACTIVE_CAMERA_READERS[cid] = NodeCameraReader(cid, cname, source)
                    logger.info(f"Started camera stream reader: {cname} ({cid}) -> {source}")
                except Exception as e:
                    logger.error(f"Failed to start camera {cid}: {e}")
            elif str(existing.rtsp_source) != str(source):
                try:
                    existing.release()
                    ACTIVE_CAMERA_READERS[cid] = NodeCameraReader(cid, cname, source)
                    logger.info(f"Updated camera stream reader: {cname} ({cid}) -> {source}")
                except Exception as e:
                    logger.error(f"Failed to update camera {cid}: {e}")

_last_cloud_cam_hash = None

def _background_camera_sync_loop():
    """Poll Cloud every 3 seconds for real-time camera additions or deletions."""
    global _last_cloud_cam_hash
    while True:
        try:
            time.sleep(3.0)
            org_id = CONFIG.get("organization_id")
            branch_id = CONFIG.get("branch_id")
            if org_id and branch_id:
                cams = fetch_all_cloud_cameras(org_id, branch_id)
                cam_hash = json.dumps([(c.get("id") or c.get("camera_id"), c.get("rtsp_url")) for c in cams], sort_keys=True)
                if _last_cloud_cam_hash is not None and cam_hash != _last_cloud_cam_hash:
                    logger.info(f"[REAL-TIME SYNC] Cloud camera changes detected ({len(cams)} cameras). Reloading streams...")
                    init_camera_streams()
                _last_cloud_cam_hash = cam_hash
        except Exception:
            pass

threading.Thread(target=_background_camera_sync_loop, daemon=True).start()


# ==============================================================================
# FLASK LIVE UI & STREAM ROUTES
# ==============================================================================

@app.route('/')
def live_node_page():
    """Dedicated Live AI Attendance Window with Dynamic Camera Grid."""
    return render_template('live_node.html', config=CONFIG)


@app.route('/api/node/health')
def api_node_health():
    """Healthcheck endpoint for standalone packaging & node monitor."""
    return jsonify({
        "status": "healthy",
        "service": "QuantEye Client AI Node",
        "port": 5005,
        "linked": CONFIG.get("linked", False),
        "organization_id": CONFIG.get("organization_id"),
        "branch_id": CONFIG.get("branch_id")
    })


@app.route('/api/node/status')
def api_node_status():
    """Status endpoint for node state and active camera count."""
    with readers_lock:
        active_count = len(ACTIVE_CAMERA_READERS)
    return jsonify({
        "success": True,
        "linked": CONFIG.get("linked", False),
        "organization_id": CONFIG.get("organization_id"),
        "branch_id": CONFIG.get("branch_id"),
        "active_cameras": active_count,
        "face_match_threshold": CONFIG.get("face_match_threshold", 0.45)
    })


@app.route('/stream/<camera_id>')
def camera_stream_feed(camera_id):
    """Multipart video feed for each camera box."""
    with readers_lock:
        reader = ACTIVE_CAMERA_READERS.get(camera_id)

    if not reader:
        return jsonify({"error": "Camera not found"}), 404

    def generate():
        blank_bytes = None
        try:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CONNECTING CAMERA...", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            _, blank_buf = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
            blank_bytes = blank_buf.tobytes()
        except Exception:
            pass

        try:
            while True:
                if not reader or not reader.running:
                    break

                ret, frame = reader.get_annotated_frame()
                if not ret or frame is None or frame.size == 0:
                    if blank_bytes:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
                    time.sleep(0.4)
                    continue

                try:
                    h, w = frame.shape[:2]
                    # Downscale 1080p/4K frames slightly to max 960px width for smooth grid streaming
                    if w > 960:
                        scale = 960.0 / w
                        frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

                    success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not success:
                        time.sleep(0.03)
                        continue
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except (GeneratorExit, ConnectionResetError, BrokenPipeError):
                    break
                except Exception:
                    time.sleep(0.04)
                    continue

                time.sleep(0.033)
        except (GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.debug(f"Stream generator completed for {camera_id}: {e}")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/node/cameras')
def get_node_cameras():
    """Return active cameras for frontend grid layout."""
    with readers_lock:
        cams = [
            {"id": cid, "name": r.camera_name, "stream_url": f"/stream/{cid}"}
            for cid, r in ACTIVE_CAMERA_READERS.items()
        ]
    return jsonify({"success": True, "cameras": cams})


@app.route('/api/node/detections')
def get_node_detections():
    """Real-time detection ticker."""
    with detections_lock:
        feed = list(LATEST_DETECTIONS_FEED)
    return jsonify({"success": True, "detections": feed})


@app.route('/api/node/cloud-status')
def get_node_cloud_status():
    """Check connectivity to Cloud Main Dashboard."""
    cloud_url = CONFIG.get('cloud_url', 'http://127.0.0.1:5100').rstrip('/')
    is_connected = False
    try:
        health_url = f"{cloud_url}/api/integration/local/status"
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                is_connected = True
    except Exception:
        is_connected = False
    return jsonify({
        "success": True,
        "cloud_connected": is_connected,
        "cloud_url": cloud_url,
        "organization_id": CONFIG.get("organization_id"),
        "branch_id": CONFIG.get("branch_id")
    })


def fetch_active_cloud_shift(org_id: str = None, branch_id: str = None) -> Dict[str, Any]:
    """
    Fetch active shifts in real time from Cloud Client Portal (/api/shifts).
    Calculates the currently active or default shift based on current time.
    """
    portal_url = CONFIG.get("client_portal_url", "https://quanteye-ai-attendence-production.up.railway.app").rstrip('/')
    shifts = []

    # 1. Fetch from Cloud Client Portal
    try:
        req_url = f"{portal_url}/api/shifts"
        if org_id:
            req_url += f"?organization_id={urllib.parse.quote(org_id)}"
        req = urllib.request.Request(req_url, headers={
            "User-Agent": "QuantEye-Node",
            "X-Client-Auth": "true"
        }, method="GET")
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                shifts = data.get("shifts", [])
    except Exception as e:
        logger.debug(f"[SHIFT SYNC] Could not fetch shifts from cloud ({e}). Falling back to local.")

    # 2. Fallback to local DB if available
    if not shifts:
        try:
            with sqlite3.connect(db.DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute("SELECT id, name, start_time, end_time, grace_period_minutes, COALESCE(is_default, 1) FROM shifts")
                for r in cur.fetchall():
                    shifts.append({
                        "id": r[0], "name": r[1], "start_time": r[2], "end_time": r[3],
                        "grace_period_minutes": r[4], "is_default": bool(r[5])
                    })
        except Exception:
            pass

    if not shifts:
        return {
            "name": "Standard Shift",
            "start_time": "09:00",
            "end_time": "17:00",
            "formatted": "09:00 AM - 05:00 PM",
            "display": "Standard Shift (09:00 AM - 05:00 PM)",
            "is_active_now": True,
            "grace_period_minutes": 15
        }

    now = datetime.now()
    cur_mins = now.hour * 60 + now.minute

    def to_mins(t_str):
        try:
            parts = str(t_str).strip().split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    def format_12h(t_str):
        try:
            parts = str(t_str).strip().split(':')
            h, m = int(parts[0]), int(parts[1])
            period = "AM" if h < 12 else "PM"
            h12 = h % 12
            if h12 == 0: h12 = 12
            return f"{h12:02d}:{m:02d} {period}"
        except Exception:
            return str(t_str)

    active_shift = None
    for s in shifts:
        s_start = to_mins(s.get("start_time", "09:00"))
        s_end = to_mins(s.get("end_time", "17:00"))
        if s_start <= s_end:
            if s_start <= cur_mins <= s_end:
                active_shift = s
                break
        else: # Overnight shift
            if cur_mins >= s_start or cur_mins <= s_end:
                active_shift = s
                break

    if not active_shift:
        default_shifts = [s for s in shifts if s.get("is_default")]
        active_shift = default_shifts[0] if default_shifts else shifts[0]

    s_name = active_shift.get("name", "Shift")
    start_12h = format_12h(active_shift.get("start_time", "09:00"))
    end_12h = format_12h(active_shift.get("end_time", "17:00"))
    grace = active_shift.get("grace_period_minutes", 15)

    return {
        "id": active_shift.get("id"),
        "name": s_name,
        "start_time": active_shift.get("start_time"),
        "end_time": active_shift.get("end_time"),
        "start_12h": start_12h,
        "end_12h": end_12h,
        "formatted": f"{start_12h} - {end_12h}",
        "grace_period_minutes": grace,
        "display": f"{s_name} ({start_12h} - {end_12h})"
    }


@app.route('/api/node/shift', methods=['GET'])
def get_node_active_shift():
    """Real-time active shift timing endpoint for header display."""
    org_id = CONFIG.get("organization_id")
    branch_id = CONFIG.get("branch_id")
    shift_info = fetch_active_cloud_shift(org_id, branch_id)
    return jsonify({
        "success": True,
        "shift": shift_info
    })


@app.route('/api/node/logout', methods=['POST'])
def api_node_logout():
    """
    Log out from current organization/branch.
    Clears runtime state, stops active camera stream readers, and resets config_node.json.
    """
    global CONFIG, ACTIVE_CAMERA_READERS
    with readers_lock:
        for cid, reader in list(ACTIVE_CAMERA_READERS.items()):
            try:
                reader.release()
            except Exception:
                pass
        ACTIVE_CAMERA_READERS.clear()

    CONFIG['linked'] = False
    CONFIG['organization_id'] = None
    CONFIG['organization_name'] = None
    CONFIG['branch_id'] = None
    CONFIG['branch_name'] = None
    CONFIG['branch_key'] = None
    CONFIG['cameras'] = []

    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        logger.warning(f"Error persisting reset config: {e}")

    logger.info("[NODE LOGOUT] Successfully logged out from current organization.")
    return jsonify({
        "success": True,
        "message": "Logged out successfully. Node is ready to bind to another organization."
    })


@app.route('/api/node/login-auth', methods=['POST'])
def api_node_login_auth():
    """
    Step 1: Authenticate Client credentials + Branch Key against Cloud.
    Accepts: username/email, password, branch_key, cloud_url.
    Authenticates against Cloud `/api/client/authenticate` and `/api/edge/validate-key`.
    Saves configuration, binds tenant, triggers camera reload and profile sync.
    """
    global CONFIG
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or data.get('email') or '').strip()
    password = (data.get('password') or '').strip()
    branch_key = (data.get('branch_key') or '').strip()
    cloud_url = (data.get('cloud_url') or CONFIG.get('cloud_url', 'http://127.0.0.1:5100')).strip().rstrip('/')

    if not branch_key and not username:
        return jsonify({"success": False, "error": "Username/Email and Branch Key are required"}), 400

    try:
        # Step 1.1: If username and password provided, authenticate against Cloud
        auth_info = {}
        if username and password:
            auth_url = f"{cloud_url}/api/client/authenticate"
            auth_body = json.dumps({
                "username": username,
                "password": password,
                "branch_key": branch_key
            }).encode('utf-8')
            req_auth = urllib.request.Request(auth_url, data=auth_body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req_auth, timeout=5.0) as resp_auth:
                    if resp_auth.status == 200:
                        auth_info = json.loads(resp_auth.read().decode('utf-8'))
                        # If branch_key wasn't provided, use the one associated with the user's branch
                        if not branch_key and auth_info.get('branch_key'):
                            branch_key = auth_info['branch_key']
            except urllib.error.HTTPError as he:
                err_text = "Invalid username or password"
                try:
                    err_text = json.loads(he.read().decode('utf-8')).get('error', err_text)
                except Exception:
                    pass
                return jsonify({"success": False, "error": err_text}), 401

        if not branch_key:
            return jsonify({"success": False, "error": "Branch Key is required"}), 400

        # Step 1.2: Validate Branch Key to get cameras and branch status
        val_url = f"{cloud_url}/api/edge/validate-key"
        val_payload = json.dumps({"branch_key": branch_key}).encode('utf-8')
        req_val = urllib.request.Request(val_url, data=val_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req_val, timeout=5.0) as resp_val:
            if resp_val.status != 200:
                return jsonify({"success": False, "error": "Branch Key rejected by Cloud"}), 401
            val_data = json.loads(resp_val.read().decode('utf-8'))

        org_id = val_data['organization_id']
        branch_id = val_data['branch_id']
        org_name = val_data['organization_name']
        branch_name = val_data['branch_name']
        cameras = val_data.get('cameras', [])

        # Update runtime config
        CONFIG['linked'] = True
        CONFIG['organization_id'] = org_id
        CONFIG['organization_name'] = org_name
        CONFIG['branch_id'] = branch_id
        CONFIG['branch_name'] = branch_name
        CONFIG['branch_key'] = branch_key
        CONFIG['cloud_url'] = cloud_url
        CONFIG['username'] = username
        CONFIG['cameras'] = cameras

        # Persist securely to config_node.json
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=2)

        # Synchronize employee profiles & face embeddings for this tenant
        sync_client_profiles_from_cloud(org_id, branch_id, cloud_url)
        reload_face_cache()

        # Reload camera streams dynamically
        threading.Thread(target=init_camera_streams, daemon=True).start()

        return jsonify({
            "success": True,
            "linked": True,
            "organization_id": org_id,
            "organization_name": org_name,
            "branch_id": branch_id,
            "branch_name": branch_name,
            "camera_count": len(cameras),
            "message": f"Successfully activated {org_name} • {branch_name}"
        }), 200

    except urllib.error.HTTPError as he:
        try:
            err_body = json.loads(he.read().decode('utf-8'))
            return jsonify({"success": False, "error": err_body.get('error', f"HTTP {he.code}")}), he.code
        except Exception:
            return jsonify({"success": False, "error": f"Server rejected login (HTTP {he.code})"}), he.code
    except Exception as e:
        return jsonify({"success": False, "error": f"Connection error: {str(e)}"}), 502


@app.route('/api/node/activate-key', methods=['POST'])
def api_node_activate_key():
    """
    Activate the Standalone Edge Node with an Organization/Branch Key.
    Delegates to full api_node_login_auth flow.
    """
    return api_node_login_auth()


@app.route('/api/node/cameras/refresh', methods=['POST', 'GET'])
def api_node_cameras_refresh():
    """Dynamically re-pull cloud camera settings and reload local streams."""
    try:
        init_camera_streams()
        with readers_lock:
            cams = [
                {"id": cid, "name": r.camera_name, "stream_url": f"/stream/{cid}"}
                for cid, r in ACTIVE_CAMERA_READERS.items()
            ]
        return jsonify({
            "success": True,
            "message": f"Successfully re-synced {len(cams)} cameras from Cloud",
            "camera_count": len(cams),
            "cameras": cams
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/employees', methods=['GET'])
def api_node_list_employees():
    """
    List enrolled employees strictly for this configured branch/node with embedding status.
    """
    org_id = CONFIG.get("organization_id", "")
    branch_id = CONFIG.get("branch_id", "")
    try:
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            # Auto-ensure columns
            try:
                cur.execute("ALTER TABLE users ADD COLUMN organization_id TEXT")
            except Exception: pass
            try:
                cur.execute("ALTER TABLE users ADD COLUMN branch_id TEXT")
            except Exception: pass

            if org_id and branch_id:
                cur.execute("""
                    SELECT u.id, u.name, COALESCE(u.department, ''), COALESCE(u.position, ''), 
                           COALESCE(u.notes, ''), COUNT(e.id) as emb_count
                    FROM users u
                    LEFT JOIN embeddings e ON u.id = e.user_id
                    WHERE u.organization_id = ? AND u.branch_id = ?
                    GROUP BY u.id
                    ORDER BY u.id DESC
                """, (org_id, branch_id))
            else:
                cur.execute("""
                    SELECT u.id, u.name, COALESCE(u.department, ''), COALESCE(u.position, ''), 
                           COALESCE(u.notes, ''), COUNT(e.id) as emb_count
                    FROM users u
                    LEFT JOIN embeddings e ON u.id = e.user_id
                    GROUP BY u.id
                    ORDER BY u.id DESC
                """)
            rows = cur.fetchall()
            employees = [{
                "id": r[0],
                "name": r[1],
                "department": r[2],
                "position": r[3],
                "employee_code": r[4] or f"EMP-{r[0]}",
                "embedding_count": r[5],
                "has_embedding": (r[5] > 0)
            } for r in rows]
            return jsonify({"success": True, "employees": employees, "count": len(employees)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/employees/enroll', methods=['POST'])
def api_node_enroll_employee():
    """
    Manually enroll an employee with face image:
    1. Extracts 512-dim InsightFace embedding
    2. Saves to local SQLite database scoped strictly to this branch
    3. Reloads in-memory face recognition cache
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    emp_code = (data.get("employee_code") or "").strip()
    dept = (data.get("department") or "General").strip()
    photo_b64 = data.get("image_base64") or data.get("photo")

    if not name:
        return jsonify({"success": False, "error": "Employee name is required"}), 400
    if not photo_b64:
        return jsonify({"success": False, "error": "Face photo is required for embedding generation"}), 400

    try:
        # Decode base64 image
        if ',' in photo_b64:
            photo_b64 = photo_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(photo_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Could not decode image file"}), 400

        # Extract InsightFace embedding
        extracted = fp.detect_and_extract_insightface(img)
        if not extracted or extracted[0].get('embedding') is None:
            return jsonify({"success": False, "error": "No clear face detected in the uploaded photo. Please ensure face is centered and well-lit."}), 400

        embedding_vec = extracted[0]['embedding'].tolist()

        org_id = CONFIG.get("organization_id", "")
        branch_id = CONFIG.get("branch_id", "")

        # Save to database
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            # Ensure columns
            try: cur.execute("ALTER TABLE users ADD COLUMN organization_id TEXT")
            except Exception: pass
            try: cur.execute("ALTER TABLE users ADD COLUMN branch_id TEXT")
            except Exception: pass

            # Insert or update user
            cur.execute("""
                INSERT INTO users (name, department, notes, active, organization_id, branch_id)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (name, dept, emp_code, org_id, branch_id))
            user_id = cur.lastrowid

            # Save embedding
            cur.execute(
                "INSERT INTO embeddings (user_id, embedding, source_video) VALUES (?, ?, ?)",
                (user_id, json.dumps(embedding_vec).encode(), "manual_enrollment")
            )
            conn.commit()

        # Instantly refresh memory cache
        reload_face_cache()
        logger.info(f"[ENROLL SUCCESS] Enrolled employee '{name}' (ID {user_id}) for {org_id} / {branch_id}")

        return jsonify({
            "success": True,
            "message": f"Successfully enrolled '{name}'! Face embedding generated and cached for live attendance.",
            "user_id": user_id,
            "name": name,
            "department": dept
        }), 201

    except Exception as e:
        logger.error(f"Error enrolling employee: {e}")
        return jsonify({"success": False, "error": f"Enrollment error: {str(e)}"}), 500


@app.route('/api/node/embeddings/import', methods=['POST'])
def api_node_import_embeddings():
    """
    Import embeddings file (JSON) to activate face recognition for specific employees.
    Supports formats:
    [
      { "name": "John Doe", "employee_code": "EMP-01", "embedding": [...] }, ...
    ]
    or dict mapping:
    {
      "EMP-01": [...], or "John Doe": [...]
    }
    """
    data = request.get_json(silent=True) or {}
    items = data.get("embeddings") or data.get("profiles") or data
    if not items:
        return jsonify({"success": False, "error": "No embeddings data found in payload"}), 400

    org_id = CONFIG.get("organization_id", "")
    branch_id = CONFIG.get("branch_id", "")
    imported = 0
    errors = []

    try:
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            try: cur.execute("ALTER TABLE users ADD COLUMN organization_id TEXT")
            except Exception: pass
            try: cur.execute("ALTER TABLE users ADD COLUMN branch_id TEXT")
            except Exception: pass

            # Handle list of employee embedding records
            if isinstance(items, list):
                for item in items:
                    name = (item.get("name") or "").strip()
                    emp_code = (item.get("employee_code") or item.get("id") or "").strip()
                    emb = item.get("embedding")
                    dept = item.get("department") or "General"

                    if not emb or not (name or emp_code):
                        continue

                    # Match existing cloud-synced user by code or name
                    cur.execute("""
                        SELECT id, name FROM users 
                        WHERE (notes = ? OR name = ?) 
                          AND (organization_id = ? OR organization_id IS NULL OR organization_id = '')
                    """, (emp_code, name, org_id))
                    row = cur.fetchone()

                    if row:
                        u_id = row[0]
                    else:
                        cur.execute("""
                            INSERT INTO users (name, department, notes, active, organization_id, branch_id)
                            VALUES (?, ?, ?, 1, ?, ?)
                        """, (name or emp_code, dept, emp_code, org_id, branch_id))
                        u_id = cur.lastrowid

                    # Replace old embeddings for this employee and insert new
                    cur.execute("DELETE FROM embeddings WHERE user_id = ?", (u_id,))
                    cur.execute(
                        "INSERT INTO embeddings (user_id, embedding, source_video) VALUES (?, ?, ?)",
                        (u_id, json.dumps(emb).encode(), "imported_embedding")
                    )
                    imported += 1

            # Handle dict format: {"EMP-01": [512 floats]}
            elif isinstance(items, dict):
                for key, emb in items.items():
                    if not isinstance(emb, list): continue
                    # Find user by code or name
                    cur.execute("""
                        SELECT id, name FROM users 
                        WHERE (notes = ? OR name = ?) 
                          AND (organization_id = ? OR organization_id IS NULL OR organization_id = '')
                    """, (key, key, org_id))
                    row = cur.fetchone()
                    if row:
                        u_id = row[0]
                    else:
                        cur.execute("""
                            INSERT INTO users (name, department, notes, active, organization_id, branch_id)
                            VALUES (?, 'General', ?, 1, ?, ?)
                        """, (key, key, org_id, branch_id))
                        u_id = cur.lastrowid

                    cur.execute("DELETE FROM embeddings WHERE user_id = ?", (u_id,))
                    cur.execute(
                        "INSERT INTO embeddings (user_id, embedding, source_video) VALUES (?, ?, ?)",
                        (u_id, json.dumps(emb).encode(), "imported_embedding")
                    )
                    imported += 1

            conn.commit()

        reload_face_cache()
        logger.info(f"[IMPORT SUCCESS] Imported and activated {imported} employee embeddings.")

        return jsonify({
            "success": True,
            "message": f"Successfully imported {imported} employee embeddings! AI will now recognize these staff members.",
            "imported_count": imported
        }), 200

    except Exception as e:
        logger.error(f"Error importing embeddings: {e}")
        return jsonify({"success": False, "error": f"Import error: {str(e)}"}), 500


@app.route('/api/node/embeddings/export', methods=['GET'])
def api_node_export_embeddings():
    """Export all currently enrolled embeddings for this branch to JSON."""
    org_id = CONFIG.get("organization_id", "")
    branch_id = CONFIG.get("branch_id", "")
    try:
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT u.id, u.name, COALESCE(u.notes, ''), COALESCE(u.department, ''), e.embedding
                FROM users u
                JOIN embeddings e ON u.id = e.user_id
                WHERE (u.organization_id = ? AND u.branch_id = ?)
                   OR (u.organization_id IS NULL OR u.organization_id = '')
            """, (org_id, branch_id))
            rows = cur.fetchall()
            export_list = []
            for r in rows:
                try:
                    emb = json.loads(r[4].decode()) if isinstance(r[4], bytes) else json.loads(r[4])
                    export_list.append({
                        "id": r[0],
                        "name": r[1],
                        "employee_code": r[2] or f"EMP-{r[0]}",
                        "department": r[3],
                        "embedding": emb
                    })
                except Exception:
                    pass
        return jsonify({
            "success": True,
            "branch_id": branch_id,
            "organization_id": org_id,
            "count": len(export_list),
            "embeddings": export_list
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/employees/delete', methods=['POST'])
def api_node_delete_employee():
    """Remove employee and their embeddings."""
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"success": False, "error": "user_id required"}), 400
    try:
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        reload_face_cache()
        return jsonify({"success": True, "message": "Employee removed successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/employees/sync', methods=['POST'])
def api_node_sync_employees():
    """Sync profiles from Cloud or local master."""
    org_id = CONFIG.get("organization_id", "")
    branch_id = CONFIG.get("branch_id", "")
    cloud_url = CONFIG.get("cloud_url", "http://127.0.0.1:5100")
    if not org_id or not branch_id:
        return jsonify({"success": False, "error": "Node must be activated with branch key first"}), 400
    try:
        sync_client_profiles_from_cloud(org_id, branch_id, cloud_url)
        reload_face_cache()
        return jsonify({"success": True, "message": "Profiles synchronized successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/pending-staff', methods=['GET'])
def api_node_pending_staff():
    """
    Return list of staff registered from Cloud for this tenant whose face embeddings are pending.
    """
    org_id = CONFIG.get("organization_id", "")
    branch_id = CONFIG.get("branch_id", "")
    cloud_url = CONFIG.get("cloud_url", "http://127.0.0.1:5100")

    # Non-blocking background sync if online (throttled to 30s)
    if org_id:
        global _last_profile_sync_t
        if '_last_profile_sync_t' not in globals():
            _last_profile_sync_t = 0
        if time.time() - _last_profile_sync_t > 30:
            _last_profile_sync_t = time.time()
            threading.Thread(target=sync_client_profiles_from_cloud, args=(org_id, branch_id, cloud_url), daemon=True).start()

    try:
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            try: cur.execute("ALTER TABLE users ADD COLUMN organization_id TEXT")
            except Exception: pass
            try: cur.execute("ALTER TABLE users ADD COLUMN branch_id TEXT")
            except Exception: pass

            cur.execute("""
                SELECT u.id, u.name, COALESCE(u.department, 'General'), COALESCE(u.position, 'Staff Member'), 
                       COALESCE(u.notes, ''), COUNT(e.id) as emb_count, COALESCE(u.email, '')
                FROM users u
                LEFT JOIN embeddings e ON u.id = e.user_id
                WHERE (u.organization_id = ? OR u.organization_id IS NULL OR u.organization_id = '' OR u.organization_id = 'DEFAULT_ORG')
                GROUP BY u.id
                ORDER BY u.id DESC
            """, (org_id,))
            rows = cur.fetchall()

            all_staff = []
            pending_staff = []
            for r in rows:
                item = {
                    "id": r[0],
                    "name": r[1],
                    "department": r[2],
                    "position": r[3],
                    "employee_code": r[4] or f"EMP-{r[0]}",
                    "embedding_count": r[5],
                    "has_embedding": (r[5] > 0),
                    "email": r[6]
                }
                all_staff.append(item)
                if r[5] == 0:
                    pending_staff.append(item)

            return jsonify({
                "success": True,
                "pending": pending_staff,
                "pending_count": len(pending_staff),
                "total_count": len(all_staff),
                "all_staff": all_staff
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/node/train-video', methods=['POST'])
def api_node_train_video():
    """
    Train an employee's face embedding from an uploaded short video (or webcam recording).
    Extracts face frames, extracts 512D InsightFace vectors, persists embeddings in SQLite,
    and refreshes the live face recognition cache immediately.
    """
    user_id = request.form.get('user_id')
    if not user_id:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')

    if not user_id:
        return jsonify({"success": False, "error": "user_id is required"}), 400

    if 'video' not in request.files:
        return jsonify({"success": False, "error": "Video file is required"}), 400

    video_file = request.files['video']
    if not video_file or video_file.filename == '':
        return jsonify({"success": False, "error": "Valid video file is required"}), 400

    temp_path = None
    try:
        import tempfile
        ext = os.path.splitext(video_file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            temp_path = tf.name
            video_file.save(temp_path)

        # Call video face biometric processor
        enroll_res = fp.process_enrollment_video(temp_path, max_frames=120)
        if not enroll_res or not enroll_res.get('success') or not enroll_res.get('embeddings'):
            err = enroll_res.get('error') or "No clear face detected in the video. Please ensure face is well-lit, centered, and visible."
            return jsonify({"success": False, "error": err}), 400

        embeddings = enroll_res['embeddings']
        saved_count = 0

        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            for emb in embeddings:
                emb_list = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                cur.execute(
                    "INSERT INTO embeddings (user_id, embedding, source_video) VALUES (?, ?, ?)",
                    (user_id, json.dumps(emb_list).encode(), f"video_enroll_{video_file.filename}")
                )
                saved_count += 1
            conn.commit()

        # Instantly refresh in-memory face matching matrix
        reload_face_cache()
        logger.info(f"[BIOMETRIC ENROLL SUCCESS] Enrolled biometric profile for user {user_id} ({saved_count} frames)")

        # Sync registered embeddings to Cloud Portal so Staff Directory marks Training: Complete
        def _sync_embeddings_to_cloud(uid, embs, src_name):
            p_url = CONFIG.get("client_portal_url", "https://quanteye-ai-attendence-production.up.railway.app").rstrip('/')
            # 1. Downsample embeddings to top 10 representative vectors to prevent network timeouts
            sub_embs = []
            if embs:
                step = max(1, len(embs) // 10)
                for i in range(0, len(embs), step):
                    e = embs[i]
                    sub_embs.append(e.tolist() if isinstance(e, np.ndarray) else list(e))
                    if len(sub_embs) >= 10:
                        break

            # 2. Push to sync-embeddings endpoint
            try:
                sync_ep = f"{p_url}/api/users/{uid}/sync-embeddings"
                post_payload = json.dumps({
                    "embeddings": sub_embs,
                    "source": src_name
                }).encode('utf-8')
                s_req = urllib.request.Request(sync_ep, data=post_payload, headers={
                    "Content-Type": "application/json",
                    "User-Agent": "QuantEye-Node",
                    "X-Client-Auth": "true",
                    "X-Organization-Id": CONFIG.get("organization_id", ""),
                    "X-Branch-Id": CONFIG.get("branch_id", "")
                }, method="POST")
                with urllib.request.urlopen(s_req, timeout=15.0) as s_resp:
                    if s_resp.status == 200:
                        logger.info(f"[CLOUD SYNC] Synced biometric profile to Cloud Portal for user {uid}")
            except Exception as se:
                logger.warning(f"[CLOUD SYNC] Note syncing biometric profile to Cloud Portal: {se}")

            # 3. Direct mark-trained call guarantees Cloud DB marks Training Complete even on high network latency
            try:
                mark_ep = f"{p_url}/api/users/{uid}/mark-trained"
                m_req = urllib.request.Request(mark_ep, data=b"{}", headers={
                    "Content-Type": "application/json",
                    "User-Agent": "QuantEye-Node",
                    "X-Client-Auth": "true",
                    "X-Organization-Id": CONFIG.get("organization_id", ""),
                    "X-Branch-Id": CONFIG.get("branch_id", "")
                }, method="POST")
                with urllib.request.urlopen(m_req, timeout=10.0) as m_resp:
                    if m_resp.status == 200:
                        logger.info(f"[CLOUD SYNC] Successfully confirmed Training Complete on Cloud Portal for user {uid}")
            except Exception as me:
                logger.debug(f"[CLOUD SYNC] Note calling mark-trained: {me}")

        threading.Thread(target=_sync_embeddings_to_cloud, args=(user_id, embeddings, f"video_enroll_{video_file.filename}"), daemon=True).start()

        return jsonify({
            "success": True,
            "message": "Staff face biometric profile registered successfully! Live recognition is now active for this member.",
            "user_id": user_id,
            "frames_registered": saved_count
        }), 200

    except Exception as e:
        logger.error(f"Error training video: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def start_client_node(port: int = 5005):
    logger.info("Initializing QuantEye Client AI Node...")
    db.init_db()
    reload_face_cache()
    threading.Thread(target=init_camera_streams, daemon=True).start()
    logger.info(f"Starting QuantEye Client AI Node on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_client_node()
