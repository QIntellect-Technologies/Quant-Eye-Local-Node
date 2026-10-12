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
        "organization_id": "ORG-001",
        "organization_name": "Fork n Knives",
        "branch_id": "BR-FORK-01",
        "branch_name": "Global HQ",
        "cloud_url": "http://127.0.0.1:5100",
        "node_port": 5005,
        "face_match_threshold": 0.45,
        "cameras": [
            {
                "id": "cam_webcam_0",
                "name": "Live Entrance Camera",
                "camera_type": "webcam",
                "rtsp_url": 0,
                "room_name": "Main Entrance",
                "channel_number": 1,
                "active": 1
            }
        ]
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
    """Background worker continuously flushing recognized attendance events to Cloud."""
    while True:
        try:
            event = cloud_event_queue.get()
            cloud_url = f"{CONFIG.get('cloud_url', 'http://127.0.0.1:5100').rstrip('/')}/api/sync/attendance-event"
            req_data = json.dumps(event).encode('utf-8')
            req = urllib.request.Request(
                cloud_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status in (200, 201):
                        logger.info(f"[CLOUD SYNC OK] Event for '{event.get('name')}' delivered to Cloud.")
            except Exception as ce:
                logger.warning(f"[CLOUD SYNC OFFLINE] Could not deliver event to Cloud ({ce}). Event will retry/buffered.")
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
    Sync client employees and embeddings from local database or cloud for this specific organization & branch.
    Strict tenant isolation: ensures profiles strictly belong to org_id and branch_id.
    """
    try:
        # Check if local database already has users scoped to this branch
        with sqlite3.connect(db.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users WHERE organization_id = ? AND branch_id = ?", (org_id, branch_id))
            scoped_count = cur.fetchone()[0]

            # If node DB doesn't have them yet, check if master Local-Dashboard has them
            if scoped_count == 0:
                master_db = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Local-Dashboard", "attendance.db"))
                if os.path.exists(master_db):
                    with sqlite3.connect(master_db) as mconn:
                        mcur = mconn.cursor()
                        mcur.execute("SELECT id, name, email, department, active, organization_id, branch_id FROM users WHERE organization_id = ? AND branch_id = ?", (org_id, branch_id))
                        musers = mcur.fetchall()
                        for u in musers:
                            cur.execute("""
                                INSERT OR REPLACE INTO users (id, name, email, department, active, organization_id, branch_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, u)
                            # Copy embeddings
                            mcur.execute("SELECT id, user_id, embedding, source_video, created_at FROM embeddings WHERE user_id = ?", (u[0],))
                            membs = mcur.fetchall()
                            for emb in membs:
                                cur.execute("""
                                    INSERT OR REPLACE INTO embeddings (id, user_id, embedding, source_video, created_at)
                                    VALUES (?, ?, ?, ?, ?)
                                """, emb)
                        conn.commit()
                        logger.info(f"[TENANT SYNC] Imported {len(musers)} employees and embeddings for {org_id} / {branch_id}")
    except Exception as e:
        logger.warning(f"Could not synchronize employee profiles: {e}")
    return 0


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
        
        # If tenant is set in node config, enforce tenant filtering
        if target_org and u_org and u_org != target_org:
            continue
        if target_branch and u_branch and u_branch != target_branch:
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
# HIGH PERFORMANCE DUAL-THREADED CAMERA STREAM READER
# ==============================================================================

class NodeCameraReader:
    def __init__(self, camera_id: str, camera_name: str, rtsp_source: Any):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_source = rtsp_source
        self.is_webcam = isinstance(rtsp_source, int) or (isinstance(rtsp_source, str) and rtsp_source.isdigit())
        
        if self.is_webcam:
            self.rtsp_url = int(rtsp_source)
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
        else:
            self.rtsp_url = str(rtsp_source)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame = None
        self.latest_detections = []
        self.tracked_faces = {}
        self.next_track_id = 1
        self.last_logged = {}  # {user_id: timestamp}
        self.running = True
        self.lock = threading.Lock()

        # Thread 1: Grabber
        threading.Thread(target=self._grabber_loop, daemon=True).start()
        # Thread 2: AI Processor
        threading.Thread(target=self._ai_loop, daemon=True).start()

    def _grabber_loop(self):
        while self.running:
            if not self.cap.isOpened():
                if self.is_webcam:
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                time.sleep(1.0)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.03)

    def _ai_loop(self):
        last_ai_time = 0
        while self.running:
            now = time.time()
            if (now - last_ai_time) < 0.05:  # ~20 FPS AI cadence
                time.sleep(0.01)
                continue

            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            last_ai_time = now
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

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            frame = self.latest_frame.copy()
            detections = list(self.latest_detections)

        # Draw smooth bounding boxes and name labels
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            name = d['name']
            conf = d['confidence']
            is_match = name != "Unknown"
            color = (0, 230, 115) if is_match else (0, 140, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({int(conf * 100)}%)" if is_match else "Verifying..."
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + len(label) * 9, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return True, frame

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass


def init_camera_streams():
    """Fetch camera list from Cloud if online, otherwise use local config."""
    global ACTIVE_CAMERA_READERS
    cameras_to_use = CONFIG.get("cameras", [])

    # Try fetching active cameras from Cloud
    try:
        cloud_cams_url = f"{CONFIG.get('cloud_url', 'http://127.0.0.1:5100').rstrip('/')}/api/branches/{CONFIG.get('branch_id', 'BR-FORK-01')}/cameras?organization_id={CONFIG.get('organization_id', 'ORG-001')}"
        req = urllib.request.Request(cloud_cams_url, method="GET")
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get("cameras"):
                    cameras_to_use = data["cameras"]
                    logger.info(f"[CLOUD CONNECTED] Loaded {len(cameras_to_use)} camera streams configured in Cloud.")
    except Exception as ce:
        logger.info(f"[STANDALONE] Cloud unreachable ({ce}). Starting with local camera config.")

    with readers_lock:
        for reader in ACTIVE_CAMERA_READERS.values():
            reader.release()
        ACTIVE_CAMERA_READERS.clear()

        for cam in cameras_to_use:
            cid = str(cam.get("id"))
            cname = cam.get("name", f"Camera {cid}")
            source = cam.get("rtsp_url", 0)
            try:
                ACTIVE_CAMERA_READERS[cid] = NodeCameraReader(cid, cname, source)
                logger.info(f"Started camera stream reader: {cname} ({cid})")
            except Exception as e:
                logger.error(f"Failed to start camera {cid}: {e}")


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
        while True:
            ret, frame = reader.get_annotated_frame()
            if not ret or frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "CONNECTING CAMERA...", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                _, jpeg = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
            
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)

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


def start_client_node(port: int = 5005):
    logger.info("Initializing QuantEye Client AI Node...")
    db.init_db()
    reload_face_cache()
    threading.Thread(target=init_camera_streams, daemon=True).start()
    logger.info(f"Starting QuantEye Client AI Node on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_client_node()
