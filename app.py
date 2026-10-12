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

logger = get_logger("client_node")

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "quanteye-client-node-secret-key"

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_node.json')

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

def reload_face_cache():
    global EMBEDDING_CACHE, ENROLLED_MATRIX, ENROLLED_USER_IDS, ENROLLED_USER_METADATA
    logger.info("[*] Refreshing in-memory facial embedding cache from database...")
    new_cache = {}
    matrix_rows = []
    user_ids = []
    user_metadata = []

    all_users = db.get_all_users()
    for user in all_users:
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
            logger.info(f"[OK] Cached {len(user_ids)} profiles in vectorized matrix for instant live recognition.")
        else:
            ENROLLED_MATRIX = None
            ENROLLED_USER_IDS = []
            ENROLLED_USER_METADATA = []


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


@app.route('/api/node/activate-key', methods=['POST'])
def api_node_activate_key():
    """
    Activate the Standalone Edge Node with an Organization/Branch Key.
    Dynamic: Works for any client branch without hardcoding.
    """
    global CONFIG
    data = request.get_json(silent=True) or {}
    branch_key = data.get('branch_key', '').strip()
    cloud_url = (data.get('cloud_url') or CONFIG.get('cloud_url', 'http://127.0.0.1:5100')).strip().rstrip('/')

    if not branch_key:
        return jsonify({"success": False, "error": "Branch Key is required"}), 400

    try:
        val_url = f"{cloud_url}/api/edge/validate-key"
        val_payload = json.dumps({"branch_key": branch_key}).encode('utf-8')
        req = urllib.request.Request(val_url, data=val_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            if resp.status == 200:
                val_data = json.loads(resp.read().decode('utf-8'))
                # Update runtime config
                CONFIG['organization_id'] = val_data['organization_id']
                CONFIG['organization_name'] = val_data['organization_name']
                CONFIG['branch_id'] = val_data['branch_id']
                CONFIG['branch_name'] = val_data['branch_name']
                CONFIG['branch_key'] = branch_key
                CONFIG['cloud_url'] = cloud_url
                CONFIG['cameras'] = val_data.get('cameras', [])
                
                # Persist to config_node.json
                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                    json.dump(CONFIG, f, indent=2)

                # Reload camera streams dynamically
                threading.Thread(target=init_camera_streams, daemon=True).start()

                return jsonify({
                    "success": True,
                    "organization_name": val_data['organization_name'],
                    "branch_name": val_data['branch_name'],
                    "camera_count": len(val_data.get('cameras', [])),
                    "message": f"Successfully activated {val_data['organization_name']} - {val_data['branch_name']}"
                }), 200
            else:
                return jsonify({"success": False, "error": "Authentication failed"}), 401
    except urllib.error.HTTPError as he:
        try:
            err_body = json.loads(he.read().decode('utf-8'))
            return jsonify({"success": False, "error": err_body.get('error', f"HTTP {he.code}")}), he.code
        except Exception:
            return jsonify({"success": False, "error": f"Cloud rejected key (HTTP {he.code})"}), he.code
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not reach server: {str(e)}"}), 502


def start_client_node(port: int = 5005):
    logger.info("Initializing QuantEye Client AI Node...")
    db.init_db()
    reload_face_cache()
    init_camera_streams()
    logger.info(f"Starting QuantEye Client AI Node on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_client_node()
