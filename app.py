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
from typing import Dict, List, Optional, Any, Tuple
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
from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS,
    MAX_CONTENT_LENGTH, FACE_MATCHING_THRESHOLD,
    MIN_ENROLLMENT_FRAMES, OPTIMAL_FACES_PER_VIDEO,
    MIN_VIDEO_DURATION, MAX_VIDEO_DURATION,
    TRACK_MAX_AGE_SECONDS, TRACK_ACTIVE_IOU_THRESHOLD, TRACK_LOST_IOU_THRESHOLD,
    TRACK_ACTIVE_DIST_FACTOR, TRACK_LOST_DIST_FACTOR,
    TRACK_ACTIVE_MIN_DIST, TRACK_LOST_MIN_DIST,
    TRACK_UNKNOWN_RETRY_INTERVAL, TRACK_AI_INTERVAL,
    IDLE_AI_INTERVAL, ATTENDANCE_CONFIRMATION_FRAMES,
    ATTENDANCE_CONFIRMATION_WINDOW, ATTENDANCE_CONFIRMATION_MIN_SIM,
    PRESENCE_ACTIVE_TIMEOUT, PRESENCE_RECENT_TIMEOUT
)
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
                with urllib.request.urlopen(preq, timeout=4.0) as presp:
                    if presp.status in (200, 201):
                        resp_content = presp.read().decode('utf-8')
                        try:
                            resp_json = json.loads(resp_content)
                            if resp_json.get("attendance_logged"):
                                logger.info(f"✓ [CLOUD PORTAL ATTENDANCE OK] Live attendance for '{event.get('name')}' ({event.get('event_type', 'CHECKIN')}) recorded: {resp_json.get('reason')}")
                            else:
                                reason = resp_json.get("reason", "Rejected by cloud rules")
                                logger.warning(f"[CLOUD PORTAL REJECTED] '{event.get('name')}': {reason}")
                                uid = event.get("user_id")
                                if uid and event.get("event_type") != "CHECKOUT":
                                    with attendance_state_lock:
                                        USER_ATTENDANCE_STATE.pop(uid, None)
                        except Exception:
                            logger.info(f"[ATTENDANCE SYNC OK] Live attendance for '{event.get('name')}' delivered to Cloud Portal.")
            except Exception as pe:
                logger.warning(f"[PORTAL ATTENDANCE ERROR] Failed syncing '{event.get('name')}': {pe}")
                uid = event.get("user_id")
                if uid and event.get("event_type") != "CHECKOUT":
                    with attendance_state_lock:
                        USER_ATTENDANCE_STATE.pop(uid, None)

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


def dispatch_attendance_event_to_cloud(user_id: int, name: str, confidence: float, camera_id: str, camera_name: str, face_crop_b64: Optional[str] = None, timestamp: Optional[str] = None, event_type: str = "CHECKIN"):
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
        "event_type": event_type,
        "timestamp": timestamp or datetime.now().isoformat()
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

            # Also sync shifts from Cloud Portal so shift timings match Cloud 100%
            try:
                shifts_url = f"{portal_url}/api/shifts"
                sreq = urllib.request.Request(shifts_url, headers={
                    "User-Agent": "QuantEye-Node",
                    "X-Client-Auth": "true",
                    "X-Organization-Id": org_id,
                    "X-Branch-Id": branch_id or "DEFAULT_BRANCH"
                }, method="GET")
                with urllib.request.urlopen(sreq, timeout=3.5) as sresp:
                    if sresp.status == 200:
                        sdata = json.loads(sresp.read().decode('utf-8'))
                        cloud_shifts = sdata.get("shifts", [])
                        with sqlite3.connect(db.DB_PATH) as sconn:
                            scur = sconn.cursor()
                            for cs in cloud_shifts:
                                sid = cs.get("id")
                                sname = cs.get("name", "Morning Shift")
                                st = cs.get("start_time", "09:00")
                                et = cs.get("end_time", "18:00")
                                gp = cs.get("grace_period_minutes", 15)
                                scur.execute("""
                                    INSERT INTO shifts (id, name, start_time, end_time, grace_period_minutes)
                                    VALUES (?, ?, ?, ?, ?)
                                    ON CONFLICT(id) DO UPDATE SET
                                        name = excluded.name,
                                        start_time = excluded.start_time,
                                        end_time = excluded.end_time,
                                        grace_period_minutes = excluded.grace_period_minutes
                                """, (sid, sname, st, et, gp))
                            sconn.commit()
            except Exception as se:
                logger.debug(f"[PORTAL SYNC] Shift sync note: {se}")
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


# =============================================================================# ==============================================================================
# ORIGINAL LOCAL-DASHBOARD STATE REGISTRY & MATCHING ENGINE
# ==============================================================================

LIVE_PRESENCE = {}
presence_lock = threading.Lock()
MANUALLY_ABSENT_USERS = set()
DETECTED_USERS_SESSION = set()
LATEST_STREAM_DETECTIONS = []
STREAM_PAUSED_STATE = {}
monitoring_state_lock = threading.Lock()
active_stream_readers = {}
active_readers_lock = threading.Lock()
today_logged_lock = threading.Lock()

ACTIVE_CAMERA_READERS = active_stream_readers
readers_lock = active_readers_lock
LATEST_DETECTIONS_FEED = LATEST_STREAM_DETECTIONS
detections_lock = cache_lock

USER_ATTENDANCE_STATE = {}
USER_ATTENDANCE_STATE_DATE = ""

# -----------------------------------------------------------------------
# ASYNCHRONOUS DATABASE WRITER QUEUE & IN-MEMORY CACHING
# Video processing thread never waits for SQLite file locks or disk I/O.
# All attendance logs and snapshots are dispatched in ~0.001ms to a background worker.
# -----------------------------------------------------------------------
DB_TASK_QUEUE = queue.Queue(maxsize=5000)
TODAY_LOGGED_ATTENDANCE = set()  # set of (user_id, source)
TODAY_LOGGED_DATE = None

def _db_writer_worker():
    """Background daemon worker executing DB writes asynchronously so AI loop never drops frames."""
    while True:
        try:
            task_type, payload = DB_TASK_QUEUE.get()
            if task_type == 'log_attendance':
                success, reason = db.log_attendance(
                    user_id=payload['user_id'],
                    detected_name=payload['detected_name'],
                    confidence=payload['confidence'],
                    source=payload['source'],
                    location=payload.get('location'),
                    device_id=payload.get('device_id'),
                    bypass_shift_check=True,
                    checkout_camera_id=payload.get('checkout_camera_id'),
                    checkout_location=payload.get('checkout_location'),
                    attendance_time=payload.get('attendance_time')
                )
                if not success:
                    clear_logged_today_in_memory(payload['user_id'])
                    logger.info(f"[Shift Validation] Skipped DB log for '{payload['detected_name']}': {reason}")
                else:
                    try:
                        dispatch_attendance_event_to_cloud(
                            user_id=payload['user_id'],
                            name=payload['detected_name'],
                            confidence=payload['confidence'],
                            camera_id=payload.get('device_id'),
                            camera_name=payload.get('location'),
                            face_crop_b64=payload.get('face_crop_b64'),
                            timestamp=(payload.get('attendance_time') or datetime.now()).isoformat(),
                            event_type=payload.get('event_type', 'CHECKIN')
                        )
                    except Exception as ce:
                        logger.debug(f"Cloud dispatch error: {ce}")
            elif task_type == 'record_locator_snapshot':
                db.record_locator_snapshot(
                    user_id=payload['user_id'],
                    name=payload['name'],
                    department=payload.get('department', ''),
                    camera_id=payload['camera_id'],
                    camera_name=payload['camera_name'],
                    room_name=payload['room_name'],
                    confidence=payload['confidence'],
                    first_seen=payload['first_seen'],
                    last_seen=payload['last_seen'],
                    interval_seconds=payload.get('interval_seconds', 300)
                )
            elif task_type == 'clear_manual_absent':
                db.clear_manual_absent_today(payload['user_id'])
            DB_TASK_QUEUE.task_done()
        except Exception as e:
            logger.error(f"[DB Writer Worker] Error executing DB task: {e}")

db_writer_thread = threading.Thread(target=_db_writer_worker, daemon=True)
db_writer_thread.start()


def is_already_logged_today(user_id: int, source: str) -> bool:
    """O(1) in-memory check to prevent repeated SQLite queries during live streams."""
    global TODAY_LOGGED_DATE, TODAY_LOGGED_ATTENDANCE
    today_str = datetime.now().strftime('%Y-%m-%d')
    with today_logged_lock:
        if TODAY_LOGGED_DATE != today_str:
            TODAY_LOGGED_DATE = today_str
            try:
                records = db.get_attendance_today()
                TODAY_LOGGED_ATTENDANCE = {(r['user_id'], r.get('source', '')) for r in records}
            except Exception:
                TODAY_LOGGED_ATTENDANCE = set()
        with cache_lock:
            if user_id in MANUALLY_ABSENT_USERS:
                to_discard = [item for item in TODAY_LOGGED_ATTENDANCE if item[0] == user_id]
                for item in to_discard:
                    TODAY_LOGGED_ATTENDANCE.discard(item)
                return False
        return (user_id, source) in TODAY_LOGGED_ATTENDANCE


def mark_logged_today_in_memory(user_id: int, source: str):
    with today_logged_lock:
        TODAY_LOGGED_ATTENDANCE.add((user_id, source))


def clear_logged_today_in_memory(user_id: Optional[int] = None):
    """Clear in-memory attendance record so re-marking or camera re-detection writes to DB."""
    global TODAY_LOGGED_ATTENDANCE, USER_ATTENDANCE_STATE
    with today_logged_lock:
        if user_id is None:
            TODAY_LOGGED_ATTENDANCE.clear()
            USER_ATTENDANCE_STATE.clear()
        else:
            to_discard = [item for item in TODAY_LOGGED_ATTENDANCE if item[0] == user_id]
            for item in to_discard:
                TODAY_LOGGED_ATTENDANCE.discard(item)
            USER_ATTENDANCE_STATE.pop(user_id, None)


def resync_logged_today_in_memory():
    """Resynchronize TODAY_LOGGED_ATTENDANCE directly from database records for today."""
    global TODAY_LOGGED_DATE, TODAY_LOGGED_ATTENDANCE, USER_ATTENDANCE_STATE
    today_str = datetime.now().strftime('%Y-%m-%d')
    with today_logged_lock:
        TODAY_LOGGED_DATE = today_str
        USER_ATTENDANCE_STATE.clear()
        try:
            records = db.get_attendance_today()
            TODAY_LOGGED_ATTENDANCE = {(r['user_id'], r.get('source', '')) for r in records}
        except Exception:
            TODAY_LOGGED_ATTENDANCE = set()


def update_live_presence(user_id, name, department, camera_id, camera_name, room_name, confidence, face_crop=None):
    """Update global presence registry when a verified person is seen in a camera/room (non-blocking)."""
    with presence_lock:
        prev = LIVE_PRESENCE.get(user_id, {})
        now = time.time()
        prev_room = prev.get('room_name')
        first_seen = prev.get('first_seen_room', now) if prev_room == room_name else now
        LIVE_PRESENCE[user_id] = {
            "user_id": user_id,
            "name": name,
            "department": department or prev.get('department', ''),
            "camera_id": camera_id,
            "camera_name": camera_name,
            "room_name": room_name,
            "confidence": float(confidence),
            "last_seen": now,
            "first_seen_room": first_seen,
            "face_crop": face_crop or prev.get('face_crop')
        }

    # Queue locator snapshot asynchronously so the video loop returns instantly
    try:
        DB_TASK_QUEUE.put_nowait(('record_locator_snapshot', {
            'user_id': user_id,
            'name': name,
            'department': department or '',
            'camera_id': camera_id,
            'camera_name': camera_name,
            'room_name': room_name,
            'confidence': confidence,
            'first_seen': first_seen,
            'last_seen': now,
            'interval_seconds': 300
        }))
    except queue.Full:
        pass


def refresh_presence_timestamp(user_id, current_time=None):
    """Keep active dwell time alive while face is continuously tracked in current room."""
    with presence_lock:
        if user_id in LIVE_PRESENCE:
            LIVE_PRESENCE[user_id]["last_seen"] = current_time or time.time()


def match_embedding_vectorized(test_embedding: np.ndarray, threshold: float = FACE_MATCHING_THRESHOLD, organization_id: Optional[str] = None):
    """
    Lightning-fast vectorized search against all enrolled users in memory.
    Uses BLAS matrix dot product: O(1) in ~0.05ms instead of Python dictionary loops.
    Strictly filters candidates by caller organization_id for Biometric Isolation.
    Returns: (best_uid, user_meta_dict, best_similarity, is_match)
    """
    if test_embedding is None:
        return None, None, 0.0, False

    with cache_lock:
        local_matrix = ENROLLED_MATRIX
        local_uids = ENROLLED_USER_IDS
        local_meta = ENROLLED_USER_METADATA

    if local_matrix is None or len(local_uids) == 0:
        return None, None, 0.0, False

    try:
        norm = float(np.linalg.norm(test_embedding))
        if norm < 1e-6:
            return None, None, 0.0, False
        query_emb = np.divide(test_embedding, norm, dtype=np.float32)

        # Single C/BLAS matrix dot product: (N, 512) . (512,) -> (N,)
        similarities = np.dot(local_matrix, query_emb)
        
        if organization_id:
            valid_indices = [
                i for i, m in enumerate(local_meta)
                if (m.get('organization_id') == organization_id or
                    (organization_id == 'DEFAULT_ORG' and (not m.get('organization_id') or m.get('organization_id') == 'DEFAULT_ORG')))
            ]
            if not valid_indices:
                return None, None, 0.0, False
            sub_similarities = similarities[valid_indices]
            sub_best = int(np.argmax(sub_similarities))
            max_sim = float(sub_similarities[sub_best])
            best_idx = valid_indices[sub_best]
        else:
            best_idx = int(np.argmax(similarities))
            max_sim = float(similarities[best_idx])

        is_match = (max_sim >= threshold)
        if is_match:
            best_uid = local_uids[best_idx]
            best_data = local_meta[best_idx]
            return best_uid, best_data, max_sim, True
        else:
            return None, None, max_sim, False
    except Exception as e:
        logger.error(f"Vectorized matching error: {e}")
        return None, None, 0.0, False


def upsert_live_detection(uid, detection_entry=None, **updates):
    """Keep ONE ticker row per employee: replace/refresh their entry and move it to the top.
    Display only - never touches attendance records."""
    global LATEST_STREAM_DETECTIONS, DETECTED_USERS_SESSION
    with cache_lock:
        existing = None
        for i, det in enumerate(LATEST_STREAM_DETECTIONS):
            if det.get('user_id') == uid:
                existing = LATEST_STREAM_DETECTIONS.pop(i)
                break
        if detection_entry is not None:
            entry = dict(detection_entry)
            if existing and not entry.get('face_crop'):
                entry['face_crop'] = existing.get('face_crop')
        elif existing is not None:
            entry = existing
            entry.update({k: v for k, v in updates.items() if v is not None})
        else:
            return False
        entry['timestamp'] = datetime.now().strftime("%I:%M:%S %p")
        DETECTED_USERS_SESSION.add(uid)
        LATEST_STREAM_DETECTIONS.insert(0, entry)
        del LATEST_STREAM_DETECTIONS[1000:]
        return True


def remove_live_detection_for_user(user_id: int):
    """Remove a user from live detections, room presence locator, and active tracking after manual absent marking."""
    global LATEST_STREAM_DETECTIONS, DETECTED_USERS_SESSION, MANUALLY_ABSENT_USERS
    with cache_lock:
        LATEST_STREAM_DETECTIONS = [
            det for det in LATEST_STREAM_DETECTIONS if det.get('user_id') != user_id
        ]
        DETECTED_USERS_SESSION.discard(user_id)
        MANUALLY_ABSENT_USERS.add(user_id)

    clear_logged_today_in_memory(user_id)

    with presence_lock:
        if user_id in LIVE_PRESENCE:
            del LIVE_PRESENCE[user_id]
            logger.info(f"✓ Cleared user ID {user_id} from LIVE_PRESENCE room locator store.")

    with active_readers_lock:
        for reader in active_stream_readers.values():
            with reader.lock:
                reader.tracked_faces = {
                    tid: t for tid, t in reader.tracked_faces.items()
                    if t.get('user_id') != user_id
                }


def restore_live_detection_for_user(user_id: int):
    """Allow a manually absent user to be recognized again without restarting the server."""
    global DETECTED_USERS_SESSION, MANUALLY_ABSENT_USERS
    with cache_lock:
        MANUALLY_ABSENT_USERS.discard(user_id)
        DETECTED_USERS_SESSION.discard(user_id)


def process_user_attendance_event(user_id: int, name: str, confidence: float, source: str, location: str, camera_id: str, shift_info: Optional[Dict] = None, face_crop_b64: Optional[str] = None) -> Tuple[str, str]:
    """
    Camera-Aware & Shift-Based State Sequence for Employee Attendance:
    
    State Progression:
    Not Checked In -> Check-In Window + Authorized Check-In Camera -> Present/Checked In -> Check-Out Window + Authorized Check-Out Camera -> Checked Out -> Completed
    
    Rules Enforced:
    1. Every employee evaluated against assigned camera permissions AND shift rules.
    2. Check-In ONLY allowed on employee's authorized Check-In camera(s).
    3. Check-Out ONLY allowed on employee's authorized Check-Out camera(s).
    4. Repeated detections on Check-In cameras do NOT create duplicate records.
    5. Detections on unauthorized cameras do NOT mark attendance.
    """
    global USER_ATTENDANCE_STATE, USER_ATTENDANCE_STATE_DATE
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    with today_logged_lock:
        if USER_ATTENDANCE_STATE_DATE != today_str:
            USER_ATTENDANCE_STATE_DATE = today_str
            USER_ATTENDANCE_STATE.clear()
            try:
                today_logs = db.get_attendance_today()
                user_min_max = {}
                for log in today_logs:
                    uid = log['user_id']
                    ts_str = str(log['timestamp'])
                    try:
                        ts_dt = datetime.strptime(ts_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts_dt = now
                    if uid not in user_min_max:
                        user_min_max[uid] = {'min': ts_dt, 'max': ts_dt}
                    else:
                        if ts_dt < user_min_max[uid]['min']:
                            user_min_max[uid]['min'] = ts_dt
                        if ts_dt > user_min_max[uid]['max']:
                            user_min_max[uid]['max'] = ts_dt

                for uid, mm in user_min_max.items():
                    USER_ATTENDANCE_STATE[uid] = {
                        'checkin_time': mm['min'],
                        'last_checkout_log': mm['max'] if mm['max'] != mm['min'] else None
                    }
            except Exception as e:
                logger.error(f"Failed initializing USER_ATTENDANCE_STATE from DB: {e}")

        # Check manual absent override
        with cache_lock:
            if user_id in MANUALLY_ABSENT_USERS:
                USER_ATTENDANCE_STATE.pop(user_id, None)

        user_state = USER_ATTENDANCE_STATE.get(user_id)

        # -------------------------------------------------------------
        # STATE 0: NOT CHECKED IN TODAY -> Check-In Camera & Shift Window
        # -------------------------------------------------------------
        if not user_state:
            # Rule: Camera authorization check for Check-In
            cam_allowed, cam_msg = db.check_user_camera_allowed(user_id, camera_id, 'checkin')
            if not cam_allowed:
                logger.info(f"[CAMERA CHECK-IN BLOCKED] Staff '{name}' (ID #{user_id}) on '{camera_id}': {cam_msg}")
                return "REJECTED_CAMERA_UNAUTHORIZED", cam_msg

            shift_allowed, shift_msg, retrieved_shift = db.check_user_shift_allowed(user_id, now)
            effective_shift = retrieved_shift or shift_info or {}

            if not shift_allowed:
                logger.info(f"[SHIFT CHECK-IN BLOCKED] Staff '{name}' (ID #{user_id}): {shift_msg}")
                return "REJECTED_OUTSIDE_SHIFT", shift_msg

            # Valid check-in window and authorized camera -> Record Check-In
            USER_ATTENDANCE_STATE[user_id] = {
                'checkin_time': now,
                'last_checkout_log': None
            }
            try:
                DB_TASK_QUEUE.put_nowait(('clear_manual_absent', {'user_id': user_id}))
                DB_TASK_QUEUE.put_nowait(('log_attendance', {
                    'user_id': user_id,
                    'detected_name': name,
                    'confidence': confidence,
                    'source': source,
                    'location': location,
                    'device_id': camera_id,
                    'face_crop_b64': face_crop_b64,
                    'attendance_time': now,
                    'event_type': 'CHECKIN'
                }))
                logger.info(f"✓ [CHECK-IN LOGGED] Staff '{name}' (ID #{user_id}) on Camera '{camera_id}' ({location}) at {now.strftime('%I:%M:%S %p')} ({effective_shift.get('name', 'Shift')})")
            except queue.Full:
                pass
            return "CHECKIN", f"Check-In recorded at {now.strftime('%I:%M %p')} on {location}"

        # -------------------------------------------------------------
        # STATE 1: ALREADY CHECKED IN TODAY -> Check-Out Validation
        # -------------------------------------------------------------
        checkin_time = user_state['checkin_time']
        seconds_since_checkin = (now - checkin_time).total_seconds()

        # Initial check-in buffer: Detections within the first 60 seconds confirm initial Check-In
        if seconds_since_checkin < 60:
            return "CHECKIN_CONFIRMED", f"Check-In active ({int(seconds_since_checkin)}s after check-in)"

        # Check user's assigned shift end time
        _, _, user_shift = db.check_user_shift_allowed(user_id, now)
        start_time_str = user_shift.get('start_time', '09:00') if user_shift else '09:00'
        end_time_str = user_shift.get('end_time', '18:00') if user_shift else '18:00'

        # STRICT: Check-Out opens EXACTLY at shift end time (e.g. 06:00 PM) - NEVER EARLIER.
        is_checkout_window = False
        try:
            cur_secs = now.hour * 3600 + now.minute * 60 + now.second
            e_parts = [int(p) for p in end_time_str.split(':')[:2]]
            e_secs = e_parts[0] * 3600 + e_parts[1] * 60
            s_parts = [int(p) for p in start_time_str.split(':')[:2]]
            s_secs = s_parts[0] * 3600 + s_parts[1] * 60
            if e_secs < s_secs:  # Overnight shift (e.g. 22:00 -> 06:00)
                is_checkout_window = (e_secs <= cur_secs < s_secs)
            else:
                is_checkout_window = (cur_secs >= e_secs)
        except Exception:
            is_checkout_window = False

        # If still within working shift hours before end time, employee is active/present (do not check out early!)
        if not is_checkout_window:
            return "PRESENT_ACTIVE", f"Staff '{name}' is active on shift (Check-In: {checkin_time.strftime('%I:%M %p')})"

        # Check-Out Event: Verify camera authorization for Check-Out
        cam_allowed, cam_msg = db.check_user_camera_allowed(user_id, camera_id, 'checkout')
        if not cam_allowed:
            logger.info(f"[CAMERA CHECK-OUT BLOCKED] Staff '{name}' (ID #{user_id}) on Camera '{camera_id}': {cam_msg}")
            return "REJECTED_CAMERA_UNAUTHORIZED", cam_msg

        last_checkout_log = user_state.get('last_checkout_log')
        # Check-Out is recorded ONCE per day (first detection after shift end) and never updated again
        if last_checkout_log is None:
            user_state['last_checkout_log'] = now
            try:
                DB_TASK_QUEUE.put_nowait(('log_attendance', {
                    'user_id': user_id,
                    'detected_name': name,
                    'confidence': confidence,
                    'source': source,
                    'location': location,
                    'device_id': camera_id,
                    'checkout_camera_id': camera_id,
                    'checkout_location': location,
                    'face_crop_b64': face_crop_b64,
                    'attendance_time': now,
                    'event_type': 'CHECKOUT'
                }))
                logger.info(f"✓ [CHECK-OUT LOGGED] Staff '{name}' (ID #{user_id}) Check-Out on Camera '{camera_id}' ({location}) updated to {now.strftime('%I:%M:%S %p')} (Check-In was {checkin_time.strftime('%I:%M:%S %p')})")
            except queue.Full:
                pass
            return "CHECKOUT", f"Check-Out recorded at {now.strftime('%I:%M %p')} on {location}"

        return "ATTENDANCE_COMPLETED", "Attendance active and completed for assigned shift"


class CameraStreamReader:
    """
    High-performance dual-threaded camera reader:
    - Thread 1 (Grabber): Continuously grabs the latest frame at 30 FPS from the NVR/DVR.
      This completely eliminates OpenCV buffer delay and lagging.
    - Thread 2 (AI Worker): Processes the latest frame asynchronously for face detection 
      and recognition in the background, updating detections without blocking the stream.
    """
    def __init__(self, camera_id, rtsp_url):
        import cv2
        import threading
        import os
        
        self.camera_id = str(camera_id)
        self.rtsp_url = rtsp_url
        
        cam_info = db.get_camera_by_id(camera_id)
        if not cam_info:
            for c in CONFIG.get('cameras', []):
                if str(c.get('id') or c.get('camera_id')) == str(camera_id):
                    cam_info = {
                        'name': c.get('name') or c.get('camera_name', f"Camera {camera_id}"),
                        'room_name': c.get('room_name') or 'Live Monitoring'
                    }
                    break
        if cam_info:
            self.room_name = cam_info.get('room_name', f"Room ({camera_id})")
            self.camera_label = cam_info.get('name', f"Camera {camera_id}")
        else:
            self.room_name = f"Room ({camera_id})"
            self.camera_label = f"Camera {camera_id}"
        self.camera_name = self.camera_label
        
        self.is_webcam = isinstance(rtsp_url, int) or (isinstance(rtsp_url, str) and str(rtsp_url).isdigit())
        if self.is_webcam:
            self.rtsp_url = int(rtsp_url)
            logger.info(f"Initializing USB Webcam (Index: {self.rtsp_url}) using CAP_DSHOW...")
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            # Set low-delay TCP parameters BEFORE initializing the VideoCapture to eliminate buffer lag
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.latest_frame = None
        self.latest_detections = []
        self.tracked_faces = {}  # track_id -> dict for locking face identities
        self.next_track_id = 1
        self.last_logged = {}  # {user_id: timestamp}
        self.running = True
        self.viewers_count = 0  # Active viewer tracking
        self.paused = False  # UI feed hidden but backend recognition remains active
        
        self.lock = threading.Lock()
        
        # Thread 1: Camera Frame Grabber (Runs at ~30-60Hz, never blocks for AI)
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.grabber_thread.start()
        
        # Thread 2: Asynchronous AI Worker (Runs at ~3Hz, processes frames in background)
        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        self.ai_thread.start()

    def _grabber_loop(self):
        import time
        import cv2
        import os
        
        # Set low-latency and TCP options for FFmpeg backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        
        consecutive_failures = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                if self.is_webcam:
                    if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('IS_CLOUD') or os.name != 'nt':
                        time.sleep(15)
                        continue
                    logger.warning(f"[{self.camera_id}] Webcam closed. Re-opening with CAP_DSHOW...")
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_DSHOW)
                    if not self.cap or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)
                continue
                
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                consecutive_failures = 0
                # Resize immediately to 1080px wide for superior distant face recognition range
                h, w = frame.shape[:2]
                max_width = 1080
                if w > max_width:
                    scale = max_width / w
                    frame = cv2.resize(frame, (max_width, int(h * scale)))
                
                with self.lock:
                    self.latest_frame = frame
            else:
                consecutive_failures += 1
                # If reading fails continuously for ~1.5 seconds (30 attempts), reset capture to reconnect
                if consecutive_failures >= 30:
                    logger.warning(f"[{self.camera_id}] RTSP stream disconnected or stalled ({consecutive_failures} failed reads). Reconnecting...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    consecutive_failures = 0
                    time.sleep(1)
                    continue
                time.sleep(0.01)

    def _ai_loop(self):
        import time
        import cv2
        import base64
        import numpy as np
        from datetime import datetime
        
        last_ai_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Process the latest frame continuously on GPU. We intentionally do not drop frames
            # for live attendance recognition because the model is CUDA-enabled and a 20 FPS loop
            # is the correct real-time cadence for attendance tracking.
            ai_interval = TRACK_AI_INTERVAL if self.viewers_count > 0 or self.paused else IDLE_AI_INTERVAL
            ai_interval = min(ai_interval, 0.05)

            # Retrieve latest frame from grabber thread under lock
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is not None and (current_time - last_ai_time) >= ai_interval:
                last_ai_time = current_time

                try:
                    # -----------------------------------------------------------------------
                    # STEP 2 (CROWD FIX): FOCUS GPU COMPUTE ON UNCONFIRMED STUDENTS
                    # Front-row students who are already confirmed and logged in DB don't
                    # need their ArcFace embeddings re-extracted every frame. We pass their
                    # active bboxes so SCRFD still tracks them smoothly, while ArcFace runs
                    # ONLY on the new and unconfirmed students walking behind them!
                    # -----------------------------------------------------------------------
                    confirmed_bboxes = [
                        t["bbox"] for t in self.tracked_faces.values()
                        if t.get("confirmed") and t.get("name") != "Unknown" and (current_time - t.get("last_seen", 0)) < 0.6
                    ]

                    # ALL-IN-ONE High-Speed Face Detection & Selective Extraction
                    face_results = fp.detect_and_extract_insightface(frame_to_process, skip_bboxes=confirmed_bboxes)
                    new_cached_detections = []
                    
                    # Sort faces by area (largest first) to prioritize closer faces
                    face_results = sorted(face_results, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
                    
                    # Get active tracked faces (allow up to TRACK_MAX_AGE_SECONDS so tracks persist smoothly during CPU inference)
                    active_tracks = {tid: t for tid, t in self.tracked_faces.items() if (current_time - t["last_seen"]) < TRACK_MAX_AGE_SECONDS}
                    
                    assigned_detections = []  # list of (face_dict, track_id)
                    used_track_ids = set()
                    
                    for face_dict in face_results:
                        x1, y1, x2, y2 = face_dict['bbox']
                        conf = face_dict['conf']
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        best_tid = None
                        best_dist = float('inf')
                        
                        # Match with closest active track
                        for tid, t in active_tracks.items():
                            if tid in used_track_ids:
                                continue
                            rx, ry = t["centroid"]
                            tx1, ty1, tx2, ty2 = t["bbox"]
                            
                            # Calculate IoU
                            ix1 = max(x1, tx1)
                            iy1 = max(y1, ty1)
                            ix2 = min(x2, tx2)
                            iy2 = min(y2, ty2)
                            i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            u_area = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - i_area
                            iou = i_area / u_area if u_area > 0 else 0.0
                            
                            dist = np.sqrt((cx - rx)**2 + (cy - ry)**2)
                            face_size = max(x2 - x1, y2 - y1)
                            
                            if (iou > TRACK_ACTIVE_IOU_THRESHOLD or dist < max(TRACK_ACTIVE_MIN_DIST, face_size * TRACK_ACTIVE_DIST_FACTOR)) and dist < best_dist:
                                best_dist = dist
                                best_tid = tid
                                
                        if best_tid is not None:
                            used_track_ids.add(best_tid)
                            assigned_detections.append((face_dict, best_tid))
                        else:
                            # Start a new track
                            inherited_name = "Unknown"
                            inherited_uid = None
                            inherited_sim = 0.0
                            last_ai_run_val = 0.0
                            inherited_history = []
                            inherited_confirmed = False
                            
                            for old_tid, old_t in list(self.tracked_faces.items()):
                                is_lost = (current_time - old_t["last_seen"]) > 0.15
                                # Allow inheriting from BOTH confirmed tracks AND in-progress ("Unknown" but
                                # already accumulating consensus) tracks. This prevents a walking student's
                                # partial match_history from being wiped out just because they got a new
                                # track ID from a brief detection gap or fast movement.
                                has_progress = old_t["name"] != "Unknown" or len(old_t.get("match_history", [])) > 0
                                if is_lost and has_progress and (current_time - old_t["last_seen"]) < TRACK_MAX_AGE_SECONDS:
                                    old_rx, old_ry = old_t["centroid"]
                                    tx1, ty1, tx2, ty2 = old_t["bbox"]
                                    
                                    ix1 = max(x1, tx1)
                                    iy1 = max(y1, ty1)
                                    ix2 = min(x2, tx2)
                                    iy2 = min(y2, ty2)
                                    i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                    u_area = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - i_area
                                    iou = i_area / u_area if u_area > 0 else 0.0
                                    
                                    spatial_dist = np.sqrt((cx - old_rx)**2 + (cy - old_ry)**2)
                                    face_size = max(x2 - x1, y2 - y1)
                                    
                                    if iou > TRACK_LOST_IOU_THRESHOLD or spatial_dist < max(TRACK_LOST_MIN_DIST, face_size * TRACK_LOST_DIST_FACTOR):
                                        inherited_name = old_t["name"]
                                        inherited_uid = old_t["user_id"]
                                        inherited_sim = old_t["similarity"]
                                        last_ai_run_val = old_t["last_ai_run"]
                                        inherited_history = list(old_t.get("match_history", []))
                                        inherited_confirmed = old_t.get("confirmed", False)
                                        break
                            
                            tid = self.next_track_id
                            self.next_track_id += 1
                            self.tracked_faces[tid] = {
                                "name": inherited_name,
                                "user_id": inherited_uid,
                                "similarity": inherited_sim,
                                "last_seen": current_time,
                                "centroid": (cx, cy),
                                "bbox": (x1, y1, x2, y2),
                                "last_ai_run": last_ai_run_val,
                                "match_history": inherited_history,
                                "confirmed": inherited_confirmed
                            }
                            assigned_detections.append((face_dict, tid))
                    
                    # Process each assigned face
                    for idx, (face_dict, tid) in enumerate(assigned_detections):
                        x1, y1, x2, y2 = face_dict['bbox']
                        test_embedding = face_dict['embedding']
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        track = self.tracked_faces[tid]
                        track["centroid"] = (cx, cy)
                        track["bbox"] = (x1, y1, x2, y2)
                        track["last_seen"] = current_time
                        
                        # If track is already confirmed and recognized, keep presence timestamp refreshed & process Check-Out events
                        if track.get("confirmed") and track.get("user_id"):
                            user_id = track["user_id"]
                            name = track.get("name", "Staff")
                            refresh_presence_timestamp(user_id, current_time)

                            # Live ticker: refresh last-detection time every ~1s (display only)
                            if (current_time - track.get("last_ticker_refresh", 0)) >= 1.0:
                                track["last_ticker_refresh"] = current_time
                                _cl = getattr(self, 'camera_label', '') or f"Camera {self.camera_id}"
                                _rn = getattr(self, 'room_name', '') or 'Live Monitoring'
                                _cd = f"{_cl} ({_rn})" if _rn not in _cl else _cl
                                if not upsert_live_detection(
                                    user_id,
                                    camera_id=self.camera_id,
                                    source=f"stream_{self.camera_id}",
                                    camera_name=_cd,
                                    camera_label=_cl,
                                    room=_rn,
                                    room_name=_rn,
                                    location=_rn,
                                    confidence=float(track.get("similarity", 0) or 0) or None,
                                ):
                                    # Entry missing (e.g. list cleared) - re-create it
                                    upsert_live_detection(user_id, {
                                        "name": name,
                                        "confidence": float(track.get("similarity", 0) or 0),
                                        "source": f"stream_{self.camera_id}",
                                        "camera_id": self.camera_id,
                                        "camera_name": _cd,
                                        "camera_label": _cl,
                                        "room": _rn,
                                        "room_name": _rn,
                                        "location": _rn,
                                        "face_crop": track.get("face_crop"),
                                        "user_id": user_id,
                                        "department": "",
                                    })
                            
                            last_eval = track.get("last_attendance_eval", 0)
                            if (current_time - last_eval) >= 5.0:
                                track["last_attendance_eval"] = current_time
                                stream_source = f'stream_{self.camera_id}'
                                action_status, action_msg = process_user_attendance_event(
                                    user_id=user_id,
                                    name=name,
                                    confidence=float(track.get("similarity", 0.9)),
                                    source=stream_source,
                                    location=self.room_name,
                                    camera_id=self.camera_id,
                                    shift_info=None,
                                    face_crop_b64=track.get("face_crop")
                                )
                            
                        # If track identity is not yet locked/confirmed, perform biometric recognition
                        if (not track.get("confirmed") or track["name"] == "Unknown") and test_embedding is not None:
                            track["last_ai_run"] = current_time

                            # -----------------------------------------------------------------------
                            # STEP 1 (CROWD FIX): INSTANT VECTORIZED MATRIX SEARCH
                            # Single C/BLAS matrix dot product: O(1) in 0.05ms instead of looping
                            # over 100+ students in Python dictionaries.
                            # -----------------------------------------------------------------------
                            best_uid, cache_data, best_similarity, is_match = match_embedding_vectorized(
                                test_embedding, threshold=FACE_MATCHING_THRESHOLD
                            )
                            best_match = (best_uid, cache_data) if (is_match and best_uid is not None) else None

                            if best_match:
                                uid, cache_data = best_match
                                with cache_lock:
                                    MANUALLY_ABSENT_USERS.discard(uid)
                                # ----------------------------------------------------
                                # TEMPORAL CONSENSUS ENGINE (100% Reliable Attendance)
                                # ----------------------------------------------------
                                history = track.setdefault("match_history", [])
                                history.append((uid, best_similarity, current_time))
                                
                                # Filter matches within confirmation window
                                cutoff = current_time - ATTENDANCE_CONFIRMATION_WINDOW
                                track["match_history"] = [m for m in history if m[2] >= cutoff]
                                
                                recent_matches = [m for m in track["match_history"] if m[0] == uid]
                                num_matches = len(recent_matches)
                                avg_sim = sum(m[1] for m in recent_matches) / num_matches if num_matches > 0 else 0.0
                                
                                # CROWD RECOGNITION GATE:
                                # is_high_conf  - single-frame fast path: no wait, fires immediately at 0.45+ similarity.
                                is_high_conf = (best_similarity >= 0.45)
                                is_consensus = (num_matches >= ATTENDANCE_CONFIRMATION_FRAMES and avg_sim >= ATTENDANCE_CONFIRMATION_MIN_SIM)

                                if track.get("confirmed", False) or is_high_conf or is_consensus:
                                    if cache_data is None:
                                        continue
                                    name = cache_data['name'] if cache_data is not None else track.get('name', 'Unknown')
                                    user_id = uid
                                    similarity_score = best_similarity

                                    was_confirmed = track.get("confirmed", False)
                                    track["confirmed"] = True
                                    track["name"] = name
                                    track["user_id"] = user_id
                                    track["similarity"] = similarity_score

                                    if was_confirmed:
                                        reason = f"ALREADY CONFIRMED (reaffirm {best_similarity:.3f})"
                                    elif is_high_conf:
                                        reason = f"HIGH CONFIDENCE ({best_similarity:.3f} >= 0.45)"
                                    else:
                                        reason = f"CONSENSUS ({num_matches}/{ATTENDANCE_CONFIRMATION_FRAMES} frames, avg {avg_sim:.3f})"
                                    logger.info(f"[_ai_loop - Stream {self.camera_id} - Track {tid}] MATCH CONFIRMED [{reason}]: '{name}' in {self.room_name}")
                                    
                                    # Face crop extraction for live snapshot
                                    face_crop_b64 = None
                                    try:
                                        fh, fw = frame_to_process.shape[:2]
                                        fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
                                        crop = frame_to_process[fy1:fy2, fx1:fx2]
                                        if crop.size > 0:
                                            crop_resized = cv2.resize(crop, (80, 80))
                                            _, buf = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                            face_crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
                                            track["face_crop"] = face_crop_b64
                                    except Exception as ce:
                                        logger.warning(f"Face crop extraction failed: {ce}")
                                    
                                    # Update Global Multi-Camera Room Presence Registry
                                    update_live_presence(
                                        user_id=uid,
                                        name=name,
                                        department=cache_data.get('department', '') if cache_data is not None else '',
                                        camera_id=self.camera_id,
                                        camera_name=self.camera_label,
                                        room_name=self.room_name,
                                        confidence=float(best_similarity),
                                        face_crop=face_crop_b64
                                    )
                                    
                                    stream_source = f'stream_{self.camera_id}'
                                    action_status, action_msg = process_user_attendance_event(
                                        user_id=user_id,
                                        name=name,
                                        confidence=float(best_similarity),
                                        source=stream_source,
                                        location=self.room_name,
                                        camera_id=self.camera_id,
                                        shift_info=None,
                                        face_crop_b64=face_crop_b64
                                    )

                                    cam_label_val = getattr(self, 'camera_label', '') or f"Camera {self.camera_id}"
                                    room_name_val = getattr(self, 'room_name', '') or 'Live Monitoring'
                                    cam_display = f"{cam_label_val} ({room_name_val})" if cam_label_val and room_name_val and room_name_val not in cam_label_val else (cam_label_val or room_name_val)

                                    detection_entry = {
                                        "name": name,
                                        "timestamp": datetime.now().strftime("%I:%M:%S %p"),
                                        "confidence": float(best_similarity),
                                        "source": f"stream_{self.camera_id}",
                                        "camera_id": self.camera_id,
                                        "camera_name": cam_display,
                                        "camera_label": cam_label_val,
                                        "room": room_name_val,
                                        "room_name": room_name_val,
                                        "location": room_name_val,
                                        "face_crop": face_crop_b64,
                                        "user_id": uid,
                                        "department": cache_data.get('department', '') if cache_data is not None else ''
                                    }
                                    # One row per employee: update time & move to top on every re-appearance
                                    upsert_live_detection(uid, detection_entry)

                                    if action_status == "REJECTED_OUTSIDE_SHIFT":
                                        logger.info(f"[_ai_loop - Stream {self.camera_id}] Shift check blocked attendance for '{name}': {action_msg}")
                                else:
                                    logger.debug(f"[_ai_loop - Stream {self.camera_id} - Track {tid}] Accumulating consensus for '{cache_data['name'] if cache_data is not None else 'Unknown'}': {num_matches}/{ATTENDANCE_CONFIRMATION_FRAMES} frames (avg {avg_sim:.3f})")
                            else:
                                if track["name"] == "Unknown":
                                    logger.debug(f"[_ai_loop - Stream {self.camera_id} - Track {tid}] Biometric check: No match (Highest: {best_similarity:.4f})")
                                    
                    # Build new_cached_detections from all active tracks (smooth grace period based on TRACK_MAX_AGE_SECONDS)
                    for tid, track in list(self.tracked_faces.items()):
                        if (current_time - track["last_seen"]) < TRACK_MAX_AGE_SECONDS:
                            name = track["name"]
                            similarity_score = track["similarity"]
                            bbox = track["bbox"]
                            color = (0, 255, 0) if (name != "Unknown" and track.get("confirmed")) else (0, 0, 255)
                            
                            display_name = f"{name} ({int(similarity_score * 100)}%)" if track.get("confirmed") else "Verifying..."
                            new_cached_detections.append({
                                'bbox': bbox,
                                'name': display_name,
                                'color': color,
                                'similarity': similarity_score
                            })
                        
                    # Clean up very old inactive tracks (>6.0 seconds)
                    self.tracked_faces = {tid: t for tid, t in self.tracked_faces.items() if (current_time - t["last_seen"]) < 6.0}

                    # If a manually absent user has left the scene entirely, allow them to be recognized again later.
                    with cache_lock:
                        active_suppressed = {
                            t.get('user_id') for t in self.tracked_faces.values() if t.get('user_id') in MANUALLY_ABSENT_USERS
                        }
                        for uid in list(MANUALLY_ABSENT_USERS):
                            if uid not in active_suppressed:
                                MANUALLY_ABSENT_USERS.discard(uid)
                    
                    with self.lock:
                        self.latest_detections = new_cached_detections
                        
                except Exception as e:
                    logger.error(f"Background AI processing failed for {self.camera_id}: {e}")
            else:
                # Keep background recognition alive when the UI feed is paused; clear only when truly idle and not paused.
                has_viewers = self.viewers_count > 0 or self.paused
                if not has_viewers and not self.paused and len(self.latest_detections) > 0:
                    with self.lock:
                        self.latest_detections = []
            
            time.sleep(0.02)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None, []
            return True, self.latest_frame.copy(), list(self.latest_detections)

    def get_annotated_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        try:
            self.grabber_thread.join(timeout=1.0)
            self.ai_thread.join(timeout=1.0)
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def release(self):
        self.stop()


def get_or_create_reader(camera_id, rtsp_url):
    """Retrieve or spin up a background RTSP stream reader."""
    global active_stream_readers
    with active_readers_lock:
        if camera_id in active_stream_readers:
            existing = active_stream_readers[camera_id]
            if existing.running and str(existing.rtsp_url) == str(rtsp_url):
                with monitoring_state_lock:
                    existing.paused = STREAM_PAUSED_STATE.get(camera_id, False)
                return existing
            else:
                logger.info(f"[*] Camera {camera_id} URL changed or stopped. Recreating stream reader...")
                try:
                    existing.stop()
                except Exception as e:
                    logger.warning(f"Error stopping old reader for {camera_id}: {e}")
                del active_stream_readers[camera_id]
        
        logger.info(f"[*] Starting background thread stream reader for: {camera_id} ({rtsp_url})")
        reader = CameraStreamReader(camera_id, rtsp_url)
        with monitoring_state_lock:
            reader.paused = STREAM_PAUSED_STATE.get(camera_id, False)
        active_stream_readers[camera_id] = reader
        return reader


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
    """Fetch camera list in real time from Cloud or Client Portal and update streams using get_or_create_reader."""
    global ACTIVE_CAMERA_READERS, active_stream_readers
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

    with active_readers_lock:
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
        stale_cids = [cid for cid in list(active_stream_readers.keys()) if cid not in valid_cams]
        for cid in stale_cids:
            reader = active_stream_readers.pop(cid, None)
            if reader:
                try:
                    reader.stop()
                    logger.info(f"[*] Safely stopped deleted camera stream: {cid}")
                except Exception as e:
                    logger.debug(f"Error stopping camera {cid}: {e}")

        # 2. Add or update cameras using exact get_or_create_reader
        for cid, (cname, source) in valid_cams.items():
            try:
                reader = get_or_create_reader(cid, source)
                logger.info(f"Active camera stream reader: {cname} ({cid}) -> {source}")
            except Exception as e:
                logger.error(f"Failed to start camera {cid}: {e}")

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
# FLASK HTTP ROUTES & STREAMING
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
    with active_readers_lock:
        active_count = len(active_stream_readers)
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
    with active_readers_lock:
        reader = active_stream_readers.get(str(camera_id))

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

        with reader.lock:
            reader.viewers_count += 1

        try:
            while True:
                if not reader or not reader.running:
                    break

                ret, frame, detections = reader.get_frame()
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

                    success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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
        finally:
            with reader.lock:
                reader.viewers_count = max(0, reader.viewers_count - 1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/node/cameras')
def get_node_cameras():
    """Return active cameras for frontend grid layout."""
    with active_readers_lock:
        cams = [
            {"id": cid, "name": getattr(r, 'camera_label', f"Camera {cid}"), "stream_url": f"/stream/{cid}"}
            for cid, r in active_stream_readers.items()
        ]
    return jsonify({"success": True, "cameras": cams})


@app.route('/api/node/detections')
def get_node_detections():
    """Real-time detection ticker."""
    global LATEST_STREAM_DETECTIONS
    with cache_lock:
        feed = list(LATEST_STREAM_DETECTIONS)
    return jsonify({"success": True, "detections": feed})


@app.route('/api/live-detections', methods=['GET'])
def get_live_detections():
    global LATEST_STREAM_DETECTIONS
    with cache_lock:
        return jsonify({"detections": LATEST_STREAM_DETECTIONS, "success": True})
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
