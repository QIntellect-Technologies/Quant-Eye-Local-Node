"""
Verification Test for Step 1 & Step 2:
1. Client Node Authentication (/api/node/login-auth)
2. Organization + Branch Binding
3. Camera Configuration Loading
4. Employee / Face Embedding Profile Synchronization
5. Attendance Event Generation & Cloud Dispatch simulation
"""
import sys
import time
import os
import json
import sqlite3
import copy
import tempfile
import unittest
import uuid
import zipfile
import atexit
from unittest.mock import patch, MagicMock

_ORIGINAL_DB_PATH_ENV = os.environ.get("QINTELLECT_DB_PATH")
_TEST_DB_DIR = tempfile.TemporaryDirectory(prefix="quanteye-node-tests-")
atexit.register(_TEST_DB_DIR.cleanup)
os.environ["QINTELLECT_DB_PATH"] = os.path.join(_TEST_DB_DIR.name, "attendance.db")

# Ensure QuantEye-Client-Node is in sys.path
client_node_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(client_node_dir)
sys.path.insert(0, client_node_dir)

# The projects reuse generic top-level module names. Drop sibling-project
# modules left behind when pytest collects multiple project suites together.
for module_name, module in list(sys.modules.items()):
    module_file = getattr(module, "__file__", None)
    if not module_file:
        continue
    module_path = os.path.normcase(os.path.abspath(module_file))
    if not module_path.startswith(os.path.normcase(project_root + os.sep)):
        continue
    if not module_path.startswith(os.path.normcase(client_node_dir + os.sep)):
        del sys.modules[module_name]

import app as client_app

class TestStep1And2ClientNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_config_path = client_app.CONFIG_PATH
        cls.original_config = copy.deepcopy(client_app.CONFIG)
        cls.test_config_dir = tempfile.TemporaryDirectory()
        client_app.CONFIG_PATH = os.path.join(cls.test_config_dir.name, 'config_node.json')
        if not client_app.db.init_db():
            raise RuntimeError("Could not initialize isolated Client Node test database")

    def setUp(self):
        client_app.app.config['TESTING'] = True
        self.client = client_app.app.test_client()
        client_app.CONFIG.clear()
        client_app.CONFIG.update(copy.deepcopy(self.original_config))

    @classmethod
    def tearDownClass(cls):
        client_app._cloud_delivery_shutdown.set()
        client_app._cloud_delivery_wakeup.set()
        client_app._cloud_delivery_thread.join(timeout=5)
        client_app.CONFIG_PATH = cls.original_config_path
        client_app.CONFIG.clear()
        client_app.CONFIG.update(cls.original_config)
        cls.test_config_dir.cleanup()
        if _ORIGINAL_DB_PATH_ENV is None:
            os.environ.pop("QINTELLECT_DB_PATH", None)
        else:
            os.environ["QINTELLECT_DB_PATH"] = _ORIGINAL_DB_PATH_ENV

    def test_missing_credentials_returns_400(self):
        resp = self.client.post('/api/node/login-auth', json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data['success'])

    @patch('urllib.request.urlopen')
    def test_successful_client_node_activation_and_binding(self, mock_urlopen):
        # Mock responses from Cloud /api/client/authenticate and /api/edge/validate-key
        class MockResponse:
            def __init__(self, data, status=200):
                self._data = data
                self.status = status
            def read(self):
                return json.dumps(self._data).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def side_effect(req, *args, **kwargs):
            url = req.full_url
            if '/api/client/authenticate' in url:
                return MockResponse({
                    "success": True,
                    "organization_id": "ORG-TEST-CLIENT",
                    "organization_name": "Test Client Corp",
                    "branch_id": "BR-TEST-01",
                    "branch_name": "Main Office",
                    "branch_key": "QE-TEST-KEY-12345"
                })
            elif '/api/edge/validate-key' in url:
                return MockResponse({
                    "success": True,
                    "organization_id": "ORG-TEST-CLIENT",
                    "organization_name": "Test Client Corp",
                    "branch_id": "BR-TEST-01",
                    "branch_name": "Main Office",
                    "cameras": [
                        {"id": "cam_1", "name": "Entrance NVR", "rtsp_url": "0", "channel_number": 1, "active": 1},
                        {"id": "cam_2", "name": "Backyard DVR", "rtsp_url": "0", "channel_number": 2, "active": 1},
                        {"id": "cam_3", "name": "Hallway Cam", "rtsp_url": "0", "channel_number": 3, "active": 1},
                        {"id": "cam_4", "name": "Parking Area", "rtsp_url": "0", "channel_number": 4, "active": 1}
                    ]
                })
            return MockResponse({"success": False}, status=404)

        mock_urlopen.side_effect = side_effect

        payload = {
            "username": "client_admin@test.com",
            "password": "Pass1234!",
            "branch_key": "QE-TEST-KEY-12345",
            "cloud_url": "http://127.0.0.1:5100"
        }

        resp = self.client.post('/api/node/login-auth', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertTrue(data['linked'])
        self.assertEqual(data['organization_id'], "ORG-TEST-CLIENT")
        self.assertEqual(data['branch_id'], "BR-TEST-01")
        self.assertEqual(data['camera_count'], 4)

        # Verify config was updated in memory
        self.assertEqual(client_app.CONFIG['organization_id'], "ORG-TEST-CLIENT")
        self.assertEqual(client_app.CONFIG['branch_id'], "BR-TEST-01")
        self.assertEqual(len(client_app.CONFIG['cameras']), 4)
        persisted_config = None
        for _ in range(40):
            try:
                with open(client_app.CONFIG_PATH, encoding="utf-8") as config_file:
                    persisted_config = json.load(config_file)
                break
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
        self.assertIsNotNone(persisted_config)
        self.assertTrue(persisted_config["linked"])
        self.assertEqual(persisted_config["branch_key"], "QE-TEST-KEY-12345")
        self.assertEqual(persisted_config["cameras"], client_app.CONFIG["cameras"])

    def test_dynamic_camera_endpoint(self):
        resp = self.client.get('/api/node/cameras')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('cameras', data)

    @patch('urllib.request.urlopen')
    def test_camera_sync_authenticates_branch_and_preserves_cache_on_failure(self, mock_urlopen):
        cached_camera = {"id": "cam-cached", "rtsp_url": "rtsp://camera.local/stream"}
        client_app.CONFIG.update({
            "cloud_url": "http://cloud.test",
            "branch_key": "QE-TEST-BRANCH-KEY",
            "cameras": [cached_camera],
        })
        mock_urlopen.side_effect = OSError("cloud is unavailable")

        cameras = client_app.fetch_all_cloud_cameras("ORG-TEST", "BR-TEST")
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]["id"], cached_camera["id"])
        self.assertEqual(cameras[0]["rtsp_url"], cached_camera["rtsp_url"])
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("X-branch-key"), "QE-TEST-BRANCH-KEY")

    @patch('urllib.request.urlopen')
    def test_authoritative_empty_camera_sync_clears_cache(self, mock_urlopen):
        class MockResponse:
            status = 200

            def read(self):
                return b'{"success": true, "cameras": []}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        client_app.CONFIG.update({
            "cloud_url": "http://cloud.test",
            "branch_key": "QE-TEST-BRANCH-KEY",
            "cameras": [{"id": "cam-stale", "rtsp_url": "rtsp://camera.local/stale"}],
        })
        mock_urlopen.return_value = MockResponse()

        self.assertEqual(client_app.fetch_all_cloud_cameras("ORG-TEST", "BR-TEST"), [])

    def test_logout_clears_active_organization_runtime_state(self):
        client_app.CONFIG.update({
            "linked": True,
            "organization_id": "ORG-OLD",
            "organization_name": "Old Organization",
            "branch_id": "BR-OLD",
            "branch_name": "Old Branch",
            "branch_key": "old-branch-key",
            "username": "old@example.test",
            "cameras": [{"id": "old-camera"}],
        })
        client_app.EMBEDDING_CACHE[7] = {"name": "Old Employee"}
        client_app.ENROLLED_MATRIX = client_app.np.ones((1, 512), dtype=client_app.np.float32)
        client_app.ENROLLED_USER_IDS = [7]
        client_app.ENROLLED_USER_METADATA = [{"id": 7, "name": "Old Employee"}]
        client_app.LATEST_STREAM_DETECTIONS.append({"name": "Old Employee"})
        client_app.TODAY_LOGGED_ATTENDANCE.add((7, "old-camera"))
        client_app.USER_ATTENDANCE_STATE[7] = {"checkin_time": "old"}

        response = self.client.post('/api/node/logout')

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        self.assertFalse(client_app.CONFIG["linked"])
        self.assertIsNone(client_app.CONFIG["organization_id"])
        self.assertIsNone(client_app.CONFIG["branch_id"])
        self.assertIsNone(client_app.CONFIG["branch_key"])
        self.assertEqual(client_app.CONFIG["username"], "")
        self.assertEqual(client_app.CONFIG["cameras"], [])
        self.assertEqual(client_app.EMBEDDING_CACHE, {})
        self.assertIsNone(client_app.ENROLLED_MATRIX)
        self.assertEqual(client_app.ENROLLED_USER_IDS, [])
        self.assertEqual(client_app.ENROLLED_USER_METADATA, [])
        self.assertEqual(client_app.LATEST_STREAM_DETECTIONS, [])
        self.assertEqual(client_app.TODAY_LOGGED_ATTENDANCE, set())
        self.assertEqual(client_app.USER_ATTENDANCE_STATE, {})

        page = self.client.get('/')
        self.assertIn(b'<body class="node-unlinked">', page.data)

    def test_face_cache_only_uses_exact_active_tenant_and_branch(self):
        client_app.CONFIG.update({
            "linked": True,
            "organization_id": "ORG-NEW",
            "branch_id": "BR-NEW",
        })
        users = [
            {"id": 1, "name": "Old tenant", "organization_id": "ORG-OLD", "branch_id": "BR-OLD"},
            {"id": 2, "name": "New tenant", "organization_id": "ORG-NEW", "branch_id": "BR-NEW"},
            {"id": 3, "name": "Unscoped", "organization_id": "DEFAULT_ORG", "branch_id": "DEFAULT_BRANCH"},
        ]
        with patch.object(client_app.db, "get_all_users", return_value=users), \
             patch.object(client_app.db, "get_embeddings_for_user", return_value=[{"embedding": [1.0, 0.0]}]), \
             patch.object(client_app.fp, "compute_aggregate_embedding", return_value=client_app.np.array([1.0, 0.0])):
            client_app.reload_face_cache()

        self.assertEqual(client_app.ENROLLED_USER_IDS, [2])
        self.assertEqual(set(client_app.EMBEDDING_CACHE), {2})

    def test_unlinked_node_keeps_face_cache_empty(self):
        client_app.CONFIG.update({
            "linked": False,
            "organization_id": None,
            "branch_id": None,
        })
        client_app.EMBEDDING_CACHE[1] = {"name": "Old employee"}
        client_app.ENROLLED_MATRIX = client_app.np.ones((1, 512), dtype=client_app.np.float32)
        with patch.object(client_app.db, "get_all_users") as get_users:
            client_app.reload_face_cache()

        get_users.assert_not_called()
        self.assertEqual(client_app.EMBEDDING_CACHE, {})
        self.assertIsNone(client_app.ENROLLED_MATRIX)
        self.assertEqual(client_app.ENROLLED_USER_IDS, [])

    def test_cloud_status_endpoint(self):
        resp = self.client.get('/api/node/cloud-status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('cloud_connected', data)

    def test_live_node_html_renders_without_crash(self):
        client_app.CONFIG['linked'] = False
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        content = resp.data.decode('utf-8')
        self.assertIn("QuantEye", content)
        self.assertIn("Sign in to your organization", content)
        self.assertIn('id="authUsernameInput"', content)
        self.assertIn('id="authPasswordInput"', content)
        self.assertIn('id="liveMultiGrid"', content)
        self.assertIn("openActivationModal();", content)

    def test_attendance_submission_uses_persisted_branch_credential(self):
        client_app.CONFIG.update({
            "linked": True,
            "branch_key": "issued-branch-key",
            "cloud_url": "https://main.example.test",
        })
        event = {"event_id": str(uuid.uuid4()), "user_id": 101, "name": "Alice"}
        response = MagicMock(status=201)
        response.__enter__.return_value = response
        with patch("app.urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(client_app._submit_attendance_event_to_main(event), 201)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://main.example.test/api/sync/attendance-event")
        self.assertEqual(request.get_header("X-branch-key"), "issued-branch-key")
        self.assertEqual(json.loads(request.data)["event_id"], event["event_id"])

    def test_attendance_submission_fails_closed_without_branch_binding(self):
        client_app.CONFIG.update({"linked": False, "branch_key": ""})
        with patch("app.urllib.request.urlopen") as urlopen:
            with self.assertRaises(RuntimeError):
                client_app._submit_attendance_event_to_main({"event_id": str(uuid.uuid4())})
        urlopen.assert_not_called()

    def test_cloud_manual_absence_is_mirrored_once_to_local_attendance_state(self):
        client_app.CONFIG.update({
            "linked": True,
            "branch_key": "issued-branch-key",
            "organization_id": "ORG-CLOUD-SYNC",
            "branch_id": "BR-CLOUD-SYNC",
        })
        user_id = client_app.db.add_user(
            name=f"Cloud Absence Sync {uuid.uuid4().hex[:8]}",
            organization_id="ORG-CLOUD-SYNC",
            branch_id="BR-CLOUD-SYNC",
        )
        self.assertIsNotNone(user_id)
        user_id = int(user_id)
        client_app._cloud_absent_users_synced.discard(user_id)
        client_app.USER_ATTENDANCE_STATE[user_id] = {
            "checkin_time": client_app.datetime.now(),
            "last_checkout_log": None,
        }
        state = {
            "success": True,
            "organization_id": "ORG-CLOUD-SYNC",
            "branch_id": "BR-CLOUD-SYNC",
            "manual_absent_user_ids": [user_id],
        }

        first_sync = client_app.apply_cloud_attendance_sync_state(state)
        second_sync = client_app.apply_cloud_attendance_sync_state(state)

        self.assertEqual(first_sync["updated_local_count"], 1)
        self.assertEqual(second_sync["updated_local_count"], 0)
        self.assertTrue(client_app.db.is_user_manually_absent_today(user_id))
        self.assertNotIn(user_id, client_app.USER_ATTENDANCE_STATE)

    def test_attendance_sync_state_request_uses_saved_branch_credential(self):
        client_app.CONFIG.update({
            "linked": True,
            "branch_key": "issued-branch-key",
            "organization_id": "ORG-CLOUD-SYNC",
            "branch_id": "BR-CLOUD-SYNC",
            "cloud_url": "https://main.example.test",
        })
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "success": True,
            "organization_id": "ORG-CLOUD-SYNC",
            "branch_id": "BR-CLOUD-SYNC",
            "manual_absent_user_ids": [],
        }).encode("utf-8")
        with patch("app.urllib.request.urlopen", return_value=response) as urlopen:
            state = client_app.fetch_cloud_attendance_sync_state()

        self.assertTrue(state["success"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://main.example.test/api/edge/sync-state")
        self.assertEqual(request.get_header("X-branch-key"), "issued-branch-key")

    def test_restart_refreshes_profiles_attendance_state_and_camera_streams(self):
        client_app.CONFIG.update({
            "linked": True,
            "branch_key": "issued-branch-key",
            "organization_id": "ORG-CLOUD-SYNC",
            "branch_id": "BR-CLOUD-SYNC",
            "cloud_url": "https://main.example.test",
            "cameras": [{"id": "camera-1", "name": "Front Door"}],
        })
        with patch.object(client_app, "sync_client_profiles_from_cloud", return_value=3) as sync_profiles, \
             patch.object(client_app, "reload_face_cache") as refresh_cache, \
             patch.object(
                 client_app,
                 "apply_cloud_attendance_sync_state",
                 return_value={"manual_absent_count": 1, "updated_local_count": 1},
             ) as sync_attendance, \
             patch.object(client_app, "init_camera_streams") as refresh_cameras:
            response = self.client.post("/api/node/cameras/refresh")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["employee_count"], 3)
        self.assertEqual(result["attendance_updates_applied"], 1)
        self.assertEqual(result["camera_count"], 1)
        sync_profiles.assert_called_once_with(
            "ORG-CLOUD-SYNC",
            "BR-CLOUD-SYNC",
            "https://main.example.test",
        )
        refresh_cache.assert_called_once_with()
        sync_attendance.assert_called_once_with()
        refresh_cameras.assert_called_once_with()

    def test_cloud_delivery_retries_persist_and_preserve_event_id(self):
        event_id = str(uuid.uuid4())
        event = {"event_id": event_id, "user_id": 101, "name": "Retry Employee"}
        client_app.db.enqueue_client_cloud_delivery(event_id, event)

        with patch.object(
            client_app, "_submit_attendance_event_to_main",
            side_effect=ConnectionError("Main unavailable"),
        ):
            self.assertEqual(client_app._deliver_due_cloud_events(), 1)

        with client_app.db.sqlite3.connect(client_app.db.DB_PATH) as conn:
            row = conn.execute(
                "SELECT event_id, payload_json, status, attempts FROM client_cloud_delivery "
                "WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            self.assertEqual(row[0], event_id)
            self.assertEqual(json.loads(row[1]), event)
            self.assertEqual(row[2], "pending")
            self.assertEqual(row[3], 1)
            conn.execute(
                "UPDATE client_cloud_delivery "
                "SET next_attempt_at = datetime('now', 'localtime', '-1 second') "
                "WHERE event_id = ?",
                (event_id,),
            )

        # Re-read from SQLite to model a process restart before Main recovers.
        due = client_app.db.get_due_client_cloud_deliveries()
        self.assertIn(event_id, {item["event_id"] for item in due})
        with patch.object(client_app, "_submit_attendance_event_to_main", return_value=200) as send:
            self.assertEqual(client_app._deliver_due_cloud_events(), 1)

        send.assert_called_once_with(event)
        with client_app.db.sqlite3.connect(client_app.db.DB_PATH) as conn:
            row = conn.execute(
                "SELECT status, attempts FROM client_cloud_delivery WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        self.assertEqual(row, ("delivered", 1))

    def test_cloud_delivery_spools_when_sqlite_is_unavailable(self):
        event_id = str(uuid.uuid4())
        event = {"event_id": event_id, "user_id": 102, "name": "Spool Employee"}
        with patch.object(
            client_app.db.sqlite3,
            "connect",
            side_effect=client_app.db.sqlite3.OperationalError("database unavailable"),
        ):
            client_app.db.enqueue_client_cloud_delivery(event_id, event)

        self.assertTrue(os.path.isdir(client_app.db._CLIENT_CLOUD_SPOOL_DIR))
        due = client_app.db.get_due_client_cloud_deliveries()
        self.assertIn(event_id, {item["event_id"] for item in due})
        with patch.object(client_app, "_submit_attendance_event_to_main", return_value=201) as send:
            self.assertEqual(client_app._deliver_due_cloud_events(), 1)
        send.assert_called_once_with(event)
        with client_app.db.sqlite3.connect(client_app.db.DB_PATH) as conn:
            row = conn.execute(
                "SELECT status FROM client_cloud_delivery WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        self.assertEqual(row, ("delivered",))

    def test_build_specs_and_distribution_configs_are_clean(self):
        clean_config = {
            "linked": False,
            "organization_id": "",
            "branch_id": "",
            "branch_key": "",
            "cameras": [],
        }
        for spec_name in (
            "QuantEye_Client_Node_CPU.spec",
            "QuantEye_Client_Node_GPU.spec",
            "QuantEye_Live_Attendance_Node.spec",
        ):
            with open(os.path.join(client_node_dir, spec_name), encoding="utf-8") as spec_file:
                spec = spec_file.read()
            self.assertNotIn("('config_node.json', '.')", spec)
            self.assertIn("'config_node_dev_backup.json'", spec)

        with open(os.path.join(client_node_dir, "config_node.example.json"), encoding="utf-8") as example_file:
            example_config = json.load(example_file)
        for field, expected in clean_config.items():
            self.assertEqual(example_config[field], expected)

        # Existing dist folders may contain a client's persisted activation state.
        dist_dir = os.path.join(client_node_dir, "dist", "camera_sync_build")
        for package_name in (
            "QuantEye_Client_Node_CPU",
            "QuantEye_Client_Node_GPU",
            "QuantEye_Live_Attendance_Node",
        ):
            for config_name in ("config_node.json", os.path.join("_internal", "config_node.json")):
                config_path = os.path.join(dist_dir, package_name, config_name)
                if os.path.isfile(config_path):
                    with open(config_path, encoding="utf-8") as config_file:
                        package_config = json.load(config_file)
                    for field, expected in clean_config.items():
                        self.assertEqual(package_config[field], expected, config_path)

            archive_path = os.path.join(dist_dir, package_name + ".zip")
            if not os.path.isfile(archive_path):
                continue
            with zipfile.ZipFile(archive_path) as archive:
                self.assertFalse(any(
                    name.endswith("config_node_dev_backup.json")
                    for name in archive.namelist()
                ))
                for name in archive.namelist():
                    if not name.endswith("/config_node.json"):
                        continue
                    package_config = json.loads(archive.read(name))
                    for field, expected in clean_config.items():
                        self.assertEqual(package_config[field], expected, name)

    def test_legacy_payroll_month_constraint_migrates_to_org_branch_scope(self):
        with tempfile.TemporaryDirectory(prefix="quanteye-node-payroll-") as temp_dir:
            legacy_path = os.path.join(temp_dir, "legacy.db")
            with sqlite3.connect(legacy_path) as conn:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.executescript("""
                    CREATE TABLE payroll_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        payroll_month TEXT NOT NULL UNIQUE,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        employee_count INTEGER NOT NULL DEFAULT 0,
                        total_basic_salary REAL NOT NULL DEFAULT 0,
                        total_deduction REAL NOT NULL DEFAULT 0,
                        total_net_salary REAL NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'Finalized',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        finalized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_allowances REAL NOT NULL DEFAULT 0,
                        total_overtime REAL NOT NULL DEFAULT 0,
                        organization_id TEXT NOT NULL DEFAULT 'DEFAULT_ORG',
                        branch_id TEXT NOT NULL DEFAULT 'DEFAULT_BRANCH'
                    );
                    CREATE TABLE payroll_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        payroll_run_id INTEGER NOT NULL,
                        FOREIGN KEY(payroll_run_id) REFERENCES payroll_runs(id)
                    );
                    INSERT INTO payroll_runs (
                        id, payroll_month, period_start, period_end, organization_id, branch_id
                    ) VALUES (7, '2026-10', '2026-10-01', '2026-10-31', 'ORG-A', 'BR-A1');
                    INSERT INTO payroll_records (payroll_run_id) VALUES (7);
                    CREATE TABLE payroll_runs_scoped (id INTEGER PRIMARY KEY);
                """)
                client_app.db._migrate_payroll_run_tenant_uniqueness(conn.cursor())
                conn.commit()
                conn.execute("PRAGMA foreign_keys = ON")

                preserved = conn.execute(
                    "SELECT id, payroll_month, organization_id, branch_id FROM payroll_runs WHERE id = 7"
                ).fetchone()
                self.assertEqual(preserved, (7, "2026-10", "ORG-A", "BR-A1"))
                self.assertEqual(
                    conn.execute("SELECT payroll_run_id FROM payroll_records").fetchone(),
                    (7,),
                )
                self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
                for organization_id, branch_id in (("ORG-B", "BR-B1"), ("ORG-A", "BR-A2")):
                    conn.execute(
                        """INSERT INTO payroll_runs (
                               payroll_month, period_start, period_end, organization_id, branch_id
                           ) VALUES ('2026-10', '2026-10-01', '2026-10-31', ?, ?)""",
                        (organization_id, branch_id),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """INSERT INTO payroll_runs (
                               payroll_month, period_start, period_end, organization_id, branch_id
                           ) VALUES ('2026-10', '2026-10-01', '2026-10-31', 'ORG-A', 'BR-A1')"""
                    )
            conn.close()

if __name__ == '__main__':
    unittest.main()
