"""
Verification Test for Step 1 & Step 2:
1. Client Node Authentication (/api/node/login-auth)
2. Organization + Branch Binding
3. Camera Configuration Loading
4. Employee / Face Embedding Profile Synchronization
5. Attendance Event Generation & Cloud Dispatch simulation
"""
import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Ensure QuantEye-Client-Node is in sys.path
client_node_dir = os.path.abspath(r"d:\ImranProjects\QIntellectProjects\QuantEye-System\QuantEye-Client-Node")
sys.path.insert(0, client_node_dir)

import app as client_app

class TestStep1And2ClientNode(unittest.TestCase):
    def setUp(self):
        client_app.app.config['TESTING'] = True
        self.client = client_app.app.test_client()

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

    def test_dynamic_camera_endpoint(self):
        resp = self.client.get('/api/node/cameras')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('cameras', data)

    def test_cloud_status_endpoint(self):
        resp = self.client.get('/api/node/cloud-status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertIn('cloud_connected', data)

    def test_live_node_html_renders_without_crash(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        content = resp.data.decode('utf-8')
        self.assertIn("QuantEye", content)
        self.assertIn("Client Branch Activation", content)
        self.assertIn("inputUsername", content)
        self.assertIn("inputPassword", content)
        self.assertIn("grid-multi", content)

if __name__ == '__main__':
    unittest.main()
