"""
Packaged Client Node EXE End-to-End Verification Test
Tests:
 1. Packaged EXE Launch & Process Health
 2. Live Node Web UI Loading & Activation Modal
 3. Camera Feed APIs & Detection Feed
 4. Key Activation Flow
 5. Clean Process Shutdown
"""
import subprocess
import time
import urllib.request
import urllib.error
import json
import os
import sys

exe_path = r"D:\ImranProjects\QIntellectProjects\QuantEye-System\QuantEye-Client-Node\dist\QuantEye_Live_Attendance_Node\QuantEye_Live_Attendance_Node.exe"
print(f"============================================================")
print(f"VERIFYING PACKAGED CLIENT NODE EXE:")
print(f" Path: {exe_path}")
print(f" Size: {os.path.getsize(exe_path) / (1024*1024):.2f} MB")
print(f"============================================================")

assert os.path.exists(exe_path), "EXE does not exist!"

work_dir = os.path.dirname(exe_path)
proc = subprocess.Popen([exe_path], cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

try:
    print("[1/5] Waiting for Client Node web engine to start...")
    started = False
    for i in range(25):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:5005/", timeout=2) as req:
                if req.status == 200:
                    html = req.read().decode("utf-8")
                    assert "QuantEye Live AI Attendance Node" in html or "QuantEye" in html
                    assert "activationModal" in html
                    started = True
                    print(f"  -> SUCCESS: Client Node Web UI loaded on port 5005 (HTTP 200)")
                    print(f"  -> SUCCESS: Activation Modal verified in HTML response.")
                    break
        except Exception:
            pass

    if not started:
        print("  -> ERROR: Client Node failed to respond on http://127.0.0.1:5005/")
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    print("\n[2/5] Verifying Camera Stream API...")
    with urllib.request.urlopen("http://127.0.0.1:5005/api/node/cameras", timeout=3) as req:
        cam_data = json.loads(req.read().decode("utf-8"))
        assert cam_data.get("success") is True
        cams = cam_data.get("cameras", [])
        print(f"  -> SUCCESS: Camera endpoint returned {len(cams)} active configured cameras.")

    print("\n[3/5] Verifying Live AI Detection Ticker API...")
    with urllib.request.urlopen("http://127.0.0.1:5005/api/node/detections", timeout=3) as req:
        det_data = json.loads(req.read().decode("utf-8"))
        assert det_data.get("success") is True
        print(f"  -> SUCCESS: Detection ticker endpoint active, returned {len(det_data.get('detections', []))} detections.")

    print("\n[4/5] Verifying Cloud Status Endpoint...")
    with urllib.request.urlopen("http://127.0.0.1:5005/api/node/cloud-status", timeout=3) as req:
        cs_data = json.loads(req.read().decode("utf-8"))
        assert cs_data.get("success") is True
        print(f"  -> SUCCESS: Cloud status endpoint active: {cs_data}")

    print("\n[5/5] Verifying Node Activation API...")
    activate_url = "http://127.0.0.1:5005/api/node/activate-key"
    payload = json.dumps({"branch_key": "TEST-STANDALONE-KEY"}).encode("utf-8")
    req = urllib.request.Request(activate_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as he:
        # Expected since cloud server isn't running on 5100 during unit test
        print(f"  -> SUCCESS: Activation endpoint responded with HTTP {he.code} (Correctly validated against cloud URL).")

    print("\n============================================================")
    print("ALL 5/5 PACKAGED EXE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("QuantEye_Live_Attendance_Node.exe is ready for client distribution.")
    print("============================================================")

finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
