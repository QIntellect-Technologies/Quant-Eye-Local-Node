import subprocess
import time
import urllib.request
import json
import sys
import os

exe_path = r"D:\ImranProjects\QIntellectProjects\QuantEye-System\QuantEye-Client-Node\dist\QuantEye_Live_Attendance_Node\QuantEye_Live_Attendance_Node.exe"
print("Testing packaged EXE:", exe_path)

if not os.path.exists(exe_path):
    print("ERROR: Packaged EXE does not exist!")
    sys.exit(1)

work_dir = os.path.dirname(exe_path)
proc = subprocess.Popen([exe_path], cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

try:
    started = False
    for i in range(45):
        time.sleep(1)
        try:
            req = urllib.request.urlopen("http://127.0.0.1:5005/api/node/health", timeout=2)
            if req.status == 200:
                data = json.loads(req.read().decode("utf-8"))
                print("PASSED [1/3] Healthcheck OK:", data)
                started = True
                break
        except Exception:
            pass

    if not started:
        print("ERROR: Could not reach /api/node/health within timeout.")
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        print("STDOUT:", stdout[:1000])
        print("STDERR:", stderr[:1000])
        sys.exit(1)

    # 2. Check UI page
    html_req = urllib.request.urlopen("http://127.0.0.1:5005/", timeout=3)
    html = html_req.read().decode("utf-8")
    assert "QuantEye AI Attendance Client Node" in html, "Page title mismatch"
    assert "activationModal" in html, "Activation modal missing"
    assert "video-grid" in html, "Video grid missing"
    print("PASSED [2/3] Packaged UI served correctly with activation modal & camera grid!")

    # 3. Check status API
    status_req = urllib.request.urlopen("http://127.0.0.1:5005/api/node/status", timeout=3)
    status_data = json.loads(status_req.read().decode("utf-8"))
    print("PASSED [3/3] Node status API OK:", status_data)

    print("\nALL VERIFICATIONS PASSED: Packaged Client Node EXE is fully functional!")

finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
