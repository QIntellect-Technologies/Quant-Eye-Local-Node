"""
QuantEye Client AI Node - Standalone Desktop Launcher
Opens the dedicated Live AI Attendance Window in native lightweight browser window.
"""
import os
import sys
import time
import webbrowser
import threading
import subprocess

def main():
    print("=====================================================")
    print("  QuantEye Client AI Node - Starting Live AI Engine  ")
    print("=====================================================")
    
    # Import and run node app
    from app import start_client_node
    
    port = 5005
    url = f"http://127.0.0.1:{port}"
    
    def open_browser():
        time.sleep(2.0)
        print(f"[OK] Opening Live AI Attendance Window at {url}")
        try:
            # Try launching in app mode (clean dedicated window without browser URL bar)
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            ]
            launched = False
            for p in chrome_paths:
                if os.path.exists(p):
                    subprocess.Popen([p, f"--app={url}"])
                    launched = True
                    break
            if not launched:
                webbrowser.open(url)
        except Exception:
            webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    start_client_node(port=port)

if __name__ == "__main__":
    main()
