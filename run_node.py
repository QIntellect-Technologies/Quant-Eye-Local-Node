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
        time.sleep(0.8)
        print(f"[OK] Opening Live AI Attendance Window at {url}")
        try:
            # Use native default browser for instant non-blocking load
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    start_client_node(port=port)

if __name__ == "__main__":
    main()
