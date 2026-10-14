# QuantEye Client AI Edge Node

**QuantEye Client AI Node** is the dedicated local edge application for real-time facial recognition attendance, multi-stream RTSP/webcam capture, and automatic synchronization with the QuantEye Cloud / Main Dashboard.

---

## Key Features

- **Multi-Camera Edge Pipeline**: High-performance multithreaded RTSP, DVR, NVR, and USB webcam capture.
- **Deep Learning Biometrics**: Face detection and recognition using InsightFace (SCRFD + ArcFace) with CUDA GPU acceleration and CPU fallback.
- **Cloud & Edge Synchronization**: Real-time event dispatching and offline-resilient local SQLite storage.
- **Desktop Window UI**: Dedicated low-latency live camera monitoring interface served at `http://127.0.0.1:5005`.
- **Tenant & Branch Aware**: Built-in support for tenant (`organization_id`) and branch (`branch_id`) context.

---

## Quick Start

### 1. Requirements

- Python 3.10+ (Recommended: Conda / Virtual Environment)
- OpenCV (`opencv-python`)
- ONNX Runtime (`onnxruntime` or `onnxruntime-gpu`)
- InsightFace & PyTorch
- Flask

### 2. Configuration

The node starts unlinked. Link it from the first-launch activation screen using the organization's issued branch credential. Activation validates the credential with Main Dashboard and writes the returned organization, branch, credential, server URL, and camera configuration to the local `config_node.json`.

```bash
cp config_node.example.json config_node.json
```

The example config is intentionally clean and contains no organization, branch key, or cameras. Do not distribute a runtime `config_node.json`; it contains the linked node's branch credential. CPU/GPU build specs exclude runtime configuration, and outbound service URLs must use HTTPS outside explicit local development/test mode.

Attendance events are persisted in the local SQLite `client_cloud_delivery` outbox before delivery. Failed requests remain pending with bounded exponential backoff and the original event ID, so retries are idempotent and survive node restarts. If SQLite itself is unavailable, the event cannot be durably queued; the node logs this as a critical delivery error rather than reporting it as delivered.

Camera configuration is fetched through the Main Dashboard using the node's saved branch key. The Main Dashboard retrieves only that branch's active cameras from the Client Portal with its server-side integration key; that integration key is never stored in or sent to the EXE. A successful empty response clears deleted cameras, while an unavailable cloud source preserves the node's last-known camera configuration.

Staff profiles and face-training vectors sync through Main Dashboard using the activated branch key. If attendance-state sync is temporarily unavailable, camera and staff refresh still completes and the node retries attendance-state sync in the background.

### 3. Launching the Node

Run the standalone desktop launcher:

```bash
python run_node.py
```

Or run the Flask server directly:

```bash
python app.py
```

The node web interface will be available at:
`http://127.0.0.1:5005`
