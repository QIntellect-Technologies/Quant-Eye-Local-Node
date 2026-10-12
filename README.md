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

Copy the example configuration:

```bash
cp config_node.example.json config_node.json
```

Adjust the RTSP URLs, camera IDs, and target dashboard URL in `config_node.json`.

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
