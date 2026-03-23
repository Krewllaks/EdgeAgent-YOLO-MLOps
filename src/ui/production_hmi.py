"""
Production HMI — FastAPI + HTML Operator Arayuzu.

Streamlit yerine 7/24 uretim icin uygun, coklu operator destekli,
dokunmatik ekran uyumlu web arayuzu.

Ozellikler:
  - Canli OK/NOK sayaclari ve vardiya istatistikleri
  - Son 10 karar goruntuleme
  - Operator geri bildirim arayuzu
  - Sistem sagligi izleme
  - Model bilgileri ve hot-swap
  - Turkce arayuz

Kullanim:
    python src/ui/production_hmi.py
    python src/ui/production_hmi.py --host 0.0.0.0 --port 8080

API Endpoints:
    GET  /           — Ana sayfa (HTML)
    GET  /api/stats  — Pipeline istatistikleri
    GET  /api/shift  — Vardiya raporu
    GET  /api/health — Sistem sagligi
    GET  /api/recent — Son N karar
    GET  /api/stream — Canli kamera akisi (MJPEG, bbox overlay)
    POST /api/feedback — Operator geri bildirimi
    POST /api/model/swap — Model degistir (hot-swap)
"""

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Uygulama ─────────────────────────────────────────────────────

app = FastAPI(title="EdgeAgent HMI", version="1.0.0")

# Global state
_recent_events: deque = deque(maxlen=50)
_pipeline = None
_shift_logger = None
_watchdog = None
_lock = threading.Lock()


@dataclass
class FeedbackPayload:
    image_id: str
    verdict: str       # correct | incorrect | partial
    operator_id: str = ""
    notes: str = ""


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Ana sayfa — operator HMI."""
    return _render_html()


@app.get("/api/stats")
async def get_stats():
    """Pipeline istatistikleri."""
    if _pipeline:
        stats = _pipeline.get_stats()
        return JSONResponse(content={
            "total": stats.total_frames,
            "ok": stats.ok_count,
            "nok": stats.nok_count,
            "quality_rate": round(
                stats.ok_count / max(stats.total_frames, 1) * 100, 2
            ),
            "avg_inference_ms": round(stats.avg_inference_ms, 2),
            "uptime_seconds": round(stats.uptime_seconds, 1),
            "last_verdict": stats.last_verdict,
            "last_verdict_time": stats.last_verdict_time,
        })
    return JSONResponse(content={
        "total": 0, "ok": 0, "nok": 0, "quality_rate": 100.0,
        "avg_inference_ms": 0, "uptime_seconds": 0,
        "last_verdict": "", "last_verdict_time": "",
    })


@app.get("/api/shift")
async def get_shift():
    """Vardiya raporu."""
    if _shift_logger:
        report = _shift_logger.get_shift_report()
        if report is not None:
            return JSONResponse(content={
                "shift_name": report.shift_name,
                "total": report.total_inspected,
                "ok": report.ok_count,
                "nok": report.nok_count,
                "quality_rate": report.quality_rate,
                "avg_inference_ms": round(report.avg_inference_ms, 2),
            })
    return JSONResponse(content={
        "shift_name": "Bilinmiyor",
        "total": 0, "ok": 0, "nok": 0,
        "quality_rate": 100.0, "avg_inference_ms": 0,
    })


@app.get("/api/health")
async def get_health():
    """Sistem sagligi."""
    health = {"status": "ok", "checks": {}}

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            total_gb = mem[1] / (1024**3)
            free_gb = mem[0] / (1024**3)
            used_pct = round((1 - free_gb / total_gb) * 100, 1)
            health["checks"]["gpu_memory"] = {
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "used_pct": used_pct,
                "ok": used_pct < 90,
            }
        else:
            health["checks"]["gpu"] = {"ok": False, "msg": "CUDA yok"}
    except ImportError:
        health["checks"]["gpu"] = {"ok": False, "msg": "torch yuklenmemis"}

    # Pipeline
    if _pipeline:
        stats = _pipeline.get_stats()
        health["checks"]["pipeline"] = {
            "ok": True,
            "frames_processed": stats.total_frames,
            "uptime_seconds": round(stats.uptime_seconds, 1),
        }

        # Kamera durumu
        camera_degraded = getattr(_pipeline, "_camera_degraded", False)
        health["checks"]["camera"] = {
            "ok": not camera_degraded,
            "msg": "MockCamera (degraded)" if camera_degraded else "Bagli",
        }

        # Model durumu
        model_degraded = getattr(_pipeline, "_model_degraded", False)
        health["checks"]["model"] = {
            "ok": not model_degraded,
            "msg": "Yuklenmedi (demo modu)" if model_degraded else "Yuklendi",
        }

        # Stream durumu
        fb = _pipeline.frame_buffer
        if fb is not None:
            health["checks"]["stream"] = {
                "ok": fb.has_frames,
                "msg": f"{fb.size} frame buffer" if fb.has_frames else "Frame bekleniyor",
            }
    else:
        health["checks"]["pipeline"] = {"ok": False, "msg": "Pipeline calismiyor"}

    # Disk
    import shutil
    disk = shutil.disk_usage(str(ROOT))
    free_gb = disk.free / (1024**3)
    health["checks"]["disk"] = {
        "free_gb": round(free_gb, 2),
        "ok": free_gb > 1.0,
    }

    # Overall
    all_ok = all(c.get("ok", False) for c in health["checks"].values())
    health["status"] = "ok" if all_ok else "warning"

    return JSONResponse(content=health)


@app.get("/api/recent")
async def get_recent(n: int = 10):
    """Son N karar."""
    with _lock:
        events = list(_recent_events)[-n:]
    return JSONResponse(content=[
        {
            "timestamp": e.get("timestamp", ""),
            "product_id": e.get("product_id", ""),
            "verdict": e.get("final_verdict", ""),
            "confidence": e.get("confidence", 0),
            "inference_ms": e.get("inference_ms", 0),
            "detections": len(e.get("detections", [])),
        }
        for e in reversed(events)
    ])


@app.post("/api/feedback")
async def post_feedback(request: Request):
    """Operator geri bildirimi kaydet."""
    try:
        body = await request.json()
        feedback_path = ROOT / "data" / "feedback" / "feedback_log.jsonl"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image": body.get("image_id", ""),
            "label": body.get("verdict", "correct"),
            "operator_id": body.get("operator_id", ""),
            "notes": body.get("notes", ""),
            "source": "hmi",
        }

        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return JSONResponse(content={"status": "ok", "message": "Geri bildirim kaydedildi"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/api/model/swap")
async def model_swap(request: Request):
    """Model hot-swap."""
    if not _pipeline:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Pipeline calismiyor"},
        )
    try:
        body = await request.json()
        model_path = body.get("model_path", "")
        if not model_path:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "model_path gerekli"},
            )
        success = _pipeline.hot_swap_model(model_path)
        if success:
            return JSONResponse(content={"status": "ok", "message": f"Model degistirildi: {model_path}"})
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Model yuklenemedi"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── Live Camera Stream ───────────────────────────────────────────

@app.get("/api/stream")
async def video_stream():
    """Canli kamera akisi (MJPEG) — bbox + verdict overlay."""
    if not _pipeline or not _pipeline.frame_buffer:
        return JSONResponse(
            status_code=503,
            content={"error": "Kamera akisi mevcut degil"},
        )
    from src.ui.stream_utils import mjpeg_generator
    return StreamingResponse(
        mjpeg_generator(_pipeline.frame_buffer, fps_limit=15, quality=80),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Event Handler ────────────────────────────────────────────────

def _on_inference_event(event):
    """Pipeline'dan gelen her inference olayini isle."""
    with _lock:
        event_dict = asdict(event) if hasattr(event, "__dataclass_fields__") else vars(event)
        _recent_events.append(event_dict)

    # Shift logger'a kaydet
    if _shift_logger:
        try:
            verdict = event_dict.get("final_verdict", "OK")
            confidence = event_dict.get("confidence", 1.0)
            _shift_logger.record_verdict(verdict, confidence=confidence)
        except Exception:
            pass


# ── HTML Template ────────────────────────────────────────────────

def _render_html() -> str:
    """Operator HMI HTML sayfasi."""
    return """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeAgent - Kalite Kontrol</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0f3460;
        }
        .header h1 { font-size: 20px; color: #e94560; }
        .header .info { font-size: 13px; color: #aaa; }
        .status-bar {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .status-dot {
            width: 12px; height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.ok { background: #00b894; }
        .status-dot.warn { background: #fdcb6e; }
        .status-dot.err { background: #e17055; }

        .main { padding: 16px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .card h2 {
            font-size: 14px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }

        .big-number {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
        }
        .big-number.ok { color: #00b894; }
        .big-number.nok { color: #e17055; }
        .big-number.neutral { color: #74b9ff; }

        .stats-row {
            display: flex;
            justify-content: space-around;
            margin-top: 12px;
        }
        .stat-item { text-align: center; }
        .stat-item .label { font-size: 11px; color: #888; }
        .stat-item .value { font-size: 22px; font-weight: bold; }

        .quality-bar {
            height: 8px;
            background: #2d3436;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .quality-bar .fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .quality-bar .fill.good { background: #00b894; }
        .quality-bar .fill.warn { background: #fdcb6e; }
        .quality-bar .fill.bad { background: #e17055; }

        .verdict-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }
        .verdict-badge.OK { background: #00b89433; color: #00b894; border: 1px solid #00b894; }
        .verdict-badge.NOK { background: #e1705533; color: #e17055; border: 1px solid #e17055; }

        .recent-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .recent-table th {
            text-align: left;
            padding: 8px 6px;
            border-bottom: 1px solid #0f3460;
            color: #888;
            font-size: 11px;
        }
        .recent-table td {
            padding: 6px;
            border-bottom: 1px solid #0f346033;
        }

        .full-width { grid-column: 1 / -1; }
        .two-col { grid-column: span 2; }

        .health-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f346033;
        }
        .health-item:last-child { border-bottom: none; }

        .btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            min-height: 44px;
        }
        .btn:active { transform: scale(0.96); }
        .btn-ok { background: #00b894; color: #fff; }
        .btn-nok { background: #e17055; color: #fff; }
        .btn-partial { background: #fdcb6e; color: #333; }
        .btn-danger { background: #d63031; color: #fff; }
        .btn:hover { opacity: 0.9; }

        .feedback-section {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-top: 16px;
        }

        .shift-info {
            display: flex;
            gap: 24px;
            margin-top: 8px;
        }
        .shift-info .item { text-align: center; }
        .shift-info .item .val { font-size: 28px; font-weight: bold; }
        .shift-info .item .lbl { font-size: 11px; color: #888; }

        @media (max-width: 900px) {
            .main { grid-template-columns: 1fr; }
            .two-col, .full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>EdgeAgent Kalite Kontrol</h1>
        <div class="status-bar">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">Baglaniyor...</span>
            <span class="info" id="clockText"></span>
        </div>
    </div>

    <div class="main">
        <!-- Son Karar -->
        <div class="card">
            <h2>Son Karar</h2>
            <div class="big-number neutral" id="lastVerdict">-</div>
            <div style="text-align:center; color:#888; font-size:12px;" id="lastVerdictTime">-</div>
        </div>

        <!-- Toplam Denetim -->
        <div class="card">
            <h2>Toplam Denetim</h2>
            <div class="big-number neutral" id="totalCount">0</div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="value ok" id="okCount">0</div>
                    <div class="label">OK</div>
                </div>
                <div class="stat-item">
                    <div class="value nok" id="nokCount">0</div>
                    <div class="label">NOK</div>
                </div>
            </div>
        </div>

        <!-- Kalite Orani -->
        <div class="card">
            <h2>Kalite Orani</h2>
            <div class="big-number ok" id="qualityRate">100%</div>
            <div class="quality-bar">
                <div class="fill good" id="qualityFill" style="width: 100%"></div>
            </div>
            <div style="text-align:center; margin-top:8px; font-size:12px; color:#888;">
                Ort. Cikarsama: <span id="avgInference">0</span> ms
            </div>
        </div>

        <!-- Canli Kamera -->
        <div class="card full-width">
            <h2>Canli Kamera</h2>
            <div id="stream-container" style="position:relative; text-align:center;">
                <img id="live-stream" src="/api/stream"
                     style="width:100%; max-height:480px; border-radius:8px; object-fit:contain; display:block; margin:0 auto;"
                     onerror="this.style.display='none'; document.getElementById('no-stream').style.display='block';">
                <div id="no-stream" style="display:none; text-align:center; padding:40px; color:#888;">
                    Kamera akisi bekleniyor...
                </div>
            </div>
            <div style="margin-top:8px; display:flex; gap:12px; justify-content:center;">
                <button class="btn" style="background:#0f3460; color:#eee;" onclick="toggleStream()">Akisi Baslat / Durdur</button>
            </div>
        </div>

        <!-- Vardiya Bilgisi -->
        <div class="card two-col">
            <h2>Vardiya Bilgisi</h2>
            <div style="font-size: 18px; font-weight: bold; color: #74b9ff;" id="shiftName">-</div>
            <div class="shift-info">
                <div class="item">
                    <div class="val" id="shiftTotal">0</div>
                    <div class="lbl">Toplam</div>
                </div>
                <div class="item">
                    <div class="val ok" id="shiftOk">0</div>
                    <div class="lbl">OK</div>
                </div>
                <div class="item">
                    <div class="val nok" id="shiftNok">0</div>
                    <div class="lbl">NOK</div>
                </div>
                <div class="item">
                    <div class="val" id="shiftPPM" style="color:#fdcb6e">0</div>
                    <div class="lbl">PPM</div>
                </div>
            </div>
        </div>

        <!-- Sistem Sagligi -->
        <div class="card">
            <h2>Sistem Sagligi</h2>
            <div id="healthChecks">
                <div class="health-item">
                    <span>Yukleniyor...</span>
                </div>
            </div>
        </div>

        <!-- Son Kararlar Tablosu -->
        <div class="card full-width">
            <h2>Son Kararlar</h2>
            <table class="recent-table">
                <thead>
                    <tr>
                        <th>Zaman</th>
                        <th>Urun ID</th>
                        <th>Karar</th>
                        <th>Guven</th>
                        <th>Cikarsama (ms)</th>
                        <th>Tespit</th>
                    </tr>
                </thead>
                <tbody id="recentBody">
                    <tr><td colspan="6" style="text-align:center; color:#888;">Veri bekleniyor...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Operator Geri Bildirim -->
        <div class="card full-width">
            <h2>Operator Geri Bildirimi</h2>
            <p style="color:#888; font-size:13px; margin-bottom:12px;">
                Son karari degerlendir:
            </p>
            <div class="feedback-section">
                <button class="btn btn-ok" onclick="sendFeedback('correct')">Dogru</button>
                <button class="btn btn-partial" onclick="sendFeedback('partial')">Kismi Dogru</button>
                <button class="btn btn-nok" onclick="sendFeedback('incorrect')">Yanlis</button>
            </div>
            <div id="feedbackMsg" style="text-align:center; margin-top:10px; font-size:13px; color:#888;"></div>
        </div>
    </div>

    <script>
        // Saat guncelle
        function updateClock() {
            const now = new Date();
            document.getElementById('clockText').textContent =
                now.toLocaleTimeString('tr-TR', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
        }
        setInterval(updateClock, 1000);
        updateClock();

        // Stats guncelle
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const d = await res.json();

                document.getElementById('totalCount').textContent = d.total;
                document.getElementById('okCount').textContent = d.ok;
                document.getElementById('nokCount').textContent = d.nok;
                document.getElementById('avgInference').textContent = d.avg_inference_ms;

                const rate = d.quality_rate;
                document.getElementById('qualityRate').textContent = rate + '%';
                const fill = document.getElementById('qualityFill');
                fill.style.width = rate + '%';
                fill.className = 'fill ' + (rate >= 98 ? 'good' : rate >= 95 ? 'warn' : 'bad');

                const qrEl = document.getElementById('qualityRate');
                qrEl.className = 'big-number ' + (rate >= 98 ? 'ok' : rate >= 95 ? 'neutral' : 'nok');

                const lv = document.getElementById('lastVerdict');
                if (d.last_verdict) {
                    lv.textContent = d.last_verdict;
                    lv.className = 'big-number ' + (d.last_verdict === 'OK' ? 'ok' : 'nok');
                }
                if (d.last_verdict_time) {
                    document.getElementById('lastVerdictTime').textContent = d.last_verdict_time;
                }

                // Status dot
                const dot = document.getElementById('statusDot');
                const txt = document.getElementById('statusText');
                if (d.total > 0) {
                    dot.className = 'status-dot ok';
                    txt.textContent = 'Calisiyor';
                } else {
                    dot.className = 'status-dot warn';
                    txt.textContent = 'Bekleniyor';
                }
            } catch(e) {
                document.getElementById('statusDot').className = 'status-dot err';
                document.getElementById('statusText').textContent = 'Baglanti hatasi';
            }
        }

        // Vardiya guncelle
        async function updateShift() {
            try {
                const res = await fetch('/api/shift');
                const d = await res.json();
                document.getElementById('shiftName').textContent = d.shift_name || '-';
                document.getElementById('shiftTotal').textContent = d.total || 0;
                document.getElementById('shiftOk').textContent = d.ok || 0;
                document.getElementById('shiftNok').textContent = d.nok || 0;

                const ppm = d.total > 0 ? Math.round((d.nok || 0) / d.total * 1000000) : 0;
                document.getElementById('shiftPPM').textContent = ppm;
            } catch(e) {}
        }

        // Saglik guncelle
        async function updateHealth() {
            try {
                const res = await fetch('/api/health');
                const d = await res.json();
                const container = document.getElementById('healthChecks');
                let html = '';
                for (const [key, check] of Object.entries(d.checks)) {
                    const name = key.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
                    const ok = check.ok;
                    const dot = '<span class="status-dot ' + (ok ? 'ok' : 'err') + '"></span>';
                    let detail = '';
                    if (check.used_pct !== undefined) detail = check.used_pct + '% kullanim';
                    else if (check.free_gb !== undefined) detail = check.free_gb + ' GB bos';
                    else if (check.frames_processed !== undefined) detail = check.frames_processed + ' kare';
                    else if (check.msg) detail = check.msg;

                    html += '<div class="health-item">' +
                        '<span>' + dot + ' ' + name + '</span>' +
                        '<span style="color:#888; font-size:12px;">' + detail + '</span>' +
                        '</div>';
                }
                container.innerHTML = html || '<div class="health-item"><span>Kontrol yapiliyor...</span></div>';
            } catch(e) {}
        }

        // Son kararlar guncelle
        async function updateRecent() {
            try {
                const res = await fetch('/api/recent?n=10');
                const events = await res.json();
                const tbody = document.getElementById('recentBody');
                if (events.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#888;">Henuz karar yok</td></tr>';
                    return;
                }
                let html = '';
                for (const e of events) {
                    const ts = e.timestamp ? e.timestamp.split('T')[1] || e.timestamp : '-';
                    const badge = '<span class="verdict-badge ' +
                        (e.verdict === 'OK' ? 'OK' : 'NOK') + '">' + e.verdict + '</span>';
                    html += '<tr>' +
                        '<td>' + ts + '</td>' +
                        '<td>' + (e.product_id || '-') + '</td>' +
                        '<td>' + badge + '</td>' +
                        '<td>' + (e.confidence * 100).toFixed(1) + '%</td>' +
                        '<td>' + (e.inference_ms || 0).toFixed(1) + '</td>' +
                        '<td>' + (e.detections || 0) + '</td>' +
                        '</tr>';
                }
                tbody.innerHTML = html;
            } catch(e) {}
        }

        // Geri bildirim gonder
        async function sendFeedback(verdict) {
            const msg = document.getElementById('feedbackMsg');
            try {
                const res = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_id: 'hmi_' + Date.now(),
                        verdict: verdict,
                        operator_id: '',
                    }),
                });
                const d = await res.json();
                msg.textContent = d.message || 'Kaydedildi';
                msg.style.color = '#00b894';
                setTimeout(() => { msg.textContent = ''; }, 3000);
            } catch(e) {
                msg.textContent = 'Hata: ' + e.message;
                msg.style.color = '#e17055';
            }
        }

        // Canli kamera toggle
        let streamActive = true;
        function toggleStream() {
            const img = document.getElementById('live-stream');
            const noStream = document.getElementById('no-stream');
            if (streamActive) {
                img.src = '';
                img.style.display = 'none';
                noStream.style.display = 'block';
                noStream.textContent = 'Akis durduruldu';
                streamActive = false;
            } else {
                img.src = '/api/stream';
                img.style.display = 'block';
                noStream.style.display = 'none';
                streamActive = true;
            }
        }

        // Periyodik guncelleme
        setInterval(updateStats, 2000);
        setInterval(updateShift, 10000);
        setInterval(updateHealth, 15000);
        setInterval(updateRecent, 3000);

        // Ilk yuklemede hemen calistir
        updateStats();
        updateShift();
        updateHealth();
        updateRecent();
    </script>
</body>
</html>"""


# ── Baslat ───────────────────────────────────────────────────────

def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Pipeline ile entegre HMI uygulamasini olustur."""
    global _pipeline, _shift_logger

    if config_path:
        try:
            from src.pipeline.inference_pipeline import InferencePipeline
            _pipeline = InferencePipeline.from_config(config_path)
            _pipeline.on_event(_on_inference_event)
            _pipeline.start()
            logger.info("Pipeline baslatildi")
        except Exception as e:
            logger.warning(f"Pipeline baslatilamadi: {e}")

    try:
        from src.pipeline.shift_logger import ShiftLogger
        _shift_logger = ShiftLogger()
    except Exception as e:
        logger.warning(f"ShiftLogger yuklenemedi: {e}")

    return app


def main():
    parser = argparse.ArgumentParser(description="EdgeAgent Production HMI")
    parser.add_argument("--host", default="0.0.0.0", help="Sunucu adresi")
    parser.add_argument("--port", type=int, default=8080, help="Sunucu portu")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "production_config.yaml"),
        help="Production config dosyasi",
    )
    parser.add_argument(
        "--no-pipeline", action="store_true",
        help="Pipeline baslatmadan sadece HMI'yi calistir (gelistirme icin)",
    )
    args = parser.parse_args()

    import uvicorn

    config_path = None if args.no_pipeline else args.config
    create_app(config_path)

    print(f"\n  EdgeAgent HMI: http://{args.host}:{args.port}")
    print(f"  Config: {args.config}")
    print(f"  Pipeline: {'Devre disi' if args.no_pipeline else 'Aktif'}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
