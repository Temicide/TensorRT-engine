# ── HTML dashboards ────────────────────────────────────────────────────────────
def cam_live_html(cam_id):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{cam_id.upper()} Live</title>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:monospace;}}
  h2{{margin:8px 12px;color:#0f0;}}
  .nav{{display:flex;gap:6px;padding:6px 12px;flex-wrap:wrap;}}
  .nav a{{color:#0af;text-decoration:none;padding:4px 8px;border:1px solid #0af;border-radius:4px;font-size:13px;}}
  .nav a:hover{{background:#0af;color:#000;}}
  .main{{display:flex;gap:12px;padding:12px;}}
  #video{{max-width:720px;width:100%;border:2px solid #0f0;border-radius:4px;}}
  #log{{flex:1;max-height:500px;overflow-y:auto;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:8px;font-size:12px;}}
  .row{{padding:3px 0;border-bottom:1px solid #222;animation:flash 0.6s ease;}}
  @keyframes flash{{from{{background:#0f02;}}to{{background:transparent;}}}}
  #stats{{display:flex;gap:16px;padding:4px 12px;font-size:13px;background:#1a1a1a;border-bottom:1px solid #333;}}
  .stat{{color:#aaa;}} .stat span{{color:#0f0;font-weight:bold;}}
</style>
</head>
<body>
<h2>📷 {cam_id.upper()} — Live Detection</h2>
<div class="nav">
  {"".join(f'<a href="/cam{i}/live">cam{i}</a>' for i in range(1,6))}
  <a href="/log/live" style="border-color:#f80;color:#f80;">📋 Central Log</a>
</div>
<div id="stats">
  <div class="stat">FPS: <span id="fps">—</span></div>
  <div class="stat">Latency: <span id="lat">—</span> ms</div>
  <div class="stat">Detections: <span id="dets">—</span></div>
  <div class="stat">Frame: <span id="frm">—</span></div>
</div>
<div class="main">
  <img id="video" src="/{cam_id}/video">
  <div id="log"><em>Waiting for detections...</em></div>
</div>
<script>
const src = new EventSource('/{cam_id}/stream');
const logEl = document.getElementById('log');
let sawFrame = false;
src.onmessage = e => {{
  const r = JSON.parse(e.data);
  document.getElementById('fps').textContent = r.fps?.toFixed(1) ?? '—';
  document.getElementById('lat').textContent = r.latency_ms ?? '—';
  document.getElementById('dets').textContent = r.num_detections ?? 0;
  document.getElementById('frm').textContent = r.frame ?? '—';
  if (!sawFrame && r.frame !== undefined) {{
    sawFrame = true;
    logEl.innerHTML = '<em>Stream active. No detections yet.</em>';
  }}
  if (r.num_detections > 0) {{
    const div = document.createElement('div');
    div.className = 'row';
    const names = r.detections.map(d => `${{d.class_name}}(${{(d.confidence*100).toFixed(0)}}%)`).join(', ');
    div.textContent = `[${{r.timestamp.slice(11,19)}}] frame ${{r.frame}}: ${{names}}`;
    if (logEl.firstChild?.tagName === 'EM') logEl.innerHTML = '';
    logEl.prepend(div);
    if (logEl.children.length > 200) logEl.lastChild.remove();
  }}
}};
src.onerror = () => setTimeout(() => location.reload(), 3000);
</script>
</body></html>"""


def log_live_html():
    cam_links = "".join(
        f'<a href="/cam{i}/live">cam{i}</a> ' for i in range(1, 6)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Central Detection Log</title>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:monospace;}}
  h2{{margin:8px 12px;color:#f80;}}
  .nav{{display:flex;gap:6px;padding:6px 12px;flex-wrap:wrap;}}
  .nav a{{color:#0af;text-decoration:none;padding:4px 8px;border:1px solid #0af;border-radius:4px;font-size:13px;}}
  .nav a:hover{{background:#0af;color:#000;}}
  #stats{{display:flex;gap:16px;padding:6px 12px;font-size:13px;background:#1a1a1a;border-bottom:1px solid #333;flex-wrap:wrap;}}
  .stat{{color:#aaa;}} .stat span{{color:#f80;font-weight:bold;}}
  #log{{padding:10px 12px;max-height:calc(100vh - 160px);overflow-y:auto;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th{{background:#222;color:#aaa;padding:5px 8px;text-align:left;position:sticky;top:0;}}
  td{{padding:4px 8px;border-bottom:1px solid #1e1e1e;}}
  tr.new{{animation:flash 0.6s ease;}}
  @keyframes flash{{from{{background:#f801;}}to{{background:transparent;}}}}
  .cam1{{color:#0f0;}} .cam2{{color:#0af;}} .cam3{{color:#f80;}}
  .cam4{{color:#f0f;}} .cam5{{color:#ff0;}}
  #chart-wrap{{padding:0 12px 8px;}}
  canvas{{background:#1a1a1a;border-radius:4px;}}
</style>
</head>
<body>
<h2>📋 Central Detection Log — All Cameras</h2>
<div class="nav">
  {cam_links}
  <a href="/log/live" style="border-color:#f80;color:#f80;">📋 Central Log</a>
  <a href="/detections" style="border-color:#aaa;color:#aaa;">🔌 API</a>
</div>
<div id="stats">
  {"".join(f'<div class="stat">cam{i}: <span id="cnt{i}">0</span></div>' for i in range(1,6))}
  <div class="stat">Total rows: <span id="total">0</span></div>
</div>
<div id="chart-wrap">
  <canvas id="chart" height="60"></canvas>
</div>
<div id="log">
  <table>
    <thead><tr>
      <th>Time</th><th>Camera</th><th>Frame</th>
      <th>Detections</th><th>Latency</th><th>Objects</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<script>
const camColors = {{cam1:'#0f0',cam2:'#0af',cam3:'#f80',cam4:'#f0f',cam5:'#ff0'}};
const counts = {{cam1:0,cam2:0,cam3:0,cam4:0,cam5:0}};
const classCounts = {{}};
let total = 0;

const src = new EventSource('/log/stream');
src.onmessage = e => {{
  const r = JSON.parse(e.data);
  const cam = r.camera_id || 'unknown';
  if (counts[cam] !== undefined) counts[cam]++;
  total++;
  document.getElementById('total').textContent = total;
  for (let i=1;i<=5;i++) {{
    const el = document.getElementById('cnt'+i);
    if (el) el.textContent = counts['cam'+i] || 0;
  }}
  r.detections?.forEach(d => {{
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  }});
  drawChart();
  if (r.num_detections === 0) return;
  const tbody = document.getElementById('tbody');
  const tr = document.createElement('tr');
  tr.className = 'new';
  const names = r.detections.map(d=>`${{d.class_name}}(${{(d.confidence*100).toFixed(0)}}%)`).join(', ');
  const camColor = camColors[cam] || '#eee';
  tr.innerHTML = `
    <td>${{r.timestamp?.slice(11,19) ?? ''}}</td>
    <td style="color:${{camColor}};font-weight:bold">${{cam}}</td>
    <td>${{r.frame ?? ''}}</td>
    <td>${{r.num_detections}}</td>
    <td>${{r.latency_ms}} ms</td>
    <td>${{names}}</td>`;
  tbody.prepend(tr);
  if (tbody.children.length > 500) tbody.lastChild.remove();
}};
src.onerror = () => setTimeout(() => location.reload(), 3000);

function drawChart() {{
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth - 24;
  canvas.width = W; canvas.height = 80;
  ctx.clearRect(0,0,W,80);
  const entries = Object.entries(classCounts).sort((a,b)=>b[1]-a[1]).slice(0,10);
  if (!entries.length) return;
  const max = entries[0][1];
  const bw = Math.max(20, Math.floor((W - 20) / entries.length) - 4);
  entries.forEach(([cls, cnt], i) => {{
    const bh = Math.max(4, Math.floor((cnt / max) * 60));
    const x = 10 + i * (bw + 4);
    const y = 70 - bh;
    ctx.fillStyle = '#0f0';
    ctx.fillRect(x, y, bw, bh);
    ctx.fillStyle = '#aaa';
    ctx.font = '9px monospace';
    ctx.fillText(cls.slice(0,7), x, 80);
  }});
}}
</script>
</body></html>"""
