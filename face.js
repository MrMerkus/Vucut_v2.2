// ══════════════════════════════════════════════════════════════════════════════
// face.js  —  Yüz Tanıma Modülü
// Mevcut vücut/el takip sistemine eklenti olarak çalışır.
// index.html'e  <script type="module" src="face.js"></script>  ekleyin.
// script.js'e dokunmayın — bu dosya kendi döngüsünü yönetir.
// ══════════════════════════════════════════════════════════════════════════════

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

// ── SABITLER ─────────────────────────────────────────────────────────────────
const FACE_COLORS = [
  "#00e5cc", // teal  (KİŞİ 1 ile aynı)
  "#ff4081", // pink
  "#ffd600", // sarı
  "#69ff47", // yeşil
  "#e040fb", // mor
  "#ff6d00", // turuncu
];

// Yüz blendshape'lerinden takip etmek istediğimiz ifadeler
const TRACKED_SHAPES = [
  { key: "eyeBlinkLeft",          label: "SOL GÖZ KIRPMA"   },
  { key: "eyeBlinkRight",         label: "SAĞ GÖZ KIRPMA"   },
  { key: "mouthSmileLeft",        label: "GÜLÜMSEME (Sol)"   },
  { key: "mouthSmileRight",       label: "GÜLÜMSEME (Sağ)"   },
  { key: "browInnerUp",           label: "KAŞ YUKARI"        },
  { key: "jawOpen",               label: "AĞIZ AÇIKLIĞI"     },
];

// ── DOM ───────────────────────────────────────────────────────────────────────
const video         = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx     = canvasElement.getContext("2d");
const metricsPanel  = document.getElementById("metrics-panel");

// ── STATE ─────────────────────────────────────────────────────────────────────
let faceLandmarker   = null;
let lastVideoTime    = -1;
let isRunning        = false;

// Yüz kartlarının DOM referansları (slotIdx → card element)
const faceCards = {};

// ── MODEL YÜKLEMESİ ──────────────────────────────────────────────────────────
async function initFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 6,
  });
  console.log("[face.js] FaceLandmarker hazır.");
  hookCameraButton();
}

// ── KAMERA BUTONUNA HOOK ──────────────────────────────────────────────────────
// script.js webcam döngüsünü yönettiği için biz de aynı video elementini
// kullanarak kendi döngümüzü başlatıp durduruyoruz.
function hookCameraButton() {
  const btn = document.getElementById("webcamButton");
  if (!btn) return;

  btn.addEventListener("click", () => {
    // Kısa gecikmeyle webcamRunning'in güncellenmesini bekle
    setTimeout(() => {
      const videoActive = !!(video.srcObject);
      if (videoActive && !isRunning) {
        isRunning = true;
        video.addEventListener("loadeddata", startLoop, { once: true });
        if (video.readyState >= 2) startLoop(); // zaten hazırsa hemen başlat
      } else if (!videoActive && isRunning) {
        isRunning = false;
        clearAllFaceCards();
      }
    }, 200);
  });
}

function startLoop() {
  if (!isRunning) return;
  predictFace();
}

// ── ANA DÖNGÜ ─────────────────────────────────────────────────────────────────
function predictFace() {
  if (!isRunning || !faceLandmarker || video.paused || video.ended) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now     = performance.now();

    // Canvas boyutlarını video ile senkronize et (script.js de aynısını yapıyor)
    if (canvasElement.width !== video.videoWidth && video.videoWidth > 0) {
      canvasElement.width  = video.videoWidth;
      canvasElement.height = video.videoHeight;
    }

    const results = faceLandmarker.detectForVideo(video, now);
    const faces   = results.faceLandmarks     ?? [];
    const shapes  = results.faceBlendshapes   ?? [];

    // ── Yüz iskeleti çiz ────────────────────────────────────────────────
    // script.js canvasCtx.save/restore yaptığı için biz de kendi save/restore'umuzu yapıyoruz
    canvasCtx.save();
    const drawUtils = new DrawingUtils(canvasCtx);

    for (let i = 0; i < faces.length; i++) {
      const color = FACE_COLORS[i % FACE_COLORS.length];
      const lm    = faces[i];

      // Sadece yüz ağı — hafif ve temiz görünüm
      drawUtils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
        color: color + "55",  // yarı saydam
        lineWidth: 0.5,
      });
      drawUtils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
        color,
        lineWidth: 1.5,
      });
      drawUtils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
        color: "#30FF30",
        lineWidth: 1.5,
      });
      drawUtils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
        color: "#FF3030",
        lineWidth: 1.5,
      });
      drawUtils.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LIPS, {
        color,
        lineWidth: 1.5,
      });

      // Yüz numarası etiketi — alnın üstüne
      const noseTip = lm[1]; // burun ucu
      if (noseTip) {
        const lx = (1 - noseTip.x) * canvasElement.width;
        const ly = noseTip.y       * canvasElement.height - 40;
        canvasCtx.save();
        canvasCtx.font          = "bold 11px 'Space Mono', monospace";
        canvasCtx.textAlign     = "center";
        canvasCtx.textBaseline  = "middle";
        const label = `YÜZ ${i + 1}`;
        const tw    = canvasCtx.measureText(label).width + 10;
        canvasCtx.fillStyle = "rgba(5,10,14,0.85)";
        canvasCtx.beginPath();
        canvasCtx.roundRect(lx - tw / 2, ly - 9, tw, 18, 4);
        canvasCtx.fill();
        canvasCtx.strokeStyle = color;
        canvasCtx.lineWidth   = 1;
        canvasCtx.stroke();
        canvasCtx.fillStyle   = color;
        canvasCtx.shadowColor = color;
        canvasCtx.shadowBlur  = 6;
        canvasCtx.fillText(label, lx, ly);
        canvasCtx.restore();
      }

      // ── Metrics panelini güncelle ──────────────────────────────────────
      const blendShapes = shapes[i]?.categories ?? [];
      ensureFaceCard(i, color);
      updateFaceCard(i, lm, blendShapes);
    }

    canvasCtx.restore();

    // Artık görünmeyen yüz kartlarını kaldır
    const visibleCount = faces.length;
    for (const idx in faceCards) {
      if (parseInt(idx) >= visibleCount) removeFaceCard(parseInt(idx));
    }
  }

  requestAnimationFrame(predictFace);
}

// ── YÜZDEN METRİK HESAPLA ────────────────────────────────────────────────────

// Göz açıklık oranı — EAR (Eye Aspect Ratio)
function calcEAR(lm, eyeIndices) {
  // Basitleştirilmiş 6 noktalı EAR
  const [p1, p2, p3, p4, p5, p6] = eyeIndices.map(i => lm[i]);
  if (!p1 || !p4) return null;
  const A = Math.hypot(p2.x - p6.x, p2.y - p6.y);
  const B = Math.hypot(p3.x - p5.x, p3.y - p5.y);
  const C = Math.hypot(p1.x - p4.x, p1.y - p4.y);
  return C > 0 ? ((A + B) / (2 * C)) : null;
}

// Baş eğim açısı (yatay — roll)
function calcHeadRoll(lm) {
  const leftEye  = lm[33];   // sol göz köşesi
  const rightEye = lm[263];  // sağ göz köşesi
  if (!leftEye || !rightEye) return null;
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  return Math.round(Math.atan2(dy, dx) * (180 / Math.PI));
}

// Baş yaw (sola/sağa dönüş) — burun ucu ile iki kulak arası oran
function calcHeadYaw(lm) {
  const nose   = lm[1];
  const leftEar  = lm[234];
  const rightEar = lm[454];
  if (!nose || !leftEar || !rightEar) return null;
  const leftDist  = Math.abs(nose.x - leftEar.x);
  const rightDist = Math.abs(nose.x - rightEar.x);
  const total     = leftDist + rightDist;
  if (total === 0) return 0;
  // -90 (tamamen sağa) → 0 (düz) → +90 (tamamen sola)
  return Math.round(((leftDist - rightDist) / total) * 90);
}

// Ağız açıklık oranı
function calcMouthOpen(lm) {
  const top    = lm[13];  // üst dudak
  const bottom = lm[14];  // alt dudak
  const left   = lm[61];  // sol ağız köşesi
  const right  = lm[291]; // sağ ağız köşesi
  if (!top || !bottom || !left || !right) return null;
  const vertical   = Math.hypot(top.x - bottom.x, top.y - bottom.y);
  const horizontal = Math.hypot(left.x - right.x, left.y - right.y);
  return horizontal > 0 ? Math.round((vertical / horizontal) * 100) : 0;
}

// Simetri skoru (sol/sağ göz mesafesi dengesi)
function calcFaceSymmetry(lm) {
  const nose      = lm[1];
  const leftEye   = lm[33];
  const rightEye  = lm[263];
  if (!nose || !leftEye || !rightEye) return null;
  const lDist = Math.hypot(nose.x - leftEye.x,  nose.y - leftEye.y);
  const rDist = Math.hypot(nose.x - rightEye.x, nose.y - rightEye.y);
  const mx    = Math.max(lDist, rDist);
  if (mx === 0) return 100;
  return Math.round((1 - Math.abs(lDist - rDist) / mx) * 100);
}

// Blendshape değerini ada göre bul
function getShape(categories, key) {
  const cat = categories.find(c => c.categoryName === key);
  return cat ? +cat.score : 0;
}

// İfade tahmini
function guessExpression(categories) {
  const smile    = (getShape(categories, "mouthSmileLeft") + getShape(categories, "mouthSmileRight")) / 2;
  const blink    = (getShape(categories, "eyeBlinkLeft")  + getShape(categories, "eyeBlinkRight"))  / 2;
  const surprised = getShape(categories, "browInnerUp");
  const jawOpen   = getShape(categories, "jawOpen");

  if (blink > 0.6)     return "GÖZLER KAPALI";
  if (jawOpen > 0.5)   return "AĞIZ AÇIK";
  if (smile > 0.4)     return "GÜLÜMSÜYOR";
  if (surprised > 0.5) return "ŞAŞIRMIŞ";
  return "NÖTR";
}

// ── KART YÖNETİMİ ────────────────────────────────────────────────────────────
function ensureFaceCard(faceIdx, color) {
  if (faceCards[faceIdx]) return;

  const id   = `face-${faceIdx}`;
  const card = document.createElement("div");
  card.className = "person-card face-card";
  card.id        = `face-card-${faceIdx}`;
  card.style.setProperty("--person-color", color);
  card.style.borderTopColor = color;

  card.innerHTML = `
    <div class="person-header">
      <div class="person-dot" style="background:${color};box-shadow:0 0 8px ${color}"></div>
      <div class="person-title" style="color:${color}">YÜZ ${faceIdx + 1}</div>
      <div style="margin-left:auto;font-size:9px;letter-spacing:2px;color:var(--text-dim)" id="${id}-expr">—</div>
    </div>
    <div class="person-card-body">

      <!-- Baş Açıları -->
      <div>
        <div class="panel-title">BAŞ AÇILARI</div>
        <div class="angles-grid">
          <div class="angle-card" id="${id}-card-roll">
            <div class="angle-label">ROLL (EĞİM)</div>
            <div class="angle-value" id="${id}-val-roll">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-roll" style="background:${color}"></div></div>
          </div>
          <div class="angle-card" id="${id}-card-yaw">
            <div class="angle-label">YAW (DÖNÜŞ)</div>
            <div class="angle-value" id="${id}-val-yaw">—°</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-yaw" style="background:${color}"></div></div>
          </div>
          <div class="angle-card wide" id="${id}-card-mouth">
            <div class="angle-label">AĞIZ AÇIKLIĞI</div>
            <div class="angle-value" id="${id}-val-mouth">—%</div>
            <div class="angle-bar-wrap"><div class="angle-bar" id="${id}-bar-mouth" style="background:${color}"></div></div>
          </div>
        </div>
      </div>

      <!-- Blendshape Göstergeleri -->
      <div>
        <div class="panel-title">YÜZLEME GÖSTERGELERİ</div>
        <div class="blend-list" id="${id}-blends">
          ${TRACKED_SHAPES.map(s => `
            <div class="blend-row" id="${id}-blend-${s.key}">
              <div class="blend-row-label">${s.label}</div>
              <div class="blend-row-bar-wrap">
                <div class="blend-row-bar" id="${id}-bbar-${s.key}" style="background:${color}"></div>
              </div>
              <div class="blend-row-val" id="${id}-bval-${s.key}">0.00</div>
            </div>
          `).join("")}
        </div>
      </div>

      <!-- İstatistikler -->
      <div>
        <div class="panel-title">VERİ AKIŞI</div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-label">NOKTA SAYISI</div>
            <div class="stat-value" id="${id}-stat-pts">0</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">SOL GÖZ</div>
            <div class="stat-value" id="${id}-stat-leye">—</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">SAĞ GÖZ</div>
            <div class="stat-value" id="${id}-stat-reye">—</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">İFADE</div>
            <div class="stat-value" id="${id}-stat-expr" style="font-size:10px">—</div>
          </div>
        </div>
      </div>

      <!-- Simetri -->
      <div>
        <div class="panel-title">YÜZ SİMETRİSİ</div>
        <div class="symmetry-display">
          <div class="symmetry-score" id="${id}-sym-score" style="color:${color};text-shadow:0 0 16px ${color}">—</div>
          <div class="symmetry-label">Sol / Sağ göz dengesi</div>
          <div class="symmetry-bar-wrap">
            <div class="symmetry-bar" id="${id}-sym-bar" style="background:${color}"></div>
          </div>
        </div>
      </div>

    </div>
  `;

  metricsPanel.appendChild(card);
  faceCards[faceIdx] = card;
}

function updateFaceCard(faceIdx, lm, categories) {
  const id = `face-${faceIdx}`;

  // Baş açıları
  const roll  = calcHeadRoll(lm);
  const yaw   = calcHeadYaw(lm);
  const mouth = calcMouthOpen(lm);

  setAngleCard(id, "roll",  roll,  90);
  setAngleCard(id, "yaw",   yaw,   90);
  setBarCard  (id, "mouth", mouth !== null ? mouth : null, 100, "%");

  // Blendshapes
  for (const s of TRACKED_SHAPES) {
    const val   = getShape(categories, s.key);
    const bbar  = document.getElementById(`${id}-bbar-${s.key}`);
    const bval  = document.getElementById(`${id}-bval-${s.key}`);
    if (bbar) bbar.style.width = `${val * 100}%`;
    if (bval) bval.textContent  = val.toFixed(2);
  }

  // Göz kırpma oranı (EAR) — MediaPipe landmark indeksleri
  // Sol göz: 33, 159, 158, 133, 153, 144
  // Sağ göz: 263, 386, 385, 362, 380, 373
  const leftEAR  = calcEAR(lm, [33, 159, 158, 133, 153, 144]);
  const rightEAR = calcEAR(lm, [263, 386, 385, 362, 380, 373]);

  const leyeEl = document.getElementById(`${id}-stat-leye`);
  const reyeEl = document.getElementById(`${id}-stat-reye`);
  if (leyeEl && leftEAR  !== null) {
    const closed = leftEAR < 0.15;
    leyeEl.textContent = closed ? "KAPALI" : "AÇIK";
    leyeEl.className   = "stat-value" + (closed ? "" : " yes");
  }
  if (reyeEl && rightEAR !== null) {
    const closed = rightEAR < 0.15;
    reyeEl.textContent = closed ? "KAPALI" : "AÇIK";
    reyeEl.className   = "stat-value" + (closed ? "" : " yes");
  }

  // Nokta sayısı
  const ptsEl = document.getElementById(`${id}-stat-pts`);
  if (ptsEl) ptsEl.textContent = lm.length;

  // İfade
  const expr    = guessExpression(categories);
  const exprEl  = document.getElementById(`${id}-stat-expr`);
  const exprBdg = document.getElementById(`${id}-expr`);
  if (exprEl)  exprEl.textContent  = expr;
  if (exprBdg) exprBdg.textContent = expr;

  // Simetri
  const sym    = calcFaceSymmetry(lm);
  const symEl  = document.getElementById(`${id}-sym-score`);
  const symBar = document.getElementById(`${id}-sym-bar`);
  if (symEl) {
    symEl.textContent  = sym !== null ? `${sym}%` : "—";
    if (symBar) symBar.style.width = sym !== null ? `${sym}%` : "0%";
  }
}

function setAngleCard(id, name, value, maxVal) {
  const valEl  = document.getElementById(`${id}-val-${name}`);
  const barEl  = document.getElementById(`${id}-bar-${name}`);
  const card   = document.getElementById(`${id}-card-${name}`);
  if (!valEl) return;
  if (value === null) {
    valEl.textContent = "—°";
    if (barEl) barEl.style.width = "0%";
    if (card)  card.classList.remove("active");
    return;
  }
  valEl.textContent = `${value}°`;
  if (barEl) barEl.style.width = `${Math.min(100, (Math.abs(value) / maxVal) * 100)}%`;
  if (card)  card.classList.add("active");
}

function setBarCard(id, name, value, maxVal, unit = "%") {
  const valEl = document.getElementById(`${id}-val-${name}`);
  const barEl = document.getElementById(`${id}-bar-${name}`);
  const card  = document.getElementById(`${id}-card-${name}`);
  if (!valEl) return;
  if (value === null) {
    valEl.textContent = `—${unit}`;
    if (barEl) barEl.style.width = "0%";
    if (card)  card.classList.remove("active");
    return;
  }
  valEl.textContent = `${value}${unit}`;
  if (barEl) barEl.style.width = `${Math.min(100, (value / maxVal) * 100)}%`;
  if (card)  card.classList.add("active");
}

function removeFaceCard(faceIdx) {
  if (faceCards[faceIdx]) {
    faceCards[faceIdx].remove();
    delete faceCards[faceIdx];
  }
}

function clearAllFaceCards() {
  for (const idx in faceCards) {
    removeFaceCard(parseInt(idx));
  }
}

// ── EKLENTI CSS ───────────────────────────────────────────────────────────────
// Blendshape göstergesi için küçük stiller
const style = document.createElement("style");
style.textContent = `
.face-card { border-top-width: 3px !important; }

.blend-list {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.blend-row {
  display: grid;
  grid-template-columns: 120px 1fr 40px;
  align-items: center;
  gap: 6px;
}

.blend-row-label {
  font-size: 7px;
  letter-spacing: 1px;
  color: var(--text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.blend-row-bar-wrap {
  height: 4px;
  background: var(--bg);
  border-radius: 2px;
  overflow: hidden;
}

.blend-row-bar {
  height: 100%;
  width: 0%;
  border-radius: 2px;
  transition: width 0.1s ease;
}

.blend-row-val {
  font-family: 'Space Mono', monospace;
  font-size: 9px;
  color: var(--text-dim);
  text-align: right;
}
`;
document.head.appendChild(style);

// ── BAŞLAT ────────────────────────────────────────────────────────────────────
initFaceLandmarker();
