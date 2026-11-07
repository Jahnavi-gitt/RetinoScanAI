const GRADE_LABELS = [
  "Grade 0: No DR",
  "Grade 1: Mild NPDR",
  "Grade 2: Moderate NPDR",
  "Grade 3: Severe NPDR",
  "Grade 4: Proliferative DR"
];

const GRADE_COLORS = [
  "#10b981", 
  "#00b0ff", 
  "#f59e0b", 
  "#ff6d00", 
  "#f43f5e"  
];

const ANATOMICAL_SHORT = [
  "Superotemporal",
  "Sup. Arcade",
  "Superonasal",
  "Nasal / Disc",
  "Central Macula",
  "Temporal",
  "Inferotemporal",
  "Inf. Arcade",
  "Inferonasal"
];

const fundusImg = document.getElementById("fundusImage");
const gradcamOverlay = document.getElementById("gradcamOverlay");
const gridOverlay = document.getElementById("gridOverlay");
const loadingOverlay = document.getElementById("loadingOverlay");
const opacityControlBar = document.getElementById("opacityControlBar");
const opacitySlider = document.getElementById("opacitySlider");
const opacityVal = document.getElementById("opacityVal");

const predictedGradeTitle = document.getElementById("predictedGradeTitle");
const urgencyBadge = document.getElementById("urgencyBadge");
const latencyBadge = document.getElementById("latencyBadge");
const followUpText = document.getElementById("followUpText");
const recommendationText = document.getElementById("recommendationText");
const icdCode = document.getElementById("icdCode");
const diagnosisCard = document.getElementById("diagnosisCard");
const probBarsList = document.getElementById("probBarsList");
const attnMiniGrid = document.getElementById("attnMiniGrid");
const peakZoneBadge = document.getElementById("peakZoneBadge");

const radMean = document.getElementById("radMean");
const radEntropy = document.getElementById("radEntropy");
const radContrast = document.getElementById("radContrast");
const radHomogeneity = document.getElementById("radHomogeneity");
const radEdge = document.getElementById("radEdge");
const radMichelson = document.getElementById("radMichelson");

const btnViewOrig = document.getElementById("btnViewOrig");
const btnViewGrid = document.getElementById("btnViewGrid");
const btnViewGradcam = document.getElementById("btnViewGradcam");

let currentViewMode = "original";
let latestPredictionData = null;
let currentModelType = "baseline";
let currentPayload = { sample_id: "grade_0" };

const activeModelTag = document.getElementById("activeModelTag");
const btnModelBaseline = document.getElementById("btnModelBaseline");
const btnModelGap3 = document.getElementById("btnModelGap3");

document.addEventListener("DOMContentLoaded", () => {
  setupViewControls();
  setupModelPicker();
  setupSamplePicker();
  setupDropZone();
  setupOpacitySlider();

  loadPredictionForSample("grade_0");
});

function setupModelPicker() {
  const modelBtns = [btnModelBaseline, btnModelGap3];
  modelBtns.forEach(btn => {
    if (btn) {
      btn.addEventListener("click", () => {
        modelBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentModelType = btn.dataset.model || "baseline";
        if (currentPayload && (currentPayload.sample_id || currentPayload.image_base64)) {
          requestPrediction(currentPayload);
        } else {
          loadPredictionForSample("grade_0");
        }
      });
    }
  });
}

function setupViewControls() {
  const btns = [btnViewOrig, btnViewGrid, btnViewGradcam];
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentViewMode = btn.dataset.view;
      applyViewMode();
    });
  });
}

function applyViewMode() {
  if (currentViewMode === "original") {
    gridOverlay.style.opacity = "0";
    gradcamOverlay.style.opacity = "0";
    opacityControlBar.style.display = "none";
  } else if (currentViewMode === "grid") {
    gridOverlay.style.opacity = "1";
    gradcamOverlay.style.opacity = "0";
    opacityControlBar.style.display = "none";
  } else if (currentViewMode === "gradcam") {
    gridOverlay.style.opacity = "0";
    gradcamOverlay.style.opacity = (opacitySlider.value / 100.0).toString();
    opacityControlBar.style.display = "flex";
  }
}

function setupOpacitySlider() {
  opacitySlider.addEventListener("input", (e) => {
    const val = e.target.value;
    opacityVal.textContent = `${val}%`;
    if (currentViewMode === "gradcam") {
      gradcamOverlay.style.opacity = (val / 100.0).toString();
    }
  });
}

function setupSamplePicker() {
  const pills = document.querySelectorAll(".sample-pill[data-sample]");
  pills.forEach(pill => {
    pill.addEventListener("click", () => {
      pills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const sampleId = pill.dataset.sample;
      if (sampleId) {
        loadPredictionForSample(sampleId);
      }
    });
  });
}

function setupDropZone() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleCustomFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleCustomFile(e.target.files[0]);
    }
  });
}

function handleCustomFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const b64 = e.target.result;
    fundusImg.src = b64;
    
    document.querySelectorAll(".sample-pill[data-sample]").forEach(p => p.classList.remove("active"));
    currentPayload = { image_base64: b64 };
    requestPrediction(currentPayload);
  };
  reader.readAsDataURL(file);
}

function loadPredictionForSample(sampleId) {
  if (!sampleId) return;
  
  const sampleMap = {
    "grade_0": "/samples/grade_0_007-0007-000.jpg",
    "grade_1": "/samples/grade_1_007-1843-100.jpg",
    "grade_2": "/samples/grade_2_007-1927-100.jpg",
    "grade_3": "/samples/grade_3_007-3548-200.jpg",
    "grade_4": "/samples/grade_4_007-3979-200.jpg"
  };
  if (sampleMap[sampleId]) {
    fundusImg.src = sampleMap[sampleId];
  }
  currentPayload = { sample_id: sampleId };
  requestPrediction(currentPayload);
}

async function requestPrediction(payload) {
  currentPayload = payload;
  const requestBody = { ...payload, model_type: currentModelType };
  loadingOverlay.classList.add("active");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody)
    });
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    const data = await response.json();
    latestPredictionData = data;
    renderResults(data);
  } catch (err) {
    console.error("Prediction error:", err);
    alert(`Inference failed: ${err.message}`);
  } finally {
    loadingOverlay.classList.remove("active");
  }
}

function renderResults(data) {
  const g = data.predicted_grade;
  const color = GRADE_COLORS[g];

  if (activeModelTag) {
    if (data.model_type === "gap3") {
      activeModelTag.textContent = "Gap 3 (Spatial Cross-Attn)";
      activeModelTag.style.background = "rgba(139, 92, 246, 0.25)";
      activeModelTag.style.borderColor = "rgba(139, 92, 246, 0.6)";
      activeModelTag.style.color = "#c084fc";
    } else {
      activeModelTag.textContent = "Baseline (Swin-Tiny + CORAL)";
      activeModelTag.style.background = "rgba(0, 229, 255, 0.15)";
      activeModelTag.style.borderColor = "rgba(0, 229, 255, 0.4)";
      activeModelTag.style.color = "#00e5ff";
    }
  }

  predictedGradeTitle.textContent = `${GRADE_LABELS[g]}`;
  predictedGradeTitle.style.color = color;

  diagnosisCard.style.border = `1px solid ${color}88`;
  diagnosisCard.style.background = `linear-gradient(135deg, ${color}15, rgba(18, 26, 45, 0.9))`;

  const guidance = data.referral_guidance;
  urgencyBadge.textContent = guidance.urgency;
  urgencyBadge.style.background = `${color}25`;
  urgencyBadge.style.color = color;
  urgencyBadge.style.borderColor = color;

  followUpText.textContent = guidance.follow_up;
  recommendationText.textContent = guidance.recommendation;
  icdCode.textContent = `ICD-10: ${guidance.icd_code}`;
  latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;

  if (data.gradcam_overlay_base64) {
    gradcamOverlay.src = data.gradcam_overlay_base64;
  }
  applyViewMode();

  probBarsList.innerHTML = "";
  const probs = data.coral_probabilities;
  probs.forEach((p, idx) => {
    const pct = (p * 100.0).toFixed(1);
    const pColor = GRADE_COLORS[idx];
    const isWinner = idx === g;

    const item = document.createElement("div");
    item.className = "prob-bar-item";
    item.innerHTML = `
      <div class="prob-bar-labels">
        <span style="font-weight: ${isWinner ? '700' : '400'}; color: ${isWinner ? pColor : 'var(--text-muted)'};">
          ${GRADE_LABELS[idx]} ${isWinner ? '✓' : ''}
        </span>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: ${isWinner ? pColor : 'var(--text-main)'};">
          ${pct}%
        </span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%; background: ${pColor};"></div>
      </div>
    `;
    probBarsList.appendChild(item);
  });

  peakZoneBadge.textContent = `Peak: ${data.peak_attended_zone.split(':')[1] || data.peak_attended_zone}`;
  attnMiniGrid.innerHTML = "";
  const regAttn = data.regional_attention;
  const maxAttn = Math.max(...regAttn);

  regAttn.forEach((weight, idx) => {
    const pct = (weight * 100.0).toFixed(1);
    const intensity = (weight / (maxAttn + 1e-6));
    const cell = document.createElement("div");
    cell.className = "attn-cell";
    cell.style.background = `rgba(0, 229, 255, ${Math.max(0.08, intensity * 0.4)})`;
    if (idx === 4) {
      cell.style.borderColor = "var(--accent-amber)";
    }
    cell.innerHTML = `
      <span class="attn-cell-val" style="color: ${intensity > 0.6 ? '#00e5ff' : '#94a3b8'}">${pct}%</span>
      <span class="attn-cell-label">${ANATOMICAL_SHORT[idx]}</span>
    `;
    attnMiniGrid.appendChild(cell);
  });

  if (data.macula_radiomics) {
    radMean.textContent = data.macula_radiomics.first_order_mean;
    radEntropy.textContent = data.macula_radiomics.first_order_entropy;
    radContrast.textContent = data.macula_radiomics.glcm_contrast;
    radHomogeneity.textContent = data.macula_radiomics.glcm_homogeneity;
    radEdge.textContent = data.macula_radiomics.canny_edge_density;
    radMichelson.textContent = data.macula_radiomics.michelson_contrast;
  }
}
