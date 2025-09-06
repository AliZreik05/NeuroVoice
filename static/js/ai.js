document.getElementById('year').textContent = new Date().getFullYear();

const analyzeBtn  = document.getElementById("analyzeBtn");
const fileInput   = document.getElementById("audioFile");
const fileNameEl  = document.getElementById("fileName");
const resultCard  = document.getElementById("resultCard");
const container   = document.querySelector(".ai-container");
const progressBar = document.getElementById("progressBar");
const progressEl  = document.getElementById("progress");

const allowed = new Set(["wav","flac","aif","aiff","aifc"]);
const maxMB   = 25;

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  fileNameEl.textContent = f ? f.name : "No file chosen";
  analyzeBtn.disabled = !f;
});

function setProgress(pct) {
  progressBar.style.display = "block";
  progressEl.style.width = `${pct}%`;
  progressEl.textContent = `${Math.floor(pct)}%`;
}
function resetProgress() {
  progressBar.style.display = "none";
  progressEl.style.width = "0%";
  progressEl.textContent = "";
}

function showResult(prediction, confidence) {
  const labelEl = document.getElementById("resultLabel");
  const confEl  = document.getElementById("confidenceText");
  const badgeEl = document.getElementById("resultBadge");

  // Map API label to your UI text
  const positive = /dementia/i.test(prediction);
  labelEl.textContent = positive ? "🚩 Alzheimer Detected" : "✅ No Alzheimer Detected";
  badgeEl.textContent = positive ? "Positive" : "Negative";
  confEl.textContent  = (confidence != null)
    ? `Confidence: ${Math.round(confidence)}%`
    : "Confidence: —";

  resultCard.classList.remove("hidden");
  container?.classList.add("has-result");
}

function showError(msg) {
  const labelEl = document.getElementById("resultLabel");
  const confEl  = document.getElementById("confidenceText");
  const badgeEl = document.getElementById("resultBadge");
  labelEl.textContent = "⚠️ Analysis Failed";
  badgeEl.textContent = "Error";
  confEl.textContent  = msg || "Please try again.";
  resultCard.classList.remove("hidden");
}

// Real upload with progress using XHR (fetch doesn't expose upload progress)
analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  // client-side checks (optional)
  const ext = file.name.split(".").pop().toLowerCase();
  if (!allowed.has(ext)) {
    showError("Unsupported file type. Use WAV/FLAC/AIFF.");
    return;
  }
  if (file.size > maxMB * 1024 * 1024) {
    showError(`File too large (> ${maxMB} MB).`);
    return;
  }

  analyzeBtn.disabled = true;
  setProgress(1);

  const form = new FormData();
  form.append("audio", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/predict");

  // If you use CSRF protection, add your token header here:
  // xhr.setRequestHeader("X-CSRFToken", "{{ csrf_token() }}");

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = (e.loaded / e.total) * 100;
      setProgress(pct);
    }
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState !== 4) return;

    analyzeBtn.disabled = false;
    resetProgress();

    try {
      const res = JSON.parse(xhr.responseText || "{}");
      if (xhr.status >= 200 && xhr.status < 300 && res.ok) {
        // res.prediction e.g. "Likely Dementia" / "Not Dementia"
        // res.confidence may be a float (0-100 or 0-1 depending on your endpoint)
        const confPct = (res.confidence != null && res.confidence <= 1)
                        ? res.confidence * 100
                        : res.confidence;
        showResult(res.prediction, confPct);
      } else {
        showError(res.error || "Server error.");
      }
    } catch (err) {
      showError("Invalid server response.");
    }
  };

  xhr.onerror = () => {
    analyzeBtn.disabled = false;
    resetProgress();
    showError("Network error.");
  };

  xhr.send(form);
});

document.getElementById("reAnalyzeBtn").addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
  resultCard.classList.add("hidden");
  fileInput.value = "";
  fileNameEl.textContent = "No file chosen";
  analyzeBtn.disabled = true;
});