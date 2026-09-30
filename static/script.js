(() => {
  let lastPrediction = null;
  let requestVersion = 0;
  let explanationController = null;
  const explanationPanel = document.getElementById("explanationPanel");
  const explainBtn = document.getElementById("explainBtn");
  const explainStatus = document.getElementById("explainStatus");
  const explainContent = document.getElementById("explainContent");
  const form = document.getElementById("riskForm");
  const submitBtn = document.getElementById("submitBtn");
  const errorNote = document.getElementById("errorNote");
  const verdict = document.getElementById("verdict");

  const incomeInput = document.getElementById("person_income");
  const amountInput = document.getElementById("loan_amnt");
  const percentInput = document.getElementById("loan_percent_income");

  const gaugeFill = document.getElementById("gaugeFill");
  const gaugeThreshold = document.getElementById("gaugeThreshold");
  const probNumber = document.getElementById("probNumber");
  const stampBadge = document.getElementById("stampBadge");
  const stampText = document.getElementById("stampText");

  const factProb = document.getElementById("factProb");
  const factThreshold = document.getElementById("factThreshold");
  const factResult = document.getElementById("factResult");

  const apiDot = document.getElementById("apiDot");
  const apiStatusText = document.getElementById("apiStatusText");

  const GAUGE_CIRCUMFERENCE = 540.35; // 2 * PI * 86, matches CSS

  // ---------- Auto-calculate loan-to-income ratio ----------
  function recalcPercent() {
    const income = parseFloat(incomeInput.value);
    const amount = parseFloat(amountInput.value);
    if (income > 0 && amount >= 0) {
      percentInput.value = (amount / income).toFixed(2);
    }
  }
  incomeInput.addEventListener("input", recalcPercent);
  amountInput.addEventListener("input", recalcPercent);
  recalcPercent();

  // ---------- Service status check ----------
  fetch("/health", { method: "GET" })
    .then((res) => {
      if (res.ok) {
        apiDot.classList.add("ok");
        apiStatusText.textContent = "service ready";
      } else {
        throw new Error("bad status");
      }
    })
    .catch(() => {
      apiDot.classList.add("down");
      apiStatusText.textContent = "service unreachable";
    });

  fetch("/models").then(res => res.ok ? res.json() : Promise.reject()).then(data => {
    const select = document.getElementById("modelChoice");
    data.models.forEach(name => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name.replaceAll("_", " ") + (name === data.default_model ? " (selected on validation)" : "");
      select.appendChild(option);
    });
  }).catch(() => {});
  // ---------- Helpers ----------
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.querySelector(".btn-label").textContent = isLoading
      ? "Reviewing file…"
      : "Assess risk";
  }

  function showError(message) {
    errorNote.textContent = message;
    errorNote.hidden = false;
  }

  function clearError() {
    errorNote.hidden = true;
    errorNote.textContent = "";
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      el.textContent = value.toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = to.toFixed(1);
    }
    requestAnimationFrame(tick);
  }

  function renderVerdict(data) {
    const probabilityPct = data.default_probability * 100;
    const thresholdPct = data.threshold * 100;
    const isHighRisk = data.default_prediction === 1;

    verdict.hidden = false;
    verdict.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Gauge fill
    const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPct / 100);
    gaugeFill.style.stroke = isHighRisk ? "var(--risk-red)" : "var(--brass)";
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
    });

    // Threshold tick
    gaugeThreshold.style.transform = `rotate(${thresholdPct * 3.6}deg)`;

    // Number readout
    animateNumber(probNumber, 0, probabilityPct, 1000);

    // Stamp
    stampBadge.classList.remove("stamp--in", "risk-high");
    void stampBadge.offsetWidth; // restart animation
    if (isHighRisk) {
      stampBadge.classList.add("risk-high");
      stampText.textContent = "HIGH RISK";
    } else {
      stampText.textContent = "LOW RISK";
    }
    requestAnimationFrame(() => stampBadge.classList.add("stamp--in"));

    // Ledger facts
    factProb.textContent = `${probabilityPct.toFixed(1)}%`;
    factThreshold.textContent = `${thresholdPct.toFixed(1)}%`;
    factResult.textContent = data.Result;
  }

  // Invalidate stale results if inputs or the selected model change.
  function invalidatePrediction() {
    requestVersion++;
    lastPrediction = null;
    if (explanationController) explanationController.abort();
    explanationPanel.hidden = true;
    verdict.hidden = true;
  }
  form.addEventListener("input", invalidatePrediction);
  form.addEventListener("change", invalidatePrediction);

  explainBtn.addEventListener("click", async () => {
    if (!lastPrediction) return;
    const snapshot = lastPrediction;
    const version = requestVersion;
    explanationController = new AbortController();
    const timeout = setTimeout(() => explanationController?.abort(), 120000);
    explainBtn.disabled = true;
    explainContent.hidden = true;
    explainStatus.textContent = "Calculating SHAP contributions… The first request may take longer.";
    try {
      const response = await fetch("/explain?model=" + encodeURIComponent(snapshot.model), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(snapshot.payload), signal: explanationController.signal
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Explanation request failed.");
      if (version !== requestVersion) return;
      if (Math.abs(data.default_probability - snapshot.probability) > 0.00001) throw new Error("Prediction changed. Please assess risk again.");
      document.getElementById("explainedModel").textContent = "Explained model: " + data.model.replaceAll("_", " ");
      document.getElementById("shapBase").textContent = (100 * data.base_probability).toFixed(2) + "%";
      document.getElementById("shapPrediction").textContent = (100 * data.default_probability).toFixed(2) + "%";
      const rows = document.getElementById("shapRows"); rows.replaceChildren();
      const maximum = Math.max(...data.features.map(f => Math.abs(f.percentage_points)), .00001);
      data.features.forEach(f => {
        const row = document.createElement("div"); row.className = "shap-row";
        const label = document.createElement("div"); label.className = "shap-label"; label.textContent = f.label;
        const input = document.createElement("small"); input.textContent = f.value === null ? "Missing (imputed)" : String(f.value); label.append(input);
        const track = document.createElement("div"); track.className = "shap-track";
        const bar = document.createElement("div"); bar.className = "shap-bar " + (f.contribution >= 0 ? "shap-up" : "shap-down");
        bar.style.width = (100 * Math.abs(f.percentage_points) / maximum) + "%"; track.append(bar);
        const value = document.createElement("div"); value.className = "shap-value";
        value.textContent = (f.percentage_points >= 0 ? "+" : "") + f.percentage_points.toFixed(2) + " pp";
        const direction = document.createElement("small"); direction.textContent = f.direction === "neutral" ? "No effect" : f.direction + " risk"; value.append(direction);
        row.append(label, track, value); rows.append(row);
      });
      const warnings = document.getElementById("shapWarnings"); warnings.replaceChildren();
      data.warnings.forEach(text => { const p = document.createElement("p"); p.className = "shap-warning"; p.textContent = text; warnings.append(p); });
      explainStatus.textContent = "Explanation ready. All 11 feature contributions are shown.";
      explainContent.hidden = false;
    } catch (error) {
      if (version === requestVersion) explainStatus.textContent = error.name === "AbortError" ? "Explanation timed out. Your prediction is still available; try again shortly." : error.message;
    } finally {
      clearTimeout(timeout);
      if (version === requestVersion) explainBtn.disabled = false;
    }
  });

  // ---------- Submit ----------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    recalcPercent();
    if (!form.reportValidity()) return;
    const version = ++requestVersion;
    if (explanationController) explanationController.abort();
    lastPrediction = null;
    explanationPanel.hidden = true;
    verdict.hidden = true;
    clearError();
    setLoading(true);

    const payload = {
      person_age: parseInt(document.getElementById("person_age").value, 10),
      person_income: parseFloat(incomeInput.value),
      person_home_ownership: document.getElementById("person_home_ownership").value,
      person_emp_length: parseFloat(document.getElementById("person_emp_length").value),
      loan_intent: document.getElementById("loan_intent").value,
      loan_grade: document.getElementById("loan_grade").value,
      loan_amnt: parseFloat(amountInput.value),
      loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
      loan_percent_income: parseFloat(percentInput.value),
      cb_person_default_on_file: document.getElementById("cb_person_default_on_file").value,
      cb_person_cred_hist_length: parseInt(document.getElementById("cb_person_cred_hist_length").value, 10),
    };

    try {
      const res = await fetch("/predict" + (document.getElementById("modelChoice").value ? "?model=" + encodeURIComponent(document.getElementById("modelChoice").value) : "") , {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = Array.isArray(body?.detail)
          ? body.detail.map(item => item.msg.replace(/^Value error, /, "")).join("; ")
          : (body?.detail || `HTTP ${res.status}`);
        throw new Error(detail);
      }

      const data = await res.json();
      if (version !== requestVersion) return;
      renderVerdict(data);
      lastPrediction = { payload: structuredClone(payload), model: data.model, probability: data.default_probability };
      explanationPanel.hidden = false;
      explainContent.hidden = true;
      explainStatus.textContent = "Explain the submitted inputs using " + data.model.replaceAll("_", " ") + ".";
      explainBtn.disabled = false;
    } catch (err) {
      showError(`Prediction unavailable: ${err.message || "Check the service is running."}`);
    } finally {
      setLoading(false);
    }
  });
})();
