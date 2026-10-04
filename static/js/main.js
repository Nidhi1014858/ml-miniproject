/**
 * static/js/main.js
 * LinkWise - Premium Frosted Glass Client Logic
 * Handles interactive prediction submission, loading states, priority styling,
 * confidence progress bar, decision timeline nodes, and realistic random packet synthesis.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Form and action elements
  const packetForm = document.getElementById("packet-form");
  const predictBtn = document.getElementById("predict-btn");
  const randomBtn = document.getElementById("random-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const predictIcon = document.getElementById("predict-icon");
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");

  // Result card views
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultLoading = document.getElementById("result-loading");
  const resultDisplay = document.getElementById("result-display");

  // Output targets
  const priorityBadge = document.getElementById("priority-badge");
  const priorityValue = document.getElementById("priority-value");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceBar = document.getElementById("confidence-bar");
  const decisionPath = document.getElementById("decision-path");
  const summaryChips = document.getElementById("summary-chips");

  // Form input elements
  const dataTypeInput = document.getElementById("data_type");
  const sizeKbInput = document.getElementById("size_kb");
  const batteryPctInput = document.getElementById("battery_pct");
  const passTimeMinInput = document.getElementById("pass_time_min");

  /**
   * Helper: Escape HTML to prevent injection
   */
  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Helper: Show or hide inline error notification
   */
  function showError(msg) {
    if (msg) {
      errorMessage.textContent = msg;
      errorBox.style.display = "flex";
      errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      errorBox.style.display = "none";
      errorMessage.textContent = "";
    }
  }

  /**
   * Helper: Set UI loading state during prediction
   */
  function setLoading(isLoading) {
    if (isLoading) {
      predictBtn.disabled = true;
      randomBtn.disabled = true;
      if (btnSpinner) btnSpinner.style.display = "inline-block";
      if (predictIcon) predictIcon.style.display = "none";
      showError(null);
      resultPlaceholder.style.display = "none";
      resultDisplay.style.display = "none";
      resultLoading.style.display = "flex";
    } else {
      predictBtn.disabled = false;
      randomBtn.disabled = false;
      if (btnSpinner) btnSpinner.style.display = "none";
      if (predictIcon) predictIcon.style.display = "inline-block";
      resultLoading.style.display = "none";
      if (resultDisplay.style.display !== "block") {
        resultPlaceholder.style.display = "flex";
      }
    }
  }

  /**
   * Helper: Get currently selected radio pill toggle value
   */
  function getRadioValue(name) {
    const checked = document.querySelector(`input[name="${name}"]:checked`);
    return checked ? checked.value : null;
  }

  /**
   * Display prediction results with soft animation & timeline nodes
   */
  function displayResult(result, packet) {
    const priority = String(result.priority || "").toUpperCase();
    priorityValue.textContent = priority;

    // Reset priority styling classes
    priorityBadge.classList.remove("priority-high", "priority-medium", "priority-low");
    confidenceBar.classList.remove("bar-high", "bar-medium", "bar-low");

    if (priority === "HIGH") {
      priorityBadge.classList.add("priority-high");
      confidenceBar.classList.add("bar-high");
    } else if (priority === "LOW") {
      priorityBadge.classList.add("priority-low");
      confidenceBar.classList.add("bar-low");
    } else {
      priorityBadge.classList.add("priority-medium");
      confidenceBar.classList.add("bar-medium");
    }

    // Confidence percentage and bar
    const confidencePct = Math.round((Number(result.confidence) || 0) * 100);
    confidenceValue.textContent = `${confidencePct}%`;
    confidenceBar.style.width = `${confidencePct}%`;

    // Populate decision path as connected reasoning nodes
    decisionPath.innerHTML = "";
    if (Array.isArray(result.path) && result.path.length > 0) {
      result.path.forEach((step, idx) => {
        const li = document.createElement("li");
        li.className = "decision-step-item";
        li.style.animationDelay = `${idx * 70}ms`;
        li.innerHTML = `
          <div class="step-node-indicator">
            <span class="step-node-dot"></span>
          </div>
          <div class="step-content">
            <span class="step-number">Step ${idx + 1}</span>
            <span class="step-text">${escapeHtml(step)}</span>
          </div>
        `;
        decisionPath.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.className = "decision-step-item";
      li.innerHTML = `
        <div class="step-node-indicator"><span class="step-node-dot"></span></div>
        <div class="step-content"><span class="step-text">Decision reached at tree root node.</span></div>
      `;
      decisionPath.appendChild(li);
    }

    // Small summary chips showing evaluated telemetry with colored accent indicators
    summaryChips.innerHTML = `
      <span class="pill-chip chip-type"><span class="chip-dot"></span><strong>Type:</strong> ${escapeHtml(packet.data_type)}</span>
      <span class="pill-chip chip-mode"><span class="chip-dot"></span><strong>Mode:</strong> ${escapeHtml(packet.sat_mode)}</span>
      <span class="pill-chip chip-size"><span class="chip-dot"></span><strong>Size:</strong> ${packet.size_kb} KB</span>
      <span class="pill-chip chip-battery"><span class="chip-dot"></span><strong>Battery:</strong> ${packet.battery_pct}%</span>
      <span class="pill-chip chip-link"><span class="chip-dot"></span><strong>Link:</strong> ${escapeHtml(packet.link_quality)}</span>
      <span class="pill-chip chip-time"><span class="chip-dot"></span><strong>Pass Time:</strong> ${packet.pass_time_min} min</span>
    `;

    // Swap view with gentle fade-in
    resultPlaceholder.style.display = "none";
    resultLoading.style.display = "none";
    resultDisplay.style.display = "block";

    if (window.innerWidth < 960) {
      resultDisplay.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /**
   * Handle prediction form submission via fetch()
   */
  packetForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    showError(null);

    const dataType = dataTypeInput.value;
    const satMode = getRadioValue("sat_mode");
    const linkQuality = getRadioValue("link_quality");

    const rawSize = sizeKbInput.value;
    const rawBattery = batteryPctInput.value;
    const rawPassTime = passTimeMinInput.value;

    if (!dataType || !satMode || !linkQuality || rawSize === "" || rawBattery === "" || rawPassTime === "") {
      showError("Please fill in all packet telemetry fields.");
      return;
    }

    // Explicitly parse numbers to numeric floats
    const sizeKb = Number(rawSize);
    const batteryPct = Number(rawBattery);
    const passTimeMin = Number(rawPassTime);

    const payload = {
      data_type: dataType,
      sat_mode: satMode,
      size_kb: sizeKb,
      battery_pct: batteryPct,
      link_quality: linkQuality,
      pass_time_min: passTimeMin,
    };

    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || `Prediction failed (${response.status})`);
      }

      displayResult(data, data.packet || payload);
    } catch (err) {
      showError(err.message || "Failed to communicate with prediction server.");
    } finally {
      setLoading(false);
    }
  });

  /**
   * Random Packet Generator
   * Generates realistic satellite telemetry and updates both inputs and pill toggles
   */
  randomBtn.addEventListener("click", () => {
    showError(null);

    const dataTypes = ["Fault alert", "Housekeeping", "SSTV image", "Voice/Data"];
    const linkQualities = ["Poor", "Fair", "Good"];

    const selectedType = dataTypes[Math.floor(Math.random() * dataTypes.length)];
    // ~15% operational probability for Safe mode
    const selectedMode = Math.random() < 0.15 ? "Safe" : "Normal";
    const selectedLink = linkQualities[Math.floor(Math.random() * linkQualities.length)];

    // Realistic size distribution based on payload type
    let size = 50;
    if (selectedType === "Fault alert") {
      size = Math.floor(Math.random() * 20) + 1; // 1-20 KB
    } else if (selectedType === "Housekeeping") {
      size = Math.floor(Math.random() * 91) + 10; // 10-100 KB
    } else if (selectedType === "SSTV image") {
      size = Math.floor(Math.random() * 601) + 200; // 200-800 KB
    } else { // Voice/Data
      size = Math.floor(Math.random() * 351) + 50; // 50-400 KB
    }

    // Realistic battery distribution based on operational mode
    let battery = 75;
    if (selectedMode === "Safe") {
      battery = Math.floor(Math.random() * 41) + 10; // 10-50%
    } else {
      battery = Math.floor(Math.random() * 91) + 10; // 10-100%
    }

    // Pass time left between 0.5 and 12.0 min in steps of 0.1
    const passTime = Number((Math.random() * (12.0 - 0.5) + 0.5).toFixed(1));

    // Update inputs
    dataTypeInput.value = selectedType;
    sizeKbInput.value = size;
    batteryPctInput.value = battery;
    passTimeMinInput.value = passTime;

    // Update pill toggle radio buttons
    const linkRadio = document.querySelector(`input[name="link_quality"][value="${selectedLink}"]`);
    if (linkRadio) linkRadio.checked = true;

    const modeRadio = document.querySelector(`input[name="sat_mode"][value="${selectedMode}"]`);
    if (modeRadio) modeRadio.checked = true;

    // Pulse animation on updated inputs
    [dataTypeInput, sizeKbInput, batteryPctInput, passTimeMinInput].forEach((el) => {
      el.classList.add("input-pulse");
      setTimeout(() => el.classList.remove("input-pulse"), 500);
    });
  });

  /**
   * Replace insight emojis and unicode characters with crisp SVG vector icons
   * coloured to match each individual card's accent palette.
   */
  function setupColoredCardIcons() {
    // 1. Noise & Expected Accuracy Card (Amber Accent)
    const rationaleIcon = document.querySelector(".rationale-icon");
    if (rationaleIcon) {
      rationaleIcon.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      `;
    }

    // 2. Metric Cards 1 to 5 (Coloured to each card's theme)
    // Card 1: Test Accuracy (Coral) - Bullseye / Precision Target
    // Card 2: Training Rows (Peach) - Stacked Dataset / Database Records
    // Card 3: Test Rows (Lilac) - Lab Test Flask / Holdout Split
    // Card 4: Tree Depth (Lavender) - Decision Tree Branch Hierarchy
    // Card 5: Number of Leaves (Mint) - Vector Leaf / Decision Terminal Nodes
    const metricCards = document.querySelectorAll(".metrics-grid .metric-card");
    const metricIcons = [
      // 1. Test Accuracy (Target / Precision)
      `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10"></circle>
        <circle cx="12" cy="12" r="6"></circle>
        <circle cx="12" cy="12" r="2"></circle>
      </svg>`,
      // 2. Training Rows (Database / Data Storage)
      `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
      </svg>`,
      // 3. Test Rows (Test Tube / Lab Flask)
      `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M10 2v7.31L4.69 17.5A2 2 0 0 0 6.27 21h11.46a2 2 0 0 0 1.58-3.5L14 9.31V2"></path>
        <line x1="8.5" y1="2" x2="15.5" y2="2"></line>
        <line x1="7" y1="14" x2="17" y2="14"></line>
      </svg>`,
      // 4. Tree Depth (Decision Tree Hierarchy)
      `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="3" width="6" height="4" rx="1"></rect>
        <rect x="3" y="17" width="6" height="4" rx="1"></rect>
        <rect x="15" y="17" width="6" height="4" rx="1"></rect>
        <path d="M12 7v5"></path>
        <path d="M6 17v-3a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v3"></path>
      </svg>`,
      // 5. Number of Leaves (Vector Leaf)
      `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"></path>
        <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"></path>
      </svg>`,
    ];

    metricCards.forEach((card, idx) => {
      const iconEl = card.querySelector(".metric-icon");
      if (iconEl && metricIcons[idx]) {
        iconEl.innerHTML = metricIcons[idx];
      }
    });

    // 3. Diagnostic Plots: Replace unicode arrow glyph with clean vector expand icon
    const expandButtons = document.querySelectorAll(".plot-expand-btn");
    const expandSvg = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>`;

    expandButtons.forEach((btn) => {
      const text = btn.textContent.replace(/[\u2922\u2190-\u21FF\u2600-\u27BF]/g, "").trim();
      btn.innerHTML = `${expandSvg} <span>${escapeHtml(text)}</span>`;
    });
  }

  // Initialize vector icons replacement immediately
  setupColoredCardIcons();
});
