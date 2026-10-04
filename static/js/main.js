/**
 * static/js/main.js
 * SatPrior - Frontend Client Logic
 * Handles interactive form submission via fetch(), loading state animations,
 * priority badge visual formatting, and random packet synthesis.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const packetForm = document.getElementById("packet-form");
  const predictBtn = document.getElementById("predict-btn");
  const randomBtn = document.getElementById("random-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const errorBox = document.getElementById("error-box");
  const errorMessage = document.getElementById("error-message");

  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultLoading = document.getElementById("result-loading");
  const resultDisplay = document.getElementById("result-display");

  const priorityBadge = document.getElementById("priority-badge");
  const priorityValue = document.getElementById("priority-value");
  const explanationText = document.getElementById("explanation-text");
  const summaryChips = document.getElementById("summary-chips");

  // Form Inputs
  const dataTypeInput = document.getElementById("data_type");
  const urgencyInput = document.getElementById("urgency");
  const dataSizeInput = document.getElementById("data_size_kb");
  const batteryLevelInput = document.getElementById("battery_level");
  const linkQualityInput = document.getElementById("link_quality");

  /**
   * Helper: Show or hide inline error message
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
   * Helper: Set UI loading state
   */
  function setLoading(isLoading) {
    if (isLoading) {
      predictBtn.disabled = true;
      randomBtn.disabled = true;
      btnSpinner.style.display = "inline-block";
      showError(null);
      resultPlaceholder.style.display = "none";
      resultDisplay.style.display = "none";
      resultLoading.style.display = "flex";
    } else {
      predictBtn.disabled = false;
      randomBtn.disabled = false;
      btnSpinner.style.display = "none";
      resultLoading.style.display = "none";
    }
  }

  /**
   * Display prediction results and update styling
   */
  function displayResult(priority, explanation, packet) {
    const p = String(priority).toUpperCase();
    priorityValue.textContent = p;

    // Reset classes
    priorityBadge.classList.remove("priority-high", "priority-medium", "priority-low");

    if (p === "HIGH") {
      priorityBadge.classList.add("priority-high");
    } else if (p === "LOW") {
      priorityBadge.classList.add("priority-low");
    } else {
      priorityBadge.classList.add("priority-medium");
    }

    explanationText.textContent = explanation;

    // Populate summary chips
    summaryChips.innerHTML = `
      <span class="chip"><strong>Type:</strong> ${packet.data_type}</span>
      <span class="chip"><strong>Urgency:</strong> ${packet.urgency}</span>
      <span class="chip"><strong>Size:</strong> ${packet.data_size_kb} KB</span>
      <span class="chip"><strong>Battery:</strong> ${packet.battery_level}%</span>
      <span class="chip"><strong>Link:</strong> ${packet.link_quality}</span>
    `;

    resultPlaceholder.style.display = "none";
    resultLoading.style.display = "none";
    resultDisplay.style.display = "block";

    // Smooth scroll into result view if on small screens
    if (window.innerWidth < 960) {
      resultDisplay.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /**
   * Handle form submission via fetch()
   */
  packetForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    showError(null);

    const dataType = dataTypeInput.value;
    const urgency = urgencyInput.value;
    const dataSize = parseFloat(dataSizeInput.value);
    const batteryLevel = parseFloat(batteryLevelInput.value);
    const linkQuality = linkQualityInput.value;

    // Client-side quick validation
    if (isNaN(dataSize) || dataSize < 1 || dataSize > 10000) {
      showError("Please enter a valid packet size between 1 and 10,000 KB.");
      return;
    }
    if (isNaN(batteryLevel) || batteryLevel < 0 || batteryLevel > 100) {
      showError("Please enter a valid battery level between 0% and 100%.");
      return;
    }

    const payload = {
      data_type: dataType,
      urgency: urgency,
      data_size_kb: dataSize,
      battery_level: batteryLevel,
      link_quality: linkQuality,
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
        throw new Error(data.error || `Server returned error (${response.status})`);
      }

      displayResult(data.priority, data.explanation, data.packet);
    } catch (err) {
      showError(err.message || "Failed to communicate with prediction server.");
      resultPlaceholder.style.display = "flex";
      resultDisplay.style.display = "none";
    } finally {
      setLoading(false);
    }
  });

  /**
   * Random Packet Generator
   * Generates a realistic SomaiyaSat telemetry packet
   */
  randomBtn.addEventListener("click", () => {
    showError(null);

    const dataTypes = ["TT&C", "Housekeeping", "SSTV", "Voice/Data"];
    const urgencies = ["Low", "Medium", "High"];
    const linkQualities = ["Poor", "Fair", "Good"];

    const selectedType = dataTypes[Math.floor(Math.random() * dataTypes.length)];
    const selectedUrgency = urgencies[Math.floor(Math.random() * urgencies.length)];
    const selectedLink = linkQualities[Math.floor(Math.random() * linkQualities.length)];

    let size = 120;
    if (selectedType === "TT&C") {
      size = Math.floor(Math.random() * 50) + 10; // 10-60 KB
    } else if (selectedType === "Housekeeping") {
      size = Math.floor(Math.random() * 140) + 40; // 40-180 KB
    } else if (selectedType === "Voice/Data") {
      size = Math.floor(Math.random() * 650) + 150; // 150-800 KB
    } else {
      size = Math.floor(Math.random() * 1800) + 500; // 500-2300 KB
    }

    const battery = Math.floor(Math.random() * 85) + 15; // 15-99%

    // Apply values to inputs
    dataTypeInput.value = selectedType;
    urgencyInput.value = selectedUrgency;
    dataSizeInput.value = size;
    batteryLevelInput.value = battery;
    linkQualityInput.value = selectedLink;

    // Visual feedback pulse on inputs
    [dataTypeInput, urgencyInput, dataSizeInput, batteryLevelInput, linkQualityInput].forEach((el) => {
      el.classList.add("input-pulse");
      setTimeout(() => el.classList.remove("input-pulse"), 600);
    });
  });

});
