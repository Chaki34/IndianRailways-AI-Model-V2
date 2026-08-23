const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("predict-form");
const resultCard = document.getElementById("result-card");
const errorBox = document.getElementById("error-box");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  resultCard.hidden = true;

  const payload = {
    department: document.getElementById("department").value,
    asset_type: document.getElementById("asset_type").value,
    maintenance_type: document.getElementById("maintenance_type").value,
    traffic_density: document.getElementById("traffic_density").value,
    risk_level: document.getElementById("risk_level").value,
    section_id: document.getElementById("section_id").value,
    asset_age_years: parseFloat(document.getElementById("asset_age_years").value),
    condition_score: parseFloat(document.getElementById("condition_score").value),
    severity: parseInt(document.getElementById("severity").value, 10),
    criticality_score: parseFloat(document.getElementById("criticality_score").value),
    urgency_score: parseInt(document.getElementById("urgency_score").value, 10),
    safety_risk_score: parseInt(document.getElementById("safety_risk_score").value, 10),
    overdue_days: parseInt(document.getElementById("overdue_days").value, 10),
    estimated_duration_hours: parseFloat(document.getElementById("estimated_duration_hours").value),
  };

  try {
    const res = await fetch(`${API_BASE}/predict-duration`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const hours = data.predicted_duration_hours;
    const wholeHours = Math.floor(hours);
    const minutes = Math.round((hours - wholeHours) * 60);

    document.getElementById("result-hours").textContent = `${hours.toFixed(2)} HOURS`;
    document.getElementById("result-human").textContent =
      `\u2248 ${wholeHours} hours ${minutes} minutes`;
    document.getElementById("result-range").textContent =
      `Estimated range: ${data.estimated_range.lower_hours}h \u2013 ${data.estimated_range.upper_hours}h`;

    resultCard.hidden = false;
  } catch (err) {
    errorBox.textContent = `Prediction failed: ${err.message}`;
    errorBox.hidden = false;
  }
});
