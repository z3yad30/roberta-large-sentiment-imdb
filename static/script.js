// Use relative URL → safest solution when frontend and API are served from same origin
const API_BASE = "/api/v1";

// Alternative (explicit) versions – use only one:
// const API_BASE = "http://127.0.0.1:8000/api/v1";
// const API_BASE = "http://localhost:8000/api/v1";

async function postData(endpoint, data = {}) {
    const url = `${API_BASE}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            let errorDetail = "Request failed";
            try {
                const err = await response.json();
                errorDetail = err.detail || errorDetail;
            } catch {
                // ignore json parse error
            }
            throw new Error(errorDetail);
        }

        return await response.json();
    } catch (err) {
        console.error("Fetch error:", err);
        throw err;
    }
}

function showSingleResult(data) {
    const box = document.getElementById("single-result");
    box.classList.remove("hidden");

    const labelEl = document.getElementById("single-label");
    const confEl  = document.getElementById("single-confidence");
    const probEl  = document.getElementById("single-prob");
    const textEl  = document.getElementById("single-text-preview");

    const isPositive = data.label === "POSITIVE";

    labelEl.textContent = data.label;
    labelEl.className = "result-label " + (isPositive ? "positive" : "negative");

    confEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    if (data.probabilities) {
        probEl.innerHTML = `
            POSITIVE: ${(data.probabilities.POSITIVE * 100).toFixed(1)}%  
              NEGATIVE: ${(data.probabilities.NEGATIVE * 100).toFixed(1)}%
        `;
    } else {
        probEl.innerHTML = "";
    }

    textEl.textContent = data.text.substring(0, 220) + (data.text.length > 220 ? "..." : "");
}

function showBatchResults(predictions) {
    const container = document.getElementById("batch-items");
    container.innerHTML = "";

    predictions.forEach(pred => {
        const div = document.createElement("div");
        div.className = "batch-item";

        const isPos = pred.label === "POSITIVE";

        div.innerHTML = `
            <div style="font-weight:600; color:${isPos ? '#2a9d8f' : '#e63946'}">
                ${pred.label} – ${(pred.confidence * 100).toFixed(1)}%
            </div>
            ${pred.probabilities ? `
                <div style="font-size:0.9rem; color:#555; margin:0.4rem 0;">
                    POS: ${(pred.probabilities.POSITIVE*100).toFixed(1)}% • 
                    NEG: ${(pred.probabilities.NEGATIVE*100).toFixed(1)}%
                </div>` : ""}
            <div style="margin-top:0.6rem; line-height:1.4;">
                ${pred.text.substring(0, 180)}${pred.text.length > 180 ? "..." : ""}
            </div>
        `;

        container.appendChild(div);
    });

    document.getElementById("batch-result").classList.remove("hidden");
}

// ── Single prediction ────────────────────────────────────────
document.getElementById("single-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = document.getElementById("single-text").value.trim();
    const returnProb = document.getElementById("single-probs").checked;

    if (!text) return;

    try {
        const data = await postData("/predict", {
            text,
            return_probabilities: returnProb
        });
        showSingleResult(data);
    } catch (err) {
        alert("Error: " + err.message);
    }
});

// ── Batch prediction ─────────────────────────────────────────
document.getElementById("batch-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const raw = document.getElementById("batch-texts").value.trim();
    if (!raw) return;

    const texts = raw.split("\n")
        .map(t => t.trim())
        .filter(t => t.length > 0);

    if (texts.length === 0) return;

    const returnProb = document.getElementById("batch-probs").checked;

    try {
        const { predictions } = await postData("/batch-predict", {
            texts,
            return_probabilities: returnProb
        });
        showBatchResults(predictions);
    } catch (err) {
        alert("Batch error: " + err.message);
    }
});