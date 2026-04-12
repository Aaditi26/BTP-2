const revealItems = document.querySelectorAll(".section, .hero-panel");

const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            entry.target.classList.add("visible");
        }
    });
}, {
    threshold: 0.12
});

revealItems.forEach((item) => {
    item.classList.add("reveal");
    observer.observe(item);
});

const optimizerForm = document.querySelector("#optimizer-form");
const statusPill = document.querySelector("#status-pill");
const bestSequence = document.querySelector("#best-sequence");
const bestTime = document.querySelector("#best-time");
const bestCost = document.querySelector("#best-cost");
const bestWear = document.querySelector("#best-wear");
const operationResults = document.querySelector("#operation-results");
const paretoResults = document.querySelector("#pareto-results");

function parseConstraints(rawValue, separator) {
    return rawValue
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
            const [left, right] = line.split(separator).map((part) => part.trim());
            return { left, right };
        })
        .filter((entry) => entry.left && entry.right);
}

function buildPayload(form) {
    const selectedOperations = Array.from(form.querySelectorAll('input[name="operations"]:checked'))
        .map((input) => input.value);

    const precedenceConstraints = parseConstraints(form.precedence_constraints.value, ">")
        .map((item) => ({ before: item.left, after: item.right }));

    const fixedPositions = parseConstraints(form.fixed_positions.value, "@")
        .map((item) => ({ operation: item.left, index: Number(item.right) }));

    return {
        job_id: form.job_id.value.trim(),
        material_type: form.material_type.value,
        material_hardness: Number(form.material_hardness.value),
        workpiece_geometry: {
            length: Number(form.length.value),
            diameter: Number(form.diameter.value),
            thickness: Number(form.thickness.value)
        },
        tolerance_requirement: Number(form.tolerance_requirement.value),
        surface_finish_requirement: Number(form.surface_finish_requirement.value),
        machine_type: form.machine_type.value,
        available_operations: selectedOperations,
        precedence_constraints: precedenceConstraints,
        fixed_positions: fixedPositions,
        machine_limits: {
            speed: [Number(form.speed_min.value), Number(form.speed_max.value)],
            feed: [Number(form.feed_min.value), Number(form.feed_max.value)],
            depth: [Number(form.depth_min.value), Number(form.depth_max.value)]
        }
    };
}

function cleanTextArtifacts(text) {
    return text
        .replace(/\u00C2\u00B7/g, "|")
        .replace(/\u00E2\u2020\u2019/g, "->")
        .replace(/\u00E2\u2020\u201C/g, "↓")
        .replace(/\u00E2\u20AC\u00B0/g, "<=")
        .replace(/\u00CE\u00A3/g, "Sum");
}

function normalizeStaticText() {
    document.querySelectorAll("pre code, .flow-arrow").forEach((node) => {
        node.textContent = cleanTextArtifacts(node.textContent);
    });
}

function renderOperations(operations) {
    operationResults.innerHTML = operations.map((entry) => `
        <article class="operation-card">
            <div class="operation-head">
                <h4>${entry.operation}</h4>
                <span>${entry.parameters.tool_type} · ${entry.parameters.coolant_condition}</span>
            </div>
            <div class="operation-grid">
                <p><strong>Speed:</strong> ${entry.parameters.cutting_speed}</p>
                <p><strong>Feed:</strong> ${entry.parameters.feed_rate}</p>
                <p><strong>Depth:</strong> ${entry.parameters.depth_of_cut}</p>
                <p><strong>Time:</strong> ${entry.predictions.machining_time}</p>
                <p><strong>Cost:</strong> ${entry.predictions.cost}</p>
                <p><strong>Wear:</strong> ${entry.predictions.tool_wear}</p>
            </div>
        </article>
    `).join("");
    operationResults.innerHTML = cleanTextArtifacts(operationResults.innerHTML);
}

function renderPareto(paretoSet) {
    paretoResults.innerHTML = paretoSet.map((entry, index) => `
        <article class="pareto-card">
            <div class="pareto-rank">Solution ${index + 1}</div>
            <p>${entry.sequence.join(" → ")}</p>
            <div class="pareto-metrics">
                <span>Time: ${entry.total_machining_time}</span>
                <span>Cost: ${entry.total_cost}</span>
                <span>Wear: ${entry.total_tool_wear}</span>
            </div>
        </article>
    `).join("");
    paretoResults.innerHTML = cleanTextArtifacts(paretoResults.innerHTML);
}

async function submitOptimization(event) {
    event.preventDefault();

    const payload = buildPayload(optimizerForm);

    if (payload.available_operations.length < 2) {
        statusPill.textContent = "Need more operations";
        operationResults.innerHTML = '<p class="placeholder-text">Select at least two machining operations.</p>';
        paretoResults.innerHTML = "";
        return;
    }

    statusPill.textContent = "Optimizing...";
    operationResults.innerHTML = '<p class="placeholder-text">Running Random Forest and NSGA-II locally. Please wait...</p>';
    paretoResults.innerHTML = "";

    try {
        const response = await fetch("/optimize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Optimization request failed.");
        }

        statusPill.textContent = "Completed";
        bestSequence.textContent = result.optimal_sequence.join(" → ");
        bestTime.textContent = result.minimum_total_machining_time;
        bestCost.textContent = result.minimum_total_cost;
        bestWear.textContent = result.minimum_total_tool_wear;
        bestSequence.textContent = cleanTextArtifacts(bestSequence.textContent);

        renderOperations(result.optimal_parameters);
        renderPareto(result.pareto_optimal_set);
    } catch (error) {
        statusPill.textContent = "Error";
        operationResults.innerHTML = `<p class="placeholder-text">${error.message}</p>`;
        paretoResults.innerHTML = "";
    }
}

optimizerForm?.addEventListener("submit", submitOptimization);
normalizeStaticText();
