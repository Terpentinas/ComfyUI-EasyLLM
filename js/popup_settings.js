/**
 * EasyLLM — Reusable Settings Panel Component
 *
 * Extracted from popup_chat.js:469-779 to be shared by:
 *   - Chat popup (openChatPopup)
 *   - Enhancer history popup (openOutputHistoryPopup)
 *   - Standalone settings popup (openSettingsPopup, accessed via gear icon)
 *
 * Exports:
 *   createSettingsPanel(node, options?)  — Creates and returns a DOM element with the full settings UI
 */

import { showToast } from "./ui_utils.js";
import { syncPopupSettingsToCanvas } from "./popup_bubble.js";

// ────────────────────────────────────────────────────────────────────────
// Helper: Create a <select> with preset options, ensuring the current value
// is always selectable even if not in the preset list.
// ────────────────────────────────────────────────────────────────────────

function _createPresetDropdown(presets, currentValue, attrs = {}) {
    const sel = document.createElement("select");
    sel.className = "llm-popup-settings-select";
    Object.assign(sel, attrs);

    let matched = false;
    for (const v of presets) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        if (String(v) === String(currentValue)) {
            opt.selected = true;
            matched = true;
        }
        sel.appendChild(opt);
    }
    // If currentValue doesn't match any preset, add it as a temporary option
    if (!matched && currentValue !== undefined && currentValue !== null && currentValue !== "") {
        const opt = document.createElement("option");
        opt.value = currentValue;
        opt.textContent = `${currentValue} (custom)`;
        opt.selected = true;
        sel.appendChild(opt);
        console.debug(`[LLM Chat] Added custom dropdown option: ${currentValue}`);
    }
    return sel;
}

// ── Friendly labels for chat template options ──
const CT_LABELS = {
    "auto": "Auto-detect (Recommended)",
    "qwen": "Qwen / ChatML",
    "llama": "Llama 3 / 3.1",
    "mistral": "Mistral",
    "phi": "Phi-3 / Phi-4",
    "deepseek": "DeepSeek",
    "gemma": "Gemma",
};

// ────────────────────────────────────────────────────────────────────────
// createSettingsPanel — creates and returns a DOM element with settings UI
//
// @param {object} node  — The LiteGraph node (EasyLLM / EasyLLMGGUF)
// @param {object} [options]
// @param {boolean} [options.showHeader=true]  — Show collapsible "⚙️ Settings" header
// @param {boolean} [options.showApplyButton=false] — Add "Apply" button that syncs to canvas
// @param {boolean} [options.compact=false] — Use compact layout (for enhancer popup)
// @returns {HTMLElement}
// ────────────────────────────────────────────────────────────────────────

export function createSettingsPanel(node, options = {}) {
    const {
        showHeader = true,
        showApplyButton = false,
        compact = false,
    } = options;

    // ── Container ──
    const container = document.createElement("div");
    container.className = compact
        ? "llm-popup-settings llm-popup-settings--compact"
        : "llm-popup-settings";

    // ── Collapsible header ──
    if (showHeader) {
        const settingsSummary = document.createElement("summary");
        settingsSummary.className = "llm-popup-settings-header";
        settingsSummary.textContent = "⚙️ Settings";
        container.appendChild(settingsSummary);

        // Restore collapsed state across popup opens
        if (node._popupSettings?.settingsOpen) {
            container.setAttribute("open", "");
        }
    } else {
        // Always expanded when no collapsible header
        container.setAttribute("open", "");
    }

    // Load current settings — read from canvas widgets FIRST (authoritative source),
    // fall back to _popupSettings (synced via callbacks).
    const popupSettings = node._popupSettings || {};
    const mlw = node.widgets?.find(w => w.name === "max_length");
    const tpw = node.widgets?.find(w => w.name === "temperature");

    const currentMaxLength = mlw?.value || popupSettings.max_length || 768;
    const currentTemperature = node.isEasyLLMGGUF
        ? (tpw?.value ?? popupSettings.temperature ?? 0.7)
        : (tpw?.value || popupSettings.temperature || "auto");

    // ── Single-column sectioned settings layout ──
    const settingsSections = document.createElement("div");
    settingsSections.className = "llm-popup-settings-sections";

    // ════════════════════════════════════════════════════════════════
    // Section: Basic — Generation parameters (always visible)
    // ════════════════════════════════════════════════════════════════
    const basicSection = document.createElement("div");
    basicSection.className = compact
        ? "llm-popup-settings-section"
        : "llm-popup-settings-section llm-popup-settings-basic";

    // Temperature
    const tempRow = document.createElement("div");
    tempRow.className = "llm-popup-settings-row";

    const tempLabel = document.createElement("span");
    tempLabel.className = "llm-popup-settings-label";
    tempLabel.textContent = "Temperature:";
    tempRow.appendChild(tempLabel);

    let tempControl;
    if (node.isEasyLLMGGUF) {
        // GGUF: Float slider (0.0–2.0, step 0.1)
        const tempInput = document.createElement("input");
        tempInput.type = "number";
        tempInput.className = "llm-popup-settings-input";
        tempInput.min = 0.0;
        tempInput.max = 2.0;
        tempInput.step = 0.1;
        tempInput.value = currentTemperature;
        tempInput.title = "Sampling temperature. 0 = greedy/deterministic.";
        tempRow.appendChild(tempInput);
        tempControl = tempInput;
    } else {
        // CLIP: Dropdown with presets
        tempControl = _createPresetDropdown(
            ["auto", "0.0", "0.3", "0.5", "0.7", "0.9"],
            currentTemperature,
            { style: "flex: 0 0 auto" }
        );
        tempRow.appendChild(tempControl);
    }
    basicSection.appendChild(tempRow);

    // Max Length
    const maxRow = document.createElement("div");
    maxRow.className = "llm-popup-settings-row";

    const maxLabel = document.createElement("span");
    maxLabel.className = "llm-popup-settings-label";
    maxLabel.textContent = "Max Length:";
    maxRow.appendChild(maxLabel);

    const maxSelect = _createPresetDropdown(
        [64, 128, 256, 384, 512, 768, 1024, 2048, 4096, 8192],
        currentMaxLength,
        { style: "flex: 0 0 auto" }
    );
    maxRow.appendChild(maxSelect);
    basicSection.appendChild(maxRow);

    // Seed
    const seedWidget = node.widgets?.find(w => w.name === "seed");
    const currentSeed = seedWidget?.value ?? 0;

    const seedRow = document.createElement("div");
    seedRow.className = "llm-popup-settings-row";

    const seedLabel = document.createElement("span");
    seedLabel.className = "llm-popup-settings-label";
    seedLabel.textContent = "Seed:";
    seedRow.appendChild(seedLabel);

    // ── Seed control strip: wider input + 🔀 auto-randomize toggle ──
    // Auto-randomize state is stored on node._llmSeedRandomize (persistent
    // custom property). Since ComfyUI's native controlWidget is N/A for this
    // node (despite control_after_generate: True), we manage it manually.
    const seedControls = document.createElement("div");
    seedControls.className = "llm-popup-seed-controls";

    const seedInput = document.createElement("input");
    seedInput.type = "number";
    seedInput.className = "llm-popup-settings-input";
    seedInput.value = currentSeed;
    seedInput.min = 0;
    seedInput.max = 0xFFFFFFFF;
    seedInput.title = "Seed value. 🔀 toggles auto-randomize on each queue.";
    seedControls.appendChild(seedInput);

    // 🔀 Auto-randomize toggle — default ON for new sessions
    // Icon: 🔀 (shuffle) when ON, 🔒 (locked) when OFF
    const toggleBtn = document.createElement("button");
    const cwMode = seedWidget?.controlWidget?.value;
    const isRandomizing = (node._llmSeedRandomize !== undefined)
        ? node._llmSeedRandomize
        : true; // Default ON
    toggleBtn.textContent = isRandomizing ? "🔀" : "🔒";
    toggleBtn.className = "llm-popup-seed-btn" + (isRandomizing ? " active" : "");
    toggleBtn.title = isRandomizing
        ? "Auto-randomize seed on each queue (currently ON)"
        : "Auto-randomize seed on each queue (currently OFF)";
    toggleBtn.addEventListener("click", () => {
        const sdW = node.widgets?.find(w => w.name === "seed");
        const nowRandomizing = !node._llmSeedRandomize;
        node._llmSeedRandomize = nowRandomizing;
        // Also sync native controlWidget if it exists
        if (sdW) {
            const cw = sdW.controlWidget;
            if (cw && typeof cw.update === "function") {
                cw.update(nowRandomizing ? "randomize" : "fixed");
            }
            if (!sdW.options) sdW.options = {};
            sdW.options.randomize = nowRandomizing;
        }
        toggleBtn.textContent = nowRandomizing ? "🔀" : "🔒";
        toggleBtn.className = "llm-popup-seed-btn" + (nowRandomizing ? " active" : "");
        toggleBtn.title = nowRandomizing
            ? "Auto-randomize seed on each queue (currently ON)"
            : "Auto-randomize seed on each queue (currently OFF)";
        console.debug(`[LLM Chat] Seed auto-randomize toggled ${nowRandomizing ? "ON" : "OFF"} for node ${node.id}`);
    });
    seedControls.appendChild(toggleBtn);

    seedRow.appendChild(seedControls);
    basicSection.appendChild(seedRow);

    // Enable Image Generation (trigger_prompt auto-queue)
    const genRow = document.createElement("div");
    genRow.className = "llm-popup-settings-row";

    const genLabel = document.createElement("span");
    genLabel.className = "llm-popup-settings-label";
    genLabel.textContent = "Image Gen:";
    genRow.appendChild(genLabel);

    const genCheckbox = document.createElement("input");
    genCheckbox.type = "checkbox";
    genCheckbox.checked = node._enableImageGeneration !== false; // default ON
    genCheckbox.title = "When ON, the model can trigger image generation (auto-queue for generate_image/edit_image). Turn OFF to disable — useful when the generator mode causes streaming or history issues.";
    genRow.appendChild(genCheckbox);

    // Store reference for syncPopupSettingsToCanvas
    node._popupGenCheckbox = genCheckbox;

    basicSection.appendChild(genRow);

    // Iterative Refinement (per-node toggle for edit_image pipeline cache)
    const iterRow = document.createElement("div");
    iterRow.className = "llm-popup-settings-row";

    const iterLabel = document.createElement("span");
    iterLabel.className = "llm-popup-settings-label";
    iterLabel.textContent = "Iterative Refine:";
    iterRow.appendChild(iterLabel);

    const iterCheckbox = document.createElement("input");
    iterCheckbox.type = "checkbox";
    iterCheckbox.checked = node._iterativeRefinement === true; // default OFF
    iterCheckbox.title = "When ON, each edit_image uses the previous generation result as its source image. When OFF, all edits use the original uploaded image.";
    iterRow.appendChild(iterCheckbox);

    // Store reference for syncPopupSettingsToCanvas
    node._popupIterRefineCheckbox = iterCheckbox;

    basicSection.appendChild(iterRow);

    settingsSections.appendChild(basicSection);

    // ════════════════════════════════════════════════════════════════
    // Section: Advanced — collapsible hardware/system settings
    // ════════════════════════════════════════════════════════════════
    const hwDetails = document.createElement("details");
    hwDetails.className = "llm-popup-settings-section";

    const hwSummary = document.createElement("summary");
    hwSummary.className = "llm-popup-settings-section-header";
    hwSummary.textContent = "⚙️ Hardware Settings";
    hwSummary.style.cursor = "pointer";
    hwDetails.appendChild(hwSummary);

    const hwContent = document.createElement("div");
    hwContent.style.cssText = "display: flex; flex-direction: column; gap: 6px; padding: 4px 0;";

    // Helper: add a settings row with label + input element
    function addHardwareRow(label, inputEl) {
        const row = document.createElement("div");
        row.className = "llm-popup-settings-row";
        const lbl = document.createElement("span");
        lbl.className = "llm-popup-settings-label";
        lbl.textContent = label;
        row.appendChild(lbl);
        row.appendChild(inputEl);
        hwContent.appendChild(row);
    }

    // VRAM Mode (shared: both CLIP and GGUF)
    const vramWidget = node.widgets?.find(w => w.name === "vram_mode");
    const vramOptions = vramWidget?.options?.values || ["unload", "keep_loaded", "aggressive_free"];
    const currentVram = vramWidget?.value || popupSettings.vram_mode || "unload";
    const vramSelect = document.createElement("select");
    vramSelect.className = "llm-popup-settings-select";
    for (const optVal of vramOptions) {
        const opt = document.createElement("option");
        opt.value = optVal;
        opt.textContent = optVal;
        if (optVal === currentVram) opt.selected = true;
        vramSelect.appendChild(opt);
    }
    node._popupVramSelect = vramSelect;
    addHardwareRow("VRAM Mode:", vramSelect);

    // use_mlock (shared: both CLIP and GGUF)
    const mlockWidget = node.widgets?.find(w => w.name === "use_mlock");
    const mlockRow = document.createElement("div");
    mlockRow.className = "llm-popup-settings-row";
    const mlockLabel = document.createElement("span");
    mlockLabel.className = "llm-popup-settings-label";
    mlockLabel.textContent = "Use mlock:";
    mlockRow.appendChild(mlockLabel);
    const mlockCheckbox = document.createElement("input");
    mlockCheckbox.type = "checkbox";
    mlockCheckbox.checked = popupSettings.use_mlock ??
        mlockWidget?.value ?? (node.isEasyLLMGGUF ? true : false);
    node._popupMlockCheckbox = mlockCheckbox;
    mlockRow.appendChild(mlockCheckbox);
    hwContent.appendChild(mlockRow);

    // ── GGUF-specific hardware fields ──
    if (node.isEasyLLMGGUF) {
        // n_gpu_layers
        const nGpuInput = document.createElement("input");
        nGpuInput.type = "number";
        nGpuInput.className = "llm-popup-settings-input";
        nGpuInput.min = -1;
        nGpuInput.max = 200;
        nGpuInput.step = 1;
        nGpuInput.value = popupSettings.n_gpu_layers ??
            node.widgets?.find(w => w.name === "n_gpu_layers")?.value ?? -1;
        nGpuInput.title = "-1 = all layers on GPU. 0 = CPU only.";
        node._popupNGpuInput = nGpuInput;
        addHardwareRow("GPU Layers:", nGpuInput);

        // n_ctx
        const nCtxInput = document.createElement("input");
        nCtxInput.type = "number";
        nCtxInput.className = "llm-popup-settings-input";
        nCtxInput.min = 512;
        nCtxInput.max = 32768;
        nCtxInput.step = 512;
        nCtxInput.value = popupSettings.n_ctx ??
            node.widgets?.find(w => w.name === "n_ctx")?.value ?? 4096;
        nCtxInput.title = "Context window size (max tokens the model can remember).";
        node._popupNCtxInput = nCtxInput;
        addHardwareRow("Context Size:", nCtxInput);

        // chat_template
        const ctWidget = node.widgets?.find(w => w.name === "chat_template");
        const ctNames = ctWidget?.options?.values || ["auto", "qwen", "llama", "mistral", "phi", "deepseek", "gemma"];
        const ctSelect = document.createElement("select");
        ctSelect.className = "llm-popup-settings-select";
        const savedCT = popupSettings.chat_template ||
            node.widgets?.find(w => w.name === "chat_template")?.value || "auto";
        for (const name of ctNames) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = CT_LABELS[name] || name;
            if (name === savedCT) opt.selected = true;
            ctSelect.appendChild(opt);
        }
        node._popupCTSelect = ctSelect;
        addHardwareRow("Chat Template:", ctSelect);

        // top_k
        const topKInput = document.createElement("input");
        topKInput.type = "number";
        topKInput.className = "llm-popup-settings-input";
        topKInput.min = 1;
        topKInput.max = 100;
        topKInput.step = 1;
        topKInput.value = popupSettings.top_k ??
            node.widgets?.find(w => w.name === "top_k")?.value ?? 50;
        node._popupTopKInput = topKInput;
        addHardwareRow("Top-K:", topKInput);

        // top_p
        const topPInput = document.createElement("input");
        topPInput.type = "number";
        topPInput.className = "llm-popup-settings-input";
        topPInput.min = 0.0;
        topPInput.max = 1.0;
        topPInput.step = 0.05;
        topPInput.value = popupSettings.top_p ??
            node.widgets?.find(w => w.name === "top_p")?.value ?? 0.9;
        node._popupTopPInput = topPInput;
        addHardwareRow("Top-P:", topPInput);

        // repetition_penalty
        const repWidget = node.widgets?.find(w => w.name === "repetition_penalty");
        const repInput = document.createElement("input");
        repInput.type = "number";
        repInput.className = "llm-popup-settings-input";
        repInput.min = 0.0;
        repInput.max = 2.0;
        repInput.step = 0.05;
        repInput.value = popupSettings.repetition_penalty ??
            repWidget?.value ?? 1.0;
        repInput.title = "Repetition penalty. 1.0 = no penalty.";
        node._popupRepPenaltyInput = repInput;
        addHardwareRow("Rep. Penalty:", repInput);
    }

    hwDetails.appendChild(hwContent);
    settingsSections.appendChild(hwDetails);

    // Store popup DOM references for syncPopupSettingsToCanvas
    node._popupMaxSelect = maxSelect;
    node._popupTempSelect = tempControl;
    node._popupSeedInput = seedInput;

    container.appendChild(settingsSections);

    // ── Apply button (optional) ──
    if (showApplyButton) {
        const applyBtn = document.createElement("button");
        applyBtn.className = "llm-popup-save-btn";
        applyBtn.textContent = "✅ Apply";
        applyBtn.style.marginTop = "8px";
        applyBtn.style.width = "100%";
        applyBtn.onclick = () => {
            syncPopupSettingsToCanvas(node);
            showToast("Settings applied", "success", 1500);
        };
        container.appendChild(applyBtn);
    }

    return container;
}
