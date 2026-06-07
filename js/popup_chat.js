/**
 * EasyLLM — Popup modal module (chat + enhancer history)
 *
 * Contains: openChatPopup, closeChatPopup, openOutputHistoryPopup.
 * These are the public entry points for opening/closing popup modals.
 */

import { createOverlayModal, showToast } from "./ui_utils.js";
import { getActivePopupNode, setActivePopupNode, clearActivePopupNode } from "./constants.js";
import { openPromptManagerDialog } from "./editor.js";
import { startProgressTracking, stopProgressTracking, stopCanvasProgressTracking } from "./websocket_bridge.js";
import { getModelName, abortStreaming, renderPopupHistory, syncPopupSettingsToCanvas, createBubbleElement, hideTypingIndicator, exportChat, exportEnhancerHistory, downloadExport, handlePopupSend, openImagePicker, removeAttachedImage } from "./popup_bubble.js";
import { createScrollToBottomBtn, updateScrollState, scrollToBottom, estimateContextTokens, autoScrollIfNeeded } from "./popup_utils.js";
import { openModelBrowserPopup, closeModelBrowserPopup } from "./popup_model_browser.js";

// ────────────────────────────────────────────────────────────────────────
// Popup: Open the chat popup modal for a given node
// ────────────────────────────────────────────────────────────────────────

export function openChatPopup(node) {
    if (node._popupOverlay) return; // Already open

    // Close any other open popup first (only one popup at a time)
    if (getActivePopupNode() && getActivePopupNode() !== node) {
        closeChatPopup(getActivePopupNode());
    }

    const nodeLabel = node.title || `Node #${node.id}`;

    // Get model name for header display
    const modelName = getModelName(node);

    const headerTitle = `🤖 EasyLLM${modelName ? ` | ${modelName}` : ""}`;

    const { overlay, panel, body, footer, header } = createOverlayModal(
        "llm-popup",
        headerTitle,
        () => closeChatPopup(node),
        { hasFooter: true }
    );

    // Store panel reference for resize persistence
    node._popupPanel = panel;

    // Restore saved popup dimensions (from CSS resize)
    const savedW = node._popupSettings?.popupWidth;
    const savedH = node._popupSettings?.popupHeight;
    if (savedW) {
        panel.style.width = savedW;
        console.debug(`[LLM Chat] Restored popup width: ${savedW}`);
    }
    if (savedH) {
        panel.style.height = savedH;
        console.debug(`[LLM Chat] Restored popup height: ${savedH}`);
    }

    // ── Camera badge: visible when IMAGE socket is connected OR image uploaded ──
    const cameraBadge = document.createElement("span");
    cameraBadge.className = "llm-popup-camera-badge";
    cameraBadge.textContent = "📷";
    cameraBadge.title = "Image input connected";
    // Check if IMAGE socket is connected or image was uploaded via popup
    const imageInput = node.inputs?.find(i => i.name === "image");
    const hasWiredImage = imageInput && imageInput.link != null;
    const hasUploadedImage = !!node._uploadedImage;
    cameraBadge.style.display = (hasWiredImage || hasUploadedImage) ? "inline-block" : "none";
    header.appendChild(cameraBadge);

    // ── Header action buttons (clear, export, stop) ──
    const headerActions = document.createElement("div");
    headerActions.className = "llm-popup-header-actions";

    // Clear button
    const clearBtn = document.createElement("button");
    clearBtn.className = "llm-popup-header-btn llm-popup-clear-btn";
    clearBtn.textContent = "🗑 Clear";
    clearBtn.title = "Clear conversation";
    clearBtn.onclick = () => {
        if (confirm("Clear the entire conversation?")) {
            node._chatHistory = [];
            // Also clear any uploaded image (conversation context is gone)
            if (node._uploadedImage) {
                removeAttachedImage(node);
            }
            renderPopupHistory(node);
            showToast("Chat cleared", "success", 2000);
        }
    };
    headerActions.appendChild(clearBtn);

    // Stop button (hidden by default, shown during generation)
    const stopBtn = document.createElement("button");
    stopBtn.className = "llm-popup-header-btn llm-popup-stop-btn";
    stopBtn.textContent = "■ Stop";
    stopBtn.title = "Stop generation";
    stopBtn.style.display = "none";
    stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping...";
        await abortStreaming(node);
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.textContent = "■ Stop";
        hideTypingIndicator(node);
        showToast("Generation stopped", "info", 2000);
    };
    node._popupStopBtn = stopBtn;
    headerActions.appendChild(stopBtn);

    // Insert header actions before the close button
    const closeBtn = header.querySelector(".llm-popup-close-btn");
    if (closeBtn) {
        header.insertBefore(headerActions, closeBtn);
    }

    // ── Progress bar (thin strip below header, hidden by default) ──
    const progressBar = document.createElement("div");
    progressBar.className = "llm-popup-progress";
    progressBar.style.display = "none";
    const progressFill = document.createElement("div");
    progressFill.className = "llm-popup-progress-fill";
    progressBar.appendChild(progressFill);
    panel.insertBefore(progressBar, body);
    node._popupProgressBar = progressBar;
    node._popupProgressFill = progressFill;

    // Start tracking progress events for this node
    startProgressTracking(node);

    // ── Chat history ──
    const historyContainer = document.createElement("div");
    historyContainer.className = "llm-popup-history";
    node._popupHistoryEl = historyContainer;
    body.appendChild(historyContainer);

    // ── Context indicator ──
    const contextEl = document.createElement("div");
    contextEl.className = "llm-chat-context-indicator";
    contextEl.style.display = "none";
    body.appendChild(contextEl);
    node._popupContextEl = contextEl;

    // Render history (async to let DOM attach)
    requestAnimationFrame(() => {
        renderPopupHistory(node);

        // If streaming was ongoing while popup was closed, pre-fill the
        // assistant bubble with all tokens accumulated so far.
        if (node._streamingAccumulatedText && node._popupStreaming) {
            const bubble = createBubbleElement("assistant", node._streamingAccumulatedText);
            node._popupHistoryEl.appendChild(bubble);
            node._currentStreamBubble = bubble;
            autoScrollIfNeeded(node._popupHistoryEl);
        }
    });

    // ── Input row ──
    const inputRow = document.createElement("div");
    inputRow.className = "llm-popup-input-row";

    // ── Image attachment button ──
    const attachBtn = document.createElement("button");
    attachBtn.className = "llm-popup-attach-btn";
    attachBtn.textContent = "📎";
    attachBtn.title = "Attach an image";
    attachBtn.onclick = () => {
        // Disable during active generation
        if (node._popupStreaming) {
            showToast("Cannot attach image during generation", "warning", 2000);
            return;
        }
        openImagePicker(node);
    };
    inputRow.appendChild(attachBtn);
    node._popupAttachBtn = attachBtn;

    const textInput = document.createElement("textarea");
    textInput.className = "llm-popup-input";
    textInput.placeholder = "Type a message... (Enter to send, Esc to close)";
    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handlePopupSend(node);
        }
    });
    // Multi-line auto-resize
    textInput.addEventListener("input", () => {
        textInput.style.height = "auto";
        textInput.style.height = Math.min(textInput.scrollHeight, 120) + "px";
    });
    // Pre-fill popup input from canvas text widget
    const canvasTextWidget = node.widgets?.find(w => w.name === "text");
    if (canvasTextWidget?.value) {
        textInput.value = canvasTextWidget.value;
    }
    node._popupInputEl = textInput;
    inputRow.appendChild(textInput);

    const sendBtn = document.createElement("button");
    sendBtn.className = "llm-popup-send-btn";
    sendBtn.textContent = "▶ Send";
    sendBtn.onclick = () => handlePopupSend(node);
    node._popupSendBtn = sendBtn;
    inputRow.appendChild(sendBtn);
    body.appendChild(inputRow);

    // ── Settings section (collapsible) ──
    const settings = document.createElement("details");
    settings.className = "llm-popup-settings";

    // Collapsible toggle header
    const settingsSummary = document.createElement("summary");
    settingsSummary.className = "llm-popup-settings-header";
    settingsSummary.textContent = "⚙️ Settings";
    settings.appendChild(settingsSummary);

    // Restore collapsed state across popup opens
    if (node._popupSettings?.settingsOpen) {
        settings.setAttribute("open", "");
    }

    /**
     * Create a <select> with preset options, ensuring the current value
     * is always selectable even if not in the preset list.
     */
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

    // Load current settings — read from canvas widgets FIRST (authoritative source),
    // fall back to _popupSettings (synced via callbacks).
    const popupSettings = node._popupSettings || {};
    const tw = node.widgets?.find(w => w.name === "prompt_template");
    const mlw = node.widgets?.find(w => w.name === "max_length");
    const tpw = node.widgets?.find(w => w.name === "temperature");
    const spw = node.widgets?.find(w => w.name === "system_prompt_text");

    const currentTemplate = tw?.value || popupSettings.prompt_template || "Custom";
    const currentCustomPrompt = spw?.value || popupSettings.system_prompt_text || "";
    const currentMaxLength = mlw?.value || popupSettings.max_length || 768;
    const currentTemperature = node.isEasyLLMGGUF
        ? (tpw?.value ?? popupSettings.temperature ?? 0.7)
        : (tpw?.value || popupSettings.temperature || "auto");

    // ── Single-column sectioned settings layout ──
    const settingsSections = document.createElement("div");
    settingsSections.className = "llm-popup-settings-sections";

    // ════════════════════════════════════════════════════════════════
    // Section: Persona — Configure the AI's behavior
    // ════════════════════════════════════════════════════════════════
    const personaSection = document.createElement("div");
    personaSection.className = "llm-popup-settings-section";

    const personaHeader = document.createElement("div");
    personaHeader.className = "llm-popup-settings-section-header";
    personaHeader.textContent = "Persona";
    personaSection.appendChild(personaHeader);

    // System Prompt dropdown row
    const sysPromptRow = document.createElement("div");
    sysPromptRow.className = "llm-popup-settings-row";

    const sysLabel = document.createElement("span");
    sysLabel.className = "llm-popup-settings-label";
    sysLabel.textContent = "System Prompt:";
    sysPromptRow.appendChild(sysLabel);

    // Fetch prompt names for dropdown
    const templateWidget = node.widgets?.find(w => w.name === "prompt_template");
    const templateNames = templateWidget?.options?.values || ["Custom"];

    const templateSelect = document.createElement("select");
    templateSelect.className = "llm-popup-settings-select";
    for (const name of templateNames) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        if (name === currentTemplate) opt.selected = true;
        templateSelect.appendChild(opt);
    }
    sysPromptRow.appendChild(templateSelect);
    personaSection.appendChild(sysPromptRow);

    // Custom prompt textarea (dimmed when template is selected)
    const customRow = document.createElement("div");
    customRow.className = "llm-popup-settings-row";
    customRow.style.flexDirection = "column";
    customRow.style.alignItems = "stretch";

    const customTextarea = document.createElement("textarea");
    customTextarea.className = "llm-popup-custom-prompt";
    customTextarea.placeholder = "Custom system prompt (empty = no system prompt sent to the model)";
    customTextarea.value = currentCustomPrompt;
    customTextarea.style.minHeight = "50px";
    customTextarea.disabled = currentTemplate !== "Custom";
    customRow.appendChild(customTextarea);
    personaSection.appendChild(customRow);

    // Dim custom textarea when template selected
    templateSelect.addEventListener("change", () => {
        customTextarea.disabled = templateSelect.value !== "Custom";
    });

    settingsSections.appendChild(personaSection);

    // ════════════════════════════════════════════════════════════════
    // Section: Tuning — Generation parameters
    // ════════════════════════════════════════════════════════════════
    const tuningSection = document.createElement("div");
    tuningSection.className = "llm-popup-settings-section";

    const tuningHeader = document.createElement("div");
    tuningHeader.className = "llm-popup-settings-section-header";
    tuningHeader.textContent = "Tuning";
    tuningSection.appendChild(tuningHeader);

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
    tuningSection.appendChild(tempRow);

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
    tuningSection.appendChild(maxRow);

    // Seed
    const seedWidget = node.widgets?.find(w => w.name === "seed");
    const currentSeed = seedWidget?.value ?? 0;

    const seedRow = document.createElement("div");
    seedRow.className = "llm-popup-settings-row";

    const seedLabel = document.createElement("span");
    seedLabel.className = "llm-popup-settings-label";
    seedLabel.textContent = "Seed:";
    seedRow.appendChild(seedLabel);

    const seedInput = document.createElement("input");
    seedInput.type = "number";
    seedInput.className = "llm-popup-settings-input";
    seedInput.value = currentSeed;
    seedInput.min = 0;
    seedInput.max = 0xFFFFFFFF;
    seedInput.title = "0 = auto-randomize";
    seedRow.appendChild(seedInput);
    tuningSection.appendChild(seedRow);

    settingsSections.appendChild(tuningSection);

    // ════════════════════════════════════════════════════════════════
    // Section: Hardware — Advanced system settings
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
    // NOTE: model_path and mmproj_path are managed via the model browser popup.
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

        // ── Friendly labels for chat template options ────────────────
        const CT_LABELS = {
            "auto": "Auto-detect (Recommended)",
            "qwen": "Qwen / ChatML",
            "llama": "Llama 3 / 3.1",
            "mistral": "Mistral",
            "phi": "Phi-3 / Phi-4",
            "deepseek": "DeepSeek",
            "gemma": "Gemma",
        };
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
    node._popupTemplateSelect = templateSelect;
    node._popupCustomTextarea = customTextarea;
    node._popupMaxSelect = maxSelect;
    node._popupTempSelect = tempControl;
    node._popupSeedInput = seedInput;

    settings.appendChild(settingsSections);
    body.appendChild(settings);

    // ── Footer (Manage Prompts + Models + Export + Save & Close) ──
    const manageBtn = document.createElement("button");
    manageBtn.className = "llm-popup-manage-btn";
    manageBtn.textContent = "⚙ Manage Prompts...";
    manageBtn.onclick = () => openPromptManagerDialog();
    footer.appendChild(manageBtn);

    // 📁 Models button (GGUF-only — opens model browser popup)
    if (node.isEasyLLMGGUF) {
        const modelBtn = document.createElement("button");
        modelBtn.className = "llm-popup-header-btn";
        modelBtn.style.marginLeft = "auto";
        modelBtn.textContent = "📁 Models";
        modelBtn.title = "Browse and select GGUF model";
        modelBtn.onclick = () => openModelBrowserPopup(node);
        footer.appendChild(modelBtn);
    }

    // 📥 Export button with dropdown
    const exportBtn = document.createElement("button");
    exportBtn.className = "llm-popup-header-btn llm-popup-export-btn llm-popup-export-btn--footer";
    if (!node.isEasyLLMGGUF) {
        exportBtn.style.marginLeft = "auto";
    }
    exportBtn.textContent = "📥 Export";
    exportBtn.title = "Export conversation";
    exportBtn.onclick = (e) => {
        e.stopPropagation();
        const existing = exportBtn.querySelector(".llm-popup-export-dropdown");
        if (existing) {
            existing.remove();
            return;
        }
        const dropdown = document.createElement("div");
        dropdown.className = "llm-popup-export-dropdown";
        dropdown.onclick = (ev) => ev.stopPropagation();

        const mdOption = document.createElement("button");
        mdOption.textContent = "Markdown (.md)";
        mdOption.onclick = () => {
            const result = exportChat(node._chatHistory, "md", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Markdown", "success", 2000);
            } else {
                showToast("No messages to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(mdOption);

        const txtOption = document.createElement("button");
        txtOption.textContent = "Plain Text (.txt)";
        txtOption.onclick = () => {
            const result = exportChat(node._chatHistory, "txt", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Plain Text", "success", 2000);
            } else {
                showToast("No messages to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(txtOption);

        exportBtn.appendChild(dropdown);
        setTimeout(() => {
            document.addEventListener("click", () => { if (dropdown.parentNode) dropdown.remove(); }, { once: true });
        }, 0);
    };
    footer.appendChild(exportBtn);

    const saveBtn = document.createElement("button");
    saveBtn.className = "llm-popup-save-btn";
    saveBtn.textContent = "💾 Save & Close";
    saveBtn.onclick = () => {
        // Sync settings to canvas FIRST (replaces _popupSettings — capture dims after).
        syncPopupSettingsToCanvas(node);

        // Capture popup panel dimensions after sync. Uses offsetWidth/offsetHeight
        // because CSS resize:both may not reflect in element.style.
        if (node._popupPanel) {
            const w = node._popupPanel.offsetWidth;
            const h = node._popupPanel.offsetHeight;
            node._popupSettings.popupWidth = w + "px";
            node._popupSettings.popupHeight = h + "px";
            console.debug(`[LLM Chat] Saved popup dimensions: ${w}x${h}`);
        } else {
            console.debug(`[LLM Chat] saveBtn.onclick: node._popupPanel is null`);
        }
        // Persist settings collapsed state
        node._popupSettings.settingsOpen = settings.hasAttribute("open");
        closeChatPopup(node);
    };
    footer.appendChild(saveBtn);

    // ── Global keyboard shortcuts ──
    const globalKeyHandler = (e) => {
        // Escape to close popup
        if (e.key === "Escape") {
            closeChatPopup(node);
        }
        // Ctrl+L or Cmd+K to clear chat
        if ((e.ctrlKey || e.metaKey) && (e.key === "l" || e.key === "k")) {
            if (node._chatHistory && node._chatHistory.length > 0) {
                e.preventDefault();
                if (confirm("Clear the entire conversation?")) {
                    node._chatHistory = [];
                    // Also clear any uploaded image (conversation context is gone)
                    if (node._uploadedImage) {
                        removeAttachedImage(node);
                    }
                    renderPopupHistory(node);
                    showToast("Chat cleared", "success", 2000);
                }
            }
        }
    };
    document.addEventListener("keydown", globalKeyHandler);
    node._popupKeyHandler = globalKeyHandler;

    // Store references and append
    node._popupOverlay = overlay;
    setActivePopupNode(node);
    document.body.appendChild(overlay);

    // Focus the input
    setTimeout(() => textInput.focus(), 100);
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Close the chat popup
// ────────────────────────────────────────────────────────────────────────

export async function closeChatPopup(node) {
    // Remove global keyboard handler
    if (node._popupKeyHandler) {
        document.removeEventListener("keydown", node._popupKeyHandler);
        node._popupKeyHandler = null;
    }

    // Streaming: do NOT abort — tokens accumulate in _streamingAccumulatedText
    // for display when popup reopens.
    try {
        await fetch(`/easyllm/popup_inactive/${node.id}`, { method: "POST" });
    } catch (_e) {
        // Non-critical
    }
    // Reset streaming bubble reference (DOM element is destroyed below)
    node._currentStreamBubble = null;

    // Stop progress tracking (popup bar + canvas green bar)
    stopProgressTracking();
    stopCanvasProgressTracking(node);

    // Close model browser popup if open
    closeModelBrowserPopup(node);

    if (node._popupOverlay && node._popupOverlay.parentElement) {
        node._popupOverlay.parentElement.removeChild(node._popupOverlay);
    }
    node._popupOverlay = null;
    node._popupPanel = null;
    node._popupHistoryEl = null;
    node._popupInputEl = null;
    node._popupProgressBar = null;
    node._popupProgressFill = null;
    node._popupContextEl = null;
    node._popupEnhancerContextEl = null;
    node._popupStopBtn = null;
    node._popupSendBtn = null;
    node._popupAttachBtn = null;
    node._popupImagePreview = null;
    if (getActivePopupNode() === node) {
        clearActivePopupNode();
    }
}

// ────────────────────────────────────────────────────────────────────────
// Enhancer Mode: Context indicator for output history popup
// ────────────────────────────────────────────────────────────────────────

/**
 * Update the context token indicator in the enhancer history popup.
 * Normalizes enhancer entries (input → user, output → assistant) and
 * estimates context using the existing estimateContextTokens() pipeline.
 * Uses per-entry systemPromptText snapshots where available.
 */
function updateEnhancerContextIndicator(node) {
    const el = node._popupEnhancerContextEl;
    if (!el) return;

    const history = node._enhancerHistory || [];
    if (history.length === 0) {
        el.textContent = "";
        el.style.display = "none";
        el.title = "";
        return;
    }

    // Normalize enhancer entries to {message, role} pairs for estimateContextTokens
    const normalizedHistory = [];
    let systemPrompt = "";
    for (const entry of history) {
        normalizedHistory.push({ message: entry.input, role: "user" });
        normalizedHistory.push({ message: entry.output, role: "assistant" });
        // Use systemPromptText from the most recent entry that has one
        if (entry.systemPromptText) {
            systemPrompt = entry.systemPromptText;
        }
    }

    // Determine chat template (same logic as getChatTemplate in popup_bubble.js).
    // "auto" is resolved server-side; use "llama" as default for token estimation.
    let chatTemplate = "llama";
    if (node.isEasyLLMGGUF) {
        const ctW = node.widgets?.find(w => w.name === "chat_template");
        if (ctW?.value && ctW.value !== "auto") chatTemplate = ctW.value;
    }

    const result = estimateContextTokens(normalizedHistory, {
        systemPrompt,
        chatTemplate,
        countImages: false,  // Enhancer entries don't carry images
    });
    const { total: used, system: sysTokens, history: historyTokens } = result;

    // Get max context: GGUF reads from n_ctx widget, CLIP uses _maxContextTokens or default 2048
    let maxCtx = node._maxContextTokens;
    if (!maxCtx && node.isEasyLLMGGUF) {
        const nCtxWidget = node.widgets?.find(w => w.name === "n_ctx");
        maxCtx = nCtxWidget ? parseInt(nCtxWidget.value) : 4096;
    }
    if (!maxCtx) maxCtx = 2048;

    el.textContent = `Context: ~${used} / ${maxCtx}`;
    el.style.display = "block";

    // Breakdown tooltip
    const tooltipParts = [];
    if (sysTokens > 0) tooltipParts.push(`System: ~${sysTokens}`);
    tooltipParts.push(`History: ~${historyTokens}`);
    el.title = `Context breakdown — ${tooltipParts.join(" | ")}`;

    // Color coding: warn at 70%, danger at 90%
    const ratio = used / maxCtx;
    if (ratio > 0.9) {
        el.style.color = "#c0392b"; // Red — critical
    } else if (ratio > 0.7) {
        el.style.color = "#d8a050"; // Amber — warning
    } else {
        el.style.color = "#94a3b8"; // Gray — normal
    }
}

// ────────────────────────────────────────────────────────────────────────
// Enhancer Mode: Open Output History Popup (read-only history log)
// ────────────────────────────────────────────────────────────────────────

/**
 * Open a read-only output history popup for enhancer mode.
 * Shows input/output pairs with scroll, progress bar, export, and stop controls.
 */
export function openOutputHistoryPopup(node) {
    if (node._popupOverlay) return; // Already open
    if (getActivePopupNode() && getActivePopupNode() !== node) {
        closeChatPopup(getActivePopupNode());
    }

    const nodeLabel = node.title || `Node #${node.id}`;
    const { overlay, panel, body, footer, header } = createOverlayModal(
        "llm-popup",
        `${nodeLabel} | 📋 Output History`,
        () => closeChatPopup(node),
        { hasFooter: true }
    );

    // ── Header action buttons (stop) ──
    const headerActions = document.createElement("div");
    headerActions.className = "llm-popup-header-actions";

    // Stop button (hidden by default, shown during generation)
    const stopBtn = document.createElement("button");
    stopBtn.className = "llm-popup-header-btn llm-popup-stop-btn";
    stopBtn.textContent = "■ Stop";
    stopBtn.title = "Stop generation";
    stopBtn.style.display = "none";
    stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping...";
        await abortStreaming(node);
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.textContent = "■ Stop";
        showToast("Generation stopped", "info", 2000);
    };
    node._popupStopBtn = stopBtn;
    headerActions.appendChild(stopBtn);

    // Insert header actions before the close button
    const closeBtn = header.querySelector(".llm-popup-close-btn");
    if (closeBtn) {
        header.insertBefore(headerActions, closeBtn);
    }

    // ── Progress bar (thin strip below header, hidden by default) ──
    const progressBar = document.createElement("div");
    progressBar.className = "llm-popup-progress";
    progressBar.style.display = "none";
    const progressFill = document.createElement("div");
    progressFill.className = "llm-popup-progress-fill";
    progressBar.appendChild(progressFill);
    panel.insertBefore(progressBar, body);
    node._popupProgressBar = progressBar;
    node._popupProgressFill = progressFill;
    startProgressTracking(node);

    // Store panel reference for resize / cleanup
    node._popupPanel = panel;

    // ── Scrollable history container ──
    const historyContainer = document.createElement("div");
    historyContainer.className = "llm-popup-history";
    node._popupHistoryEl = historyContainer;
    body.appendChild(historyContainer);

    const history = node._enhancerHistory || [];
    if (history.length === 0) {
        const empty = document.createElement("div");
        empty.className = "llm-popup-history-empty";
        empty.textContent = "🔍 Queue Prompt to generate enhanced prompts. Results appear here.";
        historyContainer.appendChild(empty);
    } else {
        // Scroll-to-bottom button
        createScrollToBottomBtn(historyContainer);

        // Scroll event handler (tracks user scroll-up state)
        historyContainer.addEventListener("scroll", () => {
            updateScrollState(historyContainer);
        });

        // Show oldest entries first (chronological order), with newest at bottom
        for (const entry of history) {
            // Build extra elements for the input bubble:
            // collapsible system prompt snapshot (if available)
            const extraElements = [];
            if (entry.systemPromptText) {
                const details = document.createElement("details");
                details.className = "llm-chat-system-prompt-details";
                const summary = document.createElement("summary");
                summary.className = "llm-chat-system-prompt-summary";
                summary.textContent = "📜 System prompt";
                details.appendChild(summary);
                const content = document.createElement("div");
                content.className = "llm-chat-system-prompt-content";
                content.textContent = entry.systemPromptText;
                details.appendChild(content);
                extraElements.push(details);
            }

            // Input bubble with copy button + paste-to-input
            const inputBubble = createBubbleElement("user", `Input: ${entry.input}`, {
                onPasteToInput: (text) => {
                    // Strip "Input: " prefix for paste
                    const cleanText = text.startsWith("Input: ") ? text.slice(7) : text;
                    const textW = node.widgets?.find(w => w.name === "text");
                    if (textW) {
                        textW.value = cleanText;
                        showToast("✏️ Pasted to text widget", "info", 2000);
                    }
                },
                extraElements: extraElements,
            });
            historyContainer.appendChild(inputBubble);

            // Output bubble with copy button + timestamp + model name badge
            const outputBubble = createBubbleElement("assistant", entry.output, {
                timestamp: entry.timestamp,
                modelName: entry.modelName,
            });
            historyContainer.appendChild(outputBubble);
        }

        // Initial scroll to bottom
        requestAnimationFrame(() => {
            scrollToBottom(historyContainer);
        });
    }

    // ── Context indicator (below history, above footer) ──
    const contextEl = document.createElement("div");
    contextEl.className = "llm-chat-context-indicator";
    contextEl.style.display = "none";
    body.appendChild(contextEl);
    node._popupEnhancerContextEl = contextEl;

    // Show context indicator after DOM is ready
    requestAnimationFrame(() => {
        updateEnhancerContextIndicator(node);
    });

    // ── Footer (Clear History + Models + Export) ──
    const clearBtn = document.createElement("button");
    clearBtn.className = "llm-popup-copy-btn";
    clearBtn.textContent = "🗑 Clear History";
    clearBtn.onclick = () => {
        node._enhancerHistory = [];
        if (node._popupOverlay) {
            closeChatPopup(node);
            // Re-open with empty state
            openOutputHistoryPopup(node);
        }
    };
    footer.appendChild(clearBtn);

    // 📁 Models button (GGUF-only — opens model browser popup)
    if (node.isEasyLLMGGUF) {
        const modelBtn = document.createElement("button");
        modelBtn.className = "llm-popup-header-btn";
        modelBtn.style.marginLeft = "auto";
        modelBtn.textContent = "📁 Models";
        modelBtn.title = "Browse and select GGUF model";
        modelBtn.onclick = () => openModelBrowserPopup(node);
        footer.appendChild(modelBtn);
    }

    // 📥 Export button with dropdown
    const exportBtn = document.createElement("button");
    exportBtn.className = "llm-popup-header-btn llm-popup-export-btn llm-popup-export-btn--footer";
    if (!node.isEasyLLMGGUF) {
        exportBtn.style.marginLeft = "auto";
    }
    exportBtn.textContent = "📥 Export";
    exportBtn.title = "Export enhancer history";
    exportBtn.onclick = (e) => {
        e.stopPropagation();
        const existing = exportBtn.querySelector(".llm-popup-export-dropdown");
        if (existing) { existing.remove(); return; }
        const dropdown = document.createElement("div");
        dropdown.className = "llm-popup-export-dropdown";
        dropdown.onclick = (ev) => ev.stopPropagation();

        const mdOption = document.createElement("button");
        mdOption.textContent = "Markdown (.md)";
        mdOption.onclick = () => {
            const result = exportEnhancerHistory(node._enhancerHistory, "md", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Markdown", "success", 2000);
            } else {
                showToast("No history to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(mdOption);

        const txtOption = document.createElement("button");
        txtOption.textContent = "Plain Text (.txt)";
        txtOption.onclick = () => {
            const result = exportEnhancerHistory(node._enhancerHistory, "txt", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Plain Text", "success", 2000);
            } else {
                showToast("No history to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(txtOption);

        exportBtn.appendChild(dropdown);
        setTimeout(() => {
            document.addEventListener("click", () => { if (dropdown.parentNode) dropdown.remove(); }, { once: true });
        }, 0);
    };
    footer.appendChild(exportBtn);

    // Store references and append
    node._popupOverlay = overlay;
    setActivePopupNode(node);
    document.body.appendChild(overlay);
}
