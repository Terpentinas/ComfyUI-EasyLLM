/**
 * EasyLLM — Popup bubble/export/image/model-name module
 *
 * Contains: chat bubble creation, history rendering, message actions,
 * context indicator, settings sync, send handler, typing indicator,
 * image upload, model name lookup, export/abort functions.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { formatTimestamp, renderMarkdown, parseThinkBlocks, createScrollToBottomBtn, updateScrollState, autoScrollIfNeeded, estimateContextTokens, formatTimingBadge, formatTimingTooltip, estimateTokens } from "./popup_utils.js";
import { startStreamListening, stopStreamListening, startCanvasProgressTracking } from "./websocket_bridge.js";
import { pushUserMessage, serializeHistoryForBackend } from "./history_store.js";
import { showToast } from "./ui_utils.js";

// ────────────────────────────────────────────────────────────────────────
// Export chat
// ────────────────────────────────────────────────────────────────────────

/**
 * Export chat history as a Markdown or Plain Text string.
 */
export function exportChat(history, format, nodeLabel) {
    if (!history || !history.length) return null;

    const now = new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC";

    if (format === "md") {
        let md = `# EasyLLM Export — ${nodeLabel || "Chat"}\n`;
        md += `Exported: ${now}\n\n`;
        for (const entry of history) {
            const role = entry.role === "user" ? "You" : "LLM";
            const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
            const header = ts ? `## ${role} (${ts})` : `## ${role}`;
            md += `${header}\n${entry.message || ""}\n\n`;
        }
        return { content: md, filename: `easyllm-chat-export-${Date.now()}.md`, mime: "text/markdown" };
    }

    // Plain Text
    let txt = `EasyLLM Export — ${nodeLabel || "Chat"}\n`;
    txt += `Exported: ${now}\n\n`;
    for (const entry of history) {
        const role = entry.role === "user" ? "You" : "LLM";
        const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
        txt += `--- ${role} ${ts ? `(${ts})` : ""} ---\n${entry.message || ""}\n\n`;
    }
    return { content: txt, filename: `easyllm-chat-export-${Date.now()}.txt`, mime: "text/plain" };
}

/**
 * Export enhancer history as Markdown or Plain Text.
 * Formats {input, output, timestamp} entries as input/output pairs, oldest first.
 *
 * @param {Array}  history   - Array of {input, output, timestamp} objects.
 * @param {string} format    - "md" or "txt".
 * @param {string} nodeLabel - Node label for the export header.
 * @returns {object|null} { content, filename, mime } or null if empty.
 */
export function exportEnhancerHistory(history, format, nodeLabel) {
    if (!history || !history.length) return null;

    const now = new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC";

    if (format === "md") {
        let md = `# EasyLLM Enhancer Export — ${nodeLabel || "Enhancer"}\n`;
        md += `Exported: ${now}\n\n`;
        // Show oldest first (chronological order — matches popup display order)
        for (const entry of history) {
            const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
            const header = ts ? `## Prompt → Output (${ts})` : `## Prompt → Output`;
            md += `${header}\n\n`;
            md += `**Input:**\n${entry.input || ""}\n\n`;
            if (entry.systemPromptText) {
                md += `**System Prompt:**\n${entry.systemPromptText}\n\n`;
            }
            if (entry.modelName) {
                md += `**Model:** ${entry.modelName}\n\n`;
            }
            md += `**Output:**\n${entry.output || ""}\n\n`;
        }
        return { content: md, filename: `easyllm-enhancer-export-${Date.now()}.md`, mime: "text/markdown" };
    }

    // Plain Text
    let txt = `EasyLLM Enhancer Export — ${nodeLabel || "Enhancer"}\n`;
    txt += `Exported: ${now}\n\n`;
    for (const entry of history) {
        const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
        txt += `--- Prompt → Output ${ts ? `(${ts})` : ""} ---\n`;
        txt += `INPUT:\n${entry.input || ""}\n\n`;
        if (entry.systemPromptText) {
            txt += `SYSTEM PROMPT:\n${entry.systemPromptText}\n\n`;
        }
        if (entry.modelName) {
            txt += `MODEL: ${entry.modelName}\n\n`;
        }
        txt += `OUTPUT:\n${entry.output || ""}\n\n`;
    }
    return { content: txt, filename: `easyllm-enhancer-export-${Date.now()}.txt`, mime: "text/plain" };
}

/**
 * Trigger a file download from an export result.
 * Uses Blob + anchor click with a window.open fallback
 * for environments where programmatic clicks may be blocked.
 */
export function downloadExport(exportResult) {
    if (!exportResult) return;
    try {
        const blob = new Blob([exportResult.content], { type: exportResult.mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = exportResult.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        // Revoke after a short delay to ensure download starts
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (_e) {
        // Fallback: open a data URI in a new tab/window
        const encoded = encodeURIComponent(exportResult.content);
        const dataUri = `data:${exportResult.mime};charset=utf-8,${encoded}`;
        window.open(dataUri, "_blank");
    }
}

// ────────────────────────────────────────────────────────────────────────
// Abort streaming
// ────────────────────────────────────────────────────────────────────────

/**
 * Abort an ongoing generation for a node.
 * Sends POST to abort endpoint and triggers cleanup.
 */
export async function abortStreaming(node) {
    if (!node) return;
    try {
        await fetch(`/easyllm/abort_stream/${node.id}`, { method: "POST" });
    } catch (_e) {
        // Non-critical
    }
    // Also send a WebSocket abort event
    try {
        const body = new URLSearchParams();
        body.set("node_id", String(node.id));
        api.fetchApi("/easyllm/abort_stream", { method: "POST", body });
    } catch (_e) {
        // Non-critical
    }
    stopStreamListening(node);
    node._popupStreaming = false;
    node._currentStreamBubble = null;
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Create a chat bubble element
// ────────────────────────────────────────────────────────────────────────

export function createBubbleElement(role, message, options = {}) {
    const isUser = role === "user";
    const bubble = document.createElement("div");
    bubble.className = `llm-chat-bubble llm-chat-bubble-${isUser ? "user" : "assistant"}`;
    if (options.error) {
        bubble.classList.add("llm-chat-bubble-error");
    }

    // Role avatar/icon
    const label = document.createElement("div");
    label.className = `llm-chat-bubble-label llm-chat-bubble-label-${isUser ? "right" : "left"}`;
    label.textContent = isUser ? "🧑 You" : "🤖 LLM";

    // Model name badge (enhancer mode output bubbles)
    if (!isUser && options.modelName) {
        const modelBadge = document.createElement("span");
        modelBadge.className = "llm-chat-model-badge";
        modelBadge.textContent = `🤖 ${options.modelName}`;
        label.appendChild(modelBadge);
    }

    bubble.appendChild(label);

    // ── Image thumbnail (user bubbles only) ──
    // Renders a clickable image preview when an image was attached to the message.
    if (isUser && options.image) {
        const imgContainer = document.createElement("div");
        imgContainer.className = "llm-chat-bubble-image";
        const img = document.createElement("img");
        img.src = options.image;
        img.alt = "Attached image";
        img.loading = "eager";
        // Re-evaluate auto-scroll after image finishes loading.
        // Network-sourced images (/view?filename=...) load asynchronously,
        // causing a layout shift that breaks the scroll position computed
        // by the requestAnimationFrame callback in renderPopupHistory.
        // This handler compensates by re-checking scrollHeight once the
        // image dimensions are known.
        img.onload = () => {
            if (img.isConnected) {
                const container = img.closest(".llm-popup-history");
                if (container) autoScrollIfNeeded(container);
            }
        };
        img.onerror = img.onload; // Same handler for error (harmless no-op)
        // Click to expand full-size in a new tab
        img.addEventListener("click", () => window.open(options.image, "_blank"));
        imgContainer.appendChild(img);
        bubble.appendChild(imgContainer);
    }

    // Parse think blocks for assistant messages
    let displayText = message;
    let thinkBlock = null;
    if (!isUser && message) {
        const parsed = parseThinkBlocks(message);
        thinkBlock = parsed.thinking;
        displayText = parsed.response || message;
    }

    // Message text — render as markdown
    const text = document.createElement("div");
    text.className = "llm-chat-bubble-text";
    if (displayText) {
        text.innerHTML = renderMarkdown(displayText);
    }


    // Collapsible thinking section
    if (thinkBlock) {
        const details = document.createElement("details");
        details.className = "llm-chat-thinking";
        const summary = document.createElement("summary");
        summary.className = "llm-chat-thinking-summary";
        summary.textContent = "🤔 Show reasoning";
        details.appendChild(summary);
        const thinkContent = document.createElement("div");
        thinkContent.className = "llm-chat-thinking-content";
        thinkContent.textContent = thinkBlock;
        details.appendChild(thinkContent);
        bubble.appendChild(details);
    }

    // Error retry button
    if (options.error && options.onRetry) {
        const retryBtn = document.createElement("button");
        retryBtn.className = "llm-chat-retry-btn";
        retryBtn.textContent = "🔄 Retry";
        retryBtn.onclick = options.onRetry;
        bubble.appendChild(retryBtn);
    }

    bubble.appendChild(text);

    // Timestamp
    if (options.timestamp) {
        const tsEl = document.createElement("div");
        tsEl.className = "llm-chat-bubble-timestamp";
        tsEl.textContent = formatTimestamp(options.timestamp);
        bubble.appendChild(tsEl);
    }

    // Action buttons row (copy / edit / delete)
    const actions = document.createElement("div");
    actions.className = "llm-chat-bubble-actions";

    // Copy button — copies display text (without think blocks) if available
    const copyBtn = document.createElement("button");
    copyBtn.className = "llm-chat-action-btn llm-chat-copy-btn";
    copyBtn.textContent = "📋";
    copyBtn.title = "Copy message";
    copyBtn.onclick = async () => {
        try {
            // Use displayText (without think blocks) if parsed, otherwise full message
            const textToCopy = displayText || message;
            await navigator.clipboard.writeText(textToCopy);
            showToast("✅ Copied!", "success", 2000);
        } catch {
            showToast("❌ Copy failed", "error", 2000);
        }
    };
    actions.appendChild(copyBtn);

    // Paste-to-input button (any user message — copies text to input field)
    if (isUser && options.onPasteToInput) {
        const pasteBtn = document.createElement("button");
        pasteBtn.className = "llm-chat-action-btn llm-chat-edit-btn";
        pasteBtn.textContent = "✏️";
        pasteBtn.title = "Paste to input";
        pasteBtn.onclick = () => options.onPasteToInput(message);
        actions.appendChild(pasteBtn);
    }

    // Delete button (any message)
    if (options.onDelete) {
        const delBtn = document.createElement("button");
        delBtn.className = "llm-chat-action-btn llm-chat-delete-btn";
        delBtn.textContent = "🗑";
        delBtn.title = "Delete message";
        delBtn.onclick = () => options.onDelete();
        actions.appendChild(delBtn);
    }

    bubble.appendChild(actions);

    // Timing badge (assistant messages only)
    if (!isUser && options.timing) {
        const timingBadge = document.createElement("div");
        timingBadge.className = "llm-chat-timing-badge";
        timingBadge.textContent = formatTimingBadge(options.timing);
        timingBadge.title = formatTimingTooltip(options.timing);
        bubble.appendChild(timingBadge);
    }

    // Attach extra elements if provided
    if (options.extraElements) {
        for (const el of options.extraElements) {
            bubble.appendChild(el);
        }
    }

    return bubble;
}

/**
 * Convert a raw uploaded filename to a ComfyUI /view URL for rendering,
 * while leaving base64 data URIs and full URLs untouched.
 *
 * @param {string|null} image - entry.image value (raw filename, data URI, URL, or null)
 * @returns {string|null} Resolved image URL or null
 */
function resolveImageUrl(image) {
    if (!image) return null;
    // If it's already a data URI, http URL, or absolute path — use as-is
    if (image.startsWith("data:") || image.startsWith("http://") || image.startsWith("https://") || image.startsWith("/")) {
        return image;
    }
    // Otherwise, treat it as a raw uploaded filename → /view endpoint URL
    // Matches the pattern used by showImagePreview().
    return `/view?filename=${encodeURIComponent(image)}&type=input&subfolder=`;
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Render chat history inside the popup's history container
// ────────────────────────────────────────────────────────────────────────

export function renderPopupHistory(node) {
    const container = node._popupHistoryEl;
    if (!container) return;

    // Reset scroll state before clearing the DOM.
    // innerHTML = '' resets scrollTop to 0, but _isUserScrolledUp
    // survives on the container object. If user had scrolled up,
    // they'd end up at the top after re-render with no auto-scroll.
    container._isUserScrolledUp = false;

    container.innerHTML = "";
    const history = node._chatHistory || [];

    if (history.length === 0) {
        const placeholder = document.createElement("div");
        placeholder.className = "llm-popup-history-empty";
        placeholder.textContent = "💬 Send a message to start chatting...";
        container.appendChild(placeholder);
        createScrollToBottomBtn(container);
        updateContextIndicator(node);
        return;
    }

    // Create scroll-to-bottom button if not exists (idempotent)
    createScrollToBottomBtn(container);

    // Add scroll event listener for smart auto-scroll
    if (!container._scrollHandlerInstalled) {
        container.addEventListener("scroll", () => updateScrollState(container));
        container._scrollHandlerInstalled = true;
    }

    for (let i = 0; i < history.length; i++) {
        const entry = history[i];

        const options = {
            timestamp: entry.timestamp,
            timing: entry.timing || entry._timing,
            // Image attached to user message (multimodal)
            // Convert raw filenames to /view URLs; pass through base64/data URIs as-is
            image: resolveImageUrl(entry.image),
            // Paste-to-input on any user message
            onPasteToInput: entry.role === "user" ? (msg) => pasteToInput(node, msg) : null,
            // Delete any message
            onDelete: () => deleteMessage(node, i),
        };

        // ── Error state: apply styling and wire Retry button ──
        if (entry.error) {
            options.error = true;
            options.onRetry = () => retryMessage(node, i);
        }

        container.appendChild(createBubbleElement(entry.role, entry.message, options));
    }

    // Update context indicator
    updateContextIndicator(node);

    // Scroll to bottom (respecting user scroll state).
    // Use requestAnimationFrame to allow the browser to compute layout
    // for eager-loaded images before calculating scroll position.
    requestAnimationFrame(() => autoScrollIfNeeded(container));
}

// ────────────────────────────────────────────────────────────────────────
// Paste-to-input and deletion
// ────────────────────────────────────────────────────────────────────────

/**
 * Paste a user message into the input field.
 * Copies the message text to the textarea and focuses it.
 */
export function pasteToInput(node, message) {
    const input = node._popupInputEl;
    if (!input) return;
    input.value = message;
    input.focus();
}

/**
 * Delete a message at the given index.
 * Removes only the clicked message — no cascade to following assistant response.
 */
export function deleteMessage(node, index) {
    const history = node._chatHistory;
    if (!history || index < 0 || index >= history.length) return;

    history.splice(index, 1);
    renderPopupHistory(node);
    showToast("🗑 Message deleted", "info", 2000);
}

// ────────────────────────────────────────────────────────────────────────
// Retry a failed generation
// ────────────────────────────────────────────────────────────────────────

/**
 * Retry a failed generation by re-sending the user message
 * that preceded the error entry at the given index.
 * Removes both the error entry and its preceding user message from history,
 * then triggers a new generation via handlePopupSend.
 */
export function retryMessage(node, errorIndex) {
    const history = node._chatHistory;
    if (!history || errorIndex < 1) return;

    // Find the preceding user message
    const userEntry = history[errorIndex - 1];
    if (!userEntry || userEntry.role !== "user") return;

    // Remove the error entry and preceding user entry
    history.splice(errorIndex - 1, 2);
    renderPopupHistory(node);

    // Write the user's original message to the text widget
    const tw = node.widgets?.find(w => w.name === "text");
    if (tw) tw.value = userEntry.message;

    // Also populate the popup input if it exists
    const input = node._popupInputEl;
    if (input) {
        input.value = userEntry.message;
        input.style.height = "36px";
    }

    // Re-send the message
    handlePopupSend(node);
}

// ────────────────────────────────────────────────────────────────────────
// Context window indicator
// ────────────────────────────────────────────────────────────────────────

// Module-level cache for system prompt templates loaded from the backend.
// Maps template name (e.g., "Prompt Engineer") → content string.
// Loaded lazily on first named-template lookup.
let _systemPromptCache = null;

/**
 * Load system prompt templates from the backend API and cache them.
 * Returns a promise that resolves when the cache is populated.
 * Subsequent calls reuse the existing cache (synchronous).
 *
 * @returns {Promise<Object<string, string>>} Map of template name → content
 */
function loadSystemPromptCache() {
    if (_systemPromptCache) return Promise.resolve(_systemPromptCache);

    return (async () => {
        try {
            const resp = await fetch("/easyllm/prompts/load");
            if (!resp.ok) {
                console.warn(`[LLM Chat] Failed to load prompts: HTTP ${resp.status}`);
                _systemPromptCache = {};
                return _systemPromptCache;
            }
            const data = await resp.json();
            const prompts = data.success ? data.prompts : [];
            const map = {};
            for (const p of prompts) {
                if (p.name && p.prompt) {
                    map[p.name] = p.prompt;
                }
            }
            _systemPromptCache = map;
            return _systemPromptCache;
        } catch (e) {
            console.warn("[LLM Chat] Failed to load prompt cache:", e);
            _systemPromptCache = {};
            return _systemPromptCache;
        }
    })();
}

/**
 * Read the effective system prompt from a node's widgets.
 * Returns empty string if no system prompt is configured.
 * Note: cannot read forceInput socket values from the frontend.
 *
 * For named templates (e.g., "Prompt Engineer") without custom text,
 * triggers a deferred cache load. The indicator will update once the
 * cache resolves (see updateContextIndicator's fire-and-forget pattern).
 */
function getSystemPromptText(node) {
    // Priority: 1) Connected socket (not readable from frontend — best effort)
    //           2) prompt_template widget (selected template name)
    //           3) system_prompt_text widget (custom text)
    const templateW = node.widgets?.find(w => w.name === "prompt_template");
    const customW = node.widgets?.find(w => w.name === "system_prompt_text");
    const templateName = templateW?.value;
    const customText = customW?.value || "";

    // If a named template (not "Custom") is selected and no custom text is set,
    // look up the template content from the cached backend prompts.
    if (templateName && templateName !== "Custom") {
        if (customText) return customText;
        // Named template without custom text — check cache synchronously
        if (_systemPromptCache) {
            return _systemPromptCache[templateName] || "";
        }
        // Cache not loaded yet — return empty, trigger deferred load
        // The caller (updateContextIndicator) handles re-counting after load.
        loadSystemPromptCache().then(() => {
            // Re-count now that cache is populated
            updateContextIndicator(node);
        });
        return "";
    }

    return customText || "";
}

/**
 * Read the chat template name from a node's widgets.
 * Falls back to "llama" if no template widget exists or if "auto" is selected.
 * "auto" is resolved server-side during generation; for frontend token
 * estimation we use "llama" as a safe middle-ground default.
 */
function getChatTemplate(node) {
    // GGUF nodes have a chat_template widget for template selection
    if (node.isEasyLLMGGUF) {
        const ctW = node.widgets?.find(w => w.name === "chat_template");
        if (ctW?.value && ctW.value !== "auto") return ctW.value;
    }
    // CLIP path always uses qwen
    return "qwen";
}

/**
 * Update the context token indicator below the history container.
 * Sets both the display text and a breakdown tooltip.
 *
 * @param {Object} node                 - The EasyLLM node
 * @param {string} [pendingMessage]     - Optional in-progress assistant message text
 *        (e.g., streaming text not yet saved to _chatHistory). Pass empty string
 *        to force display even with empty history (during streaming start).
 */
export function updateContextIndicator(node, pendingMessage) {
    const el = node._popupContextEl;
    if (!el) return;

    const history = node._chatHistory || [];

    // If no history and no pending message, hide indicator
    if (history.length === 0 && !pendingMessage) {
        el.textContent = "";
        el.style.display = "none";
        el.title = "";
        return;
    }

    // Gather options from the node's current state
    const systemPrompt = getSystemPromptText(node);
    const chatTemplate = getChatTemplate(node);

    const result = estimateContextTokens(history, {
        systemPrompt,
        chatTemplate,
        countImages: true,
        pendingAssistantMessage: pendingMessage || undefined,
    });
    const { total: used, system: sysTokens, history: historyTokens, images: imageTokens } = result;

    // Get max context: GGUF reads from n_ctx widget, CLIP uses _maxContextTokens or default 2048
    let maxCtx = node._maxContextTokens;
    if (!maxCtx && node.isEasyLLMGGUF) {
        const nCtxWidget = node.widgets?.find(w => w.name === "n_ctx");
        maxCtx = nCtxWidget ? parseInt(nCtxWidget.value) : 4096;
    }
    if (!maxCtx) maxCtx = 2048;
    el.textContent = `Context: ~${used} / ${maxCtx}`;
    el.style.display = "block";

    // Breakdown tooltip (component estimates)
    const tooltipParts = [];
    if (sysTokens > 0) tooltipParts.push(`System: ~${sysTokens}`);
    tooltipParts.push(`History: ~${historyTokens}`);
    if (imageTokens > 0) tooltipParts.push(`Images: ~${imageTokens}`);
    if (pendingMessage) tooltipParts.push(`(streaming included)`);
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
// Popup: Sync popup UI settings to canvas widget values
// ────────────────────────────────────────────────────────────────────────

export function syncPopupSettingsToCanvas(node) {
    const templateSelect = node._popupTemplateSelect;
    const customTextarea = node._popupCustomTextarea;
    const maxSelect = node._popupMaxSelect;
    const tempSelect = node._popupTempSelect;
    const seedInput = node._popupSeedInput;
    const hasPopupRefs = templateSelect && customTextarea && maxSelect && tempSelect && seedInput;

    const settingsObj = {};

    // ── GGUF-specific fields (always run — reads from canvas widgets) ──
    if (node.isEasyLLMGGUF) {
        const modelPathW = node.widgets?.find(w => w.name === "model_path");
        if (modelPathW) settingsObj.model_path = modelPathW.value;

        const nGpuW = node.widgets?.find(w => w.name === "n_gpu_layers");
        if (nGpuW) settingsObj.n_gpu_layers = nGpuW.value;

        const nCtxW = node.widgets?.find(w => w.name === "n_ctx");
        if (nCtxW) settingsObj.n_ctx = nCtxW.value;

        const chatTemplateW = node.widgets?.find(w => w.name === "chat_template");
        if (chatTemplateW) settingsObj.chat_template = chatTemplateW.value;

        const mlockW = node.widgets?.find(w => w.name === "use_mlock");
        if (mlockW) settingsObj.use_mlock = mlockW.value;

        const topKW = node.widgets?.find(w => w.name === "top_k");
        if (topKW) settingsObj.top_k = topKW.value;

        const topPW = node.widgets?.find(w => w.name === "top_p");
        if (topPW) settingsObj.top_p = topPW.value;
    }

    // ── Shared fields: conditional on popup being open ──
    if (hasPopupRefs) {
        settingsObj.prompt_template = templateSelect.value;
        settingsObj.system_prompt_text = customTextarea.value;
        settingsObj.max_length = parseInt(maxSelect.value, 10) || 768;
        settingsObj.temperature = node.isEasyLLMGGUF
            ? (parseFloat(tempSelect.value) ?? 0.7)
            : tempSelect.value;
        settingsObj.seed = parseInt(seedInput.value, 10) || 0;

        // vram_mode (shared between CLIP and GGUF)
        const vramSelect = node._popupVramSelect;
        if (vramSelect) {
            settingsObj.vram_mode = vramSelect.value;
        }

        console.debug(
            `[LLM Chat] syncPopupSettingsToCanvas: template="${settingsObj.prompt_template}", ` +
            `custom="${(settingsObj.system_prompt_text || "").substring(0, 30)}...", ` +
            `max=${settingsObj.max_length}, temp=${settingsObj.temperature}, seed=${settingsObj.seed}, ` +
            `vram=${settingsObj.vram_mode}`
        );

        // Update visible widgets so Python receives current values
        const ptW = node.widgets?.find(w => w.name === "prompt_template");
        if (ptW) {
            ptW.value = settingsObj.prompt_template;
            console.debug(`[LLM Chat] sync: prompt_template widget -> "${ptW.value}"`);
        }

        const spW = node.widgets?.find(w => w.name === "system_prompt_text");
        if (spW) {
            spW.value = settingsObj.system_prompt_text;
        }

        const mlW = node.widgets?.find(w => w.name === "max_length");
        if (mlW) {
            mlW.value = settingsObj.max_length;
        }

        const tmW = node.widgets?.find(w => w.name === "temperature");
        if (tmW) {
            tmW.value = settingsObj.temperature;
        }

        // Sync seed
        const sdW = node.widgets?.find(w => w.name === "seed");
        if (sdW) {
            sdW.value = settingsObj.seed;
        }

        // Sync vram_mode (shared between CLIP and GGUF)
        const vmW = node.widgets?.find(w => w.name === "vram_mode");
        if (vmW) {
            vmW.value = settingsObj.vram_mode;
        }
    } else {
        // ── Fallback: read shared fields from canvas widgets to preserve them ──
        const ptW = node.widgets?.find(w => w.name === "prompt_template");
        if (ptW) settingsObj.prompt_template = ptW.value;

        const spW = node.widgets?.find(w => w.name === "system_prompt_text");
        if (spW) settingsObj.system_prompt_text = spW.value;

        const mlW = node.widgets?.find(w => w.name === "max_length");
        if (mlW) settingsObj.max_length = mlW.value;

        const tmW = node.widgets?.find(w => w.name === "temperature");
        if (tmW) settingsObj.temperature = tmW.value;

        const sdW = node.widgets?.find(w => w.name === "seed");
        if (sdW) settingsObj.seed = sdW.value;

        const vmW = node.widgets?.find(w => w.name === "vram_mode");
        if (vmW) settingsObj.vram_mode = vmW.value;

        console.debug(
            `[LLM Chat] syncPopupSettingsToCanvas: popup not open, ` +
            `template="${settingsObj.prompt_template}", ` +
            `custom="${(settingsObj.system_prompt_text || "").substring(0, 30)}...", ` +
            `max=${settingsObj.max_length}, temp=${settingsObj.temperature}, seed=${settingsObj.seed}, ` +
            `vram=${settingsObj.vram_mode}`
        );
    }

    // ── Sync GGUF-specific fields to canvas widgets (always runs) ──
    if (node.isEasyLLMGGUF) {
        // model_path
        const mpInput = node._popupModelPathInput;
        if (mpInput) {
            const mpW = node.widgets?.find(w => w.name === "model_path");
            if (mpW) mpW.value = mpInput.value;
        }

        // mmproj_path
        const mmInput = node._popupMmprojInput;
        if (mmInput) {
            const mmW = node.widgets?.find(w => w.name === "mmproj_path");
            if (mmW) mmW.value = mmInput.value;
        }

        // n_gpu_layers
        const ngInput = node._popupNGpuInput;
        if (ngInput) {
            const ngW = node.widgets?.find(w => w.name === "n_gpu_layers");
            if (ngW) ngW.value = parseInt(ngInput.value, 10) || -1;
        }

        // n_ctx
        const ncInput = node._popupNCtxInput;
        if (ncInput) {
            const ncW = node.widgets?.find(w => w.name === "n_ctx");
            if (ncW) ncW.value = parseInt(ncInput.value, 10) || 4096;
        }

        // chat_template
        const ctSel = node._popupCTSelect;
        if (ctSel) {
            const ctW = node.widgets?.find(w => w.name === "chat_template");
            if (ctW) ctW.value = ctSel.value;
        }

        // top_k
        const tkInput = node._popupTopKInput;
        if (tkInput) {
            const tkW = node.widgets?.find(w => w.name === "top_k");
            if (tkW) tkW.value = parseInt(tkInput.value, 10) || 50;
        }

        // top_p
        const tpInput = node._popupTopPInput;
        if (tpInput) {
            const tpW = node.widgets?.find(w => w.name === "top_p");
            if (tpW) tpW.value = parseFloat(tpInput.value) || 0.9;
        }

        // use_mlock
        const mlCheck = node._popupMlockCheckbox;
        if (mlCheck) {
            const mlW = node.widgets?.find(w => w.name === "use_mlock");
            if (mlW) mlW.value = mlCheck.checked;
        }

        // repetition_penalty
        const repInput = node._popupRepPenaltyInput;
        if (repInput) {
            const repW = node.widgets?.find(w => w.name === "repetition_penalty");
            if (repW) repW.value = parseFloat(repInput.value) || 1.0;
        }

        // image_filename (no-wire image upload sync)
        const ifw = node.widgets?.find(w => w.name === "image_filename");
        if (ifw && node._uploadedImage) {
            ifw.value = node._uploadedImage;
        }
    }

    // ── Re-read GGUF fields from canvas widgets after sync ──
    // Canvas widgets are the authoritative source; ensure _popupSettings
    // captures the post-sync values.
    if (node.isEasyLLMGGUF) {
        const mpW = node.widgets?.find(w => w.name === "model_path");
        if (mpW) settingsObj.model_path = mpW.value;

        // mmproj_path
        const mmW = node.widgets?.find(w => w.name === "mmproj_path");
        if (mmW) settingsObj.mmproj_path = mmW.value;

        const nGpuW = node.widgets?.find(w => w.name === "n_gpu_layers");
        if (nGpuW) settingsObj.n_gpu_layers = nGpuW.value;

        const nCtxW = node.widgets?.find(w => w.name === "n_ctx");
        if (nCtxW) settingsObj.n_ctx = nCtxW.value;

        const ctW = node.widgets?.find(w => w.name === "chat_template");
        if (ctW) settingsObj.chat_template = ctW.value;

        const tkW = node.widgets?.find(w => w.name === "top_k");
        if (tkW) settingsObj.top_k = tkW.value;

        const tpW = node.widgets?.find(w => w.name === "top_p");
        if (tpW) settingsObj.top_p = tpW.value;

        const mlockW = node.widgets?.find(w => w.name === "use_mlock");
        if (mlockW) settingsObj.use_mlock = mlockW.value;

        // repetition_penalty
        const repW = node.widgets?.find(w => w.name === "repetition_penalty");
        if (repW) settingsObj.repetition_penalty = repW.value;
    }

    // Re-read vram_mode from canvas widget (shared, applies to both CLIP and GGUF)
    const vmW2 = node.widgets?.find(w => w.name === "vram_mode");
    if (vmW2) settingsObj.vram_mode = vmW2.value;

    node._popupSettings = settingsObj;
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Handle Send button click
// ────────────────────────────────────────────────────────────────────────

export async function handlePopupSend(node) {
    // Sync popup settings to canvas widgets before execution
    syncPopupSettingsToCanvas(node);

    const input = node._popupInputEl;
    if (!input) return;

    const message = input.value.trim();
    if (!message) return;

    // Append user message to local history (with dedup guard).
    // Pass uploaded image if attached, so it renders in the history bubble.
    pushUserMessage(node, message, node._uploadedImage || undefined);

    // Store chat history server-side before queueing.
    // Python retrieves it via unique_id lookup during execution.
    const historyJson = serializeHistoryForBackend(node);
    if (historyJson) {
        try {
            const history = JSON.parse(historyJson);
            fetch(`/easyllm/store_history/${node.id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ history }),
            }).catch(() => {});
        } catch (_e) {
            // Non-critical: first turn has empty history
        }
    }

    // Write user message to hidden text widget (Python receives this)
    const tw = node.widgets?.find(w => w.name === "text");
    if (tw) tw.value = message;

    // Write/clear uploaded image filename in hidden widget (Python receives this)
    // Always write — even when no image — to prevent stale filenames from persisting
    // across sends.
    const ifw = node.widgets?.find(w => w.name === "image_filename");
    if (ifw) ifw.value = node._uploadedImage || "";

    // Re-render popup history (shows user message, "..." pending)
    renderPopupHistory(node);

    // Clear input and reset height
    input.value = "";
    input.style.height = "36px";

    // Show typing indicator
    showTypingIndicator(node);

    // Streaming setup
    // 1. Signal popup as active to Python (so it selects streaming path)
    try {
        await fetch(`/easyllm/popup_active/${node.id}`, { method: "POST" });
    } catch (_e) {
        // Non-critical: streaming will fall back to blocking if this fails
    }
    // 2. Start listening for streaming token events
    startStreamListening(node);
    // 3. Track progress on the canvas green bar
    startCanvasProgressTracking(node);
    // 4. Mark this node as streaming (so onExecuted avoids duplicate bubble)
    node._popupStreaming = true;
    // 5. Reset streaming-saved flag for new generation
    node._streamingSavedHistory = false;
    // 6. Show Stop button
    const stopBtn = node._popupStopBtn;
    if (stopBtn) stopBtn.style.display = "inline-flex";

    // Trigger partial execution (EasyLLM node only, not downstream)
    app.queuePrompt(0, 1, [String(node.id)]);

    // ── Clear uploaded image after sending ──
    // The image data is already captured in the history entry, written to
    // the hidden widget, and stored server-side. Clearing here prevents
    // stale images from being re-sent with subsequent messages.
    if (node._uploadedImage) {
        await removeAttachedImage(node);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Typing indicator
// ────────────────────────────────────────────────────────────────────────

/**
 * Show a typing indicator in the history area while waiting for first token.
 */
export function showTypingIndicator(node) {
    const container = node._popupHistoryEl;
    if (!container) return;
    // Remove any existing indicator
    hideTypingIndicator(node);
    const el = document.createElement("div");
    el.className = "llm-chat-typing-indicator";
    el.textContent = "⏳ Processing";
    // Animated dots
    const dots = document.createElement("span");
    dots.className = "llm-chat-typing-dots";
    dots.textContent = "...";
    el.appendChild(dots);
    container.appendChild(el);
    node._typingIndicatorEl = el;
    autoScrollIfNeeded(container);
}

/**
 * Hide the typing indicator.
 */
export function hideTypingIndicator(node) {
    if (node._typingIndicatorEl) {
        node._typingIndicatorEl.remove();
        node._typingIndicatorEl = null;
    }
}


// ────────────────────────────────────────────────────────────────────────
// Image Upload (No-Wire Chat Mode)
// ────────────────────────────────────────────────────────────────────────

export function openImagePicker(node) {
    /** Open a file picker to upload an image for no-wire chat mode.
     *
     * 1. Creates a hidden <input type="file" accept="image/*">
     * 2. On file selection, uploads to ComfyUI's /upload/image endpoint
     * 3. Stores the filename server-side via POST /easyllm/upload_image/{node_id}
     * 4. Updates frontend state and shows thumbnail preview
     */
    // Guard: prevent upload during active generation
    if (node._popupStreaming) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;

        // Step 1: Upload to ComfyUI's native endpoint
        const formData = new FormData();
        formData.append("image", file);
        try {
            const uploadResponse = await fetch("/upload/image", {
                method: "POST",
                body: formData,
            });
            if (!uploadResponse.ok) {
                throw new Error(`Upload failed: ${uploadResponse.status}`);
            }
            const uploadResult = await uploadResponse.json();
            // uploadResult = { name: "photo_123.png", subfolder: "", type: "input" }
            const filename = uploadResult.name;

            // Step 2: Store filename server-side
            await fetch(`/easyllm/upload_image/${node.id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename }),
            });

            // Step 3: Store filename frontend-side
            node._uploadedImage = filename;

            // Step 4: Show thumbnail preview
            showImagePreview(node, filename);

            // Step 5: Update attach button state
            const attachBtn = node._popupAttachBtn;
            if (attachBtn) {
                attachBtn.textContent = "📷";
                attachBtn.title = "Image attached (click to remove)";
                attachBtn.onclick = () => removeAttachedImage(node);
            }

            showToast("✅ Image attached", "success", 2000);
        } catch (err) {
            showToast(`❌ Upload failed: ${err.message}`, "error", 4000);
        }
    };
    input.click();
}

export function showImagePreview(node, filename) {
    /** Show a thumbnail preview of the uploaded image in the popup input area. */
    // Remove existing preview if any
    const existing = node._popupImagePreview;
    if (existing) existing.remove();

    const container = document.createElement("div");
    container.className = "llm-popup-image-preview";

    const img = document.createElement("img");
    // Display from ComfyUI's /view endpoint
    img.src = `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`;
    img.alt = "Uploaded image preview";

    const removeBtn = document.createElement("button");
    removeBtn.className = "llm-popup-image-remove-btn";
    removeBtn.textContent = "✕";
    removeBtn.title = "Remove image";
    removeBtn.onclick = () => removeAttachedImage(node);

    container.appendChild(img);
    container.appendChild(removeBtn);

    // Insert preview above the input row
    const inputRow = node._popupInputEl?.closest(".llm-popup-input-row");
    if (inputRow) {
        inputRow.parentNode.insertBefore(container, inputRow);
    }
    node._popupImagePreview = container;
}

export async function removeAttachedImage(node) {
    /** Remove the attached image: clear server, frontend state, UI, and widget. */
    // ── Guard: nothing to do if no image attached ──
    if (!node._uploadedImage) return;

    // Clear server-side
    try {
        await fetch(`/easyllm/clear_uploaded_image/${node.id}`, { method: "POST" });
    } catch (_e) { /* non-critical */ }

    // Clear frontend state
    node._uploadedImage = null;

    // Remove preview UI
    const preview = node._popupImagePreview;
    if (preview) {
        preview.remove();
        node._popupImagePreview = null;
    }

    // Reset attach button
    const attachBtn = node._popupAttachBtn;
    if (attachBtn) {
        attachBtn.textContent = "📎";
        attachBtn.title = "Attach an image";
        attachBtn.onclick = () => openImagePicker(node);
    }

    // Clear the image_filename widget value to prevent Python from picking up
    // stale filenames on the next Queue Prompt execution.
    const ifw = node.widgets?.find(w => w.name === "image_filename");
    if (ifw) ifw.value = "";

    showToast("Image removed", "info", 2000);
}


/**
 * Widget names on CLIP loader nodes that contain the model name.
 * These are checked in priority order when traversing the graph
 * from an EasyLLM node's "clip" input socket.
 */
const CLIP_MODEL_WIDGET_NAMES = ["clip_name", "clip_name1", "clip_name2", "clip_name3", "model_name"];

/**
 * Get model name for header display by inspecting the model source.
 *
 * For EasyLLMGGUF: reads the model_path widget and extracts the filename.
 * For EasyLLM: traverses the graph from the clip input socket to find
 *   the upstream CLIP loader node, then reads its model-name widget.
 * For other node types or when the source cannot be determined: returns "".
 *
 * @param {object} node - The ComfyUI/LiteGraph node instance
 * @returns {string} The model display name, or "" if unavailable
 */
export function getModelName(node) {
    // ── EasyLLMGGUF: read from model_path widget ──
    if (node.type === "EasyLLMGGUF") {
        const mpw = node.widgets?.find(w => w.name === "model_path");
        if (mpw?.value) {
            // Extract filename from path, trim extension
            let name = String(mpw.value).split(/[/\\]/).pop() || "";
            name = name.replace(/\.(gguf|safetensors|pt|pth|bin)$/i, "");
            // Append 🖼️ badge when mmproj_path is set
            const mmw = node.widgets?.find(w => w.name === "mmproj_path");
            if (mmw?.value) name += " 🖼️";
            return name;
        }
        return "";
    }

    // ── EasyLLM: traverse clip input connection to find upstream CLIP loader ──
    if (node.type === "EasyLLM") {
        try {
            const clipInput = node.inputs?.find(inp => inp.name === "clip");
            if (clipInput?.link != null && app?.graph?.links) {
                const link = app.graph.links[clipInput.link];
                if (link) {
                    const sourceNode = app.graph.getNodeById(link.origin_id);
                    if (sourceNode?.widgets) {
                        for (const w of sourceNode.widgets) {
                            if (CLIP_MODEL_WIDGET_NAMES.includes(w.name) && w.value) {
                                return String(w.value).replace(/\.(safetensors|pt|pth|bin)$/i, "");
                            }
                        }
                    }
                }
            }
        } catch (_e) { /* ignore — graceful fallback */ }
    }

    return "";
}
