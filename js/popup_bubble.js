/**
 * EasyLLM — Popup bubble/export/image/model-name module
 *
 * Contains: chat bubble creation, history rendering, message actions,
 * context indicator, settings sync, send handler, typing indicator,
 * image upload, model name lookup, export/abort functions.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { formatTimestamp, renderMarkdown, parseThinkBlocks, parseAttachedTextBlocks, autoScrollIfNeeded, estimateContextTokens, formatTimingBadge, formatTimingTooltip, estimateTokens } from "./popup_utils.js";
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

    /** Strip attachment markers from message text for export. */
    function stripAttachments(msg) {
        if (!msg) return msg;
        // Replace [Attached EXT: name]\n...content...\n[/EXT] with a brief note
        return msg.replace(
            /\[Attached\s+\w+:\s*[^\]]+\]\n[\s\S]*?\n\[\/\w+\]/g,
            (match) => {
                const nameMatch = match.match(/\[Attached\s+\w+:\s*([^\]]+)\]/);
                const name = nameMatch ? nameMatch[1].trim() : "file";
                return `📎 [Attached: ${name}]`;
            }
        );
    }

    if (format === "md") {
        let md = `# EasyLLM Export — ${nodeLabel || "Chat"}\n`;
        md += `Exported: ${now}\n\n`;
        for (const entry of history) {
            const role = entry.role === "user" ? "You" : "LLM";
            const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
            const header = ts ? `## ${role} (${ts})` : `## ${role}`;
            md += `${header}\n${stripAttachments(entry.message) || ""}\n\n`;
        }
        return { content: md, filename: `easyllm-chat-export-${Date.now()}.md`, mime: "text/markdown" };
    }

    // Plain Text
    let txt = `EasyLLM Export — ${nodeLabel || "Chat"}\n`;
    txt += `Exported: ${now}\n\n`;
    for (const entry of history) {
        const role = entry.role === "user" ? "You" : "LLM";
        const ts = entry.timestamp ? formatTimestamp(entry.timestamp) : "";
        txt += `--- ${role} ${ts ? `(${ts})` : ""} ---\n${stripAttachments(entry.message) || ""}\n\n`;
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

    // ── Image thumbnails (both user and assistant bubbles) ──
    // Renders clickable image previews with type labels when images were attached.
    // Uses the `options.images` array (list of {type, data, filename} objects).
    const renderableImages = [];
    if (options.images && options.images.length > 0) {
        for (const imgObj of options.images) {
            renderableImages.push({
                src: imgObj.data || resolveImageUrl(imgObj.filename),
                label: imgObj.type === "generated" ? "🎨 Generated" : "📷 Uploaded",
            });
        }
    }
    for (const ri of renderableImages) {
        const imgContainer = document.createElement("div");
        imgContainer.className = "llm-chat-bubble-image";
        // Image type label
        const imgLabel = document.createElement("div");
        imgLabel.className = "llm-chat-bubble-image-label";
        imgLabel.textContent = ri.label;
        imgContainer.appendChild(imgLabel);
        const img = document.createElement("img");
        img.src = ri.src;
        img.alt = ri.label;
        img.loading = "eager";
        img.onload = () => {
            if (img.isConnected) {
                const container = img.closest(".llm-popup-history");
                if (container) autoScrollIfNeeded(container);
            }
        };
        img.onerror = img.onload;
        img.addEventListener("click", () => openLightbox(ri.src));
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

    // Parse attached text file blocks (both user and assistant messages)
    const attachParsed = parseAttachedTextBlocks(displayText);
    displayText = attachParsed.displayText;
    const attachments = attachParsed.attachments;

    // Message text — render as markdown
    const text = document.createElement("div");
    text.className = "llm-chat-bubble-text";
    if (displayText) {
        text.innerHTML = renderMarkdown(displayText);
    }

    // Collapsible attachment sections
    for (const att of attachments) {
        const details = document.createElement("details");
        details.className = "llm-chat-attachment";
        const summary = document.createElement("summary");
        summary.className = "llm-chat-attachment-summary";
        summary.textContent = `📄 ${att.filename}`;
        details.appendChild(summary);
        const attContent = document.createElement("div");
        attContent.className = "llm-chat-attachment-content";
        attContent.textContent = att.content;
        details.appendChild(attContent);
        bubble.appendChild(details);
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

    // Edit button (assistant messages only — opens inline editor)
    if (!isUser && options.onEdit) {
        const editBtn = document.createElement("button");
        editBtn.className = "llm-chat-action-btn llm-chat-edit-btn";
        editBtn.textContent = "✏️";
        editBtn.title = "Edit message";
        editBtn.onclick = () => enterEditMode(bubble, message, options.onEdit, options.onCancelEdit);
        actions.appendChild(editBtn);
    }

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
    // Already a data URI — use as-is (base64 fallback)
    if (image.startsWith("data:")) return image;
    // Already a full http/https URL
    if (image.startsWith("http://") || image.startsWith("https://")) return image;
    // Already a /easyllm/db/image/ or /view URL (served by backend)
    if (image.startsWith("/easyllm/db/image/") || image.startsWith("/view?")) return image;
    // Phase 3: DB-stored image filename pattern: <32 hex chars>_<type>.png
    // These are stored in easyllm_db/images/ and served via /easyllm/db/image/{filename}
    if (/^[0-9a-f]{32}_(input|generated)\.png$/.test(image)) {
        return `/easyllm/db/image/${encodeURIComponent(image)}`;
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
        updateContextIndicator(node);
        return;
    }

    for (let i = 0; i < history.length; i++) {
        const entry = history[i];

        const options = {
            timestamp: entry.timestamp,
            timing: entry.timing || entry._timing,
            // Typed images array (list of {type, data, filename} objects)
            // Pass through base64 data URIs as-is; resolve filenames in createBubbleElement
            images: entry.images || null,
            // Paste-to-input on any user message
            onPasteToInput: entry.role === "user" ? (msg) => pasteToInput(node, msg) : null,
            // Delete any message
            onDelete: () => deleteMessage(node, i),
            // Edit assistant messages
            onEdit: entry.role !== "user" ? (newText) => saveEdit(node, i, newText) : null,
            onCancelEdit: entry.role !== "user" ? () => cancelEdit(node) : null,
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
export async function deleteMessage(node, index) {
    const history = node._chatHistory;
    if (!history || index < 0 || index >= history.length) return;

    try {
        // Delete from server FIRST (server is source of truth)
        const resp = await fetch(`/easyllm/db/history/${node.id}/entry?index=${index}&type=chat`, {
            method: "DELETE",
        });
        const result = await resp.json();

        if (!result.success) {
            showToast(`Failed to delete message: ${result.error || "Unknown error"}`, "error", 3000);
            return;
        }

        // Only modify local cache after server confirms
        history.splice(index, 1);
        renderPopupHistory(node);
        showToast("🗑 Message deleted", "info", 2000);
    } catch (e) {
        showToast(`Failed to delete message: ${e.message}`, "error", 3000);
    }
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

/**
 * Read the effective system prompt from a node.
 * System prompt is now provided via the forceInput socket only,
 * which cannot be read from the frontend. Returns empty string.
 * History entries still have systemPromptText snapshots for display.
 */
function getSystemPromptText(node) {
    // System prompt is provided via wired forceInput socket.
    // Socket values are not readable from frontend JavaScript.
    // However, if the popup dropdown selected a system prompt,
    // we can use that for the context token estimate.
    return node._popupSystemPrompt || "";
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
    const maxSelect = node._popupMaxSelect;
    const tempSelect = node._popupTempSelect;
    const seedInput = node._popupSeedInput;
    const hasPopupRefs = maxSelect && tempSelect && seedInput;

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
        settingsObj.max_length = parseInt(maxSelect.value, 10) || 768;
        settingsObj.temperature = node.isEasyLLMGGUF
            ? (parseFloat(tempSelect.value) ?? 0.7)
            : tempSelect.value;
        settingsObj.seed = parseInt(seedInput.value, 10) || 0;

        // Check auto-randomize state (node._llmSeedRandomize)
        const sdW_check = node.widgets?.find(w => w.name === "seed");
        const isAutoRandomize = node._llmSeedRandomize !== false;
        console.debug(
            `[LLM Chat] syncPopupSettingsToCanvas: ` +
            `seedInput=${settingsObj.seed}, canvasWidget=${sdW_check?.value}, ` +
            `autoRandomize=${isAutoRandomize ? "🔀ON" : "🔒OFF"}`
        );

        // vram_mode (shared between CLIP and GGUF)
        const vramSelect = node._popupVramSelect;
        if (vramSelect) {
            settingsObj.vram_mode = vramSelect.value;
        }

        console.debug(
            `[LLM Chat] syncPopupSettingsToCanvas: ` +
            `max=${settingsObj.max_length}, temp=${settingsObj.temperature}, seed=${settingsObj.seed}, ` +
            `vram=${settingsObj.vram_mode}`
        );

        // Update visible widgets so Python receives current values
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

        // Sync enableImageGeneration (per-node toggle for trigger_prompt auto-queue)
        const genCheck = node._popupGenCheckbox;
        if (genCheck) {
            const was = node._enableImageGeneration;
            node._enableImageGeneration = genCheck.checked;
            if (was !== node._enableImageGeneration) {
                console.debug(
                    `[LLM Chat] Image Generation toggled ${node._enableImageGeneration ? "ON" : "OFF"} for node ${node.id}`
                );
            }
        }

        // Sync iterativeRefinement (per-node toggle for edit_image pipeline cache)
        const iterCheck = node._popupIterRefineCheckbox;
        if (iterCheck) {
            const was = node._iterativeRefinement;
            node._iterativeRefinement = iterCheck.checked;
            if (was !== node._iterativeRefinement) {
                console.debug(
                    `[LLM Chat] Iterative Refinement toggled ${node._iterativeRefinement ? "ON" : "OFF"} for node ${node.id}`
                );
            }
            // Update canvas widget so Python receives the value on queue
            const irW = node.widgets?.find(w => w.name === "iterative_refinement");
            if (irW) {
                irW.value = node._iterativeRefinement;
            }
        }
    } else {
        // ── Fallback: read shared fields from canvas widgets to preserve them ──
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
    // ── Guard: enhancer mode should not create chat history ──
    const modeW = node.widgets?.find(w => w.name === "mode");
    if (modeW && modeW.value === "enhancer") {
        showToast("Cannot send chat messages in enhancer mode", "warning", 3000);
        return;
    }

    // Sync popup settings to canvas widgets before execution
    syncPopupSettingsToCanvas(node);

    const input = node._popupInputEl;
    if (!input) return;

    const message = input.value.trim();
    if (!message) return;

    // ── Server-first write pattern ──
    // Build the user entry object first, persist to server disk,
    // then update browser cache only on success.

    const imagesArr = node._uploadedImage
        ? [{ type: "input", filename: node._uploadedImage, data: null }]
        : undefined;

    // Build the entry that will be stored
    const userEntry = {
        role: "user",
        message: message,
        timestamp: Date.now(),
        _sessionUuid: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2),
    };
    if (imagesArr) {
        userEntry.images = imagesArr;
    }

    // ── Disk persistence first (server is source of truth) ──
    try {
        const appendResp = await fetch(`/easyllm/db/history/${node.id}/append`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                entry: userEntry,
                type: "chat",
            }),
        });
        if (!appendResp.ok) {
            showToast("Failed to save message to server", "error", 3000);
            return;
        }
    } catch (_appendErr) {
        console.debug("[LLM Chat DB] handlePopupSend append failed:", _appendErr);
        showToast("Failed to save message to server", "error", 3000);
        return;
    }

    // ── Server confirmed — now update browser cache ──
    pushUserMessage(node, message, imagesArr);

    // Store chat history for backend execution (ephemeral store)
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

    // Write/clear uploaded image filename in hidden widget
    const ifw = node.widgets?.find(w => w.name === "image_filename");
    if (ifw) ifw.value = node._uploadedImage || "";

    // Write popup system prompt to hidden widget (fallback for non-wired socket)
    const spw = node.widgets?.find(w => w.name === "system_prompt_popup");
    if (spw) spw.value = node._popupSystemPrompt || "";

    // Re-render popup history (shows user message, "..." pending)
    renderPopupHistory(node);

    // Clear input and reset height
    input.value = "";
    input.style.height = "36px";

    // Show typing indicator
    showTypingIndicator(node);

    // Streaming setup
    try {
        await fetch(`/easyllm/popup_active/${node.id}`, { method: "POST" });
    } catch (_e) {
        // Non-critical: streaming will fall back to blocking if this fails
    }
    startStreamListening(node);
    startCanvasProgressTracking(node);
    node._popupStreaming = true;
    node._streamingSavedHistory = false;
    const stopBtn = node._popupStopBtn;
    if (stopBtn) stopBtn.style.display = "inline-flex";

    // Reset auto-queue guard for new turn
    node._triggerQueued = false;

    // Trigger partial execution
    app.queuePrompt(0, 1, [String(node.id)]);

    // ── Clear uploaded image after sending ──
    // keepWidget=true defers clearing image_filename so the auto-queue
    // (which fires later in onExecuted) can still read the filename
    // for the downstream edit_image pipeline's cache replay.
    if (node._uploadedImage) {
        await removeAttachedImage(node, { keepWidget: true });
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

/**
 * Upload an image file to ComfyUI, store the filename server-side,
 * and update the frontend state (preview, button, toast).
 *
 * Shared by openImagePicker (file dialog) and handleDroppedFile (drag-and-drop).
 *
 * @param {object} node - The ComfyUI/LiteGraph node instance
 * @param {File} file - The image File object
 */
async function uploadImageFile(node, file) {
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

        // ── Phase 3: Also save a copy to the DB so it's available when
        // loading history from disk, even if the ComfyUI/input/ file is cleaned up.
        try {
            const reader = new FileReader();
            reader.onload = async (evt) => {
                const base64Data = evt.target.result;
                try {
                    await fetch("/easyllm/db/image", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ base64: base64Data, type: "input" }),
                    });
                } catch (_e) {
                    // Non-critical — image is still available via /view
                }
            };
            reader.readAsDataURL(file);
        } catch (_e) {
            // Non-critical — image is still available via /view
        }

        showToast("✅ Image attached", "success", 2000);
    } catch (err) {
        showToast(`❌ Upload failed: ${err.message}`, "error", 4000);
    }
}

/**
 * Read a text-based file (.txt, .md, .json) and prepend its content
 * to the popup text input, wrapped in a descriptive marker.
 *
 * Does NOT touch the image upload pipeline — purely client-side text insertion.
 *
 * @param {object} node - The EasyLLM node
 * @param {File}   file - The text File object
 */
async function attachTextFile(node, file) {
    // Guard: prevent during active generation
    if (node._popupStreaming) {
        showToast("Cannot attach file during generation", "warning", 2000);
        return;
    }

    try {
        const text = await file.text();
        const extension = file.name.split('.').pop().toUpperCase();
        const wrapper = `\n\n[Attached ${extension}: ${file.name}]\n${text}\n[/${extension}]\n\n`;

        const input = node._popupInputEl;
        if (input) {
            // Insert at cursor position or append
            const cursorPos = input.selectionStart;
            const before = input.value.substring(0, cursorPos);
            const after = input.value.substring(cursorPos);
            input.value = before + wrapper + after;
            // Trigger input event for auto-resize
            input.dispatchEvent(new Event("input"));
            input.focus();
        }

        showToast(`✅ ${file.name} attached`, "success", 2000);
    } catch (err) {
        showToast(`❌ Failed to read ${file.name}: ${err.message}`, "error", 4000);
    }
}

export function openImagePicker(node) {
    /** Open a file picker to upload an image or attach a text file.
     *
     * 1. Creates a hidden <input type="file" accept="image/*,.txt,.md,.json">
     * 2. On file selection, branches:
     *    - image/* → uploadImageFile (existing upload pipeline)
     *    - text/*, .md, .json → attachTextFile (client-side insertion)
     */
    // Guard: prevent upload during active generation
    if (node._popupStreaming) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*,.txt,.md,.json";
    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;

        if (file.type.startsWith("image/")) {
            await uploadImageFile(node, file);
        } else {
            await attachTextFile(node, file);
        }
    };
    input.click();
}

/**
 * Classify a dropped file as image or text based on MIME type and extension.
 *
 * @param {File} file - The File object to classify
 * @returns {'image'|'text'|null} The file category, or null if unsupported
 */
function classifyDroppedFile(file) {
    if (!file) return null;
    if (file.type.startsWith("image/")) return "image";
    // Also check extension for files with non-standard MIME types
    const name = file.name.toLowerCase();
    if (file.type.startsWith("text/") || name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".json")) {
        return "text";
    }
    return null;
}

/**
 * Handle a dropped file from drag-and-drop.
 *
 * Routes images through the upload pipeline, text files through attachTextFile.
 *
 * @param {object} node - The ComfyUI/LiteGraph node instance
 * @param {File} file - The dropped File object
 */
export async function handleDroppedFile(node, file) {
    // Guard: prevent upload during active generation
    if (node._popupStreaming) {
        showToast("Cannot drop file during generation", "warning", 2000);
        return;
    }

    const category = classifyDroppedFile(file);
    if (category === "image") {
        await uploadImageFile(node, file);
    } else if (category === "text") {
        await attachTextFile(node, file);
    } else {
        showToast("❌ Unsupported file. Drop image, .txt, .md, or .json", "error", 3000);
    }
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

export async function removeAttachedImage(node, options = {}) {
    /**
     * Remove the attached image: clear server, frontend state, UI, and widget.
     *
     * @param {object}  node                 - The EasyLLM node
     * @param {object}  [options]            - Optional behavior flags
     * @param {boolean} [options.keepWidget] - If true, do NOT clear the
     *        image_filename widget value. Used by handlePopupSend to defer
     *        widget clearing until after the auto-queue fires, so the
     *        downstream edit_image pipeline can still read the uploaded
     *        filename during cache replay.
     */
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
        attachBtn.title = "Attach an image or text file";
        attachBtn.onclick = () => openImagePicker(node);
    }

    // Clear the image_filename widget value to prevent Python from picking up
    // stale filenames on the next Queue Prompt execution.
    // When keepWidget is true, defer this to after auto-queue (caller
    // is responsible for clearing it later via onExecuted).
    if (!options.keepWidget) {
        const ifw = node.widgets?.find(w => w.name === "image_filename");
        if (ifw) ifw.value = "";
    }

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

    // ── EasyLLMText: traverse graph from clip input to find CLIP loader ──
    const clipInput = node.inputs?.find(inp => inp.name === "clip");
    if (clipInput && clipInput.link != null) {
        const link = app.graph.links[clipInput.link];
        if (link) {
            const sourceNode = app.graph.getNodeById(link.origin_id);
            if (sourceNode) {
                // Check CLIP_MODEL_WIDGET_NAMES in priority order
                for (const wName of CLIP_MODEL_WIDGET_NAMES) {
                    const w = sourceNode.widgets?.find(w => w.name === wName);
                    if (w?.value) return String(w.value);
                }
            }
        }
    }
    return "";
}

// ═══════════════════════════════════════════════════════════════════════
// Enhancer History — Card Layout Rendering
// ═══════════════════════════════════════════════════════════════════════

/**
 * Build the images comparison section of an enhancer card.
 * Returns a .enhancer-card-images div, or null if no images exist.
 *
 * @param {Object} entry - Enhancer history entry { input, output, images?, ... }
 * @returns {HTMLElement|null}
 */
export function buildImagesSection(entry) {
    const images = entry.images;
    if (!images || !Array.isArray(images) || images.length === 0) return null;

    const container = document.createElement("div");
    container.className = "enhancer-card-images";

    // Separate input vs generated images
    const inputImgs = images.filter(img => img.type === "input");
    const genImgs = images.filter(img => img.type === "generated");

    // Show first input image
    for (const img of inputImgs.slice(0, 1)) {
        const wrapper = document.createElement("div");
        wrapper.className = "enhancer-card-image-wrapper";
        const label = document.createElement("div");
        label.className = "enhancer-card-image-label";
        label.textContent = "INPUT";
        wrapper.appendChild(label);
        const imgEl = document.createElement("img");
        imgEl.src = img.data || resolveImageUrl(img.filename);
        imgEl.alt = "Input image";
        imgEl.loading = "eager";
        imgEl.width = 72;
        imgEl.height = 72;
        imgEl.onerror = function () { this.style.display = "none"; };
        imgEl.onload = function () {
            const container = this.closest(".llm-popup-history");
            if (container) autoScrollIfNeeded(container);
        };
        imgEl.addEventListener("click", () => openLightbox(imgEl.src));
        wrapper.appendChild(imgEl);
        container.appendChild(wrapper);
    }

    // Arrow separator
    if (inputImgs.length > 0 && genImgs.length > 0) {
        const arrow = document.createElement("div");
        arrow.className = "enhancer-card-image-arrow";
        arrow.textContent = "→";
        container.appendChild(arrow);
    }

    // Show first generated image
    for (const img of genImgs.slice(0, 1)) {
        const wrapper = document.createElement("div");
        wrapper.className = "enhancer-card-image-wrapper";
        const label = document.createElement("div");
        label.className = "enhancer-card-image-label";
        label.textContent = "GENERATED";
        wrapper.appendChild(label);
        const imgEl = document.createElement("img");
        imgEl.src = img.data || resolveImageUrl(img.filename);
        imgEl.alt = "Generated image";
        imgEl.loading = "eager";
        imgEl.width = 72;
        imgEl.height = 72;
        imgEl.onerror = function () { this.style.display = "none"; };
        imgEl.onload = function () {
            const container = this.closest(".llm-popup-history");
            if (container) autoScrollIfNeeded(container);
        };
        imgEl.addEventListener("click", () => openLightbox(imgEl.src));
        wrapper.appendChild(imgEl);
        container.appendChild(wrapper);
    }

    return container;
}

/**
 * Rebuild the enhancer card output actions bar (edit + copy buttons).
 * Used after save or cancel to restore the full set of action buttons.
 *
 * @param {HTMLElement} container - The .enhancer-card-output element
 * @param {Object}      entry     - Enhancer history entry { output }
 * @param {Object}      options   - { onEdit }
 * @param {string}      displayText - Processed display text for copy fallback
 */
export function rebuildEnhancerActions(container, entry, options, displayText) {
    const actionsEl = container.querySelector(".enhancer-card-output-actions");
    if (!actionsEl) return;
    actionsEl.innerHTML = "";

    // Edit button (when onEdit callback is provided)
    if (options.onEdit) {
        const editBtn = document.createElement("button");
        editBtn.className = "llm-chat-action-btn llm-chat-edit-btn";
        editBtn.textContent = "✏️";
        editBtn.title = "Edit output";
        const origEntryOutput = entry.output;
        editBtn.onclick = () => {
            enterEnhancerEditMode(container, origEntryOutput || entry.output, options.onEdit, () => {
                // Cancel: restore original markdown
                const textEl2 = container.querySelector(".enhancer-card-output-text");
                if (textEl2 && origEntryOutput) {
                    const parsed2 = parseThinkBlocks(origEntryOutput);
                    const dt2 = parsed2.response || origEntryOutput;
                    const ap2 = parseAttachedTextBlocks(dt2);
                    textEl2.innerHTML = renderMarkdown(ap2.displayText);
                }
                // Restore action buttons (edit + copy)
                rebuildEnhancerActions(container, entry, options, displayText);
            });
        };
        actionsEl.appendChild(editBtn);
    }

    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "llm-chat-action-btn llm-chat-copy-btn";
    copyBtn.textContent = "📋";
    copyBtn.title = "Copy output";
    copyBtn.onclick = async () => {
        try {
            await navigator.clipboard.writeText(displayText || entry.output || "");
            showToast("✅ Copied!", "success", 2000);
        } catch {
            showToast("❌ Copy failed", "error", 2000);
        }
    };
    actionsEl.appendChild(copyBtn);
}

/**
 * Build the output text section of an enhancer card.
 *
 * @param {Object} entry - Enhancer history entry
 * @param {Object} options - { onCopy, onPasteToInput }
 * @returns {HTMLElement}
 */
function buildOutputSection(entry, options = {}) {
    const container = document.createElement("div");
    container.className = "enhancer-card-output";

    // Parse think blocks from output
    let displayText = entry.output || "";
    let thinkBlock = null;
    if (entry.output) {
        const parsed = parseThinkBlocks(entry.output);
        thinkBlock = parsed.thinking;
        displayText = parsed.response || entry.output;
    }

    // Parse attached text file blocks
    const attachParsed = parseAttachedTextBlocks(displayText);
    displayText = attachParsed.displayText;

    // Render text as markdown
    const textEl = document.createElement("div");
    textEl.className = "enhancer-card-output-text";
    if (displayText) {
        textEl.innerHTML = renderMarkdown(displayText);
    }
    container.appendChild(textEl);

    // Action buttons row
    const actions = document.createElement("div");
    actions.className = "enhancer-card-output-actions";

    // Edit button (when onEdit callback is provided)
    if (options.onEdit) {
        const editBtn = document.createElement("button");
        editBtn.className = "llm-chat-action-btn llm-chat-edit-btn";
        editBtn.textContent = "✏️";
        editBtn.title = "Edit output";
        editBtn.onclick = () => {
            enterEnhancerEditMode(container, entry.output, options.onEdit, () => {
                // Cancel: restore original markdown
                const textEl2 = container.querySelector(".enhancer-card-output-text");
                if (textEl2 && entry.output) {
                    const parsed2 = parseThinkBlocks(entry.output);
                    const dt2 = parsed2.response || entry.output;
                    const ap2 = parseAttachedTextBlocks(dt2);
                    textEl2.innerHTML = renderMarkdown(ap2.displayText);
                }
                // Restore action buttons (edit + copy) via shared helper
                rebuildEnhancerActions(container, entry, options, displayText);
            });
        };
        actions.appendChild(editBtn);
    }

    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "llm-chat-action-btn llm-chat-copy-btn";
    copyBtn.textContent = "📋";
    copyBtn.title = "Copy output";
    copyBtn.onclick = async () => {
        try {
            const textToCopy = displayText || entry.output || "";
            await navigator.clipboard.writeText(textToCopy);
            showToast("✅ Copied!", "success", 2000);
        } catch {
            showToast("❌ Copy failed", "error", 2000);
        }
    };
    actions.appendChild(copyBtn);

    // Attached text file sections (if any)
    for (const att of attachParsed.attachments) {
        const details = document.createElement("details");
        details.className = "llm-chat-attachment";
        const summary = document.createElement("summary");
        summary.className = "llm-chat-attachment-summary";
        summary.textContent = `📄 ${att.filename}`;
        details.appendChild(summary);
        const attContent = document.createElement("div");
        attContent.className = "llm-chat-attachment-content";
        attContent.textContent = att.content;
        details.appendChild(attContent);
        container.appendChild(details);
    }

    // Collapsible thinking section (above output text)
    if (thinkBlock) {
        const thinkDetails = document.createElement("details");
        thinkDetails.className = "enhancer-card-thinking";
        const thinkSummary = document.createElement("summary");
        thinkSummary.className = "enhancer-card-thinking-summary";
        thinkSummary.textContent = "💭 Thinking Process...";
        thinkDetails.appendChild(thinkSummary);
        const thinkContent = document.createElement("div");
        thinkContent.className = "enhancer-card-thinking-content";
        thinkContent.textContent = thinkBlock;
        thinkDetails.appendChild(thinkContent);
        // Insert before textEl so it appears above the output
        container.insertBefore(thinkDetails, textEl);
    }

    container.appendChild(actions);

    // Timestamp
    if (entry.timestamp) {
        const tsEl = document.createElement("div");
        tsEl.className = "enhancer-card-timestamp";
        tsEl.textContent = formatTimestamp(entry.timestamp);
        container.appendChild(tsEl);
    }

    return container;
}

/**
 * Build the collapsible details section of an enhancer card.
 *
 * @param {Object} entry - Enhancer history entry
 * @param {Object} options - { tokenCount }
 * @returns {HTMLElement}
 */
function buildDetailsSection(entry, options = {}) {
    const details = document.createElement("details");
    details.className = "enhancer-card-details";

    const summary = document.createElement("summary");
    summary.textContent = "📋 Details";
    details.appendChild(summary);

    const content = document.createElement("div");
    content.className = "enhancer-card-details-content";

    // 1. Input text (with optional paste-to-input button)
    if (entry.input) {
        const row = createDetailRow("📝", "Input:", entry.input, "enhancer-card-detail-input");
        if (options.onPasteToInput) {
            const pasteBtn = document.createElement("button");
            pasteBtn.className = "llm-chat-action-btn llm-chat-edit-btn";
            pasteBtn.textContent = "✏️";
            pasteBtn.title = "Paste to input";
            pasteBtn.style.marginLeft = "4px";
            pasteBtn.style.flexShrink = "0";
            pasteBtn.onclick = (e) => {
                e.stopPropagation();
                options.onPasteToInput(entry.input);
            };
            row.appendChild(pasteBtn);
        }
        content.appendChild(row);
    }

    // 2. Model name
    if (entry.modelName) {
        const row = createDetailRow("🤖", "Model:", entry.modelName);
        content.appendChild(row);
    }

    // 3. System prompt
    if (entry.systemPromptText) {
        const row = createDetailRow("⚙️", "System:", entry.systemPromptText, "enhancer-card-detail-system");
        content.appendChild(row);
    }

    // 4. Timestamp + token count
    const metaRow = document.createElement("div");
    metaRow.className = "enhancer-card-detail-row";
    const icon = document.createElement("span");
    icon.className = "enhancer-card-detail-icon";
    icon.textContent = "⏱️";
    metaRow.appendChild(icon);
    const metaValue = document.createElement("span");
    metaValue.className = "enhancer-card-detail-value";

    let metaText = "";
    if (entry.timestamp) {
        const d = new Date(entry.timestamp);
        metaText += d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    if (options.tokenCount != null) {
        metaText += metaText ? " · " : "";
        metaText += `~${options.tokenCount} tokens`;
    }
    metaValue.textContent = metaText || "";
    metaRow.appendChild(metaValue);
    content.appendChild(metaRow);

    details.appendChild(content);
    return details;
}

/**
 * Create a single detail row: icon + label + value.
 *
 * @param {string} iconText - Emoji icon
 * @param {string} label - Label text (e.g. "Input:")
 * @param {string} value - Value text
 * @param {string} [valueClass] - Optional additional class for the value element
 * @returns {HTMLElement}
 */
function createDetailRow(iconText, label, value, valueClass) {
    const row = document.createElement("div");
    row.className = "enhancer-card-detail-row";

    const icon = document.createElement("span");
    icon.className = "enhancer-card-detail-icon";
    icon.textContent = iconText;
    row.appendChild(icon);

    const labelEl = document.createElement("span");
    labelEl.className = "enhancer-card-detail-label";
    labelEl.textContent = label;
    row.appendChild(labelEl);

    const valueEl = document.createElement("span");
    valueEl.className = "enhancer-card-detail-value";
    if (valueClass) {
        valueEl.classList.add(valueClass);
    }
    valueEl.textContent = value;
    row.appendChild(valueEl);

    return row;
}

/**
 * Create a single enhancer card element for the output history popup.
 *
 * The card layout is:
 * ┌───────────────────────────────────────────────────────────────┐
 * │ ┌─ Images ────┬─ [💭 Thinking Process...] (collapsible) ──┐  │
 * │ │  input→gen  │  [generated text]                          │  │
 * │ │             │  [✏️] [📋]                         [time]  │  │
 * │ └─────────────┴───────────────────────────────────────────┘  │
 * │ ┌─ 📋 Details ▼ ───────────────────────────────────────────┐ │
 * │ │  Input, Model, System, Timestamp, Tokens                 │ │
 * │ └───────────────────────────────────────────────────────────┘ │
 * └───────────────────────────────────────────────────────────────┘
 *
 * @param {Object} entry - Enhancer history entry
 *   { input, output, timestamp, systemPromptText, modelName, images? }
 * @param {Object} [options] - { isSelected, onSelect }
 * @returns {HTMLElement}
 */
export function createEnhancerCardElement(entry, options = {}) {
    const card = document.createElement("div");
    card.className = "enhancer-card";

    if (options.isSelected) {
        card.classList.add("enhancer-card-selected");
    }

    // Store _sessionUuid on the card element for targeted DOM updates
    // (e.g., ImageCapture appends images after the card was already rendered)
    if (entry && entry._sessionUuid) {
        card.dataset.sessionUuid = entry._sessionUuid;
    }

    // ── Defensive: warn if entry is missing expected fields ──
    // This helps diagnose issues where _loadEnhancerHistoryFromDisk overwrites
    // in-memory data with stale/partial disk entries, causing the details
    // section to be missing expected data.
    if (!entry || !entry.output) {
        console.warn(
            `[LLM Chat] createEnhancerCardElement: entry missing 'output' field. ` +
            `Entry keys: ${Object.keys(entry || {}).join(", ") || "(empty)"}`
        );
    }
    if (entry && !entry.input) {
        console.debug(
            `[LLM Chat] createEnhancerCardElement: entry missing 'input' field ` +
            `(ok for legacy entries without stored input).`
        );
    }

    // ── Main content row: images + output ──
    const main = document.createElement("div");
    main.className = "enhancer-card-main";

    // Images section (conditional — null when no images data)
    const imagesSection = buildImagesSection(entry);
    if (imagesSection) {
        main.appendChild(imagesSection);
    } else {
        card.classList.add("enhancer-card-no-images");
    }

    // Output section (always present)
    const outputSection = buildOutputSection(entry, {
        onEdit: options.onEdit,
    });
    main.appendChild(outputSection);

    card.appendChild(main);

    // ── Collapsible details section (metadata only — thought is in output section) ──
    const details = buildDetailsSection(entry, {
        tokenCount: options.tokenCount,
        onPasteToInput: options.onPasteToInput,
    });
    card.appendChild(details);

    // ── Selection checkbox / click handler ──
    if (options.onSelect) {
        card.addEventListener("click", (e) => {
            // Don't toggle on details toggle, link clicks, or button clicks
            if (e.target.closest("details") || e.target.closest("button") || e.target.closest("a") || e.target.closest("img")) return;
            const isSelected = card.classList.toggle("enhancer-card-selected");
            options.onSelect(entry, isSelected);
        });
    }

    return card;
}

// ═══════════════════════════════════════════════════════════════════════
// Lightbox — full-screen image preview (feature 2)
// ═══════════════════════════════════════════════════════════════════════

let _lightboxOverlay = null;

/**
 * Open a lightbox overlay displaying the given image src.
 * Replaces window.open image links with an inline modal.
 */
export function openLightbox(src) {
    // Remove any existing lightbox first
    closeLightbox();

    const overlay = document.createElement("div");
    overlay.className = "llm-lightbox-overlay";

    const img = document.createElement("img");
    img.className = "llm-lightbox-image";
    img.src = src;
    img.alt = "Preview";
    overlay.appendChild(img);

    const closeBtn = document.createElement("button");
    closeBtn.className = "llm-lightbox-close";
    closeBtn.textContent = "✕";
    closeBtn.title = "Close (Esc)";
    overlay.appendChild(closeBtn);

    // Close handlers
    const handleClose = () => closeLightbox();
    const handleKey = (e) => { if (e.key === "Escape") closeLightbox(); };

    closeBtn.addEventListener("click", (e) => { e.stopPropagation(); handleClose(); });
    overlay.addEventListener("click", handleClose);
    document.addEventListener("keydown", handleKey);

    _lightboxOverlay = overlay;
    document.body.appendChild(overlay);
}

/**
 * Close the currently open lightbox overlay.
 */
export function closeLightbox() {
    if (_lightboxOverlay) {
        _lightboxOverlay.remove();
        _lightboxOverlay = null;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Inline edit — edit assistant message in-place (feature 3)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Enter inline edit mode on an assistant message bubble.
 * Replaces the rendered markdown text with a textarea.
 *
 * @param {HTMLElement} bubble - The bubble element to edit
 * @param {string} originalMessage - The raw message text (pre-markdown)
 * @param {Function} onSave - Callback with the edited text
 * @param {Function} onCancel - Callback to cancel editing
 */
function enterEditMode(bubble, originalMessage, onSave, onCancel) {
    const textEl = bubble.querySelector(".llm-chat-bubble-text");
    const actionsEl = bubble.querySelector(".llm-chat-bubble-actions");
    if (!textEl || !actionsEl) return;

    // Replace text content with textarea
    const textarea = document.createElement("textarea");
    textarea.className = "llm-chat-edit-textarea";
    textarea.value = originalMessage;
    textEl.innerHTML = "";
    textEl.appendChild(textarea);
    textarea.focus();

    // Replace action buttons with save/cancel
    const existingHTML = actionsEl.innerHTML;
    actionsEl.innerHTML = "";

    const saveBtn = document.createElement("button");
    saveBtn.className = "llm-chat-action-btn llm-chat-save-btn";
    saveBtn.textContent = "💾";
    saveBtn.title = "Save changes";
    saveBtn.onclick = () => {
        const newText = textarea.value.trim();
        if (newText) onSave(newText);
    };
    actionsEl.appendChild(saveBtn);

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "llm-chat-action-btn llm-chat-cancel-btn";
    cancelBtn.textContent = "❌";
    cancelBtn.title = "Cancel";
    cancelBtn.onclick = () => onCancel();
    actionsEl.appendChild(cancelBtn);
}

/**
 * Save an edited assistant message to history and persist to disk.
 */
async function saveEdit(node, index, newText) {
    const history = node._chatHistory;
    if (!history || !history[index]) return;

    // Optimistic local update for responsive UI
    history[index].message = newText;
    renderPopupHistory(node);

    try {
        // Persist to server (source of truth)
        const resp = await fetch(`/easyllm/db/history/${node.id}/entry`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                index: index,
                entry: history[index],
                type: "chat",
            }),
        });
        const result = await resp.json();

        if (!result.success) {
            showToast(`Failed to save edit: ${result.error || "Unknown error"}`, "error", 3000);
        } else {
            showToast("✅ Message updated", "success", 2000);
        }
    } catch (e) {
        showToast(`Failed to save edit: ${e.message}`, "error", 3000);
    }
}

/**
 * Cancel editing — re-render the popup history to restore original state.
 */
function cancelEdit(node) {
    renderPopupHistory(node);
}

// ═══════════════════════════════════════════════════════════════════════
// Enhancer card inline edit — edit output text in-place (feature)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Enter inline edit mode on an enhancer card's output text.
 * Replaces the rendered markdown textarea with a textarea and
 * swaps action buttons for save/cancel.
 *
 * @param {HTMLElement} container - The .enhancer-card-output element
 * @param {string} originalMessage - The raw output text (pre-markdown)
 * @param {Function} onSave - Callback with the edited text
 * @param {Function} onCancel - Callback to cancel editing
 */
function enterEnhancerEditMode(container, originalMessage, onSave, onCancel) {
    const textEl = container.querySelector(".enhancer-card-output-text");
    const actionsEl = container.querySelector(".enhancer-card-output-actions");
    if (!textEl || !actionsEl) return;

    // Replace text content with textarea
    const textarea = document.createElement("textarea");
    textarea.className = "llm-chat-edit-textarea";
    textarea.value = originalMessage;
    textEl.innerHTML = "";
    textEl.appendChild(textarea);
    textarea.focus();

    // Replace action buttons with save/cancel
    const existingHTML = actionsEl.innerHTML;
    actionsEl.innerHTML = "";

    const saveBtn = document.createElement("button");
    saveBtn.className = "llm-chat-action-btn llm-chat-save-btn";
    saveBtn.textContent = "💾";
    saveBtn.title = "Save changes";
    saveBtn.onclick = () => {
        const newText = textarea.value.trim();
        if (newText) onSave(newText);
    };
    actionsEl.appendChild(saveBtn);

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "llm-chat-action-btn llm-chat-cancel-btn";
    cancelBtn.textContent = "❌";
    cancelBtn.title = "Cancel";
    cancelBtn.onclick = () => {
        actionsEl.innerHTML = existingHTML;
        onCancel();
    };
    actionsEl.appendChild(cancelBtn);
}

// ────────────────────────────────────────────────────────────────────────
// Enhancer Export Dialog — Redesigned v2
// ────────────────────────────────────────────────────────────────────────

/**
 * Open the enhancer export dialog modal (redesigned v2).
 * Features structure dropdown, conditional format/separator, simple naming.
 *
 * @param {object}   node          - The EasyLLM node
 * @param {Set}      selectedEntries - Set of enhancer entry objects
 * @param {string}   nodeLabel     - Display label for the node
 */
export function openEnhancerExportDialog(node, selectedEntries, nodeLabel) {
    const entries = Array.from(selectedEntries);
    if (!entries || entries.length === 0) {
        showToast("No entries selected", "error", 2000);
        return;
    }

    // ── Overlay ──
    const overlay = document.createElement("div");
    overlay.className = "llm-export-overlay";

    const dialog = document.createElement("div");
    dialog.className = "llm-export-dialog";

    // ── Header ──
    const header = document.createElement("div");
    header.className = "llm-export-dialog-header";
    const title = document.createElement("span");
    title.className = "llm-export-dialog-title";
    title.textContent = `📥 Export ${entries.length} Entry${entries.length > 1 ? "ies" : ""}`;
    header.appendChild(title);
    const closeBtn = document.createElement("button");
    closeBtn.className = "llm-export-dialog-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = () => overlay.remove();
    header.appendChild(closeBtn);
    dialog.appendChild(header);

    // ── Body (scrollable) ──
    const body = document.createElement("div");
    body.className = "llm-export-dialog-body";

    // ── Warning ──
    const warning = document.createElement("div");
    warning.className = "llm-export-warning";
    warning.textContent = "⚠️ Export writes files to the server filesystem. Choose carefully.";
    body.appendChild(warning);

    // ═══════════════════════════════════════════════════════════════
    // 1. Structure Dropdown
    // ═══════════════════════════════════════════════════════════════
    const structSection = document.createElement("div");
    structSection.className = "llm-export-section";

    const structTitle = document.createElement("div");
    structTitle.className = "llm-export-section-title";
    structTitle.textContent = "📁 Structure";
    structSection.appendChild(structTitle);

    const structLabel = document.createElement("div");
    structLabel.className = "llm-export-dropdown-label";
    structLabel.textContent = "Output structure:";
    structSection.appendChild(structLabel);

    const structDropdown = createDropdown("structure", [
        { value: "pairs", label: "Per-Entry Pairs" },
        { value: "log", label: "Single File Log" },
    ], "pairs");
    structDropdown.className = "llm-export-dropdown";
    structSection.appendChild(structDropdown);

    const modeIndicator = document.createElement("div");
    modeIndicator.className = "llm-export-mode-indicator";
    modeIndicator.textContent = "📄 Creates separate numbered files for each entry + images";
    structSection.appendChild(modeIndicator);

    body.appendChild(structSection);

    // ═══════════════════════════════════════════════════════════════
    // 2. Format Selection (conditional on structure)
    // ═══════════════════════════════════════════════════════════════
    const formatSection = document.createElement("div");
    formatSection.className = "llm-export-section";

    const formatTitle = document.createElement("div");
    formatTitle.className = "llm-export-section-title";
    formatTitle.textContent = "📄 Format";
    formatSection.appendChild(formatTitle);

    const formatLabel = document.createElement("div");
    formatLabel.className = "llm-export-dropdown-label";
    formatLabel.textContent = "File format:";
    formatSection.appendChild(formatLabel);

    const formatGroup = document.createElement("div");
    formatGroup.className = "llm-export-radio-group";
    formatGroup.id = "llm-export-format-group";

    // Per-Entry formats: .txt (default), .md
    const pairsTxtRadio = createRadio("export_format_v2", "txt", "Plain Text (.txt)", true);
    const pairsMdRadio = createRadio("export_format_v2", "md", "Markdown (.md)", false);
    // Single File formats: .md (default), .txt, .jsonl
    const logMdRadio = createRadio("export_format_v2", "md", "Markdown (.md)", true);
    const logTxtRadio = createRadio("export_format_v2", "txt", "Plain Text (.txt)", false);
    const logJsonlRadio = createRadio("export_format_v2", "jsonl", "JSONL (.jsonl)", false);

    // Start with Per-Entry formats visible
    formatGroup.appendChild(pairsTxtRadio);
    formatGroup.appendChild(pairsMdRadio);
    // Log-only formats (hidden initially)
    logMdRadio.style.display = "none";
    logTxtRadio.style.display = "none";
    logJsonlRadio.style.display = "none";
    formatGroup.appendChild(logMdRadio);
    formatGroup.appendChild(logTxtRadio);
    formatGroup.appendChild(logJsonlRadio);

    formatSection.appendChild(formatGroup);

    // ═══════════════════════════════════════════════════════════════
    // 3. Separator Dropdown (pairs + .txt only)
    // ═══════════════════════════════════════════════════════════════
    const separatorRow = document.createElement("div");
    separatorRow.className = "llm-export-separator-row";
    separatorRow.style.display = "none";

    const separatorLabel = document.createElement("div");
    separatorLabel.className = "llm-export-dropdown-label";
    separatorLabel.textContent = "🔀 Separator:";
    separatorRow.appendChild(separatorLabel);

    const separatorDropdown = createDropdown("separator", [
        { value: "\n", label: "Newline (\\n)" },
        { value: ",", label: "Comma (,)" },
        { value: ".", label: "Period (.)" },
        { value: " ", label: "Space" },
    ], "\n");
    separatorDropdown.className = "llm-export-dropdown";
    separatorDropdown.id = "llm-export-separator-dropdown";
    separatorRow.appendChild(separatorDropdown);

    formatSection.appendChild(separatorRow);
    body.appendChild(formatSection);

    // ═══════════════════════════════════════════════════════════════
    // 4. Naming Input (differs by mode)
    // ═══════════════════════════════════════════════════════════════
    const namingSection = document.createElement("div");
    namingSection.className = "llm-export-section";
    namingSection.id = "llm-export-naming-section";

    const namingTitle = document.createElement("div");
    namingTitle.className = "llm-export-section-title";
    namingTitle.textContent = "🏷️ Base Name";
    namingSection.appendChild(namingTitle);

    const namingDesc = document.createElement("div");
    namingDesc.className = "llm-export-naming-desc-updated";
    namingDesc.textContent = "Enter a base name. Files will be named: {base_name}{counter}.{ext}";
    namingSection.appendChild(namingDesc);

    const namingInput = document.createElement("input");
    namingInput.type = "text";
    namingInput.className = "llm-export-naming-input";
    namingInput.id = "llm-export-base-name-input";
    namingInput.placeholder = "img-name";
    namingInput.value = "";
    namingSection.appendChild(namingInput);

    body.appendChild(namingSection);

    // ═══════════════════════════════════════════════════════════════
    // 5. Includes Checkboxes (same as before)
    // ═══════════════════════════════════════════════════════════════
    const contentSection = document.createElement("div");
    contentSection.className = "llm-export-section";

    const contentTitle = document.createElement("div");
    contentTitle.className = "llm-export-section-title";
    contentTitle.textContent = "☐ Include";
    contentSection.appendChild(contentTitle);

    const checkboxes = [
        ["User Prompt", "input", true],
        ["Answer (response only)", "output", true],
        ["Thinking/Reasoning", "thinking", true],
        ["System Prompt", "systemPrompt", false],
        ["Model Name", "modelName", false],
        ["Images (always separate files)", "images", true],
        ["Metadata (timestamp)", "metadata", false],
    ];

    const checkboxStates = {};
    for (const [label, key, def] of checkboxes) {
        const row = document.createElement("label");
        row.className = "llm-export-checkbox-row";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = def;
        checkboxStates[key] = cb;
        const labelSpan = document.createElement("span");
        labelSpan.textContent = label;
        row.appendChild(cb);
        row.appendChild(labelSpan);
        contentSection.appendChild(row);
    }
    body.appendChild(contentSection);

    // ═══════════════════════════════════════════════════════════════
    // 6. Destination Folder
    // ═══════════════════════════════════════════════════════════════
    const destSection = document.createElement("div");
    destSection.className = "llm-export-section";

    const destTitle = document.createElement("div");
    destTitle.className = "llm-export-section-title";
    destTitle.textContent = "📁 Destination Folder";
    destSection.appendChild(destTitle);

    const folderRow = document.createElement("div");
    folderRow.className = "llm-export-folder-row";
    const folderInput = document.createElement("input");
    folderInput.type = "text";
    folderInput.className = "llm-export-folder-input";
    folderInput.placeholder = "Server directory path...";
    const savedPath = localStorage.getItem("easyllm_export_path") || "";
    folderInput.value = savedPath;
    folderRow.appendChild(folderInput);

    const browseBtn = document.createElement("button");
    browseBtn.className = "llm-export-browse-btn";
    browseBtn.textContent = "Browse…";
    browseBtn.title = "List server directories";
    browseBtn.onclick = async () => {
        const currentPath = folderInput.value.trim();
        try {
            const resp = await fetch(`/easyllm/export/list_dir?path=${encodeURIComponent(currentPath)}`);
            const data = await resp.json();
            if (data.error) {
                showToast(`❌ ${data.error}`, "error", 3000);
                return;
            }
            pickFolderFromList(data, folderInput);
        } catch (e) {
            showToast("❌ Failed to list directory", "error", 3000);
        }
    };
    folderRow.appendChild(browseBtn);
    destSection.appendChild(folderRow);
    body.appendChild(destSection);

    // ═══════════════════════════════════════════════════════════════
    // 7. Preview Section
    // ═══════════════════════════════════════════════════════════════
    const previewSection = document.createElement("div");
    previewSection.className = "llm-export-section";

    const previewTitle = document.createElement("div");
    previewTitle.className = "llm-export-section-title";
    previewTitle.textContent = "👁️ Preview";
    previewSection.appendChild(previewTitle);

    const previewContent = document.createElement("div");
    previewContent.className = "llm-export-preview";
    previewContent.id = "llm-export-preview-content";

    const first = entries[0];
    const previewText = first?.input
        ? `Input: ${first.input.substring(0, 100)}${first.input.length > 100 ? "…" : ""}`
        : "(empty)";
    previewContent.textContent = `📄 ${entries.length} entries selected\n${previewText}\n📁 Mode: Per-Entry Pairs — numbered files`;
    previewSection.appendChild(previewContent);
    body.appendChild(previewSection);

    dialog.appendChild(body);

    // ── Footer ──
    const footer = document.createElement("div");
    footer.className = "llm-export-dialog-footer";

    const cancelFooterBtn = document.createElement("button");
    cancelFooterBtn.className = "llm-popup-close-btn";
    cancelFooterBtn.textContent = "Cancel";
    cancelFooterBtn.onclick = () => overlay.remove();
    footer.appendChild(cancelFooterBtn);

    const exportBtn = document.createElement("button");
    exportBtn.className = "llm-popup-header-btn";
    exportBtn.textContent = "📥 Export";
    exportBtn.onclick = async () => {
        const outputDir = folderInput.value.trim();
        if (!outputDir) {
            showToast("❌ Please specify an output directory", "error", 3000);
            return;
        }

        localStorage.setItem("easyllm_export_path", outputDir);

        const include = {};
        for (const key of Object.keys(checkboxStates)) {
            include[key] = checkboxStates[key].checked;
        }

        const structure = structDropdown.value;
        const exportFormat = document.querySelector('input[name="export_format_v2"]:checked')?.value || "txt";
        const separator = structure === "pairs" && exportFormat === "txt"
            ? separatorDropdown.value
            : "\n";
        const baseName = namingInput.value.trim() || (structure === "log" ? "llm_generation_log" : "img-name");

        exportBtn.disabled = true;
        exportBtn.textContent = "⏳ Exporting...";

        try {
            const resp = await fetch("/easyllm/export/enhancer_v2", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    entries,
                    options: {
                        output_dir: outputDir,
                        structure,
                        include,
                        format: exportFormat,
                        separator,
                        base_name: baseName,
                        node_label: nodeLabel,
                    },
                }),
            });
            const result = await resp.json();
            if (result.success) {
                const imgMsg = result.images_written > 0 ? ` + ${result.images_written} image(s)` : "";
                showToast(`✅ Exported ${result.file_count} file(s)${imgMsg} to ${result.output_path}`, "success", 4000);
                overlay.remove();
            } else {
                showToast(`❌ Export failed: ${result.error || "Unknown error"}`, "error", 5000);
            }
        } catch (e) {
            showToast(`❌ Export failed: ${e.message}`, "error", 5000);
        } finally {
            exportBtn.disabled = false;
            exportBtn.textContent = "📥 Export";
        }
    };
    footer.appendChild(exportBtn);

    dialog.appendChild(footer);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // ═══════════════════════════════════════════════════════════════
    // Event wiring: structure change → re-render conditional sections
    // ═══════════════════════════════════════════════════════════════
    structDropdown.addEventListener("change", () => {
        const isPairs = structDropdown.value === "pairs";
        updateFormatOptions(isPairs);
        updateSeparatorVisibility(isPairs);
        updateNamingSection(isPairs);
        updatePreview(isPairs);
        updateModeIndicator(isPairs);
    });

    formatGroup.addEventListener("change", () => {
        const isPairs = structDropdown.value === "pairs";
        updateSeparatorVisibility(isPairs);
        updatePreview(isPairs);
    });

    // ── Conditional UI helpers ──
    function updateFormatOptions(isPairs) {
        if (isPairs) {
            pairsTxtRadio.style.display = "";
            pairsMdRadio.style.display = "";
            logMdRadio.style.display = "none";
            logTxtRadio.style.display = "none";
            logJsonlRadio.style.display = "none";
            const selected = document.querySelector('input[name="export_format_v2"]:checked');
            if (!selected || selected.value === "jsonl") {
                pairsTxtRadio.querySelector("input").checked = true;
            }
        } else {
            pairsTxtRadio.style.display = "none";
            pairsMdRadio.style.display = "none";
            logMdRadio.style.display = "";
            logTxtRadio.style.display = "";
            logJsonlRadio.style.display = "";
            const selected = document.querySelector('input[name="export_format_v2"]:checked');
            if (!selected || selected.value === "jsonl") {
                logMdRadio.querySelector("input").checked = true;
            } else if (selected === pairsTxtRadio.querySelector("input") || selected === pairsMdRadio.querySelector("input")) {
                logMdRadio.querySelector("input").checked = true;
            }
        }
    }

    function updateSeparatorVisibility(isPairs) {
        const fmt = document.querySelector('input[name="export_format_v2"]:checked')?.value || "txt";
        separatorRow.style.display = (isPairs && fmt === "txt") ? "" : "none";
    }

    function updateNamingSection(isPairs) {
        const label = namingSection.querySelector(".llm-export-naming-desc-updated");
        const input = document.getElementById("llm-export-base-name-input");
        const secTitle = namingSection.querySelector(".llm-export-section-title");
        if (isPairs) {
            if (secTitle) secTitle.textContent = "🏷️ Base Name";
            if (label) label.textContent = "Enter a base name. Files will be named: {base_name}{counter}.{ext}";
            if (input) input.placeholder = "img-name";
        } else {
            if (secTitle) secTitle.textContent = "🏷️ File Name";
            if (label) label.textContent = "Enter a file name (without extension). Images go to {name}_images/ subfolder.";
            if (input) input.placeholder = "llm_generation_log";
        }
    }

    function updatePreview(isPairs) {
        const previewEl = document.getElementById("llm-export-preview-content");
        if (!previewEl) return;
        const firstEntry = entries[0];
        const previewEntryText = firstEntry?.input
            ? `Input: ${firstEntry.input.substring(0, 100)}${firstEntry.input.length > 100 ? "…" : ""}`
            : "(empty)";
        const fmt = document.querySelector('input[name="export_format_v2"]:checked')?.value || "txt";
        const modeName = isPairs ? "Per-Entry Pairs" : "Single File Log";
        const modeDesc = isPairs
            ? "numbered files per entry + images"
            : `single .${fmt} file + images subfolder`;
        previewEl.textContent = `📄 ${entries.length} entries selected\n${previewEntryText}\n📁 Mode: ${modeName} — ${modeDesc}`;
    }

    function updateModeIndicator(isPairs) {
        const indicator = structSection.querySelector(".llm-export-mode-indicator");
        if (indicator) {
            indicator.textContent = isPairs
                ? "📄 Creates separate numbered files for each entry + images"
                : "📄 Appends all entries into a single file + images subfolder";
        }
    }

    // Initial UI state
    updateModeIndicator(true);
}

/**
 * Create a radio button row inside the export dialog.
 * @param {string} name - Radio group name
 * @param {string} value - Radio value
 * @param {string} label - Display label
 * @param {boolean} checked - Whether selected by default
 * @returns {HTMLLabelElement}
 */
function createRadio(name, value, label, checked) {
    const row = document.createElement("label");
    row.className = "llm-export-radio-row";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = name;
    radio.value = value;
    radio.checked = !!checked;
    row.appendChild(radio);
    const span = document.createElement("span");
    span.textContent = label;
    row.appendChild(span);
    return row;
}

/**
 * Create a styled dropdown (select) element.
 * @param {string} name - Select name attribute
 * @param {Array<{value: string, label: string}>} options - Dropdown options
 * @param {string} defaultVal - Default selected value
 * @returns {HTMLSelectElement}
 */
function createDropdown(name, options, defaultVal) {
    const select = document.createElement("select");
    select.name = name;
    for (const opt of options) {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === defaultVal) {
            option.selected = true;
        }
        select.appendChild(option);
    }
    return select;
}

/**
 * Show a simple folder picker popup listing directory contents.
 * @param {object} data - Response from /easyllm/export/list_dir
 * @param {HTMLInputElement} folderInput - The folder path input to update
 */
function pickFolderFromList(data, folderInput) {
    // Remove any existing picker
    const existing = document.querySelector(".llm-export-folder-picker");
    if (existing) existing.remove();

    const picker = document.createElement("div");
    picker.className = "llm-export-folder-picker";

    // Current path display
    const pathDisplay = document.createElement("div");
    pathDisplay.className = "llm-export-folder-picker-path";
    pathDisplay.textContent = data.path || "/";
    picker.appendChild(pathDisplay);

    // Parent directory link
    if (data.parent) {
        const parentLink = document.createElement("button");
        parentLink.className = "llm-export-folder-picker-item";
        parentLink.textContent = "📂 .. (parent)";
        parentLink.onclick = async () => {
            try {
                const resp = await fetch(`/easyllm/export/list_dir?path=${encodeURIComponent(data.parent)}`);
                const newData = await resp.json();
                if (!newData.error) {
                    picker.remove();
                    pickFolderFromList(newData, folderInput);
                }
            } catch (e) {}
        };
        picker.appendChild(parentLink);
    }

    // Subdirectories
    if (data.contents && data.contents.length > 0) {
        const dirs = data.contents.filter(c => c.type === "dir");
        for (const dir of dirs) {
            const item = document.createElement("button");
            item.className = "llm-export-folder-picker-item";
            item.textContent = `📁 ${dir.name}`;
            const fullPath = data.path ? `${data.path}/${dir.name}` : dir.name;
            item.onclick = async () => {
                // Navigate into
                try {
                    const resp = await fetch(`/easyllm/export/list_dir?path=${encodeURIComponent(fullPath)}`);
                    const newData = await resp.json();
                    if (!newData.error) {
                        picker.remove();
                        pickFolderFromList(newData, folderInput);
                    }
                } catch (e) {}
            };
            // Double-click selects
            item.ondblclick = () => {
                folderInput.value = fullPath;
                picker.remove();
            };
            picker.appendChild(item);
        }
    }

    // Select current folder button
    if (data.path) {
        const selectBtn = document.createElement("button");
        selectBtn.className = "llm-export-folder-picker-item llm-export-folder-picker-select";
        selectBtn.textContent = "✅ Select this folder";
        selectBtn.onclick = () => {
            folderInput.value = data.path;
            picker.remove();
        };
        picker.appendChild(selectBtn);
    }

    // Close button
    const closePickerBtn = document.createElement("button");
    closePickerBtn.className = "llm-export-folder-picker-item";
    closePickerBtn.textContent = "✕ Cancel";
    closePickerBtn.onclick = () => picker.remove();
    picker.appendChild(closePickerBtn);

    // Position near the folder input
    const parent = folderInput.closest(".llm-export-folder-row") || folderInput.parentElement;
    parent.style.position = "relative";
    parent.appendChild(picker);
}
