/**
 * 📚 EasyLLM Prompt Select — Frontend Extension
 * ===============================================
 *
 * Provides dynamic category→prompt_name cascading dropdown,
 * node surface prompt preview, and a refresh button.
 *
 * Imports:
 *     fetchPrompts – from ./api.js
 *     findWidgetByName – from ./constants.js
 */

import { fetchPrompts } from "./api.js";
import { findWidgetByName } from "./constants.js";

// ── Constants ──────────────────────────────────────────────────────────

const NODE_NAME = "LLM_PromptSelect";
const RANDOM_LABEL = "🔀 Random";
const MOVING_LABEL = "🔄 Moving";
const ALL_LABEL = "All";
const PREVIEW_MAX_CHARS = 280;
const PREVIEW_MAX_LINES = 4;

// ── Module-level prompt cache (minimises redundant fetches) ────────────
// Invalidate by setting to null (done by updatePromptNames after fetching).
let _promptStructCache = null;

// ── Extension Registration ─────────────────────────────────────────────

app.registerExtension({
    name: "Comfy.EasyLLM.PromptSelect",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        // ────────────────────────────────────────────────────────────────
        // onNodeCreated: set up cascading widgets + callbacks
        // ────────────────────────────────────────────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const catWidget = findWidgetByName(this, "category");
            const nameWidget = findWidgetByName(this, "prompt_name");

            if (!catWidget || !nameWidget) return result;

            // Store reference to the node for async callbacks
            const node = this;
            node._previewText = "";

            // ── Category change: re-filter prompt_name dropdown ──
            const prevCatCb = catWidget.callback;
            catWidget.callback = function (val) {
                if (prevCatCb) prevCatCb.call(this, val);
                // 'this' in ComfyUI widget callbacks is the node
                updatePromptNames(this, val);
            };

            // ── Prompt name change: update preview text ──
            const prevNameCb = nameWidget.callback;
            nameWidget.callback = function (val) {
                if (prevNameCb) prevNameCb.call(this, val);
                updatePreviewText(this);
            };

            // ── Initial population (deferred to let widgets settle) ──
            setTimeout(() => {
                updatePromptNames(node, catWidget.value);
            }, 50);

            return result;
        };

        // ────────────────────────────────────────────────────────────────
        // onDrawForeground: node surface preview + refresh icon
        // ────────────────────────────────────────────────────────────────
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            origOnDrawForeground?.apply(this, arguments);

            const node = this;
            const w = node.size[0];

            // ── Refresh icon (↻) in top-right corner ──
            ctx.save();
            ctx.font = "12px sans-serif";
            ctx.textAlign = "right";
            ctx.textBaseline = "top";
            ctx.fillStyle = "rgba(167, 139, 250, 0.45)";
            ctx.fillText("↻", w - 6, 4);
            ctx.restore();

            // ── Prompt preview at the bottom of the node ──
            const preview = node._previewText;
            if (!preview) return;

            ctx.save();

            const pad = 6;
            const fontSize = 10;
            const maxWidth = w - pad * 2;
            const lineHeight = fontSize + 2;

            ctx.font = `bold ${fontSize}px monospace`;

            // Truncate very long text
            const displayText = preview.length > PREVIEW_MAX_CHARS
                ? preview.substring(0, PREVIEW_MAX_CHARS) + "…"
                : preview;

            // Wrap to fit node width
            const lines = _wrapText(ctx, displayText, maxWidth);
            const clampedLines = lines.slice(0, PREVIEW_MAX_LINES);

            const boxH = clampedLines.length * lineHeight + pad * 2;
            const boxY = Math.max(node.size[1] - boxH - 2, 30); // don't overlap title

            // ── Background pill ──
            ctx.fillStyle = "rgba(167, 139, 250, 0.07)";
            ctx.beginPath();
            if (typeof ctx.roundRect === "function") {
                ctx.roundRect(1, boxY, w - 2, boxH, 4);
            } else {
                ctx.rect(1, boxY, w - 2, boxH);
            }
            ctx.fill();

            // ── Text ──
            ctx.fillStyle = "rgba(167, 139, 250, 0.55)";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";

            clampedLines.forEach((line, i) => {
                ctx.fillText(line, pad, boxY + pad + i * lineHeight);
            });

            // Show truncated indicator if content was shortened
            if (clampedLines.length < lines.length) {
                ctx.fillStyle = "rgba(167, 139, 250, 0.35)";
                ctx.fillText("…", pad, boxY + pad + clampedLines.length * lineHeight);
            }

            ctx.restore();
        };

        // ────────────────────────────────────────────────────────────────
        // onMouseDown: detect click on refresh icon
        // ────────────────────────────────────────────────────────────────
        const origOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localX, localY) {
            const ret = origOnMouseDown
                ? origOnMouseDown.apply(this, arguments)
                : undefined;

            // Check if click is within the refresh icon area (top-right 24×20)
            if (localX > this.size[0] - 24 && localY < 20) {
                const catWidget = findWidgetByName(this, "category");
                if (catWidget) {
                    // Invalidate cache and refresh
                    _promptStructCache = null;
                    updatePromptNames(this, catWidget.value);
                }
            }

            return ret;
        };
    },
});

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/**
 * Update the prompt_name widget's dropdown options based on the
 * selected category. Fetches fresh prompt data from the backend.
 *
 * @param {object} node – ComfyUI node instance
 * @param {string} category – Selected category value
 */
async function updatePromptNames(node, category) {
    try {
        const struct = await _fetchWithCache();
        const allPrompts = struct.prompts || [];

        // Filter by category
        const filtered = category === ALL_LABEL
            ? allPrompts
            : allPrompts.filter(p => p.category === category);

        // Build unique name list: Random → Moving → named prompts
        const seen = new Set();
        const names = [RANDOM_LABEL, MOVING_LABEL];
        for (const p of filtered) {
            const n = p.name?.trim();
            if (n && !seen.has(n)) {
                names.push(n);
                seen.add(n);
            }
        }

        const nameWidget = findWidgetByName(node, "prompt_name");
        if (!nameWidget) return;

        // Store full name list for preview lookup
        node._allPromptNames = names;

        // ── Update the ComfyUI widget options ──
        nameWidget.options.values = names;

        // If current selection is no longer valid, reset to first option
        if (!names.includes(nameWidget.value)) {
            nameWidget.value = names[0] || RANDOM_LABEL;
        }

        // Trigger canvas redraw
        node.setDirtyCanvas(true, true);

        // Refresh preview text
        updatePreviewText(node);
    } catch (err) {
        console.warn("[LLM Prompt Select] Failed to update prompt names:", err);
    }
}

/**
 * Update the node surface preview text based on current
 * category + prompt_name selection.
 *
 * @param {object} node – ComfyUI node instance
 */
async function updatePreviewText(node) {
    const catWidget = findWidgetByName(node, "category");
    const nameWidget = findWidgetByName(node, "prompt_name");
    if (!catWidget || !nameWidget) return;

    const cat = catWidget.value;
    const name = nameWidget.value;

    // Random / Moving mode — show descriptive label
    if (name === RANDOM_LABEL) {
        node._previewText = "🔀 Random selection — changes each execution";
        node.setDirtyCanvas(true, true);
        return;
    }
    if (name === MOVING_LABEL) {
        node._previewText = "🔄 Sequential cycle through prompts";
        node.setDirtyCanvas(true, true);
        return;
    }

    // Named prompt — look up the prompt text
    try {
        const struct = await _fetchWithCache();
        const prompts = struct.prompts || [];

        const filtered = cat === ALL_LABEL
            ? prompts
            : prompts.filter(p => p.category === cat);

        const match = filtered.find(p => p.name === name);
        node._previewText = match?.prompt
            || prompts.find(p => p.name === name)?.prompt
            || "";
    } catch (err) {
        node._previewText = "";
    }

    node.setDirtyCanvas(true, true);
}

/**
 * Fetch prompts with a simple module-level cache so consecutive
 * calls (category change → preview update) don't double-fetch.
 *
 * @returns {Promise<{categories: string[], prompts: object[]}>}
 */
async function _fetchWithCache() {
    if (_promptStructCache) return _promptStructCache;
    const struct = await fetchPrompts();
    _promptStructCache = struct;
    return struct;
}

/**
 * Wrap text to fit within a pixel width using the current canvas context font.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} text
 * @param {number} maxWidth – available pixel width
 * @returns {string[]} lines
 */
function _wrapText(ctx, text, maxWidth) {
    const lines = [];
    let current = "";

    for (const char of text) {
        // Preserve explicit newlines
        if (char === "\n") {
            if (current) lines.push(current);
            current = "";
            continue;
        }
        const test = current + char;
        if (ctx.measureText(test).width > maxWidth && current.length > 0) {
            lines.push(current);
            current = char;
        } else {
            current = test;
        }
    }
    if (current) lines.push(current);

    return lines;
}
