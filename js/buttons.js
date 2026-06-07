/**
 * EasyLLM — Canvas node DOM button widget and widget hiding
 *
 * - createButtonDOMWidget: Creates the "Open LLM Lab & Chat" / "📋 View Output History"
 *   and "Manage Prompts" buttons as a LiteGraph DOM widget.
 * - hideCanvasWidgets: Hides setting widget rows (prompt_template,
 *   system_prompt_text, max_length, temperature) from the canvas node.
 * - refreshButtonLabel: Updates button text/behavior when mode changes dynamically.
 */

import { VISIBLE_WIDGET_NAMES, VISIBLE_WIDGET_NAMES_GGUF } from "./constants.js";
import { openChatPopup, openOutputHistoryPopup, openModelBrowserPopup } from "./popup.js";
import { openPromptManagerDialog } from "./editor.js";

// ────────────────────────────────────────────────────────────────────────
// Canvas Node: Hide widget rows not in VISIBLE_WIDGET_NAMES
// ────────────────────────────────────────────────────────────────────────

export function hideCanvasWidgets(node) {
    const visibleNames = node.isEasyLLMGGUF ? VISIBLE_WIDGET_NAMES_GGUF : VISIBLE_WIDGET_NAMES;
    for (const w of (node.widgets || [])) {
        if (visibleNames.includes(w.name)) continue;
        if (w.name.startsWith("_")) continue; // Hidden widgets already invisible
        if (w.name === "llm_chat_buttons") continue; // DOM widget — not a widget row

        // Override type/computeSize/draw to hide from canvas rendering (survives serialization).
        w.type = "converted-widget";
        w.hidden = true;
        w.computeSize = () => [0, -4];
        w.draw = () => {};
    }
}

// ────────────────────────────────────────────────────────────────────────
// Canvas Node: Create LLM Lab + Manage Prompts as DOM widget buttons
// DOM widget — created immediately via addDOMWidget.
// ────────────────────────────────────────────────────────────────────────

/**
 * Create a styled button with shared CSS base and per-button overrides.
 * @param {object} opts
 * @param {string} opts.text - Button label
 * @param {string} opts.title - Tooltip text
 * @param {string} opts.bg - Background color
 * @param {string} opts.border - Border color
 * @param {string} opts.hoverBg - Hover background color
 * @param {Function} opts.onClick - Click handler
 * @returns {HTMLButtonElement}
 */
function createStyledButton({ text, title, bg, border, hoverBg, onClick }) {
    const btn = document.createElement("button");
    btn.textContent = text;
    btn.title = title;
    btn.style.cssText = [
        "flex: 1; color: #e0e0f0;",
        `background: ${bg};`,
        `border: 1px solid ${border}; border-radius: 4px;`,
        "padding: 6px 8px; cursor: pointer;",
        "font-family: monospace; font-size: 12px;",
        "transition: background 0.1s;",
    ].join(" ");
    btn.addEventListener("mouseenter", () => { btn.style.background = hoverBg; });
    btn.addEventListener("mouseleave", () => { btn.style.background = bg; });
    btn.type = "button";
    btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        onClick(e);
    });
    return btn;
}

export function createButtonDOMWidget(node) {
    const nodeId = node.id ?? "?";
    console.debug(`[LLM Chat] createButtonDOMWidget called for node ${nodeId} (type=${node.type})`);

    // Check if already created and element is still attached to DOM.
    const existing = node.widgets?.find(w => w.name === "llm_chat_buttons");
    if (existing) {
        console.debug(`[LLM Chat] Node ${nodeId} — existing llm_chat_buttons widget found, isConnected=${existing.element?.isConnected}`);
        // If the element is still connected (in the DOM), no-op.
        // If the Vue frontend has detached the element during re-render,
        // remove the stale widget from the array so we can re-create it
        // with a fresh DOM element.
        if (existing.element?.isConnected) {
            return existing;
        }
        // Element was detached — remove the stale widget before re-creating
        const idx = node.widgets.indexOf(existing);
        if (idx !== -1) {
            node.widgets.splice(idx, 1);
        }
        console.debug(`[LLM Chat] Node ${nodeId} — Re-creating detached button DOM widget`);
    }

    // Create container for both buttons (styling via CSS class .llm-chat-button-container)
    const container = document.createElement("div");
    container.className = "llm-chat-button-container";

    // ── EasyLLM / Output History button — label/behavior depends on mode ──
    const isLLMChat = node.type === "EasyLLM" || node.comfyClass?.nodeData?.name === "EasyLLM";
    const isGGUF = node.isEasyLLMGGUF || node.type === "EasyLLMGGUF";
    if (isLLMChat || isGGUF) {
        const modeW = node.widgets?.find(w => w.name === "mode");
        const isEnhancer = modeW && modeW.value === "enhancer";

        // ── Mode indicator pill ──
        const modePill = document.createElement("span");
        modePill.className = "llm-mode-pill";
        modePill.textContent = isEnhancer ? "✨" : (isGGUF ? "⚡" : "💬");
        modePill.title = isEnhancer
            ? "Enhancer mode: direct prompt-to-output pipeline"
            : (isGGUF ? "GGUF Chat mode: interactive conversation via popup (C++ engine)" : "Chat mode: interactive conversation via popup");
        container.appendChild(modePill);

        const labBtn = createStyledButton({
            text: isEnhancer ? "📋 View History" : "💬 Open Chat",
            title: isEnhancer
                ? "View, scroll, and copy recent enhanced prompt outputs"
                : "Open the interactive chat popup",
            bg: isGGUF ? "#6a3a8a" : "#3a5a8a",
            border: isGGUF ? "#8a5aaa" : "#5a7aaa",
            hoverBg: isGGUF ? "#7a4a9a" : "#4a6a9a",
            onClick: () => {
                if (isEnhancer) {
                    openOutputHistoryPopup(node);
                } else {
                    openChatPopup(node);
                }
            },
        });
        container.appendChild(labBtn);

        // 📁 Models button (GGUF-only — opens model browser popup)
        if (isGGUF) {
            const modelBtn = createStyledButton({
                text: "📁 Models",
                title: "Browse and select GGUF model",
                bg: "#6a3a8a",
                border: "#8a5aaa",
                hoverBg: "#7a4a9a",
                onClick: () => openModelBrowserPopup(node),
            });
            container.appendChild(modelBtn);
        }
    }

    // "Manage Prompts" button (nodes with prompt_template)
    const hasTemplate = node.widgets?.find(w => w.name === "prompt_template");
    if (hasTemplate) {
        const mgrBtn = createStyledButton({
            text: "⚙ Manage Prompts...",
            title: "Add, edit, or delete system prompt templates",
            bg: "#3a3a5a",
            border: "#5a5a7a",
            hoverBg: "#4a4a7a",
            onClick: () => openPromptManagerDialog(),
        });
        container.appendChild(mgrBtn);
    }

    // Create the DOM widget — DOM is created immediately.
    // getMinHeight reserves vertical space; hideOnZoom: false keeps buttons visible when zoomed out.
    console.debug(`[LLM Chat] Node ${nodeId} — calling addDOMWidget`);
    const widget = node.addDOMWidget("llm_chat_buttons", "llm_chat_buttons", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 40,
    });
    console.debug(`[LLM Chat] Node ${nodeId} — addDOMWidget returned:`, widget ? `widget type=${widget.type}, element exists=${!!widget.element}, element.isConnected=${widget.element?.isConnected}` : "UNDEFINED!");

    // ── Pin widget width to node width ───────────────────────────────────
    try {
        Object.defineProperty(widget, "width", {
            configurable: true,
            enumerable: true,
            get() {
                const w = node.size && node.size[0] > 0 ? node.size[0] : 300;
                return w;
            },
            set(_v) { /* ignore — width is derived from the node size */ },
        });
    } catch (e) {
        console.warn("[LLM Chat] Could not pin widget width:", e);
    }

    console.debug(`[LLM Chat] Node ${nodeId} — widget creation complete, total widgets=${node.widgets?.length}`);
    return widget;
}

/**
 * Refresh the LLM Lab button label and click handler based on current mode.
 * Called when mode changes dynamically (from mode widget callback).
 * Replaces the existing button with a fresh one to rebind the click handler.
 */
export function refreshButtonLabel(node) {
    const container = node.widgets?.find(w => w.name === "llm_chat_buttons");
    if (!container?.element) return;
    const btn = container.element.querySelector("button");
    if (!btn) return;

    const modeW = node.widgets?.find(w => w.name === "mode");
    const isEnhancer = modeW && modeW.value === "enhancer";
    const isGGUF = node.isEasyLLMGGUF;

    // Create replacement button with correct label and handler
    const newBtn = document.createElement("button");
    newBtn.textContent = isEnhancer ? "📋 View History" : "💬 Open Chat";
    newBtn.title = isEnhancer
        ? "View, scroll, and copy recent enhanced prompt outputs"
        : "Open the interactive chat popup";
    newBtn.style.cssText = btn.style.cssText;
    newBtn.type = "button";
    const hoverBg = isGGUF ? "#7a4a9a" : "#4a6a9a";
    const normalBg = isGGUF ? "#6a3a8a" : "#3a5a8a";
    newBtn.addEventListener("mouseenter", () => { newBtn.style.background = hoverBg; });
    newBtn.addEventListener("mouseleave", () => { newBtn.style.background = normalBg; });
    newBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (isEnhancer) {
            openOutputHistoryPopup(node);
        } else {
            openChatPopup(node);
        }
    });

    btn.parentNode.replaceChild(newBtn, btn);
}
