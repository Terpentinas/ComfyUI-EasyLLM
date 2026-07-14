/**
 * EasyLLM — Canvas node DOM button widget and widget hiding
 *
 * - createButtonDOMWidget: Creates the "Open LLM Lab & Chat" / "📋 View Output History"
 *   and "Manage Prompts" buttons as a LiteGraph DOM widget.
 * - hideCanvasWidgets: Hides setting widget rows (max_length, temperature, etc.)
 *   from the canvas node.
 * - refreshButtonLabel: Updates button text/behavior when mode changes dynamically.
 */

import { VISIBLE_WIDGET_NAMES, VISIBLE_WIDGET_NAMES_GGUF, getWidgetEl } from "./constants.js";
import { openChatPopup, openOutputHistoryPopup, openModelBrowserPopup, openSettingsPopup, openDatabaseManagerPopup } from "./popup.js";
import { openPromptManagerDialog } from "./editor.js";

// ────────────────────────────────────────────────────────────────────────
// Canvas Node: Hide widget rows not in VISIBLE_WIDGET_NAMES
// ────────────────────────────────────────────────────────────────────────

export function hideCanvasWidgets(node, visibleNames) {
    if (!visibleNames) {
        visibleNames = node.isEasyLLMGGUF ? VISIBLE_WIDGET_NAMES_GGUF : VISIBLE_WIDGET_NAMES;
    }
    for (const w of (node.widgets || [])) {
        if (visibleNames.includes(w.name)) continue;
        if (w.name.startsWith("_")) continue; // Hidden widgets already invisible
        if (w.name === "llm_chat_buttons") continue; // DOM widget — not a widget row

        // Override type/computeSize/draw to hide from canvas rendering (survives serialization).
        w.type = "converted-widget";
        w.hidden = true;
        w.computeSize = () => [0, 0];
        w.draw = () => {};
        // Also hide the widget's DOM element so it contributes zero layout space
        const el = getWidgetEl(w);
        if (el) {
            el.classList.add("llm-widget-hidden");
            el.style.display = "none";
        }
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
function createStyledButton({ text, title, onClick, extraClass }) {
    const btn = document.createElement("button");
    btn.className = ["llm-canvas-btn", extraClass].filter(Boolean).join(" ");
    btn.textContent = text;
    btn.title = title;
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

    // Create container for both rows (styling via CSS class .llm-chat-button-container)
    const container = document.createElement("div");
    container.className = "llm-chat-button-container";

    // Create two rows inside the container
    const rowTop = document.createElement("div");
    rowTop.className = "llm-button-row llm-button-row-top";
    const rowBottom = document.createElement("div");
    rowBottom.className = "llm-button-row llm-button-row-bottom";

    // ── EasyLLM / Output History button — label/behavior depends on mode ──
    const isLLMChat = node.type === "EasyLLM" || node.comfyClass?.nodeData?.name === "EasyLLM";
    const isGGUF = node.isEasyLLMGGUF || node.type === "EasyLLMGGUF";
    if (isLLMChat || isGGUF) {
        const modeW = node.widgets?.find(w => w.name === "mode");
        const isEnhancer = modeW && modeW.value === "enhancer";

        // ── Row 1: Primary action buttons (Open Chat, Models, mode pill) ──
        const labBtn = createStyledButton({
            text: isEnhancer ? "📋 View History" : "💬 Open Chat",
            title: isEnhancer
                ? "View, scroll, and copy recent enhanced prompt outputs"
                : "Open the interactive chat popup",
            onClick: () => {
                if (isEnhancer) {
                    openOutputHistoryPopup(node);
                } else {
                    openChatPopup(node);
                }
            },
            extraClass: "llm-chat-main-btn",
        });
        rowTop.appendChild(labBtn);

        // 📁 Models button (GGUF-only — opens model browser popup)
        if (isGGUF) {
            const modelBtn = createStyledButton({
                text: "📁 Models",
                title: "Browse and select GGUF model",
                onClick: () => openModelBrowserPopup(node),
            });
            rowTop.appendChild(modelBtn);
        }

        // ── Mode toggle arrow badge (right side of top row) ──
        const modePill = document.createElement("span");
        modePill.className = "llm-mode-pill";
        modePill.textContent = isEnhancer ? "<" : ">";
        modePill.title = isEnhancer
            ? "Enhancer mode - click to switch to Chat"
            : "Chat mode - click to switch to Enhancer";
        modePill.dataset.mode = modeW?.value || "chat";
        rowTop.appendChild(modePill);

        // ── Mode toggle: click toggles chat ↔ enhancer ──
        modePill.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const modeW2 = node.widgets?.find(w => w.name === "mode");
            if (!modeW2) return;
            const newVal = modeW2.value === "enhancer" ? "chat" : "enhancer";
            modeW2.value = newVal;
            if (modeW2.callback) modeW2.callback(newVal);
        });

        // ── Row 2: Utility action buttons (Settings, Database, Prompt Library) ──
        // ⚙️ Gear icon — opens settings-only popup (available in all modes)
        const settingsBtn = document.createElement("button");
        settingsBtn.className = "llm-settings-btn";
        settingsBtn.textContent = "⚙️";
        settingsBtn.title = "Open settings (temperature, VRAM, sampling params)";
        settingsBtn.type = "button";
        settingsBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            openSettingsPopup(node);
        });
        rowBottom.appendChild(settingsBtn);

        // 🗄️ DataBase icon — opens database manager popup (available in all modes)
        const dbBtn = document.createElement("button");
        dbBtn.className = "llm-settings-btn";
        dbBtn.textContent = "🗄️ DataBase";
        dbBtn.title = "Open database manager";
        dbBtn.type = "button";
        dbBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            openDatabaseManagerPopup(node);
        });
        rowBottom.appendChild(dbBtn);

        // 📚 Prompt Library button (available on all EasyLLM/EasyLLMGGUF nodes)
        const mgrBtn = createStyledButton({
            text: "📚 Prompt Library",
            title: "Open the Prompt Library to manage system prompt templates",
            onClick: () => openPromptManagerDialog(),
        });
        rowBottom.appendChild(mgrBtn);
    }

    // Append rows to container
    container.appendChild(rowTop);
    container.appendChild(rowBottom);

    // Create the DOM widget — DOM is created immediately.
    // getMinHeight reserves vertical space; hideOnZoom: false keeps buttons visible when zoomed out.
    console.debug(`[LLM Chat] Node ${nodeId} — calling addDOMWidget`);
    const widget = node.addDOMWidget("llm_chat_buttons", "llm_chat_buttons", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 80,
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
    const btn = container.element.querySelector(".llm-chat-main-btn");
    if (!btn) return;

    const modeW = node.widgets?.find(w => w.name === "mode");
    const isEnhancer = modeW && modeW.value === "enhancer";
    const isGGUF = node.isEasyLLMGGUF;

    // Create replacement button with correct label and handler
    const newBtn = document.createElement("button");
    newBtn.className = "llm-canvas-btn llm-chat-main-btn";
    newBtn.textContent = isEnhancer ? "📋 View History" : "💬 Open Chat";
    newBtn.title = isEnhancer
        ? "View, scroll, and copy recent enhanced prompt outputs"
        : "Open the interactive chat popup";
    newBtn.type = "button";
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

    // Sync the mode badge arrow and color
    const pill = container.element.querySelector(".llm-mode-pill");
    if (pill) {
        const modeW3 = node.widgets?.find(w => w.name === "mode");
        const isEnh3 = modeW3 && modeW3.value === "enhancer";
        pill.textContent = isEnh3 ? "<" : ">";
        pill.title = isEnh3
            ? "Enhancer mode - click to switch to Chat"
            : "Chat mode - click to switch to Enhancer";
        pill.dataset.mode = isEnh3 ? "enhancer" : "chat";
    }
}
