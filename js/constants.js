/**
 * EasyLLM — Constants, CSS injection, and shared helpers
 *
 * Separated from the main extension file so it can be imported by
 * multiple focused modules without circular dependencies.
 */

// ── Node names this extension applies to ────────────────────────────────
export const NODE_NAMES = ["EasyLLM", "EasyLLMText", "EasyLLMGGUF"];

// ── Visible widget names ────────────────────────────────────────────────
// Widgets kept visible on canvas for direct editing without popup.
// "text" enables typing on the node; socket inputs (clip, system_prompt) are not widget rows.
// Order matches the logical grouping: Setup → Persona → Interaction → Tuning → Hardware
export const VISIBLE_WIDGET_NAMES = [
    // Section 1: Setup
    "mode",
    // Section 2: Persona & Input
    "text",
    "prompt_template",
    "system_prompt_text",
    // Section 3: Tuning
    "temperature",
    "max_length",
    "seed",
    // Section 4: Hardware (Less Used)
    "vram_mode",
    "use_mlock",
];

// ── Visible widget names for GGUF nodes ────────────────────────────────
// Includes GGUF-specific widgets (model_path, n_gpu_layers, n_ctx, chat_template)
// in addition to the shared settings widgets.
// Order matches the logical grouping: Setup → Persona → Interaction → Tuning → Hardware
export const VISIBLE_WIDGET_NAMES_GGUF = [
    // Section 1: Setup
    "model_path",
    "mode",
    // Section 2: Persona & Input
    "text",
    "prompt_template",
    "system_prompt_text",
    // Section 3: Tuning
    "temperature",
    "max_length",
    "seed",
    // Section 4: Hardware (Less Used)
    "chat_template",
    "n_ctx",
    "n_gpu_layers",
    "vram_mode",
    "use_mlock",
];

// ── Global popup tracking (only one popup open at a time) ───────────────
let _activePopupNode = null;

export function getActivePopupNode() {
    return _activePopupNode;
}

export function setActivePopupNode(node) {
    _activePopupNode = node;
}

export function clearActivePopupNode() {
    _activePopupNode = null;
}

// ── DOM element helper: supports both new (element) and old (inputEl) APIs ──
export function getWidgetEl(widget) {
    return widget?.element || widget?.inputEl || null;
}

// ── Widget lookup helper: find a widget by its `.name` property ──
export function findWidgetByName(node, name) {
    return node.widgets?.find(w => w.name === name);
}

// ── Inject CSS styles ──
// Source of truth: js/llm_chat.css (served as /extensions/llm-chat/llm_chat.css).
// We load it via <link> to eliminate duplication drift risk.
// A minimal inline fallback ensures critical layout styles are available
// even if the CSS file fails to load.
(function loadStylesheet() {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/extensions/llm-chat/llm_chat.css";
    link.onload = () => console.log("[LLM Chat] CSS loaded from stylesheet");
    link.onerror = () => console.warn("[LLM Chat] CSS stylesheet failed to load — using inline fallback");
    document.head.appendChild(link);

    // Minimal fallback: only critical positioning/z-index/font styles.
    // All visual polish lives in llm_chat.css (the single source of truth).
    const fallback = document.createElement("style");
    fallback.textContent = `
.llm-popup-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6); z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    font-family: monospace; font-size: 13px; color: #e0e0f0;
}
.llm-popup-panel {
    background: #2a2a3e; border: 1px solid #5a5a7a; border-radius: 8px;
    padding: 0; width: 760px; max-width: 90vw; max-height: 90vh;
    display: flex; flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5); overflow: hidden;
    box-sizing: border-box;
}
.llm-manager-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6); z-index: 10000;
    display: flex; align-items: center; justify-content: center;
    font-family: monospace; font-size: 13px; color: #e0e0f0;
}
.llm-chat-button-container {
    display: flex; gap: 6px; padding: 4px 8px;
    align-items: center; width: 100%; box-sizing: border-box;
}
.llm-toast {
    position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
    padding: 10px 24px; border-radius: 8px; font-family: monospace;
    font-size: 13px; color: #e0e0f0; z-index: 99999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5); pointer-events: none;
    max-width: 80vw; text-align: center;
}
`;
    document.head.appendChild(fallback);
})();
