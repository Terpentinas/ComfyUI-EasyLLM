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
// Source of truth: js/llm_chat.css (co-located in the js/ directory).
// We load it via <link> to eliminate duplication drift risk.
// The URL is resolved dynamically via import.meta.url so it works
// regardless of the custom_node folder name on any machine.
// A comprehensive inline fallback ensures all popup components render
// properly even if the external CSS file fails to load.
(function loadStylesheet() {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    // Resolve llm_chat.css relative to this module's URL — works with ANY folder name.
    link.href = new URL("llm_chat.css", import.meta.url).href;
    link.onload = () => console.log("[LLM Chat] CSS loaded from stylesheet");
    link.onerror = () => console.warn("[LLM Chat] CSS stylesheet failed to load — using inline fallback");
    document.head.appendChild(link);

    // Comprehensive fallback: covers ALL popup component styles.
    // If the external CSS fails to load (wrong folder name, network issue, etc.),
    // this ensures popups still render with proper sizing, colors, and layout.
    const fallback = document.createElement("style");
    fallback.textContent = `
/* ═══════════════════════════════════════════════════════════════════════
   FALLBACK STYLES — Activated when llm_chat.css fails to load.
   Covers all popup components: chat popup, prompt manager, model browser,
   toast notifications, and interactive elements.
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Popup Overlay (full-screen backdrop) ──────────────────────────── */
.llm-popup-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; color: #e8e8ed;
}

/* ── Popup Panel ─────────────────────────────────────────────────────── */
.llm-popup-panel {
    background: #1a1b1e; border: 1px solid #2d3139; border-radius: 10px;
    padding: 0; width: 760px; max-width: 90vw; max-height: 98vh;
    min-width: 400px; min-height: 300px;
    display: flex; flex-direction: column;
    box-shadow: 0 20px 40px rgba(0,0,0,0.65); overflow: hidden;
    box-sizing: border-box; resize: both;
}

/* ── Popup Header ────────────────────────────────────────────────────── */
.llm-popup-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2a2d32;
    background: #1e1f22; flex-shrink: 0;
}
.llm-popup-header-title {
    font-size: 14px; font-weight: 600; letter-spacing: -0.01em;
}
.llm-popup-header-actions {
    display: flex; gap: 4px; align-items: center; flex-shrink: 0;
}
.llm-popup-header-btn {
    background: #22252e; color: #9a9aa5;
    border: 1px solid #2a2d32; border-radius: 6px;
    padding: 4px 10px; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 11px; white-space: nowrap;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-popup-header-btn:hover {
    color: #e8e8ed; border-color: #3a3e45; background: #2a2d36;
}

/* ── Popup Close Button ──────────────────────────────────────────────── */
.llm-popup-close-btn {
    background: rgba(239,68,68,0.15); color: #f87171;
    border: none; border-radius: 50%; width: 32px; height: 32px;
    padding: 0; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 14px; font-weight: 600;
    display: inline-flex; align-items: center; justify-content: center;
    line-height: 1; transition: background 0.15s, color 0.15s;
}
.llm-popup-close-btn:hover {
    background: rgba(239,68,68,0.35); color: #fca5a5;
}

/* ── Popup Body (scrollable content area) ──────────────────────────── */
.llm-popup-body {
    flex: 1; overflow-y: auto; padding: 12px 16px;
    display: flex; flex-direction: column; gap: 12px; min-height: 0;
}

/* ── Chat History Container ──────────────────────────────────────────── */
.llm-popup-history {
    flex: 1; overflow-y: auto; background: transparent;
    border: none; border-radius: 0; padding: 4px 0;
    min-height: 80px; user-select: text; -webkit-user-select: text;
}

/* ── Empty Chat Placeholder ──────────────────────────────────────────── */
.llm-popup-history-empty {
    color: #5c5c66; text-align: center;
    padding: 40px 12px; font-size: 12px; font-style: italic;
}

/* ── Message Bubbles ─────────────────────────────────────────────────── */
.llm-chat-bubble {
    position: relative; margin-bottom: 6px;
    padding: 6px 10px; border-radius: 6px;
    word-break: break-word; white-space: pre-wrap;
    transition: background 0.15s;
}
.llm-chat-bubble:last-child { margin-bottom: 0; }
.llm-chat-bubble-user {
    background: #1e222b; text-align: right; padding: 12px;
    border-radius: 4px;
}
.llm-chat-bubble-user:hover { background: rgba(30,34,43,0.8); }
.llm-chat-bubble-assistant {
    background: transparent; text-align: left; padding: 8px 12px;
}
.llm-chat-bubble-assistant:hover { background: rgba(255,255,255,0.04); }

.llm-chat-bubble-label {
    font-weight: 500; font-size: 10px; opacity: 0.6;
    margin-bottom: 4px; letter-spacing: 0.02em;
}
.llm-chat-bubble-label-left { text-align: left; color: #34d399; opacity: 0.9; font-weight: 600; }
.llm-chat-bubble-label-right { text-align: right; color: #9a9aa5; }
.llm-chat-bubble-text {
    word-break: break-word; white-space: pre-wrap;
    user-select: text; -webkit-user-select: text;
}

.llm-chat-model-badge {
    font-size: 1em; color: #a78bfa; margin-left: 8px; font-weight: normal;
}

/* ── Bubble Timestamp ─────────────────────────────────────────────── */
.llm-chat-bubble-timestamp {
    font-size: 10px; color: #5c5c66; margin-top: 2px; opacity: 0.7;
}

/* ── Bubble Action Buttons ─────────────────────────────────────────── */
.llm-chat-bubble-actions {
    display: flex; gap: 2px; margin-top: 4px;
    opacity: 0; transition: opacity 0.15s;
}
.llm-chat-bubble:hover .llm-chat-bubble-actions { opacity: 1; }
.llm-chat-action-btn {
    background: none; border: none; color: #6c6c78;
    cursor: pointer; padding: 2px 4px;
    font-size: 13px; line-height: 1; border-radius: 3px;
    transition: background 0.1s, color 0.1s;
}
.llm-chat-action-btn:hover {
    background: rgba(255,255,255,0.08); color: #e8e8ed;
}

/* ── Bubble Error State ────────────────────────────────────────────── */
.llm-chat-bubble-error {
    background: rgba(200,74,74,0.08) !important;
    border: 1px solid rgba(200,74,74,0.2) !important;
    border-radius: 4px !important;
}
.llm-chat-bubble-error-icon { font-size: 12px; margin-right: 4px; opacity: 0.7; }
.llm-chat-retry-btn {
    background: #22252e; color: #9a9aa5;
    border: 1px solid #3a3e45; border-radius: 4px;
    padding: 4px 10px; cursor: pointer; font-size: 11px;
    font-family: inherit; margin-top: 6px;
}
.llm-chat-retry-btn:hover { background: #2a2d36; color: #e8e8ed; }

/* ── Timing Badge ─────────────────────────────────────────────────── */
.llm-chat-timing-badge {
    font-size: 10px; color: #94a3b8; margin-top: 2px; opacity: 0.8; cursor: help;
}

/* ── Image Thumbnail in Bubbles ───────────────────────────────────── */
.llm-chat-bubble-image { margin: 6px 0; max-width: 100%; }
.llm-chat-bubble-image img {
    max-width: 240px; max-height: 180px; border-radius: 6px;
    border: 1px solid #3a3e45; cursor: zoom-in; display: block;
    background: #131417; object-fit: contain;
}

/* ── Input Row ──────────────────────────────────────────────────────── */
.llm-popup-input-row {
    display: flex; gap: 4px; align-items: flex-start; flex-shrink: 0;
    background: #1e222b; border: 1px solid #3a3e45;
    border-radius: 8px; padding: 6px 8px;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.llm-popup-input-row:focus-within {
    border-color: #a78bfa; box-shadow: 0 0 0 1px #a78bfa;
}
.llm-popup-input {
    flex: 1; background: transparent; color: #e8e8ed;
    border: none; border-radius: 0; padding: 4px 4px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; resize: none; height: 36px; min-height: 36px;
    max-height: 120px; box-sizing: border-box; outline: none;
}
.llm-popup-send-btn {
    background: #a78bfa; color: #fff; border: none;
    border-radius: 6px; padding: 6px 16px; cursor: pointer;
    font-size: 12px; font-weight: 600;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    height: 36px; flex-shrink: 0; line-height: 1;
    transition: background 0.15s, box-shadow 0.15s;
}
.llm-popup-send-btn:hover { background: #b99afb; }

/* ── Attach Image Button ────────────────────────────────────────────── */
.llm-popup-attach-btn {
    background: none; border: 1px solid #2a2d32; border-radius: 6px;
    color: #5c5c66; cursor: pointer; font-size: 16px;
    padding: 4px 8px; margin-right: 4px; flex-shrink: 0;
    align-self: flex-end; height: 30px; line-height: 1;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-popup-attach-btn:hover {
    background: #22252e; border-color: #3a3e45; color: #e8e8ed;
}

/* ── Image Preview ──────────────────────────────────────────────────── */
.llm-popup-image-preview {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 10px; background: #1a1b1e;
    border-bottom: 1px solid #2d3139;
}
.llm-popup-image-preview img {
    max-width: 80px; max-height: 60px; border-radius: 4px;
    object-fit: cover; border: 1px solid #2d3139;
}
.llm-popup-image-remove-btn {
    background: none; border: none; color: #34d399;
    cursor: pointer; font-size: 14px; padding: 2px 6px;
    line-height: 1; border-radius: 3px; transition: background 0.15s;
}
.llm-popup-image-remove-btn:hover { background: #2a1f1f; }

/* ── Settings Section ──────────────────────────────────────────────────── */
.llm-popup-settings {
    border: 1px solid #2a2d32; border-radius: 6px;
    padding: 6px 12px; background: #1e1f22; flex-shrink: 0;
}
.llm-popup-settings > summary.llm-popup-settings-header {
    font-weight: bold; font-size: 12px; text-transform: uppercase;
    letter-spacing: 1px; color: #9a9aa5; cursor: pointer;
    user-select: none; padding: 4px 0; outline: none; list-style: none;
}
.llm-popup-settings > summary.llm-popup-settings-header::-webkit-details-marker { display: none; }
.llm-popup-settings-row {
    display: flex; gap: 8px; align-items: center; margin-bottom: 8px;
}
.llm-popup-settings-row:last-child { margin-bottom: 0; }
.llm-popup-settings-label {
    color: #9a9aa5; font-size: 11px; min-width: 90px; flex-shrink: 0;
}
.llm-popup-settings-select,
.llm-popup-settings-input {
    background: #131417; color: #e8e8ed;
    border: 1px solid #3a3e45; border-radius: 4px;
    padding: 5px 8px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px; flex: 1; outline: none;
    transition: border-color 0.15s;
}
.llm-popup-settings-select:hover,
.llm-popup-settings-input:hover { border-color: #4a9eff; }
.llm-popup-settings-input[type="number"] { max-width: 80px; flex: 0 0 80px; }
.llm-popup-settings-sections { display: flex; flex-direction: column; gap: 12px; }
.llm-popup-settings-section { display: flex; flex-direction: column; gap: 6px; padding: 0; }
.llm-popup-settings-section-header {
    font-weight: bold; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px; color: #9a9aa5;
    padding: 6px 0 2px 0; border-bottom: 1px solid #2a2d32; margin-bottom: 4px;
}
.llm-popup-custom-prompt {
    background: #1a1b1e; color: #e8e8ed;
    border: 1px solid #2d3139; border-radius: 3px;
    padding: 6px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px; resize: vertical; min-height: 50px;
    width: 100%; box-sizing: border-box; outline: none;
}
.llm-popup-custom-prompt:focus { border-color: #4a9eff; }
.llm-popup-custom-prompt:disabled { opacity: 0.4; background: rgba(20,20,30,0.5); }

/* ── Manage Prompts Button ─────────────────────────────────────────── */
.llm-popup-manage-btn {
    background: #22252e; color: #9a9aa5;
    border: 1px solid #2a2d32; border-radius: 6px;
    padding: 4px 10px; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 11px;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-popup-manage-btn:hover {
    color: #e8e8ed; border-color: #3a3e45; background: #2a2d36;
}

/* ── Popup Footer ────────────────────────────────────────────────────── */
.llm-popup-footer {
    display: flex; justify-content: space-between; align-items: center;
    gap: 8px; padding: 10px 16px;
    border-top: 1px solid #2a2d32; background: #1e1f22; flex-shrink: 0;
}
.llm-popup-save-btn {
    background: rgba(52,211,153,0.15); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3); border-radius: 6px;
    padding: 6px 16px; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; font-weight: 500;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.llm-popup-save-btn:hover {
    background: rgba(52,211,153,0.25); border-color: #a78bfa; color: #6ee7b7;
}

/* ── Stop Button ──────────────────────────────────────────────────────── */
.llm-popup-stop-btn { background: #22252e; border-color: #2a2d32; }
.llm-popup-stop-btn:hover {
    color: #f87171 !important; border-color: #ef4444 !important;
    background: rgba(239,68,68,0.25) !important;
}

/* ── Export Dropdown ────────────────────────────────────────────────── */
.llm-popup-export-btn { position: relative; }
.llm-popup-export-dropdown {
    position: absolute; top: 100%; right: 0;
    background: #1e1f22; border: 1px solid #3a3e45;
    border-radius: 4px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    z-index: 100; min-width: 140px; overflow: hidden;
}
.llm-popup-export-btn--footer .llm-popup-export-dropdown {
    top: auto; bottom: 100%;
}
.llm-popup-export-dropdown button {
    display: block; width: 100%; padding: 8px 14px;
    background: none; border: none; color: #e8e8ed;
    font-family: monospace; font-size: 12px; text-align: left;
    cursor: pointer; transition: background 0.1s;
}
.llm-popup-export-dropdown button:hover { background: rgba(255,255,255,0.04); }

/* ── Progress Bar ──────────────────────────────────────────────────── */
.llm-popup-progress {
    height: 4px; background: rgba(255,255,255,0.08);
    overflow: hidden; flex-shrink: 0;
}
.llm-popup-progress-fill {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #4a9eff, #34d399);
    transition: width 0.2s ease;
}

/* ── Thinking Blocks ────────────────────────────────────────────────── */
.llm-chat-thinking {
    margin: 6px 0; background: #111318;
    border: 1px solid #2d3139; border-radius: 4px; overflow: hidden;
}
.llm-chat-thinking-summary {
    padding: 4px 8px; cursor: pointer; font-size: 11px;
    color: #9a9aa5; user-select: none;
}
.llm-chat-thinking-summary:hover { background: rgba(255,255,255,0.06); }
.llm-chat-thinking-content {
    padding: 4px 8px 8px; font-size: 11px; color: #e8e8ed;
    white-space: pre-wrap; word-break: break-word;
    border-top: 1px solid #2a2d32;
}

/* ── System Prompt Details ─────────────────────────────────────────── */
.llm-chat-system-prompt-details { margin-top: 6px; font-size: 11px; }
.llm-chat-system-prompt-summary {
    cursor: pointer; color: #5c5c66; font-size: 11px;
    user-select: none; padding: 2px 0;
}
.llm-chat-system-prompt-summary:hover { color: #9a9aa5; }
.llm-chat-system-prompt-content {
    margin-top: 4px; padding: 6px 8px;
    background: rgba(255,255,255,0.04); border-radius: 4px;
    font-size: 11px; line-height: 1.4; color: #9a9aa5;
    white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto;
}

/* ── Scroll-to-Bottom Button ──────────────────────────────────────── */
.llm-chat-scroll-bottom-btn {
    position: sticky; bottom: 8px; float: right; z-index: 10;
    display: none; align-items: center; gap: 4px;
    padding: 6px 14px; background: #22252e; color: #e8e8ed;
    border: 1px solid #3a3e45; border-radius: 20px; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transition: background 0.15s, opacity 0.2s;
    opacity: 0; pointer-events: none;
}
.llm-chat-scroll-bottom-btn.llm-scroll-bottom-visible {
    display: flex; opacity: 1; pointer-events: auto;
}
.llm-chat-scroll-bottom-btn:hover { background: #2a2d36; }
.llm-chat-scroll-badge {
    display: none; background: #4a7abf; color: #fff;
    font-size: 10px; min-width: 16px; height: 16px;
    border-radius: 10px; text-align: center; line-height: 16px; padding: 0 4px;
}
.llm-chat-scroll-badge:not(:empty) { display: inline-block; }

/* ── Typing Indicator ─────────────────────────────────────────────── */
.llm-chat-typing-indicator {
    display: flex; align-items: center; gap: 4px;
    padding: 8px 12px; font-size: 12px; color: #34d399;
    background: #1e222b; border-radius: 4px; margin-bottom: 6px;
}
.llm-chat-typing-dots {
    display: inline-block;
    animation: llm-typing-blink 1.2s steps(3,end) infinite;
}
@keyframes llm-typing-blink {
    0% { opacity: 1; } 33% { opacity: 0.5; } 66% { opacity: 0.2; } 100% { opacity: 1; }
}

/* ── Context Indicator ─────────────────────────────────────────────── */
.llm-chat-context-indicator {
    font-size: 10px; color: #94a3b8; text-align: center;
    padding: 2px 8px; opacity: 0.8; flex-shrink: 0;
}

/* ── Toast Notification ──────────────────────────────────────────────── */
.llm-toast {
    position: fixed; top: 30px; left: 50%; transform: translateX(-50%);
    padding: 4px 16px; border-radius: 6px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; line-height: 1.3; color: #e8e8ed;
    z-index: 99999; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    pointer-events: none; opacity: 1; transition: opacity 0.3s;
    max-width: 80vw; text-align: center;
    box-sizing: border-box; height: auto; min-height: 0;
    max-height: 38px; overflow: hidden; white-space: nowrap;
    text-overflow: ellipsis; display: block; margin: 0;
    border: 1px solid #2d3139; border-left: 3px solid #4a9eff; outline: none;
}
.llm-toast-info { background: #1e1f22; border-left-color: #4a9eff; }
.llm-toast-success { background: #1e1f22; border-left-color: #34d399; }
.llm-toast-error { background: #2a1f1f; border-left-color: #ef4444; }
.llm-toast-warning { background: #2a241a; border-left-color: #f59e0b; }

/* ── Camera Badge ────────────────────────────────────────────────────── */
.llm-popup-camera-badge {
    display: none; font-size: 16px; cursor: default;
    margin-right: 8px; vertical-align: middle; line-height: 1; opacity: 0.9;
}

/* ── Mode Pill ──────────────────────────────────────────────────────── */
.llm-mode-pill {
    font-size: 14px; cursor: help; flex-shrink: 0;
    padding: 0 2px; opacity: 0.8; transition: opacity 0.15s; line-height: 1;
}
.llm-mode-pill:hover { opacity: 1; }

/* ── Canvas Button Container ────────────────────────────────────────── */
.llm-chat-button-container {
    display: flex; gap: 6px; padding: 4px 8px;
    align-items: center; width: 100%; box-sizing: border-box;
}

/* ═══════════════════════════════════════════════════════════════════════
   Prompt Manager Dialog (.llm-manager-*)
   ═══════════════════════════════════════════════════════════════════════ */
.llm-manager-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 10000;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; color: #e8e8ed;
}
.llm-manager-panel {
    background: #1a1b1e; border: 1px solid #2d3139; border-radius: 10px;
    padding: 0; width: 620px; max-width: 90vw; max-height: 85vh;
    display: flex; flex-direction: column;
    box-shadow: 0 20px 40px rgba(0,0,0,0.65); overflow: hidden;
    resize: both; min-width: 400px; min-height: 300px;
}
.llm-manager-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid #2a2d32;
    background: #1e1f22; flex-shrink: 0;
}
.llm-manager-header-title { font-size: 14px; font-weight: 600; }
.llm-manager-body {
    flex: 1; overflow-y: auto; padding: 14px 18px;
    display: flex; flex-direction: column; gap: 8px; min-height: 200px;
}
.llm-manager-footer {
    display: flex; justify-content: flex-end; gap: 8px;
    padding: 10px 16px; border-top: 1px solid #2a2d32;
    background: #1e1f22; flex-shrink: 0;
}
.llm-manager-search {
    width: 100%; background: #1a1b1e; color: #e8e8ed;
    border: 1px solid #2d3139; border-radius: 4px;
    padding: 8px 10px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; box-sizing: border-box; outline: none;
}
.llm-manager-search:focus { border-color: #a78bfa; box-shadow: 0 0 0 2px rgba(167,139,250,0.15); }
.llm-manager-search::placeholder { color: #5c5c66; }

.llm-manager-toolbar { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
.llm-manager-toolbar .llm-manager-search { flex: 1; }
.llm-manager-list { display: flex; flex-direction: column; gap: 4px; flex: 1; }

.llm-manager-prompt-row {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 10px; background: #1a1b1e; border-radius: 4px;
    border: 1px solid #2a2d32; transition: border-color 0.1s;
}
.llm-manager-prompt-row:hover { border-color: #3a3e45; }
.llm-manager-checkbox {
    appearance: none; -webkit-appearance: none;
    width: 16px; height: 16px; border: 1px solid #64748b;
    border-radius: 3px; background: rgba(0,0,0,0.2);
    cursor: pointer; flex-shrink: 0; position: relative;
    transition: background 0.15s, border-color 0.15s;
}
.llm-manager-checkbox:checked { background: #a78bfa; border-color: #a78bfa; }
.llm-manager-checkbox:checked::after {
    content: "\\2713"; color: #fff; position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; line-height: 1;
}
.llm-manager-prompt-name {
    flex: 0 0 auto; font-weight: bold; color: #e8e8ed;
    min-width: 100px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; cursor: default;
}
.llm-manager-prompt-preview {
    flex: 1; color: #9a9aa5; font-size: 11px; cursor: pointer;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    padding: 2px 4px; border-radius: 2px; transition: background 0.1s;
}
.llm-manager-prompt-preview:hover { background: rgba(255,255,255,0.04); }
.llm-manager-prompt-preview.expanded {
    white-space: pre-wrap; word-break: break-word;
    max-height: 200px; overflow-y: auto; background: rgba(20,20,35,0.5);
}
.llm-manager-badge { color: #34d399; font-size: 10px; flex-shrink: 0; min-width: 50px; text-align: right; }
.llm-manager-row-btns { display: flex; gap: 2px; flex-shrink: 0; }
.llm-manager-empty {
    color: #9a9aa5; text-align: center; padding: 30px 12px;
    font-size: 12px; font-style: italic;
}
.llm-manager-count { color: #9a9aa5; font-size: 11px; flex-shrink: 0; }

/* ── Manager Buttons ──────────────────────────────────────────────── */
.llm-manager-btn {
    background: #22252e; color: #9a9aa5;
    border: 1px solid #3a3f4d; border-radius: 6px;
    padding: 6px 14px; cursor: pointer;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px; flex-shrink: 0;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-manager-btn:hover {
    color: #e8e8ed; border-color: #3a3e45; background: #2a2d36;
}
.llm-manager-btn:active { background: #2a2d36; }
.llm-manager-btn-primary {
    background: rgba(52,211,153,0.15); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3); padding: 8px; font-weight: 500;
}
.llm-manager-btn-primary:hover {
    background: rgba(52,211,153,0.25);
    border-color: rgba(52,211,153,0.5); color: #6ee7b7;
}
.llm-manager-btn-danger {
    background: rgba(239,68,68,0.12); border-color: transparent; color: #f87171;
}
.llm-manager-btn-danger:hover {
    background: rgba(239,68,68,0.25); color: #fca5a5; border-color: transparent;
}
.llm-manager-btn-small { padding: 2px 8px; font-size: 11px; }
.llm-manager-btn-icon {
    background: none; border: none; color: #a78bfa; cursor: pointer;
    padding: 2px 4px; font-size: 13px; line-height: 1; transition: color 0.1s;
}
.llm-manager-btn-icon:hover { color: #c4b5fd; }
.llm-manager-btn-icon:disabled { color: #444; cursor: default; }
.llm-manager-btn-ghost {
    background: none; border: none; color: #a78bfa; cursor: pointer;
    padding: 2px 6px; font-size: 13px; line-height: 1;
    border-radius: 4px; transition: color 0.15s, background 0.15s;
}
.llm-manager-btn-ghost:hover {
    color: #c4b5fd; background: rgba(167,139,250,0.15);
}

/* ── Manager Form ────────────────────────────────────────────────── */
.llm-manager-form { display: flex; flex-direction: column; gap: 10px; flex: 1; }
.llm-manager-label { color: #e8e8ed; font-size: 12px; margin-bottom: 2px; }
.llm-manager-input {
    width: 100%; background: #1a1b1e; color: #e8e8ed;
    border: 1px solid #2d3139; border-radius: 4px;
    padding: 6px 8px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; box-sizing: border-box; outline: none;
}
.llm-manager-input:focus { border-color: #4a9eff; }
.llm-manager-textarea {
    width: 100%; min-height: 120px; resize: vertical;
    flex: 1; height: auto;
    background: #1a1b1e; color: #e8e8ed;
    border: 1px solid #2d3139; border-radius: 4px;
    padding: 8px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; box-sizing: border-box; outline: none;
}
.llm-manager-textarea:focus { border-color: #4a9eff; }
.llm-manager-warning { color: #c8a050; font-size: 11px; display: none; }

/* ── Manager Confirm Dialog ───────────────────────────────────────── */
.llm-manager-confirm-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 10001;
    display: flex; align-items: center; justify-content: center;
}
.llm-manager-confirm-panel {
    background: #1a1b1e; border: 1px solid #2d3139;
    border-radius: 10px; padding: 20px; min-width: 320px;
    max-width: 450px; box-shadow: 0 20px 40px rgba(0,0,0,0.65);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; color: #e8e8ed;
}
.llm-manager-confirm-title { font-weight: bold; font-size: 15px; margin-bottom: 12px; }
.llm-manager-confirm-message { margin-bottom: 16px; line-height: 1.4; }
.llm-manager-confirm-btns { display: flex; justify-content: flex-end; gap: 8px; }

/* ── Manager Import Preview ────────────────────────────────────────── */
.llm-manager-import-preview {
    background: rgba(20,20,35,0.6); border: 1px solid #2a2d32;
    border-radius: 4px; padding: 8px; max-height: 200px;
    overflow-y: auto; font-size: 11px; color: #e8e8ed;
}
.llm-manager-select {
    background: #1a1b1e; color: #e8e8ed;
    border: 1px solid #2d3139; border-radius: 3px;
    padding: 5px 8px; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 12px; outline: none;
}
.llm-manager-select:focus { border-color: #4a9eff; }
.llm-manager-file-input { display: none; }
.llm-manager-feedback {
    padding: 8px 12px; border-radius: 4px; font-size: 12px;
    text-align: center;
    animation: llm-manager-fade 3s ease forwards;
}
.llm-manager-feedback-success {
    background: rgba(42,90,62,0.3); border: 1px solid #2a2d36; color: #8ad8aa;
}
.llm-manager-feedback-error {
    background: rgba(90,42,42,0.3); border: 1px solid #3a2a2a; color: #d8a0a0;
}
@keyframes llm-manager-fade {
    0% { opacity: 1; } 70% { opacity: 1; } 100% { opacity: 0; }
}

/* ═══════════════════════════════════════════════════════════════════════
   Model Browser (.llm-model-browser-*)
   ═══════════════════════════════════════════════════════════════════════ */
.llm-model-browser-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; color: #e8e8ed;
}
.llm-model-browser-panel {
    background: #1a1b1e; border: 1px solid #2d3139; border-radius: 10px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.65);
    width: 620px; max-width: 90vw; max-height: 90vh;
    display: flex; flex-direction: column;
    color: #e8e8ed; font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; resize: both; overflow: hidden;
    min-width: 420px; min-height: 300px;
}
.llm-model-browser-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid #2a2d32;
    background: #1e1f22; border-radius: 10px 10px 0 0;
    font-weight: 600; font-size: 14px; color: #e8e8ed;
}
.llm-model-browser-close-btn {
    background: rgba(239,68,68,0.15); border: none; color: #f87171;
    font-size: 14px; cursor: pointer; width: 32px; height: 32px;
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; line-height: 1;
    transition: background 0.15s, color 0.15s;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.llm-model-browser-close-btn:hover {
    background: rgba(239,68,68,0.35); color: #fca5a5;
}
.llm-model-browser-body {
    flex: 1; overflow: hidden; padding: 12px 16px;
    display: flex; flex-direction: column;
}
.llm-model-browser-footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; border-top: 1px solid #2a2d32;
    background: #1e1f22; border-radius: 0 0 10px 10px;
}
.llm-model-browser-footer-left { display: flex; gap: 6px; align-items: center; }
.llm-model-browser-footer-right { display: flex; gap: 6px; align-items: center; }

/* ── Search Bar ────────────────────────────────────────────────────── */
.llm-model-search-wrapper {
    position: relative; width: 100%; display: flex; align-items: center;
    background: #1e222b; border: 1px solid #3a3e45;
    border-radius: 8px; padding: 0 8px;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.llm-model-search-wrapper:focus-within {
    border-color: #a78bfa; box-shadow: 0 0 0 1px #a78bfa;
}
.llm-model-search-wrapper input {
    flex: 1; background: transparent; color: #e8e8ed;
    border: none; outline: none; padding: 8px 4px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px; box-sizing: border-box;
}
.llm-model-search-wrapper input::placeholder { color: #5c5c66; }
.llm-model-search-clear {
    font-size: 13px; color: #5c5c66; cursor: pointer;
    padding: 0 4px; line-height: 1; border: none; background: none;
    display: none; user-select: none; font-family: inherit; flex-shrink: 0;
}
.llm-model-search-clear:hover { color: #9a9aa5; }
.llm-model-search-clear.visible { display: inline; }

/* ── Filter + Sort Bar ──────────────────────────────────────────────── */
.llm-model-filter-bar {
    display: flex; align-items: center; gap: 6px;
    margin-top: 8px; flex-wrap: wrap;
    padding: 6px 8px; background: #131417;
    border: 1px solid #2a2d32; border-radius: 6px;
}
.llm-model-filter-group { display: flex; align-items: center; gap: 2px; }
.llm-model-filter-label {
    font-size: 11px; color: #5c5c66; margin-right: 2px;
    user-select: none; white-space: nowrap; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.3px;
}
.llm-model-filter-btn {
    padding: 2px 8px; font-size: 11px; cursor: pointer;
    border: 1px solid transparent; border-radius: 4px;
    background: transparent; color: #5c5c66;
    transition: background 0.1s, color 0.1s, border-color 0.1s;
    user-select: none; white-space: nowrap; font-family: inherit; font-weight: 500;
}
.llm-model-filter-btn:hover {
    background: rgba(255,255,255,0.04); color: #9a9aa5; border-color: #2a2d32;
}
.llm-model-filter-btn.active {
    background: rgba(167,139,250,0.15); border-color: #a78bfa; color: #a78bfa;
}
.llm-model-sort-btn {
    padding: 2px 6px; font-size: 11px; cursor: pointer;
    border: none; border-radius: 4px; background: transparent;
    color: #5c5c66; transition: background 0.1s, color 0.1s;
    user-select: none; font-family: inherit; font-weight: 500;
}
.llm-model-sort-btn:hover { background: rgba(255,255,255,0.04); color: #9a9aa5; }
.llm-model-sort-btn.active { color: #a78bfa; background: rgba(167,139,250,0.15); }
.llm-model-filter-sep {
    width: 1px; height: 14px; background: #2a2d32; margin: 0 4px; flex-shrink: 0;
}

/* ── Card List ──────────────────────────────────────────────────────── */
.llm-model-card-list {
    display: flex; flex-direction: column; gap: 3px;
    flex: 1; min-height: 60px; overflow-y: auto;
    margin-top: 8px; border: 1px solid #2d3139;
    border-radius: 6px; padding: 4px; background: #131417;
}
.llm-model-card-list:empty {
    display: flex; align-items: center; justify-content: center;
}
.llm-model-card-list-loading {
    display: flex; align-items: center; justify-content: center;
    gap: 6px; font-size: 11px; color: #5c5c66; padding: 20px 0;
}
.llm-model-card-list-empty {
    font-size: 11px; color: #5c5c66; text-align: center;
    font-style: italic; padding: 20px 0;
}
.llm-model-card {
    display: flex; flex-direction: column;
    padding: 6px 10px; background: #131417;
    border: 1px solid #2d3139; border-radius: 4px;
    cursor: pointer; transition: background 0.12s, border-color 0.12s;
}
.llm-model-card:hover {
    background: rgba(255,255,255,0.04); border-color: #3a3e45;
}
.llm-model-card.selected {
    border-color: #a78bfa; background: rgba(167,139,250,0.15);
    box-shadow: inset 0 0 0 1px #a78bfa;
}
.llm-model-card-name {
    font-size: 11px; font-weight: 600; color: #e8e8ed;
    word-break: break-all; margin-bottom: 3px; line-height: 1.3;
}
.llm-model-card-badges { display: flex; flex-wrap: wrap; gap: 3px; }

/* ── Dropdown ────────────────────────────────────────────────────────── */
.llm-model-search-container { position: relative; width: 100%; }
.llm-model-search-input { width: 100%; box-sizing: border-box; padding-right: 24px; }
.llm-model-dropdown {
    position: absolute; top: 100%; left: 0; right: 0; z-index: 1000;
    max-height: 300px; overflow-y: auto;
    background: #1a1b1e; border: 1px solid #2d3139;
    border-top: none; border-radius: 0 0 4px 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.llm-model-dropdown-item {
    padding: 6px 10px; font-size: 12px; color: #e8e8ed;
    cursor: pointer; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; transition: background 0.1s;
}
.llm-model-dropdown-item:hover,
.llm-model-dropdown-item.selected { background: #22252e; color: #e8e8ed; }
.llm-model-dropdown-empty {
    padding: 10px; font-size: 11px; color: #5c5c66; text-align: center; font-style: italic;
}

/* ── Settings Panel ────────────────────────────────────────────────── */
.llm-model-settings-panel {
    border: 1px solid #2a2d32; border-radius: 6px;
    padding: 8px 10px; background: #1e1f22; margin-top: 8px;
}
.llm-model-settings-panel:hover { border-color: #3a3e45; }
.llm-model-settings-toggle {
    display: flex; align-items: center; gap: 4px; cursor: pointer;
    font-size: 11px; color: #9a9aa5; padding: 3px 0;
    user-select: none; border: none; background: none;
    font-family: inherit; font-weight: 500; transition: color 0.15s;
}
.llm-model-settings-toggle:hover { color: #a78bfa; }
.llm-model-settings-content { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a2d32; }
.llm-model-settings-expanded .llm-model-settings-content { display: block; }
.llm-model-cache-row { display: flex; align-items: center; gap: 6px; margin-top: 8px; padding: 4px 0; }

/* ── Info Panel ────────────────────────────────────────────────────── */
.llm-model-info-panel {
    display: flex; flex-direction: column; gap: 4px;
    margin-top: 6px; padding: 8px 12px; font-size: 11px;
    color: #9a9aa5; line-height: 1.5;
    background: #131417; border-radius: 4px; border: 1px solid #2a2d32;
}
.llm-model-info-text { overflow-wrap: break-word; word-break: break-word; }
.llm-model-info-path-row { display: flex; align-items: center; gap: 6px; min-width: 0; }
.llm-model-info-path-text {
    flex: 1; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; word-break: break-all;
}
.llm-model-info-copy-btn {
    flex-shrink: 0; background: #22252e; border: 1px solid #2a2d32;
    border-radius: 4px; color: #9a9aa5; cursor: pointer;
    font-size: 10px; padding: 2px 6px; font-family: inherit;
    line-height: 1.4; white-space: nowrap;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-model-info-copy-btn:hover {
    color: #e8e8ed; border-color: #3a3e45; background: #2a2d36;
}
.llm-model-info-badges { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.llm-model-info-sep { color: #2d3139; user-select: none; font-size: 10px; }

/* ── Generic Pill Badge ──────────────────────────────────────────────── */
.llm-pill {
    display: inline-block; font-size: 9px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px;
    background: rgba(167,139,250,0.15); color: #a78bfa;
    text-transform: uppercase; letter-spacing: 0.3px;
    vertical-align: middle; line-height: 1.4;
}
.llm-pill-mmproj { background: rgba(60,170,100,0.2); color: #34d399; }
.llm-pill-size { background: #22252e; color: #9a9aa5; }
.llm-arch-pill {
    display: inline-block; font-size: 9px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px;
    background: rgba(167,139,250,0.15); color: #a78bfa;
    margin-right: 5px; text-transform: uppercase; letter-spacing: 0.3px;
    vertical-align: middle; line-height: 1.4;
}

/* ── Spinner ────────────────────────────────────────────────────────── */
.llm-spinner {
    display: inline-block; width: 12px; height: 12px;
    border: 2px solid #2d3139; border-top-color: #a78bfa;
    border-radius: 50%; animation: llm-spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 6px;
}
@keyframes llm-spin { to { transform: rotate(360deg); } }

/* ── Section Divider ────────────────────────────────────────────────── */
.llm-model-section-divider {
    font-size: 11px; color: #9a9aa5;
    margin: 10px 0 4px; padding: 0; user-select: none;
}
.llm-model-warning { font-size: 11px; color: #d8a050; margin-top: 4px; margin-bottom: 2px; line-height: 1.4; }

/* ── Model Browser Buttons ──────────────────────────────────────────── */
.llm-model-browser-btn {
    padding: 2px 8px; font-size: 10px; cursor: pointer;
    border: 1px solid #2a2d32; border-radius: 6px;
    background: #22252e; color: #9a9aa5;
    white-space: nowrap; flex-shrink: 0; font-family: inherit;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-model-browser-btn:hover {
    background: #2a2d36; border-color: #3a3e45; color: #e8e8ed;
}
.llm-model-browser-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.llm-model-browser-btn-primary {
    padding: 2px 8px; font-size: 11px; cursor: pointer;
    border: 1px solid #2a2d32; border-radius: 6px;
    background: #22252e; color: #e8e8ed; font-weight: 500;
    white-space: nowrap; font-family: inherit;
    transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
}
.llm-model-browser-btn-primary:hover {
    background: #2a2d36; border-color: #a78bfa;
}
.llm-model-browser-btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
.llm-model-browser-btn-danger {
    padding: 4px 8px; font-size: 11px; cursor: pointer;
    border: 1px solid rgba(239,68,68,0.25); border-radius: 6px;
    background: transparent; color: rgba(239,68,68,0.85); font-family: inherit;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-model-browser-btn-danger:hover {
    background: rgba(239,68,68,0.1); border-color: #ef4444; color: #f87171;
}
.llm-model-browser-btn-ghost {
    padding: 4px 8px; font-size: 11px; cursor: pointer;
    border: 1px solid #2d3139; border-radius: 4px;
    background: transparent; color: #9a9aa5; font-family: inherit;
    transition: background 0.15s, color 0.15s;
}
.llm-model-browser-btn-ghost:hover {
    background: rgba(167,139,250,0.15); color: #a78bfa;
}
.llm-model-browser-btn-danger-sm {
    padding: 0 6px; font-size: 10px; cursor: pointer;
    border: none; border-radius: 3px; background: transparent;
    color: #ef4444; font-family: inherit; transition: background 0.15s;
}
.llm-model-browser-btn-danger-sm:hover { background: #2a1f1f; color: #f87171; }
.llm-model-browser-btn-icon {
    padding: 0 2px; font-size: 10px; cursor: pointer;
    border: none; border-radius: 2px; background: transparent;
    line-height: 1; font-family: inherit; transition: background 0.15s;
}
.llm-model-browser-btn-icon:hover { background: #22252e; }
.llm-model-browser-btn-cancel {
    padding: 4px 12px; font-size: 12px; cursor: pointer;
    border: 1px solid #2a2d32; border-radius: 6px;
    background: #22252e; color: #9a9aa5; font-family: inherit;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
}
.llm-model-browser-btn-cancel:hover {
    background: #2a2d36; border-color: #3a3e45; color: #e8e8ed;
}
.llm-model-browser-btn-apply {
    background: rgba(52,211,153,0.15); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3); border-radius: 6px;
    padding: 6px 20px; font-size: 13px; font-weight: 500;
    cursor: pointer; font-family: inherit; margin-left: 8px;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}
.llm-model-browser-btn-apply:hover {
    background: rgba(52,211,153,0.25); border-color: #a78bfa; color: #6ee7b7;
}

/* ── Browser Layout Helpers ────────────────────────────────────────── */
.llm-model-browser-section-label {
    font-size: 12px; font-weight: 600; color: #e8e8ed;
    margin-top: 10px; margin-bottom: 4px;
    display: flex; align-items: center; justify-content: space-between;
    letter-spacing: 0.3px;
}
.llm-model-browser-hint {
    font-size: 11px; color: #9a9aa5; margin-top: 4px;
    font-style: italic; line-height: 1.4; padding: 2px 0;
}
.llm-model-browser-row { display: flex; flex-direction: row; gap: 4px; align-items: center; }
.llm-model-browser-dir-row { display: flex; align-items: center; gap: 4px; padding: 2px 0; font-size: 11px; }
.llm-model-browser-dir-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #a78bfa; }
.llm-model-browser-dir-label-excluded { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #5c5c66; text-decoration: line-through; }
.llm-model-browser-empty-text { font-size: 11px; color: #9a9aa5; padding: 4px 0; }
.llm-model-browser-error-text { font-size: 11px; color: #2a1f1f; padding: 4px 0; }
.llm-model-browser-dir-toggle {
    display: flex; align-items: center; gap: 4px; cursor: pointer;
    font-size: 12px; font-weight: 600; color: #e8e8ed;
    padding: 6px 0 4px 0; user-select: none;
    transition: color 0.15s; border-radius: 3px; letter-spacing: 0.3px;
}
.llm-model-browser-dir-toggle:hover { color: #a78bfa; }
.llm-model-browser-dir-container { display: none; margin-top: 4px; max-height: 150px; overflow-y: auto; border: 1px solid #2d3139; border-radius: 4px; padding: 4px; background: #131417; }
.llm-model-browser-expand-icon { font-size: 8px; transition: transform 0.15s; }
.llm-model-browser-dir-list { margin-top: 6px; max-height: 120px; overflow-y: auto; }
.llm-model-browser-dot { color: #a78bfa; margin-right: 4px; }
.llm-model-browser-custom-badge { font-size: 10px; margin-left: 2px; }

/* ── Cache label ────────────────────────────────────────────────────── */
#llm-model-cache-label { font-size: 11px; color: #5c5c66; user-select: none; }
`;
    document.head.appendChild(fallback);
})();
