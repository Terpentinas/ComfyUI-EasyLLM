/**
 * Shared UI utilities for overlay modals and toast notifications.
 *
 * Exports:
 *   - createOverlayModal() — used by popup.js and editor.js
 *   - showToast()         — temporary notification overlay
 */

/**
 * Create a standardized overlay modal with header (title + close button),
 * body, and optional footer.
 *
 * Caller must append `overlay` to `document.body`. The returned references
 * are not auto-appended.
 *
 * Backdrop click closes the modal by calling `onClose`.
 *
 * @param {string}   prefix    CSS class prefix (e.g. "llm-popup" or "llm-manager").
 * @param {string}   title     Header title text.
 * @param {Function} onClose   Callback invoked when the close button or backdrop is clicked.
 * @param {object}   [options]
 * @param {boolean}  [options.hasFooter=false]  Whether to create a `<div class="{prefix}-footer">`.
 * @param {string}   [options.closeBtnClass]    Class(es) for the close button.
 *                                              Defaults to `"{prefix}-close-btn"`.
 * @returns {{ overlay: HTMLElement, panel: HTMLElement, header: HTMLElement,
 *             body: HTMLElement, footer: HTMLElement|null, closeBtn: HTMLElement }}
 */
export function createOverlayModal(prefix, title, onClose, options = {}) {
    const { hasFooter = false, closeBtnClass } = options;

    // ── Overlay ──
    const overlay = document.createElement("div");
    overlay.className = `${prefix}-overlay`;

    // ── Panel ──
    const panel = document.createElement("div");
    panel.className = `${prefix}-panel`;

    // ── Header ──
    const header = document.createElement("div");
    header.className = `${prefix}-header`;

    const titleSpan = document.createElement("span");
    titleSpan.className = `${prefix}-header-title`;
    titleSpan.textContent = title;
    header.appendChild(titleSpan);

    const closeBtn = document.createElement("button");
    closeBtn.className = closeBtnClass || `${prefix}-close-btn`;
    closeBtn.textContent = "✕";
    closeBtn.onclick = onClose;
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // ── Body ──
    const body = document.createElement("div");
    body.className = `${prefix}-body`;
    panel.appendChild(body);

    // ── Footer (optional) ──
    let footer = null;
    if (hasFooter) {
        footer = document.createElement("div");
        footer.className = `${prefix}-footer`;
        panel.appendChild(footer);
    }

    overlay.appendChild(panel);

    // ── Close on backdrop click ──
    overlay.addEventListener("mousedown", (e) => {
        if (e.target === overlay) onClose();
    });

    return { overlay, panel, header, body, footer, closeBtn };
}

/**
 * Show a temporary toast notification overlay.
 * Automatically removed after `duration` ms.
 *
 * @param {string} message   - The message text.
 * @param {string} [type='info'] - 'info' | 'success' | 'error' | 'warning'
 * @param {number} [duration=2500] - Auto-dismiss time in milliseconds.
 */
export function showToast(message, type = "info", duration = 2500) {
    // Remove any existing toast
    const existing = document.querySelector(".llm-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = `llm-toast llm-toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger reflow for CSS transition
    void toast.offsetHeight;

    // Auto-remove after duration
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s";
        setTimeout(() => toast.remove(), 300);
    }, duration);
}
