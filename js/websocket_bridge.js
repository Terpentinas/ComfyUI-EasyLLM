/**
 * Centralises ComfyUI WebSocket listeners and LGraphCanvas hooks for
 * streaming tokens, generation progress, and canvas progress bars.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { createBubbleElement, autoScrollIfNeeded, renderMarkdown, parseThinkBlocks, formatTimingBadge, formatTimingTooltip, detectError, formatErrorMessage, updateContextIndicator } from "./popup.js";
import { saveStreamedAssistantMessage } from "./history_store.js";
import { showToast } from "./ui_utils.js";

// ── Module-level state ──────────────────────────────────────────────

/** Map of nodeId → { percent } for nodes with active canvas progress bars. */
const _canvasProgressMap = new Map();

/** Whether our onDrawForeground hook has been installed on LGraphCanvas. */
let _canvasHookInstalled = false;

/** Popup progress handler (api.addEventListener). */
let _progressHandler = null;

/**
 * Map of nodeId → streaming token handler (api.addEventListener).
 * Supports concurrent streaming for multiple nodes.
 * @type {Map<string, Function>}
 */
const _streamHandlers = new Map();

/** Minimum interval (ms) between real-time context indicator updates during streaming. */
const CONTEXT_UPDATE_INTERVAL_MS = 2000;

// ── Shared event listener helpers ──────────────────────────────────
// Primary: api.addEventListener; falls back to document.addEventListener when api is unavailable.

/**
 * Register a custom event listener via api.addEventListener with
 * automatic fallback to document.addEventListener.
 *
 * @param {string}   apiEvent        - Event name for api.addEventListener
 * @param {Function} handler         - The event handler function
 * @param {string}   [fallbackEvent] - Fallback event name for document.addEventListener
 *                                    (defaults to apiEvent if not provided)
 * @returns {Function} The registered handler (store for later removal).
 */
function addCustomEventListener(apiEvent, handler, fallbackEvent) {
    if (fallbackEvent === undefined) fallbackEvent = apiEvent;
    try {
        api.addEventListener(apiEvent, handler);
    } catch (_e) {
        console.warn(
            `[LLM Chat DIAG] api.addEventListener("${apiEvent}") threw — `,
            `falling back to document listener`, _e
        );
        document.addEventListener(fallbackEvent, handler);
    }
    return handler;
}

/**
 * Remove a custom event listener registered by addCustomEventListener.
 *
 * @param {string}   apiEvent        - Event name for api.removeEventListener
 * @param {Function} handler         - The handler returned by addCustomEventListener
 * @param {string}   [fallbackEvent] - Fallback event name for document.removeEventListener
 *                                    (defaults to apiEvent if not provided)
 */
function removeCustomEventListener(apiEvent, handler, fallbackEvent) {
    if (!handler) return;
    if (fallbackEvent === undefined) fallbackEvent = apiEvent;
    try {
        api.removeEventListener(apiEvent, handler);
    } catch (_e) {
        document.removeEventListener(fallbackEvent, handler);
    }
}

// ────────────────────────────────────────────────────────────────────
// Canvas green progress bar
//
// The Vue-based ComfyUI frontend v1.43.18+ ignores node._progress.
// Progress is rendered via LGraphCanvas.onDrawForeground instead.
// ────────────────────────────────────────────────────────────────────

/**
 * Install the onDrawForeground hook on the LGraphCanvas instance.
 * Chains with any existing hook so we don't break other extensions.
 */
function ensureCanvasDrawHook() {
    if (_canvasHookInstalled) return;

    const lc = app.canvas;
    if (!lc) {
        // Canvas not ready yet; retry on next frame
        requestAnimationFrame(ensureCanvasDrawHook);
        return;
    }

    // Save any existing onDrawForeground to chain with it
    const origDraw = lc.onDrawForeground;

    lc.onDrawForeground = function (ctx, canvasEl) {
        // Call original hook first (e.g. SelectionBorder)
        if (origDraw) {
            origDraw.call(this, ctx, canvasEl);
        }

        // Draw progress bars for all tracked nodes
        if (_canvasProgressMap.size === 0) return;

        const toDelete = [];

        _canvasProgressMap.forEach((data, nodeId) => {
            // Look up the node by ID in the current graph
            const node = app.graph?._nodes?.find(
                (n) => String(n.id) === String(nodeId)
            );
            if (!node) {
                toDelete.push(nodeId);
                return;
            }

            const percent = data.percent;
            // Only draw in-progress (0 < percent < 1)
            if (percent == null || percent <= 0 || percent >= 1) {
                toDelete.push(nodeId);
                return;
            }

            // Node dimensions in graph coordinate space
            const [nx, ny] = node.pos;
            const [nw, nh] = node.size || [200, 100];

            // 4px green progress bar below title area
            const barH = 4;
            const titleH = node.constructor?.NODE_TITLE_HEIGHT || 2;
            const barY = ny + titleH;
            const barW = nw;

            ctx.save();

            // Transparent dark background
            ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
            ctx.fillRect(nx, barY, barW, barH);

            // Green fill — matches ComfyUI's native progress colour
            ctx.fillStyle = "#4CAF50";
            ctx.fillRect(nx, barY, barW * Math.min(percent, 1.0), barH);

            ctx.restore();
        });

        // Clean up stale entries
        for (const id of toDelete) {
            _canvasProgressMap.delete(id);
        }
    };

    _canvasHookInstalled = true;
}

/**
 * Listen for "progress" WebSocket events filtered by node.id.
 *
 * Falls back to document.addEventListener("llmchat-progress", ...) when
 * api.addEventListener is unavailable.
 *
 * @param {object}   node       - The EasyLLM node.
 * @param {Function} onProgress - Called with { value, max, percent } (percent 0–1).
 * @returns {Function} The registered handler (store for later removal).
 */
function listenForProgress(node, onProgress) {
    const handler = (event) => {
        const detail = event.detail || event;
        const { value, max, node: execNodeId } = detail;

        if (execNodeId !== undefined && Number(execNodeId) !== Number(node.id)) return;

        const percent = max > 0 ? Math.min(value / max, 1.0) : 0;
        onProgress({ value, max, percent });
    };

    return addCustomEventListener("progress", handler, "llmchat-progress");
}

/**
 * Remove a progress event listener registered by listenForProgress().
 * @param {Function} handler - The handler returned by listenForProgress().
 */
function _removeProgressListener(handler) {
    removeCustomEventListener("progress", handler, "llmchat-progress");
}

export function startCanvasProgressTracking(node) {
    // Stop any previous canvas tracking for this node
    stopCanvasProgressTracking(node);

    // Ensure our canvas draw hook is installed
    ensureCanvasDrawHook();

    // Initialize handler storage on the node
    if (!node._canvasProgressHandlers) node._canvasProgressHandlers = [];

    const handler = listenForProgress(node, ({ value, max, percent }) => {
        // Store progress in our module-level map → rendered by onDrawForeground hook
        _canvasProgressMap.set(String(node.id), { percent });

        // Request canvas redraw so onDrawForeground gets called
        node.setDirtyCanvas(true, true);

        // Auto-clear on completion
        if (value >= max) {
            setTimeout(() => {
                _canvasProgressMap.delete(String(node.id));
                node.setDirtyCanvas(true, true);
            }, 500);
        }
    });

    node._canvasProgressHandlers.push(handler);
}

/**
 * Stop canvas progress tracking for a node and clear its progress bar.
 */
export function stopCanvasProgressTracking(node) {
    if (node._canvasProgressHandlers) {
        for (const handler of node._canvasProgressHandlers) {
            _removeProgressListener(handler);
        }
        node._canvasProgressHandlers = null;
    }
    _canvasProgressMap.delete(String(node.id));
}

// ────────────────────────────────────────────────────────────────────
// Popup: Progress tracking during generation
// ────────────────────────────────────────────────────────────────────

/**
 * Start listening to ComfyUI progress events for this node.
 * Updates the progress bar element in the popup header.
 */
export function startProgressTracking(node) {
    // Stop any previous listener first
    stopProgressTracking();

    const progressBar = node._popupProgressBar;
    const progressFill = node._popupProgressFill;
    if (!progressBar || !progressFill) return;

    _progressHandler = listenForProgress(node, ({ value, max, percent }) => {
        const percent100 = percent * 100;
        progressBar.style.display = "block";
        progressFill.style.width = percent100 + "%";

        // Auto-hide on completion
        if (value >= max) {
            setTimeout(() => {
                if (progressBar && progressFill) {
                    progressFill.style.width = "100%";
                    setTimeout(() => {
                        if (progressBar) progressBar.style.display = "none";
                        if (progressFill) progressFill.style.width = "0%";
                    }, 400);
                }
            }, 300);
        }
    });
}

/**
 * Stop listening to progress events and reset the progress bar.
 */
export function stopProgressTracking() {
    _removeProgressListener(_progressHandler);
    _progressHandler = null;
}

// ────────────────────────────────────────────────────────────────────
// Streaming token listener — shared helper
// ────────────────────────────────────────────────────────────────────

/**
 * Listen for "llm_lab_token" WebSocket events filtered by node.id.
 *
 * Falls back to document.addEventListener("llm_lab_token", ...) when
 * api.addEventListener is unavailable.
 *
 * @param {object}   node     - The EasyLLM node.
 * @param {Function} onToken  - Called with event detail for each token.
 * @param {Function} onDone   - Called with event detail on completion.
 * @returns {Function} The registered handler (store for later removal).
 */
export function listenForTokens(node, onToken, onDone) {
    const handler = (event) => {
        const detail = event.detail;
        if (!detail || String(detail.node_id) !== String(node.id)) return;

        if (detail.done) {
            console.debug(
                `[LLM Chat DIAG] Token DONE for node ${node.id}: `,
                `timing=`, detail?.timing
            );
            onDone?.(detail);
        } else if (detail.token) {
            onToken?.(detail);
        }
    };

    console.debug(
        `[LLM Chat DIAG] listenForTokens registered for node ${node.id}`
    );
    return addCustomEventListener("llm_lab_token", handler);
}

// ────────────────────────────────────────────────────────────────────
// Popup: Streaming support — WebSocket listener for per-token updates
// ────────────────────────────────────────────────────────────────────

/**
 * Start listening to "llm_lab_token" WebSocket events for this node.
 * Each event contains { node_id, token, done } — appends token text
 * to the current assistant chat bubble in real time.
 */
export function startStreamListening(node) {
    console.debug(
        `[LLM Chat DIAG] startStreamListening for node ${node.id}`
    );
    // Stop any existing streaming handler for this specific node only
    stopStreamListening(node);

    // Initialize accumulated text buffer for this streaming session
    node._streamingAccumulatedText = "";

    // Throttle re-renders via requestAnimationFrame to avoid debounce timer starvation.
    let _pendingRender = false;

    /**
     * Re-render the streaming bubble's content using markdown + think tag parsing.
     * Updates the text area and think section in-place.
     */
    function updateStreamingBubble() {
        const historyEl = node._popupHistoryEl;
        if (!historyEl) return;

        let bubble = node._currentStreamBubble;
        if (!bubble) {
            bubble = createBubbleElement("assistant", "");
            historyEl.appendChild(bubble);
            node._currentStreamBubble = bubble;
        }

        const fullText = node._streamingAccumulatedText || "";

        // Parse think blocks
        const parsed = parseThinkBlocks(fullText);
        // Use parsed.response (including empty string) when set; fall back to fullText
        // only when response is null. Prevents think content duplication during
        // streaming when the tag is open but unclosed.
        const displayText = parsed.response !== null ? parsed.response : fullText;
        const thinkBlock = parsed.thinking;

        // Update the text area with markdown rendering
        const textEl = bubble.querySelector(".llm-chat-bubble-text");
        if (textEl) {
            // Strip U+FFFD replacement chars from partial multi-byte token decode
            const sanitizedText = displayText.replace(/\ufffd/g, "");
            textEl.innerHTML = renderMarkdown(sanitizedText);
        }

        // Update or create the collapsible thinking section
        let thinkSection = bubble.querySelector(".llm-chat-thinking");
        if (thinkBlock) {
            if (!thinkSection) {
                // Create new think section
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
                // Insert after label, before text
                const label = bubble.querySelector(".llm-chat-bubble-label");
                if (label) {
                    label.after(details);
                } else {
                    bubble.prepend(details);
                }
            } else {
                // Update existing think section content
                const thinkContent = thinkSection.querySelector(".llm-chat-thinking-content");
                if (thinkContent) {
                    thinkContent.textContent = thinkBlock;
                }
            }
        } else if (thinkSection) {
            // No think block — remove the section
            thinkSection.remove();
        }
    }

    const handler = listenForTokens(node,
        // onToken — append each token with debounced re-render
        (detail) => {
            const tokenLen = detail.token?.length || 0;
            const accLen = (node._streamingAccumulatedText || "").length;
            console.debug(
                `[LLM Chat DIAG] Token rcvd for node ${node.id}: ` +
                `token_len=${tokenLen}, accumulated_len=${accLen}`
            );
            node._streamingAccumulatedText += detail.token;

            const historyEl = node._popupHistoryEl;
            if (!historyEl) return;

            // Ensure the streaming bubble exists (empty state)
            let bubble = node._currentStreamBubble;
            if (!bubble) {
                bubble = createBubbleElement("assistant", "");
                historyEl.appendChild(bubble);
                node._currentStreamBubble = bubble;
            }

            // rAF-throttled re-render: accumulate tokens between frames, render once.
            if (!_pendingRender) {
                _pendingRender = true;
                requestAnimationFrame(() => {
                    _pendingRender = false;
                    updateStreamingBubble();
                    // Re-evaluate auto-scroll after rAF render so scrollHeight is current.
                    autoScrollIfNeeded(historyEl);
                });
            }

            // ── Throttled real-time context indicator update ──
            // Updates the context count every ~2s during streaming so users
            // can see the growing estimate in real time. Avoids expensive
            // token estimation on every single token event.
            const now = Date.now();
            if (!node._lastContextIndicatorUpdate) {
                node._lastContextIndicatorUpdate = now;
            } else if (now - node._lastContextIndicatorUpdate >= CONTEXT_UPDATE_INTERVAL_MS) {
                node._lastContextIndicatorUpdate = now;
                updateContextIndicator(node, node._streamingAccumulatedText || "");
            }

            // Mark canvas context as dirty so onDrawForeground re-computes
            node._contextDirty = true;

            // Use smart auto-scroll: only scroll if user is not scrolled up
            autoScrollIfNeeded(historyEl);
        },
        // onDone — finalize streaming: flush render, persist to history
        (detail) => {
            // Flush any pending render so final content is up-to-date
            _pendingRender = false;
            updateStreamingBubble();

            // ── Capture timing data from the done event ──
            if (detail?.timing) {
                node._lastTiming = detail.timing;
            }

            // ── Detect error state (MUST happen before _currentStreamBubble is cleared) ──
            const bubbleBeforeClear = node._currentStreamBubble;
            const isError = detectError(node._streamingAccumulatedText);

            if (isError && bubbleBeforeClear) {
                bubbleBeforeClear.classList.add("llm-chat-bubble-error");

                // Format the error message for display
                const formattedMsg = formatErrorMessage(node._streamingAccumulatedText);

                // Update bubble text with formatted error
                const textEl = bubbleBeforeClear.querySelector(".llm-chat-bubble-text");
                if (textEl) {
                    textEl.innerHTML = renderMarkdown(formattedMsg);
                }

                // Add error icon if not present
                if (!bubbleBeforeClear.querySelector(".llm-chat-bubble-error-icon")) {
                    const errIcon = document.createElement("span");
                    errIcon.className = "llm-chat-bubble-error-icon";
                    errIcon.textContent = "⚠️";
                    errIcon.title = "Generation encountered an error";
                    bubbleBeforeClear.prepend(errIcon);
                }
            }

            // ── Add timing badge to the streaming bubble ──
            if (bubbleBeforeClear && node._lastTiming) {
                // Remove any existing timing badge first (avoid duplicates)
                const existingBadge = bubbleBeforeClear.querySelector(".llm-chat-timing-badge");
                if (existingBadge) existingBadge.remove();

                const timingBadge = document.createElement("div");
                timingBadge.className = "llm-chat-timing-badge";
                timingBadge.textContent = formatTimingBadge(node._lastTiming);
                timingBadge.title = formatTimingTooltip(node._lastTiming);
                bubbleBeforeClear.appendChild(timingBadge);
            }

            // Clear streaming bubble reference after error detection
            node._currentStreamBubble = null;

            // ── Persist streamed response to chat history ──
            // Persists via streaming path when onExecuted is unavailable. Error flag restores error styling on reload.
            if (node._streamingAccumulatedText) {
                saveStreamedAssistantMessage(node, node._streamingAccumulatedText, node._lastTiming, isError);
                node._streamingAccumulatedText = "";
            }

            // ── Hide typing indicator if present ──
            if (node._typingIndicatorEl) {
                node._typingIndicatorEl.style.display = "none";
            }

            // ── Hide stop button if present ──
            if (node._popupStopBtn) {
                node._popupStopBtn.style.display = "none";
            }

            // ── Update context indicator after streaming completes ──
            // The new assistant message was just saved to _chatHistory,
            // so the context count needs to reflect the additional tokens.
            updateContextIndicator(node);

            // ── Final scroll to bottom ──
            const historyEl = node._popupHistoryEl;
            if (historyEl) {
                autoScrollIfNeeded(historyEl);
            }
        }
    );
    // Store handler in the map keyed by node ID
    _streamHandlers.set(String(node.id), handler);
}

/**
 * Stop listening to streaming events and clean up handler reference.
 */
export function stopStreamListening(node) {
    if (node) {
        // ── Stop streaming for a specific node ──
        const key = String(node.id);
        const handler = _streamHandlers.get(key);
        if (handler) {
            console.debug(
                `[LLM Chat DIAG] stopStreamListening for node ${node.id} — removing handler`
            );
            removeCustomEventListener("llm_lab_token", handler);
            _streamHandlers.delete(key);
        }
    } else {
        // ── Full cleanup: stop ALL streaming handlers ──
        if (_streamHandlers.size > 0) {
            console.debug(
                `[LLM Chat DIAG] stopStreamListening (full cleanup) — removing ${_streamHandlers.size} handler(s)`
            );
            for (const handler of _streamHandlers.values()) {
                removeCustomEventListener("llm_lab_token", handler);
            }
            _streamHandlers.clear();
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// EasyLLMText: Streaming preview helpers
// ────────────────────────────────────────────────────────────────────

/**
 * Ensure the live preview widget exists on an EasyLLMText node.
 * Created once on first call, updated per-token during streaming.
 */
function ensureLivePreviewWidget(node) {
    let w = node.widgets?.find(w => w.name === "llm_live_preview");
    if (!w) {
        w = ComfyWidgets["STRING"](
            node,
            "llm_live_preview",
            ["STRING", { multiline: true }],
            app
        ).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.opacity = 0.8;
        w.inputEl.style.backgroundColor = "rgba(40, 80, 60, 0.15)";
        w.inputEl.placeholder = "Waiting for response...";
    }
    return w;
}

/**
 * Update the live preview widget with accumulated streaming text.
 */
function updateLivePreviewWidget(node, text) {
    const w = ensureLivePreviewWidget(node);
    w.value = text;
    if (w.inputEl) {
        w.inputEl.value = text;
        w.inputEl.scrollTop = w.inputEl.scrollHeight;
    }
}

/**
 * Register a "llm_lab_token" WebSocket listener that updates the
 * live preview widget on an EasyLLMText node with streaming tokens
 * from its upstream EasyLLM node(s).
 */
export function setupStreamingPreview(showtextNode, upstreamNodeIds) {
    if (!upstreamNodeIds.length) return;

    const handler = (event) => {
        const detail = event.detail;
        if (!detail || !upstreamNodeIds.includes(String(detail.node_id))) return;

        // If populateWidgets already cleared the preview (onExecuted handled), no-op
        if (showtextNode._previewCleaned) return;

        if (detail.done) {
            showtextNode._streamingPreviewDone = true;
        } else if (detail.token) {
            if (!showtextNode._livePreviewText) showtextNode._livePreviewText = "";
            showtextNode._livePreviewText += detail.token;
            updateLivePreviewWidget(showtextNode, showtextNode._livePreviewText);
        }
    };

    api.addEventListener("llm_lab_token", handler);

    // Store for cleanup on node removal
    if (!showtextNode._streamPreviewHandlers) showtextNode._streamPreviewHandlers = [];
    showtextNode._streamPreviewHandlers.push(handler);
}

// ── Background Model Pre-load Notification ─────────────────────────
// Listens for "easyllm_model_ready" WebSocket events pushed by the
// /easyllm/preload_model endpoint on background pre-load completion.
// Displays a toast notification with status and model name.
api.addEventListener("easyllm_model_ready", (event) => {
    const detail = event.detail || event;
    if (!detail) return;

    if (detail.status === "ready") {
        const displayName = detail.model_path
            ? detail.model_path.split(/[\\/]/).pop()
            : "Model";
        showToast(`✅ ${displayName} loaded — ready to generate`, "success", 3000);
    } else if (detail.status === "error") {
        const displayName = detail.model_path
            ? detail.model_path.split(/[\\/]/).pop()
            : "Model";
        showToast(`❌ ${displayName} failed to load: ${detail.error || "unknown error"}`, "error", 5000);
    }
});
