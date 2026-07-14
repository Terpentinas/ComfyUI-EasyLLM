/**
 * EasyLLM Frontend Extension — Pure Popup Architecture
 *
 * This is the main entry point. It imports focused helpers from sibling
 * modules and registers the ComfyUI extension.
 *
 * Features per node type:
 * - EasyLLM: Canvas node with visible widgets + popup for chat/settings.
 *            All interactive UI (chat history, text input, settings)
 *            lives in a popup modal outside LiteGraph's DOM.
 *            Supports chat and enhancer modes.
 *
 * Common: Management dialog to add/edit/delete system prompt templates,
 *         global dropdown refresh across all nodes.
 */

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { api } from "../../../scripts/api.js";
import { NODE_NAMES, VISIBLE_WIDGET_NAMES_IMAGE_CAPTURE } from "./constants.js";
import { createButtonDOMWidget, hideCanvasWidgets, refreshButtonLabel } from "./buttons.js";
import { extractGeneratedText, extractRawText, createBubbleElement, detectError, formatErrorMessage, hideTypingIndicator, autoScrollIfNeeded, renderPopupHistory, getModelName, buildImagesSection } from "./popup.js";
import { startStreamListening, stopStreamListening, startCanvasProgressTracking, setupStreamingPreview } from "./websocket_bridge.js";
import { initHistory, getSerializableState, restoreSerializedState, pushUserMessage, pushAssistantMessage, pushEnhancerEntry, capChatHistory, cacheGeneratedText, resetStreamingState, serializeHistoryForBackend } from "./history_store.js";
import { showToast } from "./ui_utils.js";
import { estimateContextTokens } from "./popup_utils.js";

// ── Shared helper: detect any EasyLLM node type (CLIP or GGUF) ──
function isChatNode(nodeType) {
    return nodeType === "EasyLLM" || nodeType === "EasyLLMGGUF";
}

// ── Graph traversal helpers for auto-detecting upstream LLM node ──

/**
 * Follow a link ID to its origin node, recursively passing through
 * reroute and primitive nodes. Returns null if the link is invalid.
 * Follows the same LiteGraph API pattern as findUpstreamEasyLLMNodes().
 */
function resolveUpstreamNode(graph, linkId) {
    const link = graph.links[linkId];
    if (!link) return null;
    const originNode = graph.getNodeById(link.origin_id);
    if (!originNode) return null;
    // Pass through reroute and primitive nodes
    if (originNode.type === "Reroute" || (originNode.type && /^Primitive/i.test(originNode.type))) {
        for (const inp of originNode.inputs || []) {
            if (inp.link != null) {
                return resolveUpstreamNode(graph, inp.link);
            }
        }
    }
    return originNode;
}

/**
 * Auto-detect the upstream EasyLLM node from an Image Capture node by
 * walking the graph backward along the expected wiring path:
 *
 *   Image Capture (session_uuid input) → Trigger Router (trigger_prompt input) → EasyLLM node
 *
 * Returns the LLM node's ID as a string, or null if detection fails
 * (in which case the manual node_id widget value is used as fallback).
 */
function autoDetectLLMNodeId(imageCaptureNode) {
    const graph = app.graph;
    if (!graph) return null;

    // Step 1: Follow the "session_uuid" input link to find the Trigger Router
    const sessionUuidInput = imageCaptureNode.inputs?.find(
        i => i.name === "session_uuid"
    );
    if (!sessionUuidInput || sessionUuidInput.link == null) {
        console.debug("[LLM Chat] ImageCapture: no session_uuid link — cannot auto-detect node_id");
        return null;
    }

    const triggerRouter = resolveUpstreamNode(graph, sessionUuidInput.link);
    if (!triggerRouter || triggerRouter.type !== "LLM_TriggerRouter") {
        console.debug("[LLM Chat] ImageCapture: upstream of session_uuid is not a Trigger Router — fallback to manual node_id");
        return null;
    }

    // Step 2: Follow the Trigger Router's "trigger_prompt" input link to find the LLM node
    const triggerPromptInput = triggerRouter.inputs?.find(
        i => i.name === "trigger_prompt"
    );
    if (!triggerPromptInput || triggerPromptInput.link == null) {
        console.debug("[LLM Chat] ImageCapture: Trigger Router has no trigger_prompt link — fallback to manual node_id");
        return null;
    }

    const llmNode = resolveUpstreamNode(graph, triggerPromptInput.link);
    if (!llmNode || !isChatNode(llmNode.type)) {
        console.debug("[LLM Chat] ImageCapture: upstream of trigger_prompt is not an EasyLLM node — fallback to manual node_id");
        return null;
    }

    console.debug(`[LLM Chat] ImageCapture: auto-detected LLM node ${llmNode.id} via graph traversal`);
    return String(llmNode.id);
}

/**
 * Scan the graph for the Image Capture node matching the given action.
 *
 * Iterates all EasyLLM_ImageCapture nodes, reads their `capture_mode` widget,
 * and returns the node ID of the first specific match. Falls back to any node
 * with capture_mode="all" if no specific match is found.
 *
 * This enables the "pull" architecture: by queueing only the target Capture
 * node, ComfyUI's lazy evaluation automatically prunes chains that don't
 * feed into it — no group bypass required.
 *
 * @param {string} action - The LLM action: generate_image, edit_image, or just_chat.
 * @returns {string|null} The target Capture node's string ID, or null if no match.
 */
function findCaptureNodeForAction(action) {
    const actionToMode = {
        generate_image: "generate",
        edit_image: "edit",
        just_chat: "chat",
    };
    const targetMode = actionToMode[action];
    if (!targetMode) return null;

    let fallbackNodeId = null;

    for (const node of app.graph._nodes) {
        if (node.type !== "EasyLLM_ImageCapture") continue;

        const modeWidget = node.widgets?.find(w => w.name === "capture_mode");
        const captureMode = modeWidget?.value || "all";

        // Exact match: this Capture node is configured for this action
        if (captureMode === targetMode) {
            console.debug(
                `[LLM Chat] findCaptureNodeForAction: found exact match ` +
                `node ${node.id} for action="${action}" capture_mode="${captureMode}"`
            );
            return String(node.id);
        }

        // Fallback: remember the first "all" mode node
        if (captureMode === "all" && fallbackNodeId === null) {
            fallbackNodeId = String(node.id);
        }
    }

    if (fallbackNodeId) {
        console.debug(
            `[LLM Chat] findCaptureNodeForAction: no exact match for action="${action}", ` +
            `falling back to "all" capture node ${fallbackNodeId}`
        );
    } else {
        console.debug(
            `[LLM Chat] findCaptureNodeForAction: no Capture node found for action="${action}"`
        );
    }

    return fallbackNodeId;
}

/**
 * Compute a context indicator string for the canvas node display.
 * Reads the node's current state (chat history, widgets) and returns
 * a formatted string like "Context: ~1522 / 4096" or empty if no history.
 * Results are cached on the node via _contextCache to avoid recomputation
 * every frame (set _contextDirty = true when history changes).
 *
 * During streaming, also includes the in-progress assistant message
 * (node._streamingAccumulatedText) in the estimate for real-time accuracy.
 */
function computeCanvasContext(node) {
    const history = node._chatHistory || [];
    const streamingText = node._streamingAccumulatedText || "";

    // Hide when no history or streaming
    if (history.length === 0 && !streamingText) return "";

    // System prompt is provided via wired forceInput socket — not readable from frontend.
    // Context estimation uses empty string since the actual value is resolved server-side.
    const systemPrompt = "";

    // Read chat template from widget; "auto" resolved server-side
    let chatTemplate = "llama";
    if (node.isEasyLLMGGUF) {
        const ctW = node.widgets?.find(w => w.name === "chat_template");
        if (ctW?.value && ctW.value !== "auto") chatTemplate = ctW.value;
    }

    const result = estimateContextTokens(history, {
        systemPrompt,
        chatTemplate,
        countImages: true,
        pendingAssistantMessage: streamingText || undefined,
    });
    const used = result.total;

    // Get max context from node state
    let maxCtx = node._maxContextTokens;
    if (!maxCtx && node.isEasyLLMGGUF) {
        const nCtxWidget = node.widgets?.find(w => w.name === "n_ctx");
        maxCtx = nCtxWidget ? parseInt(nCtxWidget.value) : 4096;
    }
    if (!maxCtx) maxCtx = 2048;

    node._contextResult = result;
    node._contextMaxCtx = maxCtx;
    return `Context: ~${used} / ${maxCtx}`;
}

// ────────────────────────────────────────────────────────────────────────
// Mode helpers (used across multiple if (isEasyLLM) blocks)
// ────────────────────────────────────────────────────────────────────────

/** Get the current mode value ("chat" or "enhancer") from the node's widgets. */
function getNodeMode(node) {
    const modeW = node.widgets?.find(w => w.name === "mode");
    return modeW?.value || "chat";
}

/** Update the node title to reflect the current mode and engine type. */
function updateNodeTitle(node) {
    const mode = getNodeMode(node);
    const isGGUF = node.isEasyLLMGGUF;
    if (mode === "enhancer") {
        node.title = isGGUF ? "✨ EasyLLM GGUF · Enhancer" : "✨ EasyLLM · Enhancer";
    } else if (isGGUF) {
        node.title = "🤖 EasyLLM GGUF · Chat";
    } else {
        node.title = "🤖 EasyLLM · Chat";
    }
}

// ────────────────────────────────────────────────────────────────────────
// Register the extension
// ────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "Comfy.EasyLLM",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_NAMES.includes(nodeData.name)) return;
        console.log(`[LLM Chat] beforeRegisterNodeDef for ${nodeData.name}`);

        const isEasyLLM = (nodeData.name === "EasyLLM");
        const isEasyLLMGGUF = (nodeData.name === "EasyLLMGGUF");
        const isEasyLLMText = (nodeData.name === "EasyLLMText");
        const isImageCapture = (nodeData.name === "EasyLLM_ImageCapture");

        // ── Capture raw widget values before Vue frontend strips them ──
        const VALUES = Symbol("llmchat_widgets_values");
        const VALUES_MAP = Symbol("llmchat_widgets_values_map");

        // ── Custom serialization: persist state via workflow JSON to avoid polluting
        // ── the node.widgets array (Vue strips/transforms widget values on configure).
        const origSerialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const data = origSerialize ? origSerialize.apply(this, arguments) : {};
            if (isChatNode(nodeData.name)) {
                data.llmchat = getSerializableState(this);
            }
            return data;
        };

        // ── configure hook: capture widget values before base configure runs (Vue
        // ── strips/transforms them). After configure, restore values BY NAME via a
        // ── pre-configure name→index mapping to defeat Vue's widget reordering.
        const needsWidgetPinning = isChatNode(nodeData.name) || isImageCapture;
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            const data = arguments[0];
            const nodeId = this.id ?? "?";
            const hasData = !!data;
            if (needsWidgetPinning) {
                console.debug(`[LLM Chat] configure hook for node ${nodeId} — hasData=${hasData}, widgets=${this.widgets?.length}`);
                // Capture widget values + name→index mapping before configure modifies them.
                this[VALUES] = data?.widgets_values;
                this[VALUES_MAP] = {};
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const w = this.widgets[i];
                        if (w.name && !w.name.startsWith("_")) {
                            this[VALUES_MAP][w.name] = i;
                        }
                    }
                }
                if (isChatNode(nodeData.name)) {
                    restoreSerializedState(this, data);
                }
            }
            const ret = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            if (needsWidgetPinning) {
                // Restore widget values by name using pre-configure mapping.
                const vals = this[VALUES];
                const nameIdxMap = this[VALUES_MAP];
                if (vals && nameIdxMap) {
                    for (const w of (this.widgets || [])) {
                        if (w.name && nameIdxMap.hasOwnProperty(w.name)) {
                            const idx = nameIdxMap[w.name];
                            if (idx >= 0 && idx < vals.length) {
                                w.value = vals[idx];
                            }
                        }
                    }
                }
                // Clean up captured state
                delete this[VALUES];
                delete this[VALUES_MAP];
                if (isChatNode(nodeData.name)) {
                    console.debug(`[LLM Chat] configure hook for node ${nodeId} — after origConfigure, widgets=${this.widgets?.length}, has llm_chat_buttons=${!!this.widgets?.find(w => w.name === "llm_chat_buttons")}`);
                }
            }
            return ret;
        };

        // ────────────────────────────────────────────────────────────────────
        // onNodeCreated: Hide canvas widgets + init state + schedule DOM widget.
        // DOM widget creation is deferred via requestAnimationFrame to ensure
        // Vue's widget mount containers exist before addDOMWidget attaches.
        // ────────────────────────────────────────────────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // rAF callback checks !chatNode.graph as safety guard

            console.debug(`[LLM Chat] onNodeCreated for node ${this.id} (${nodeData.name})`);

            // ── EasyLLM/GGUF-specific: Minimal canvas + popup ──
            if (isChatNode(nodeData.name)) {
                // Set GGUF flag for other modules (buttons.js, popup.js)
                this.isEasyLLMGGUF = isEasyLLMGGUF;

                // Initialize state — restored from configure hook or empty defaults
                initHistory(this);
                updateNodeTitle(this);

                // Seed auto-randomize is handled natively by ComfyUI's frontend
                // via `control_after_generate: True` in the Python widget definition
                // (like KSampler). The 🎲 toggle appears next to the seed input on canvas
                // and in the popup settings panel.

                const modeW = this.widgets?.find(w => w.name === "mode");
                const isEnhancer = modeW && modeW.value === "enhancer";
                if (!isEnhancer) {
                    hideCanvasWidgets(this);
                }

                // Defer DOM widget creation to rAF so Vue widget mount containers exist
                const chatNode = this;
                requestAnimationFrame(() => {
                    if (!chatNode.graph) {
                        console.debug(`[LLM Chat] Node ${chatNode.id} removed before rAF fired — skipping DOM widget`);
                        return;
                    }
                    console.debug(`[LLM Chat] Node ${chatNode.id} — rAF firing, creating button DOM widget`);
                    createButtonDOMWidget(chatNode);
                    const ms = chatNode.computeSize();
                    chatNode.size[1] = ms[1];
                    console.debug(`[LLM Chat] Node ${chatNode.id} — button DOM widget created, height=${chatNode.size[1]}`);
                });

                // ── Mode change callback: toggle widget visibility dynamically ──
                const modeWidget = this.widgets?.find(w => w.name === "mode");
                if (modeWidget) {
                    const modeNode = this;
                    const currentModeCb = modeWidget.callback;
                    modeWidget.callback = function (val) {
                        if (currentModeCb) currentModeCb.call(this, val);
                        if (val === "enhancer") {
                            // Show only enhancer-relevant widgets on canvas
                            // First, un-hide all widgets (undo previous hideCanvasWidgets call)
                            for (const w of modeNode.widgets || []) {
                                if (w.name.startsWith("_")) continue;
                                if (w.name === "llm_chat_buttons") continue;
                                if (w.type === "converted-widget") {
                                    w.type = w._originalType || "STRING";
                                    w.hidden = false;
                                    delete w.computeSize;
                                    delete w.draw;
                                    // Restore DOM element from hidden state
                                    const el = w.element || w.inputEl;
                                    if (el) {
                                        el.classList.remove("llm-widget-hidden");
                                        el.style.display = "";
                                        el.style.height = "";
                                        el.style.margin = "";
                                        el.style.padding = "";
                                    }
                                }
                            }
                            // All modes use the same visible widget list
                            // (temp/max_length/seed visible for quick tuning)
                            hideCanvasWidgets(modeNode);
                        } else {
                            // Chat mode: hide settings widgets (managed in popup)
                            hideCanvasWidgets(modeNode);
                        }
                        // Refresh the canvas button label and node title
                        refreshButtonLabel(modeNode);
                        updateNodeTitle(modeNode);
                        modeNode.setDirtyCanvas(true, true);
                    };
                    // Store original type for restoration
                    for (const w of this.widgets || []) {
                        if (w.type === "converted-widget") {
                            w._originalType = "STRING";
                        }
                    }
                }
            }

            // ── EasyLLM_ImageCapture: Hide hidden framework fields from canvas ──
            if (isImageCapture) {
                // unique_id, prompt, extra_pnginfo are framework-injected metadata
                // (UNIQUE_ID, PROMPT, EXTRA_PNGINFO) that ComfyUI's Vue frontend
                // creates widget slots for, consuming ~84px of vertical space.
                // Hide them so users can resize the node down to just capture_mode.
                hideCanvasWidgets(this, VISIBLE_WIDGET_NAMES_IMAGE_CAPTURE);
            }

            // ── Bidirectional sync: canvas widget changes → _popupSettings ──
            // Wraps existing callbacks to sync widget values to popup on change.
            if (isChatNode(nodeData.name)) {
                const syncNode = this;
                // GGUF nodes need additional widget sync for their extra settings
                // Order matches the section grouping: Persona → Tuning → Hardware
                const SYNC_WIDGETS = isEasyLLMGGUF
                    ? ["temperature", "max_length", "seed",
                       "chat_template", "n_ctx", "n_gpu_layers",
                       "use_mlock", "vram_mode",
                       "top_k", "top_p", "repetition_penalty"]
                    : ["temperature", "max_length", "seed"];
                for (const w of syncNode.widgets || []) {
                    if (SYNC_WIDGETS.includes(w.name)) {
                        const currentCb = w.callback;
                        w.callback = function (val) {
                            if (currentCb) currentCb.call(this, val);
                            if (!syncNode._popupSettings) syncNode._popupSettings = {};
                            syncNode._popupSettings[w.name] = val;
                        };
                    }
                }
            }

            return result;
        };

        // ────────────────────────────────────────────────────────────────────
        // Mode visual distinction + CLIP connection indicator
        // - Mode-colored bottom accent bar (always drawn)
        // - Top-right mode badge (always drawn)
        // - Red border + ⚠️ icon when CLIP is disconnected (unchanged)
        // ────────────────────────────────────────────────────────────────────
        if (isChatNode(nodeData.name)) {
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                origOnDrawForeground?.apply(this, arguments);

                const mode = getNodeMode(this);
                // GGUF nodes get a purple accent for visual distinction
                const modeColor = mode === "enhancer"
                    ? "#ff9800"
                    : (this.isEasyLLMGGUF ? "#9b59b6" : "#3a7bd5"); // Purple for GGUF, Blue for CLIP
                const modeLabel = mode === "enhancer"
                    ? "✨ ENHANCER"
                    : (this.isEasyLLMGGUF ? "🤖 GGUF CHAT" : "🤖 CHAT");

                // ── Mode-colored top accent bar (between title and widgets) ──
                ctx.save();
                ctx.fillStyle = modeColor;
                const titleH = this.constructor.NODE_TITLE_HEIGHT || 2;
                ctx.fillRect(0, titleH - 2, this.size[0], 2);
                ctx.restore();

                // ── GGUF model info display (below mode badge) ──
                if (this.isEasyLLMGGUF && this._ggufModelInfo) {
                    const info = this._ggufModelInfo;
                    const arch = info.architecture || info.name || "";
                    const ctxLen = info.context_length ? `${info.context_length} ctx` : "";
                    const infoText = [arch, ctxLen].filter(Boolean).join(" · ");
                    if (infoText) {
                        ctx.save();
                        ctx.font = "9px monospace";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "top";
                        ctx.fillStyle = "#9b59b6";
                        ctx.fillText(infoText, this.size[0] / 2, 30);
                        ctx.restore();
                    }
                }

                // ── Mode badge (below accent bar, between title area and widgets) ──
                ctx.save();
                ctx.font = "bold 9px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                const badgeX = this.size[0] / 2;
                const badgeY = 12;
                const textWidth = ctx.measureText(modeLabel).width;
                const pillW = textWidth + 14;
                const pillH = 16;
                // Background pill (20% opacity)
                ctx.fillStyle = modeColor + "33";
                ctx.beginPath();
                if (typeof ctx.roundRect === "function") {
                    ctx.roundRect(badgeX - pillW / 2, badgeY, pillW, pillH, 8);
                } else {
                    ctx.rect(badgeX - pillW / 2, badgeY, pillW, pillH);
                }
                ctx.fill();
                // Text
                ctx.fillStyle = modeColor;
                ctx.fillText(modeLabel, badgeX, badgeY + 3);
                ctx.restore();

                // ── Canvas context indicator (below mode badge, above widgets) ──
                // Only show when there's chat history to display.
                if (this._chatHistory && this._chatHistory.length > 0) {
                    // Recompute cache only when dirty (avoids O(n) work every frame)
                    if (this._contextDirty || !this._contextCache) {
                        this._contextCache = computeCanvasContext(this);
                        this._contextDirty = false;
                    }

                    if (this._contextCache) {
                        ctx.save();
                        ctx.font = "9px monospace";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "top";
                        // Color-code based on usage ratio
                        const ratio = this._contextResult
                            ? this._contextResult.total / this._contextMaxCtx
                            : 0;
                        if (ratio > 0.9) {
                            ctx.fillStyle = "#c0392b"; // Red — critical
                        } else if (ratio > 0.7) {
                            ctx.fillStyle = "#d8a050"; // Amber — warning
                        } else {
                            ctx.fillStyle = "#666";     // Gray — normal
                        }
                        ctx.fillText(this._contextCache, this.size[0] / 2, 46);
                        ctx.restore();
                    }
                }
            };
        }

        // ────────────────────────────────────────────────────────────────────
        // onExecuted: Update chat history + popup if open
        // ────────────────────────────────────────────────────────────────────
        if (isChatNode(nodeData.name)) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);

                const generatedText = extractGeneratedText(message);
                if (!generatedText) return;

                // Extract raw (uncleaned) text for popup display (preserves think tags)
                const rawText = extractRawText(message);

                // ── Extract attached image data from backend ──
                // Phase 3: Backend sends images with both "filename" (DB path, preferred)
                // and "data" (base64, fallback) fields in the images array.
                const imagesArr = message?.images || null;

                // ── Attach returned image data to the last user entry ──
                let enrichedUserIndex = -1;
                if (imagesArr && imagesArr.length > 0 && this._chatHistory && this._chatHistory.length > 0) {
                    // Streaming may have pushed an assistant entry; walk back for the user entry
                    for (let i = this._chatHistory.length - 1; i >= 0; i--) {
                        if (this._chatHistory[i].role === "user") {
                            // Store full typed images array with filename + data fields
                            this._chatHistory[i].images = imagesArr.map(img => ({
                                type: img.type,
                                filename: img.filename || null,  // DB-stored file (preferred)
                                data: img.data || null,          // base64 fallback
                            }));
                            enrichedUserIndex = i;
                            break;
                        }
                    }
                    // Re-render popup with resolved image; skip during streaming
                    if (this._popupHistoryEl && !this._popupStreaming) {
                        renderPopupHistory(this);
                    }

                    // ── Sync enriched images to server ──
                    // Non-ImageCapture workflows enrich images client-side in onExecuted,
                    // but the server is never notified. This sync ensures images survive reload.
                    if (enrichedUserIndex >= 0) {
                        const userEntry = this._chatHistory[enrichedUserIndex];
                        if (userEntry && userEntry._sessionUuid) {
                            // Use append-images if sessionUuid is available (popup path)
                            fetch(`/easyllm/db/history/${this.id}/append-images`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    session_uuid: userEntry._sessionUuid,
                                    images: imagesArr,
                                    type: "chat",
                                }),
                            }).catch(() => {});
                        } else {
                            // Fallback: update the entry directly by index (canvas queue path)
                            fetch(`/easyllm/db/history/${this.id}/entry`, {
                                method: "PUT",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    index: enrichedUserIndex,
                                    entry: userEntry,
                                    type: "chat",
                                }),
                            }).catch(() => {});
                        }
                    }
                }

                // ── Store GGUF model info for canvas display ──
                if (this.isEasyLLMGGUF && message?.model_info?.[0]) {
                    this._ggufModelInfo = message.model_info[0];
                }

                // ── Detect whether this response is an error ──
                const isError = detectError(rawText);

                // ── Determine mode — affects text clearing and history behavior ──
                const modeW = this.widgets?.find(w => w.name === "mode");
                const isEnhancer = modeW && modeW.value === "enhancer";

                // ── Use Python's resolved input_text; fall back to canvas text widget value ──
                const tw = this.widgets?.find(w => w.name === "text");
                const inputMsg = message?.input_text?.[0] || tw?.value;
                const userMessage = inputMsg?.trim();
                let assistantDeduped = false;

                if (isEnhancer) {
                    // ── ENHANCER MODE: Save to output history, keep text widget ──
                    // Use backend input_text; fall back to enhancer JSON, then widget
                    const enhancerData = message?.enhancer?.[0] ? JSON.parse(message.enhancer[0]) : null;
                    const enhancerInput = enhancerData?.input || "";
                    const enhancerSystemPrompt = enhancerData?.system_prompt || "";
                    const inputMsg = message?.input_text?.[0] || enhancerInput || tw?.value;
                    const userMessage = inputMsg?.trim();
                    const systemPrompt = message?.system_prompt?.[0] || enhancerSystemPrompt || "";

                    // Extract model name: 1) GGUF name (if descriptive) 2) model_path filename 3) architecture
                    // Validate GGUF metadata name — some models have bogus short names like "Src"
                    // which would bypass the more descriptive filename from getModelName().
                    const ggufMetaName = this._ggufModelInfo?.name;
                    const validGgufName = ggufMetaName && ggufMetaName.length >= 4 ? ggufMetaName : null;
                    const modelName = validGgufName
                        || getModelName(this)
                        || this._ggufModelInfo?.architecture
                        || "";

                    if (userMessage && generatedText) {
                        pushEnhancerEntry(this, userMessage, rawText, systemPrompt, modelName, imagesArr || undefined);
                        // Tag the last enhancer entry with the session_uuid
                        const enhHistory = this._enhancerHistory;
                        if (enhHistory && enhHistory.length > 0) {
                            enhHistory[enhHistory.length - 1]._sessionUuid = message?.session_uuid?.[0];
                        }
                    }
                    // Do NOT clear text widget — keep it for re-queue
                } else {
                    // ── CHAT MODE: Save user/assistant to chat history, clear text ──
                    // Push user message only when NOT already pre-pushed by streaming
                    if (userMessage && !this._popupStreaming) {
                        pushUserMessage(this, userMessage);
                    }

                    // Append assistant response with raw text (preserves think tags)
                    assistantDeduped = pushAssistantMessage(this, rawText, isError);
                    // Tag the assistant entry with the session_uuid
                    const chatHistory = this._chatHistory;
                    const sessionUuid = message?.session_uuid?.[0];
                    if (chatHistory && chatHistory.length > 0 && sessionUuid) {
                        const lastIdx = chatHistory.length - 1;
                        chatHistory[lastIdx]._sessionUuid = sessionUuid;
                        // Sync UUID to server so ImageCapture /append-images can find this entry
                        fetch(`/easyllm/db/history/${this.id}/entry`, {
                            method: "PUT",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                index: lastIdx,
                                entry: chatHistory[lastIdx],
                                type: "chat",
                            }),
                        }).catch(() => {});
                    }

                    // Cap at 50 entries to prevent memory issues
                    capChatHistory(this);

                    // Mark canvas context cache as dirty for next onDrawForeground
                    this._contextDirty = true;

                    // Store last output for auto-mode (use cleaned text, not raw)
                    cacheGeneratedText(this, generatedText);

                    // ── Fallback DB persistence for assistant message ──
                    // When popup is closed before streaming starts, the backend takes the
                    // blocking path (no WebSocket events). The streaming done handler never
                    // fires, so we must persist here as fallback.
                    if (!this._streamingSavedHistory) {
                        const chatHistory = this._chatHistory;
                        if (chatHistory && chatHistory.length > 0) {
                            const lastEntry = chatHistory[chatHistory.length - 1];
                            if (lastEntry && lastEntry.role === "assistant") {
                                fetch(`/easyllm/db/history/${this.id}/append`, {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ entry: lastEntry, type: "chat" }),
                                }).catch(() => {});
                            }
                        }
                    }

                    // Clear text widget (auto-mode detection: empty = Queue Prompt)
                    if (tw) tw.value = "";

                    // Clear chat_history hidden widget to prevent stale data
                    const chw = this.widgets?.find(w => w.name === "chat_history");
                    if (chw) chw.value = "";
                }

                // ── AUTO-QUEUE + PULL-BASED CAPTURE ROUTING ──
                // Parse the trigger_prompt JSON to get the action, then:
                // 1. Find the target Capture node for this action via
                //    findCaptureNodeForAction()
                // 2. Queue ONLY that Capture node — ComfyUI's lazy evaluation
                //    walks backward and prunes chains that don't feed it.
                //    No group bypass or Group Router needed.
                //
                // Guarded by this._enableImageGeneration (default true).
                // When disabled, the auto-queue is skipped — useful to work
                // around streaming/history issues caused by the generator mode.
                const triggerPromptRaw = message?.trigger_prompt?.[0];
                // Skip auto-queue in enhancer mode: the full graph already executed,
                // so the generation pipeline (KSampler → VAE → Capture) already ran.
                // Auto-queue is only needed in chat mode where only the LLM node
                // was queued via handlePopupSend().
                if (this._enableImageGeneration !== false && !isEnhancer && triggerPromptRaw) {
                    try {
                        const tp = JSON.parse(triggerPromptRaw);
                        const action = tp.action;

                        if (action === "generate_image" || action === "edit_image") {
                            const targetCaptureId = findCaptureNodeForAction(action);

                            if (targetCaptureId && !this._triggerQueued) {
                                this._triggerQueued = true;
                                
                                // ── DIAG: dump Capture node wiring ──
                                const targetNode = app.graph.getNodeById(Number(targetCaptureId));
                                if (targetNode) {
                                    console.debug(
                                        `[LLM Chat DIAG] Pull-queue for ${action}: ` +
                                        `Capture node ${targetCaptureId} ` +
                                        `(capture_mode=${(targetNode.widgets?.find(w => w.name === "capture_mode")?.value) || "all"})`
                                    );
                                    // Log inputs to the Capture node
                                    for (const inp of (targetNode.inputs || [])) {
                                        if (inp.link != null) {
                                            const srcLink = app.graph.links[inp.link];
                                            const srcNode = srcLink ? app.graph.getNodeById(srcLink.origin_id) : null;
                                            console.debug(
                                                `  [DIAG] Capture.input["${inp.name}"] → ` +
                                                `link ${inp.link} → node ${srcNode?.id} (${srcNode?.type})`
                                            );
                                        }
                                    }
                                } else {
                                    console.debug(
                                        `[LLM Chat DIAG] Pull-queue for ${action}: ` +
                                        `Capture node ${targetCaptureId} NOT FOUND in graph`
                                    );
                                }

                                // Also log the LLM node's outputs and where they go
                                const tpLinks = (this.outputs || [])[2]?.links || [];
                                const ioLinks = (this.outputs || [])[3]?.links || [];
                                console.debug(
                                    `[LLM Chat DIAG] LLM node ${this.id} outputs → ` +
                                    `trigger_prompt(2)=${JSON.stringify(tpLinks)}, ` +
                                    `image_output(3)=${JSON.stringify(ioLinks)}`
                                );

                                // Dump where image_output links point
                                for (const lId of ioLinks) {
                                    const l = app.graph.links[lId];
                                    if (l) {
                                        const destNode = app.graph.getNodeById(l.target_id);
                                        const destInput = destNode?.inputs?.[l.target_slot];
                                        console.debug(
                                            `[DIAG] link ${lId}: LLM.image_output → ` +
                                            `node ${l.target_id} (${destNode?.type}) ` +
                                            `input["${destInput?.name}"] slot ${l.target_slot}`
                                        );
                                    }
                                }

                                // Dump all IMAGE inputs of node 145 (the failing node)
                                const node145 = app.graph.getNodeById(145);
                                if (node145) {
                                    console.debug(
                                        `[DIAG] Node 145 (${node145.type}): ` +
                                        `${node145.inputs?.length} inputs`
                                    );
                                    for (const inp of (node145.inputs || [])) {
                                        const src = inp.link != null
                                            ? (() => {
                                                const sl = app.graph.links[inp.link];
                                                const sn = sl ? app.graph.getNodeById(sl.origin_id) : null;
                                                return `${sn?.id}(${sn?.type})`;
                                              })()
                                            : "UNCONNECTED";
                                        console.debug(
                                            `  [DIAG] node145.input["${inp.name}"] ` +
                                            `(type=${inp.type}) link=${inp.link} ← ${src}`
                                        );
                                    }
                                }

                                // ── DIAG: check image_filename widget right before queue ──
                                const _ifw2 = this.widgets?.find(w => w.name === "image_filename");
                                console.debug(
                                    `[DIAG] auto-queue: image_filename widget value=${JSON.stringify(_ifw2?.value)}, ` +
                                    `_uploadedImage=${JSON.stringify(this._uploadedImage)}`
                                );

                                console.debug(
                                    `[LLM Chat] Pull-queue for ${action}: ` +
                                    `targeting Capture node ${targetCaptureId}`
                                );
                                // Queue ONLY the target Capture node.
                                // seed=0, steps=1 — full graph executes, LLM uses cache replay.
                                // ComfyUI evaluates backward from this node,
                                // automatically skipping chains that don't feed it.
                                app.queuePrompt(0, 1, [targetCaptureId]);
                            } else if (!targetCaptureId) {
                                console.debug(
                                    `[LLM Chat] No Capture node found for ` +
                                    `action=${action} — images will not be captured`
                                );
                            }
                        }
                        // just_chat: no auto-queue needed (no images to generate)
                    } catch (_e) {
                        // Invalid JSON — not a trigger prompt
                    }
                }

                // ── Clear deferred image_filename after auto-queue ──
                // handlePopupSend defers clearing image_filename (keepWidget=true)
                // so the auto-queue / cache replay can still read the uploaded
                // filename for the image_output passthrough. Now that auto-queue
                // has fired (or wasn't needed), clear the widget to prevent
                // stale filenames on the next Queue Prompt execution.
                const _ifw = this.widgets?.find(w => w.name === "image_filename");
                if (_ifw && _ifw.value) {
                    _ifw.value = "";
                }

                // Update popup display if open; skip during streaming unless error.
                // Also skip if pushAssistantMessage dedup'd (cache replay from auto-queue trigger),
                // preventing duplicate bubbles when the same data is returned a second time.
                if (this._popupHistoryEl && (!this._popupStreaming || isError) && !assistantDeduped) {
                    const bubbleOptions = {};
                    let displayText = rawText;
                    if (isError) {
                        bubbleOptions.error = true;
                        displayText = formatErrorMessage(rawText);
                    }
                    const bubble = createBubbleElement("assistant", displayText, bubbleOptions);
                    this._popupHistoryEl.appendChild(bubble);
                    autoScrollIfNeeded(this._popupHistoryEl);

                    // ── Hide typing indicator on error ──
                    if (isError) {
                        hideTypingIndicator(this);
                    }
                }

                // Clean up streaming state and signal backend if active
                if (this._popupStreaming) {
                    resetStreamingState(this);
                    // Clean up the streaming event listener from the Map
                    stopStreamListening(this);
                    // ── Hide stop button; error cases skip llm_lab_complete event ──
                    if (this._popupStopBtn) {
                        this._popupStopBtn.style.display = "none";
                    }
                    try {
                        fetch(`/easyllm/popup_inactive/${this.id}`, { method: "POST" });
                    } catch (_e) {
                        // Non-critical
                    }
                    // ── Re-render popup — _chatHistory now has correct rawText ──
                        renderPopupHistory(this);
                    }
    
                    // NOTE: Assistant message persistence is handled by the WebSocket
                    // done handler in websocket_bridge.js (not here). That path fires
                    // for ALL execution paths, not just those with onExecuted.
                    // See: plans/server-db-cleanup-remaining-code.md
    
                    // ── Also persist enhancer history incrementally ──
                // The enhancer entry was just pushed to _enhancerHistory by
                // pushEnhancerEntry(). We append only the latest entry (the last one
                // in the array) rather than rewriting the entire history.
                if (isEnhancer) {
                    const enhancerHistory = this._enhancerHistory || [];
                    if (enhancerHistory.length > 0) {
                        const lastEnhancerEntry = enhancerHistory[enhancerHistory.length - 1];
                        try {
                            await fetch(`/easyllm/db/history/${this.id}/append`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    entry: lastEnhancerEntry,
                                    type: "enhancer",
                                }),
                            });
                        } catch (_appendErr) {
                            console.debug("[LLM Chat DB] onExecuted append enhancer failed:", _appendErr);
                        }
                    }
                }

                console.debug(`[LLM Chat] onExecuted node ${this.id} — response stored (raw=${rawText.length} chars, cleaned=${generatedText.length} chars)`);
            };
        }

        // ── EasyLLMText-specific: Streaming preview + display widgets ──
        if (isEasyLLMText) {
            // ── Find upstream EasyLLM nodes via LiteGraph link traversal ──
            function findUpstreamEasyLLMNodes(node) {
                const upstreamIds = [];
                for (const input of node.inputs || []) {
                    if (input.link != null) {
                        const link = app.graph.links[input.link];
                        if (link) {
                            const sourceNode = app.graph.getNodeById(link.origin_id);
                            if (sourceNode && isChatNode(sourceNode.type)) {
                                upstreamIds.push(String(sourceNode.id));
                            }
                        }
                    }
                }
                return upstreamIds;
            }

            // ── Clear all widgets and create read-only STRING display widgets ──
            function populateWidgets(textArr) {
                if (!textArr || !textArr.length) return;
                // Clear all existing widgets (including the text input widget and live preview)
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 0;
                }
                // Mark as cleaned so streaming handler doesn't recreate preview widget
                this._previewCleaned = true;
                // Create read-only STRING display widgets for each text entry
                for (const t of textArr) {
                    const w = ComfyWidgets["STRING"](
                        this,
                        "text_" + (this.widgets?.length ?? 0),
                        ["STRING", { multiline: true }],
                        app
                    ).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                    w.value = t;
                }
                // Auto-resize node to fit content
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) sz[0] = this.size[0];
                    if (sz[1] < this.size[1]) sz[1] = this.size[1];
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // ── onExecuted: receive text from Python execution ──
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                populateWidgets.call(this, message?.text);
                // Clean up streaming state
                this._streamingPreviewDone = false;
                this._livePreviewText = "";
            };

            // ── onNodeCreated: Set up streaming preview for new nodes ──
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = origOnNodeCreated?.apply(this, arguments);
                const upstreamIds = findUpstreamEasyLLMNodes(this);
                if (upstreamIds.length) {
                    setupStreamingPreview(this, upstreamIds);
                }
                // ── Set a reasonable default canvas size ──
                // Prevents the node from appearing as a tiny sliver before content arrives.
                // Respects user-resized dimensions and loaded workflow sizes (guard clause).
                if (!this.size || (this.size[0] < 350 && this.size[1] < 80)) {
                    this.size = [350, 100];
                }
                return ret;
            };

            // ── configure: capture widget values before Vue strips them (for workflow load) ──
            const SHOWTEXT_VALUES = Symbol("llmshowtext_widgets_values");
            const origConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                this[SHOWTEXT_VALUES] = arguments[0]?.widgets_values;
                return origConfigure?.apply(this, arguments);
            };

            // ── Restore display widgets from saved values; prevents duplicate text boxes on refresh ──
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                const widgets_values = this[SHOWTEXT_VALUES];
                if (widgets_values?.length) {
                    requestAnimationFrame(() => {
                        populateWidgets.call(this, widgets_values);
                    });
                }
                // Re-setup streaming preview for saved workflows
                const upstreamIds = findUpstreamEasyLLMNodes(this);
                if (upstreamIds.length) {
                    setupStreamingPreview(this, upstreamIds);
                }
            };
        }

        // ── EasyLLM_ImageCapture: Reconcile generated images with LLM node history ──
        if (isImageCapture) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);

                const sessionUuid = message?.session_uuid?.[0];
                const images = message?._easyllm_images || [];

                // Auto-detect the target LLM node via graph backward traversal.
                // The node_id widget was removed in favor of this graph-walking approach.
                // If graph traversal fails (e.g. broken wiring), reconciliation is skipped.
                const targetNodeId = autoDetectLLMNodeId(this);

                if (!sessionUuid || !targetNodeId || images.length === 0) {
                    console.debug(
                        `[LLM Chat] ImageCapture: skipped — session=${!!sessionUuid}, ` +
                        `target=${targetNodeId}, images=${images.length}`
                    );
                    return;
                }

                const targetNode = app.graph.getNodeById(Number(targetNodeId));
                if (!targetNode || !isChatNode(targetNode.type)) {
                    console.debug(`[LLM Chat] ImageCapture: target node ${targetNodeId} not found or not a chat node`);
                    return;
                }

                console.debug(`[LLM Chat] ImageCapture: reconciling ${images.length} image(s) with node ${targetNodeId}, session ${sessionUuid}`);

                // Normalize images array (same format as LLM node's images)
                const normalizedImages = images.map(img => ({
                    type: img.type || "generated",
                    filename: img.filename || null,
                    data: img.data || null,
                }));

                // ── Server-first: send images to server, which attaches them ──
                // Phase 6: Instead of in-memory enrichment + fire-and-forget
                // full-replace, we POST images to the server via /append-images.
                // The server finds the matching entry by session_uuid, appends
                // the images, and returns the updated entry. We then sync the
                // local cache from the server response.
                //
                // We try enhancer first, then chat (matching the existing logic
                // for which history type the entry lives in).
                let matchedHistoryType = null;

                // Try enhancer history first
                const enhHistory = targetNode._enhancerHistory || [];
                for (let i = enhHistory.length - 1; i >= 0; i--) {
                    if (enhHistory[i]._sessionUuid === sessionUuid) {
                        matchedHistoryType = "enhancer";
                        break;
                    }
                }

                // Then try chat history
                if (!matchedHistoryType) {
                    const chatHistory = targetNode._chatHistory || [];
                    for (let i = chatHistory.length - 1; i >= 0; i--) {
                        if (chatHistory[i]._sessionUuid === sessionUuid) {
                            matchedHistoryType = "chat";
                            break;
                        }
                    }
                }

                if (matchedHistoryType) {
                    try {
                        const resp = await fetch(`/easyllm/db/history/${targetNode.id}/append-images`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                session_uuid: sessionUuid,
                                images: normalizedImages,
                                type: matchedHistoryType,
                            }),
                        });
                        if (resp.ok) {
                            const data = await resp.json();
                            // Sync local cache from server response (authoritative)
                            if (data.entry) {
                                const historyArr = matchedHistoryType === "enhancer"
                                    ? targetNode._enhancerHistory
                                    : targetNode._chatHistory;
                                if (historyArr) {
                                    for (let i = historyArr.length - 1; i >= 0; i--) {
                                        if (historyArr[i]._sessionUuid === sessionUuid) {
                                            historyArr[i].images = data.entry.images || normalizedImages;
                                            break;
                                        }
                                    }
                                }
                            }
                            // Re-render if popup is open
                            if (targetNode._popupHistoryEl) {
                                // Detect if enhancer popup is open via the enhancer context element
                                const isEnhancerPopup = targetNode._popupEnhancerContextEl
                                    && document.contains(targetNode._popupEnhancerContextEl);

                                if (isEnhancerPopup && matchedHistoryType === "enhancer") {
                                    // ── Enhancer popup: targeted card update ──
                                    // Find the card whose dataset.sessionUuid matches, then
                                    // replace its images section with a fresh buildImagesSection().
                                    const historyContainer = targetNode._popupHistoryEl;
                                    const cards = historyContainer.querySelectorAll(".enhancer-card");
                                    for (const card of cards) {
                                        if (card.dataset.sessionUuid === sessionUuid) {
                                            // Remove old images section
                                            const oldImages = card.querySelector(".enhancer-card-images");
                                            if (oldImages) oldImages.remove();
                                            card.classList.remove("enhancer-card-no-images");
                                            // Find the matching in-memory entry to build fresh DOM
                                            const enhHistory = targetNode._enhancerHistory || [];
                                            const match = enhHistory.find(e => e._sessionUuid === sessionUuid);
                                            if (match) {
                                                const newImages = buildImagesSection(match);
                                                if (newImages) {
                                                    const main = card.querySelector(".enhancer-card-main");
                                                    if (main) {
                                                        main.insertBefore(newImages, main.firstChild);
                                                    }
                                                } else {
                                                    card.classList.add("enhancer-card-no-images");
                                                }
                                            }
                                            break;
                                        }
                                    }
                                } else {
                                    // ── Chat popup: full re-render ──
                                    renderPopupHistory(targetNode);
                                }
                                showToast(`🖼️ Generated image attached`, "success", 2000);
                            }
                        } else {
                            console.debug(`[LLM Chat] ImageCapture: server append-images returned ${resp.status}`);
                        }
                    } catch (_err) {
                        console.debug(`[LLM Chat] ImageCapture: append-images request failed:`, _err);
                    }
                } else {
                    console.debug(`[LLM Chat] ImageCapture: no entry with sessionUuid=${sessionUuid} found in cache`);
                }
            };
        }

        // ────────────────────────────────────────────────────────────────────
        // onConfigure: Create DOM widget for loaded nodes + sync state.
        // onNodeCreated may not fire during deserialization, so create here too.
        // rAF ensures Vue mount containers exist; createButtonDOMWidget guards
        // against duplicates.
        // ────────────────────────────────────────────────────────────────────
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);

            const nodeId = this.id ?? "?";
            console.debug(`[LLM Chat] onConfigure for node ${nodeId} — widgets=${this.widgets?.length}`);

            // EasyLLMText display widgets handled by its own onConfigure hook above

            // ── Create DOM widget for loaded nodes (non-ShowText) via rAF ──
            const confNodeId = this.id;
            const confNode = this;
            requestAnimationFrame(() => {
                if (!confNode.graph) {
                    console.debug(`[LLM Chat] Node ${confNodeId} removed before onConfigure rAF — skipping DOM widget`);
                    return;
                }
                if (!isEasyLLMText) {
                    const hasButtons = !!confNode.widgets?.find(w => w.name === "llm_chat_buttons");
                    console.debug(`[LLM Chat] Node ${confNodeId} — onConfigure rAF firing, hasButtons=${hasButtons}`);
                    // createButtonDOMWidget guards against duplicates
                    createButtonDOMWidget(confNode);
                    if (isChatNode(nodeData.name)) {
                        const ms = confNode.computeSize();
                        confNode.size[1] = ms[1];
                    }
                }
            });

            // ── EasyLLM/GGUF-specific: Ensure state arrays + GGUF flag ──
            if (isChatNode(nodeData.name)) {
                // Restore GGUF flag for loaded workflows
                this.isEasyLLMGGUF = isEasyLLMGGUF;
                if (!Array.isArray(this._chatHistory)) this._chatHistory = [];
                if (!this._popupSettings) this._popupSettings = {};
            }

            // ── Re-apply mode visibility after widget values restored from workflow ──
            if (isChatNode(nodeData.name)) {
                const modeW = this.widgets?.find(w => w.name === "mode");
                if (modeW) {
                    const actualMode = modeW.value;
                    if (actualMode === "enhancer") {
                        // Show only enhancer-relevant widgets on canvas
                        for (const w of this.widgets || []) {
                            if (w.name.startsWith("_")) continue;
                            if (w.name === "llm_chat_buttons") continue;
                            if (w.type === "converted-widget") {
                                w.type = w._originalType || "STRING";
                                w.hidden = false;
                                delete w.computeSize;
                                delete w.draw;
                                // Restore DOM element from hidden state
                                const el = w.element || w.inputEl;
                                if (el) {
                                    el.classList.remove("llm-widget-hidden");
                                    el.style.display = "";
                                    el.style.height = "";
                                    el.style.margin = "";
                                    el.style.padding = "";
                                }
                            }
                        }
                        // All modes use the same visible widget list
                        // (temp/max_length/seed visible for quick tuning)
                        hideCanvasWidgets(this);
                    } else {
                        // Chat mode: hide settings widgets
                        hideCanvasWidgets(this);
                    }
                }
                // Update node title to match restored mode
                updateNodeTitle(this);
            }

            // ── Bidirectional sync callbacks (restored for loaded nodes) ──
            if (isChatNode(nodeData.name)) {
                const syncNode = this;
                // GGUF nodes sync additional hardware/tuning widgets
                const SYNC_WIDGETS = isEasyLLMGGUF
                    ? ["temperature", "max_length", "seed",
                       "chat_template", "n_ctx", "n_gpu_layers",
                       "use_mlock", "vram_mode",
                       "top_k", "top_p", "repetition_penalty"]
                    : ["temperature", "max_length", "seed"];
                for (const w of syncNode.widgets || []) {
                    if (SYNC_WIDGETS.includes(w.name)) {
                        // Only wrap if not already wrapped (check for our _synced marker)
                        if (w._syncedToPopup) continue;
                        const currentCb = w.callback;
                        w.callback = function (val) {
                            if (currentCb) currentCb.call(this, val);
                            if (!syncNode._popupSettings) syncNode._popupSettings = {};
                            syncNode._popupSettings[w.name] = val;
                        };
                        w._syncedToPopup = true;
                    }
                }
            }

            // ── EasyLLM_ImageCapture: Hide hidden fields on workflow load ──
            if (isImageCapture) {
                // Re-apply hiding after deserialization so hidden fields
                // (unique_id, prompt, extra_pnginfo) remain collapsed to zero height.
                hideCanvasWidgets(this, VISIBLE_WIDGET_NAMES_IMAGE_CAPTURE);
            }
        };

        // ────────────────────────────────────────────────────────────────────
        // onRemoved: Clean up DOM widget element + streaming listeners
        // ────────────────────────────────────────────────────────────────────
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            // Remove DOM widget element from the DOM to prevent stale references
            const bw = this.widgets?.find(w => w.name === "llm_chat_buttons");
            if (bw) {
                const el = bw.element || bw.inputEl;
                if (el?.parentElement) {
                    el.parentElement.removeChild(el);
                }
            }
            // Clean up streaming WS listeners (EasyLLMText)
            if (this._streamPreviewHandlers) {
                for (const handler of this._streamPreviewHandlers) {
                    try {
                        api.removeEventListener("llm_lab_token", handler);
                    } catch (_e) {
                        // Non-critical
                    }
                }
                this._streamPreviewHandlers = null;
            }
            // Clean up GGUF state (abort stream, signal inactive)
            if (this.isEasyLLMGGUF) {
                try {
                    fetch(`/easyllm/abort_stream/${this.id}`, { method: "POST" }).catch(() => {});
                    fetch(`/easyllm/popup_inactive/${this.id}`, { method: "POST" }).catch(() => {});
                } catch (_e) {
                    // Non-critical
                }
                // Unload cached model from VRAM to prevent orphaned VRAM
                // when a node with vram_mode=keep_loaded is deleted.
                fetch("/easyllm/unload_model_cache", { method: "POST" }).catch(() => {});
            }
        };
    },
});

// ── Wrap app.queuePrompt to activate streaming / progress ──
// Seed auto-randomize is handled natively by ComfyUI's frontend via
// `control_after_generate: True` in the Python widget definition (like KSampler).
// The 🎲 toggle appears next to the seed input on canvas and in popup settings.

const origQueuePrompt = app.queuePrompt.bind(app);
app.queuePrompt = function (...args) {
    if (app.graph) {
        for (const node of app.graph._nodes) {
            // Only EasyLLM nodes — skip non-LLM nodes
            if (!isChatNode(node.type)) continue;

            // Skip bypassed nodes — they don't execute
            if (node.mode === 2) continue;

            const textW = node.widgets?.find(w => w.name === "text");
            const modeW = node.widgets?.find(w => w.name === "mode");
            const isEnhancer = modeW && modeW.value === "enhancer";

            // For enhancer mode: text may come from text_input socket, not widget.
            // Skip text widget guard so we still start canvas progress tracking.
            // For chat mode: skip if text widget is empty (nothing to generate).
            if (!isEnhancer && (!textW || !textW.value || !textW.value.trim())) continue;

            if (isEnhancer) {
                // ── ENHANCER MODE: Start canvas progress bar ──
                // Backend emits progress events for all execution paths.
                startCanvasProgressTracking(node);

                console.debug(
                    `[LLM Chat] Canvas queue: started progress tracking for enhancer node ${node.id}`
                );
                continue;
            }

            // ── CHAT MODE: Activate streaming; guard against double-activation ──
            if (node._popupStreaming) continue;

            // Only activate for nodes in the execution set (args[2]); full graph if undefined
            const outputNodeIds = args[2];
            if (outputNodeIds && Array.isArray(outputNodeIds) && !outputNodeIds.includes(String(node.id))) {
                continue;
            }

            // Push user message before streaming starts for correct [user, assistant] ordering
            pushUserMessage(node, textW.value.trim());

            // Persist user message to database incrementally (canvas queue path)
            // NOTE: Popup path already does this in handlePopupSend() via /append.
            const userText = textW.value.trim();
            if (userText) {
                fetch(`/easyllm/db/history/${node.id}/append`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        entry: { role: "user", message: userText },
                        type: "chat",
                    }),
                }).catch(() => {});
            }

            // Store chat history server-side; Python retrieves via unique_id lookup
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
                    // Non-critical
                }
            }

            // Activate streaming + canvas progress
            fetch(`/easyllm/popup_active/${node.id}`, { method: "POST" }).catch(() => {});
            startStreamListening(node);
            startCanvasProgressTracking(node);
            node._popupStreaming = true;
            node._streamingSavedHistory = false;

            console.debug(
                `[LLM Chat] Canvas queue: activated streaming + progress for node ${node.id}`
            );
        }
    }
    // ── Pre-queue: snapshot seed values to detect post-queue mutations ──
    // Some ComfyUI internal mechanism or client/server round-trip can
    // mutate the canvas widget seed value during queuePrompt execution.
    // We snapshot the seed BEFORE the queue and restore it after, regardless
    // of what mechanism caused the change. If the 🎲 toggle is ON (native
    // controlWidget mode === "randomize"), we let the new seed through.
    const seedSnapshots = new Map();
    if (app.graph) {
        for (const node of app.graph._nodes) {
            if (!isChatNode(node.type)) continue;
            const sdW = node.widgets?.find(w => w.name === "seed");
            if (sdW && sdW.value !== undefined) {
                const cw = sdW.controlWidget;
                // Check THREE sources for 🎲 state, in priority order:
                // 1. node._llmSeedRandomize — persistent property set by popup_settings.js 🎲 toggle
                //    (most reliable, nothing else resets it)
                // 2. cw?.value === "randomize" — native ComfyUI controlWidget (N/A for this node)
                // 3. sdW.options?.randomize === true — legacy fallback (may get reset by Vue internals)
                const isRandomizing = node._llmSeedRandomize !== false
                    || cw?.value === "randomize"
                    || sdW.options?.randomize === true;
                seedSnapshots.set(node.id, { value: sdW.value, isRandomizing });
            }
        }
    }

    const result = origQueuePrompt.apply(app, args);
    if (result && typeof result.then === "function") {
        result.then(() => {
            if (app.graph) {
                for (const node of app.graph._nodes) {
                    if (!isChatNode(node.type)) continue;
                    const snapshot = seedSnapshots.get(node.id);
                    if (!snapshot) continue;

                    const sdW = node.widgets?.find(w => w.name === "seed");
                    if (!sdW) continue;

                    // If 🎲 is explicitly ON (native controlWidget says randomize),
                    // let the new seed through — the user wants randomization.
                    if (snapshot.isRandomizing) {
                        console.debug(
                            `[LLM Chat] Post-queue: 🎲 ON — allowing seed change ${snapshot.value} → ${sdW.value} for node ${node.id}`
                        );
                    } else {
                        // 🎲 OFF — restore the pre-queue seed value.
                        if (sdW.value !== snapshot.value) {
                            console.debug(
                                `[LLM Chat] Post-queue: 🎲 OFF — restoring seed from ${sdW.value} → ${snapshot.value} for node ${node.id}`
                            );
                            sdW.value = snapshot.value;
                        }
                    }

                    // Sync popup seedInput to canvas widget value (restored or native)
                    const seedInput = node._popupSeedInput;
                    if (seedInput) {
                        const popupSeed = parseInt(seedInput.value, 10);
                        const canvasSeed = sdW.value;
                        if (popupSeed !== canvasSeed) {
                            seedInput.value = canvasSeed;
                            console.debug(
                                `[LLM Chat] Post-queue: synced popup seedInput from ${popupSeed} → ${canvasSeed} for node ${node.id}`
                            );
                        }
                    }
                }
            }
        }).catch(() => {});
    }
    return result;
};
