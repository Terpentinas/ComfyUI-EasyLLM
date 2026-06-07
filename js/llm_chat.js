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
import { NODE_NAMES, getWidgetEl, findWidgetByName } from "./constants.js";
import { createButtonDOMWidget, hideCanvasWidgets, refreshButtonLabel } from "./buttons.js";
import { extractGeneratedText, extractRawText, createBubbleElement, detectError, formatErrorMessage, hideTypingIndicator, autoScrollIfNeeded, renderPopupHistory, getModelName } from "./popup.js";
import { startStreamListening, stopStreamListening, startCanvasProgressTracking, setupStreamingPreview } from "./websocket_bridge.js";
import { initHistory, getSerializableState, restoreSerializedState, pushUserMessage, pushAssistantMessage, pushEnhancerEntry, capChatHistory, cacheGeneratedText, resetStreamingState, serializeHistoryForBackend } from "./history_store.js";
import { estimateContextTokens } from "./popup_utils.js";

// ── Shared helper: detect any EasyLLM node type (CLIP or GGUF) ──
function isChatNode(nodeType) {
    return nodeType === "EasyLLM" || nodeType === "EasyLLMGGUF";
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

    // Read system prompt from widget
    const customW = node.widgets?.find(w => w.name === "system_prompt_text");
    const systemPrompt = customW?.value || "";

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
// Template selection: dim the custom prompt widget when a template is active
// ────────────────────────────────────────────────────────────────────────

function setupTemplateDimming(node, templateWidget) {
    const customWidget = findWidgetByName(node, "system_prompt_text")
        || findWidgetByName(node, "custom_prompt");

    function applyDim() {
        const isTemplate = templateWidget.value !== "Custom";
        const customEl = getWidgetEl(customWidget);
        if (customEl) {
            if (isTemplate) {
                customEl.style.opacity = "0.4";
                customEl.style.backgroundColor = "rgba(20, 20, 30, 0.5)";
                customEl.readOnly = true;
                customEl.title = "Disabled because a template is selected above.";
            } else {
                customEl.style.opacity = "1";
                customEl.style.backgroundColor = "";
                customEl.readOnly = false;
                customEl.title = "";
            }
        }
    }

    const origCallback = templateWidget.callback;
    templateWidget.callback = function (val) {
        if (origCallback) origCallback.call(this, val);
        applyDim();
    };

    setTimeout(applyDim, 200);
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
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            const data = arguments[0];
            const nodeId = this.id ?? "?";
            const hasData = !!data;
            if (isChatNode(nodeData.name)) {
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
                restoreSerializedState(this, data);
            }
            const ret = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            if (isChatNode(nodeData.name)) {
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
                console.debug(`[LLM Chat] configure hook for node ${nodeId} — after origConfigure, widgets=${this.widgets?.length}, has llm_chat_buttons=${!!this.widgets?.find(w => w.name === "llm_chat_buttons")}`);
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
                    if (chatNode.size[1] < ms[1]) chatNode.size[1] = ms[1];
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
                            // Show all widgets on canvas for direct editing
                            // (reverses hideCanvasWidgets by restoring computeSize/draw)
                            for (const w of modeNode.widgets || []) {
                                if (w.name.startsWith("_")) continue;
                                if (w.name === "llm_chat_buttons") continue;
                                if (w.type === "converted-widget") {
                                    w.type = w._originalType || "STRING";
                                    w.hidden = false;
                                    delete w.computeSize;
                                    delete w.draw;
                                }
                            }
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

            // ── Common: prompt_template management button ──
            const templateWidget = this.widgets?.find(w => w.name === "prompt_template");
            if (templateWidget) {
                this._templateWidget = templateWidget;

                // Create DOM widget for Manage Prompts button — only if NOT already done by EasyLLM section
                if (!isEasyLLM) {
                    const nonChatNode = this;
                    requestAnimationFrame(() => {
                        if (!nonChatNode.graph) {
                            console.debug(`[LLM Chat] Node ${nonChatNode.id} removed before rAF fired — skipping DOM widget`);
                            return;
                        }
                        console.debug(`[LLM Chat] Node ${nonChatNode.id} — rAF firing, creating button DOM widget (non-chat)`);
                        createButtonDOMWidget(nonChatNode);
                    });
                }
            }

            // ── Template selection: dim custom prompt widget ──
            if (templateWidget) {
                setupTemplateDimming(this, templateWidget);
            }

            // ── Bidirectional sync: canvas widget changes → _popupSettings ──
            // Wraps existing callbacks to sync widget values to popup on change.
            if (isChatNode(nodeData.name)) {
                const syncNode = this;
                // GGUF nodes need additional widget sync for their extra settings
                // Order matches the section grouping: Persona → Tuning → Hardware
                const SYNC_WIDGETS = isEasyLLMGGUF
                    ? ["prompt_template", "system_prompt_text",
                       "temperature", "max_length", "seed",
                       "chat_template", "n_ctx", "n_gpu_layers",
                       "use_mlock", "vram_mode",
                       "top_k", "top_p", "repetition_penalty"]
                    : ["prompt_template", "system_prompt_text",
                       "temperature", "max_length", "seed"];
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

                // ── CLIP disconnected indicator ──
                const clipInput = this.inputs?.find(inp => inp.name === "clip");
                if (clipInput && !clipInput.link) {
                    // ── Red border ──
                    ctx.save();
                    ctx.strokeStyle = "#d84040";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(0, 0, this.size[0], this.size[1]);
                    ctx.restore();

                    // ── Warning icon ──
                    ctx.save();
                    ctx.font = "20px sans-serif";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "top";
                    ctx.fillText("⚠️", 35, 4);
                    ctx.restore();
                }

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
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                const generatedText = extractGeneratedText(message);
                if (!generatedText) return;

                // Extract raw (uncleaned) text for popup display (preserves think tags)
                const rawText = extractRawText(message);

                // ── Extract attached base64 image data URI from backend ──
                const b64Image = message?.image?.[0] || null;

                // ── Attach returned image data URI to the last user entry ──
                if (b64Image && this._chatHistory && this._chatHistory.length > 0) {
                    const lastEntry = this._chatHistory[this._chatHistory.length - 1];
                    // Streaming may have pushed an assistant entry; walk back for the user entry
                    for (let i = this._chatHistory.length - 1; i >= 0; i--) {
                        if (this._chatHistory[i].role === "user") {
                            this._chatHistory[i].image = b64Image;
                            break;
                        }
                    }
                    // Re-render popup with resolved image; skip during streaming
                    if (this._popupHistoryEl && !this._popupStreaming) {
                        renderPopupHistory(this);
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
                        pushEnhancerEntry(this, userMessage, rawText, systemPrompt, modelName);
                    }
                    // Do NOT clear text widget — keep it for re-queue
                } else {
                    // ── CHAT MODE: Save user/assistant to chat history, clear text ──
                    // Push user message only when NOT already pre-pushed by streaming
                    if (userMessage && !this._popupStreaming) {
                        pushUserMessage(this, userMessage);
                    }

                    // Append assistant response with raw text (preserves think tags)
                    pushAssistantMessage(this, rawText, isError);

                    // Cap at 50 entries to prevent memory issues
                    capChatHistory(this);

                    // Mark canvas context cache as dirty for next onDrawForeground
                    this._contextDirty = true;

                    // Store last output for auto-mode (use cleaned text, not raw)
                    cacheGeneratedText(this, generatedText);

                    // Clear text widget (auto-mode detection: empty = Queue Prompt)
                    if (tw) tw.value = "";

                    // Clear chat_history hidden widget to prevent stale data
                    const chw = this.widgets?.find(w => w.name === "chat_history");
                    if (chw) chw.value = "";
                }

                // Update popup display if open; skip during streaming unless error
                if (this._popupHistoryEl && (!this._popupStreaming || isError)) {
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
                        if (confNode.size[1] < ms[1]) confNode.size[1] = ms[1];
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
                        // Re-show all widgets (undo any previous hideCanvasWidgets call)
                        for (const w of this.widgets || []) {
                            if (w.name.startsWith("_")) continue;
                            if (w.name === "llm_chat_buttons") continue;
                            if (w.type === "converted-widget") {
                                w.type = w._originalType || "STRING";
                                w.hidden = false;
                                delete w.computeSize;
                                delete w.draw;
                            }
                        }
                    } else {
                        // Chat mode: hide settings widgets
                        hideCanvasWidgets(this);
                    }
                }
                // Update node title to match restored mode
                updateNodeTitle(this);
            }

            // Re-apply template dimming
            const tw = findWidgetByName(this, "prompt_template");
            if (tw) {
                const customWidget = findWidgetByName(this, "system_prompt_text")
                    || findWidgetByName(this, "custom_prompt");
                const customEl = getWidgetEl(customWidget);
                if (customEl) {
                    const isTemplate = tw.value !== "Custom";
                    if (isTemplate) {
                        customEl.style.opacity = "0.4";
                        customEl.style.backgroundColor = "rgba(20, 20, 30, 0.5)";
                    } else {
                        customEl.style.opacity = "1";
                        customEl.style.backgroundColor = "";
                    }
                }
            }

            // ── Bidirectional sync callbacks (restored for loaded nodes) ──
            if (isChatNode(nodeData.name)) {
                const syncNode = this;
                // GGUF nodes sync additional hardware/tuning widgets
                const SYNC_WIDGETS = isEasyLLMGGUF
                    ? ["prompt_template", "system_prompt_text",
                       "temperature", "max_length", "seed",
                       "chat_template", "n_ctx", "n_gpu_layers",
                       "use_mlock", "vram_mode",
                       "top_k", "top_p", "repetition_penalty"]
                    : ["prompt_template", "system_prompt_text",
                       "temperature", "max_length", "seed"];
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
            }
        };
    },
});

// ── Wrap app.queuePrompt to activate streaming / progress ──

const origQueuePrompt = app.queuePrompt.bind(app);
app.queuePrompt = function (...args) {
    if (app.graph) {
        for (const node of app.graph._nodes) {
            // Only EasyLLM nodes with non-empty text
            if (!isChatNode(node.type)) continue;
            const textW = node.widgets?.find(w => w.name === "text");
            if (!textW || !textW.value || !textW.value.trim()) continue;

            const modeW = node.widgets?.find(w => w.name === "mode");
            const isEnhancer = modeW && modeW.value === "enhancer";

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
    return origQueuePrompt.apply(app, args);
};
