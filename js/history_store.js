/**
 * EasyLLM — Chat History Store
 *
 * Centralizes _chatHistory / _enhancerHistory / _cachedText / _popupSettings
 * state management: serialization, push operations with dedup guards,
 * history capping, and cache management.
 */

// ────────────────────────────────────────────────────────────────────────
// Initialization
// ────────────────────────────────────────────────────────────────────────

/**
 * Initialize all history/cache/settings state on a node.
 * Safe to call multiple times (only sets if missing).
 */
export function initHistory(node) {
    if (!node._chatHistory) node._chatHistory = [];
    if (!node._enhancerHistory) node._enhancerHistory = [];
    if (!node._cachedText) node._cachedText = "";
    if (!node._popupSettings) node._popupSettings = {};
    if (!node._uploadedImage) node._uploadedImage = null;
}

// ────────────────────────────────────────────────────────────────────────
// Serialization
// ────────────────────────────────────────────────────────────────────────

/**
 * Build the serializable state object for workflow JSON persistence.
 * Called from the serialize hook in llm_chat.js.
 */
export function getSerializableState(node) {
    // Phase 3: History is now DB-backed — no longer embedded in workflow JSON.
    // This keeps workflow files small (KB, not MB) and avoids base64 bloat.
    // History is loaded on demand from the DB when the popup opens.
    return {
        chatHistory: [],                    // Was: node._chatHistory || []
        enhancerHistory: [],                // Was: node._enhancerHistory || []
        cachedText: node._cachedText || "",
        popupSettings: node._popupSettings || {},
        uploadedImage: null,                // Was: node._uploadedImage || null
        seedRandomize: node._llmSeedRandomize,  // 🔀 state survives page refresh
    };
}

/**
 * Restore serialized state onto a node from workflow JSON data.
 * Called from the configure hook in llm_chat.js.
 */
export function restoreSerializedState(node, data) {
    if (data?.llmchat) {
        // Phase 3: History is now DB-backed — loaded on popup open via
        // _loadHistoryFromDisk() and _loadEnhancerHistoryFromDisk().
        // Keep small settings for UI continuity.
        node._chatHistory = [];             // Was: data.llmchat.chatHistory || []
        node._enhancerHistory = [];         // Was: data.llmchat.enhancerHistory || []
        node._cachedText = data.llmchat.cachedText || "";
        node._popupSettings = data.llmchat.popupSettings || {};
        node._uploadedImage = null;         // Was: data.llmchat.uploadedImage || null
        // Restore 🔀 seed-randomize state (true/false); undefined = default ON
        if (data.llmchat.seedRandomize !== undefined) {
            node._llmSeedRandomize = data.llmchat.seedRandomize;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Push operations (with dedup guards)
// ────────────────────────────────────────────────────────────────────────

/**
 * Push a user message to _chatHistory with dedup guard.
 * Skips if the last entry is already this same user message (prevents
 * duplicate pushes when both handlePopupSend and onExecuted try to save).
 * @param {object}  node     - The EasyLLM node
 * @param {string}  message  - The user message text
 * @param {Array}   [images] - Optional array of {type, filename?, data?} image objects
 */
export function pushUserMessage(node, message, images) {
    if (!node._chatHistory) node._chatHistory = [];
    const lastEntry = node._chatHistory[node._chatHistory.length - 1];
    if (!lastEntry || lastEntry.role !== "user" || lastEntry.message !== message) {
        const entry = { role: "user", message, timestamp: Date.now() };
        if (images && images.length > 0) {
            // Preserve both filename and data fields from image objects
            entry.images = images.map(img => ({
                type: img.type,
                filename: img.filename || null,  // DB-stored file (preferred)
                data: img.data || null,           // Base64 fallback
            }));
        }
        node._chatHistory.push(entry);
    }
}

/**
 * Push an assistant message to _chatHistory.
 * Skips if streaming has already saved the response (via
 * saveStreamedAssistantMessage), preventing duplicates.
 * @param {object}  node     - The EasyLLM node
 * @param {string}  message  - The assistant message text
 * @param {boolean} [isError] - Whether this message represents an error
 */
export function pushAssistantMessage(node, message, isError) {
    if (!node._chatHistory) node._chatHistory = [];

    // Dedup: if the last entry is already an assistant message,
    // update it in-place instead of pushing a duplicate.
    // This handles the race condition where onExecuted fires AFTER
    // saveStreamedAssistantMessage already pushed this entry.
    const lastEntry = node._chatHistory[node._chatHistory.length - 1];
    if (lastEntry && lastEntry.role === "assistant") {
        lastEntry.message = message;
        if (isError) lastEntry.error = true;
        return true;  // true = dedup'd, was already present
    }

    // Normal path: no duplicate present — push new entry.
    const entry = { role: "assistant", message, timestamp: Date.now() };
    if (isError) entry.error = true;
    node._chatHistory.push(entry);
    return false;  // false = new entry pushed
}

/**
 * Save a streamed assistant message from the WebSocket streaming done handler.
 * Sets _streamingSavedHistory = true so onExecuted's pushAssistantMessage
 * skips its own push, preventing duplicate assistant entries.
 * @param {object}  node     - The EasyLLM node
 * @param {string}  text     - The assistant message text
 * @param {object}  [timing] - Optional timing data
 * @param {boolean} [isError] - Whether this message represents an error
 */
export function saveStreamedAssistantMessage(node, text, timing, isError) {
    if (!node._chatHistory) node._chatHistory = [];

    // Dedup: if the last entry is already an assistant message,
    // update it in-place instead of pushing a duplicate.
    // This handles the race condition where onExecuted fires BEFORE
    // the WebSocket onDone event (the reverse of pushAssistantMessage's guard).
    const lastEntry = node._chatHistory[node._chatHistory.length - 1];
    if (lastEntry && lastEntry.role === "assistant") {
        lastEntry.message = text;
        if (timing) lastEntry.timing = timing;
        if (isError) lastEntry.error = true;
        node._streamingSavedHistory = true;
        return;
    }

    const entry = { role: "assistant", message: text, timestamp: Date.now() };
    if (timing) {
        entry.timing = timing;
    }
    if (isError) entry.error = true;
    node._chatHistory.push(entry);
    node._streamingSavedHistory = true;
}

/**
 * Push an enhancer entry (input → output) with a 100-entry cap.
 * @param {object} node             - The EasyLLM node
 * @param {string} input            - The user's input text
 * @param {string} output           - The generated output text
 * @param {string} [systemPromptText] - Full text snapshot of the effective system prompt used
 * @param {string} [modelName]       - Name of the LLM model that generated this output
 */
export function pushEnhancerEntry(node, input, output, systemPromptText, modelName, images) {
    if (!node._enhancerHistory) node._enhancerHistory = [];

    // ── Dedup guard: prevent duplicate entries from auto-queue cache replay ──
    // When the LLM node re-executes (ComfyUI cache replay triggered by
    // app.queuePrompt for image capture), onExecuted fires a second time
    // with the same data. Without this guard, duplicate entries pile up.
    // Compare with pushAssistantMessage() which dedups via lastEntry.role.
    const lastEntry = node._enhancerHistory[node._enhancerHistory.length - 1];
    if (lastEntry && lastEntry.input === input && lastEntry.output === output) {
        // Same content as last entry — update timestamp to reflect re-execution
        // but don't push a duplicate card.
        lastEntry.timestamp = Date.now();
        return;
    }

    node._enhancerHistory.push({
        input,
        output,
        timestamp: Date.now(),
        systemPromptText: systemPromptText || "",
        modelName: modelName || "",
        images: images || [],  // Optional: [{ type: "input"|"generated", filename, data }]
    });
    // Phase 3: Cap removed — DB is the source of truth for full history.
    // In-memory array is just a display cache for the current popup session.
    // Safety cap at 1000 entries (~200 KB at ~200 bytes each).
    if (node._enhancerHistory.length > 1000) {
        node._enhancerHistory = node._enhancerHistory.slice(-1000);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Trimming / Capping
// ────────────────────────────────────────────────────────────────────────

/**
 * Trim a node's history array to the specified maximum length.
 * @param {object} node - The ComfyUI node object
 * @param {string} key - Property name (e.g., "_chatHistory", "_enhancerHistory")
 * @param {number} maxLen - Maximum number of entries to keep
 */
function capHistory(node, key, maxLen) {
    if (node[key] && node[key].length > maxLen) {
        node[key] = node[key].slice(-maxLen);
    }
}

/** Trim chat history to the last 50 entries. */
export function capChatHistory(node) {
    capHistory(node, "_chatHistory", 50);
}

// ────────────────────────────────────────────────────────────────────────
// Cache Management
// ────────────────────────────────────────────────────────────────────────

/** Store generated text for auto-mode reuse. */
export function cacheGeneratedText(node, text) {
    node._cachedText = text;
}

// ────────────────────────────────────────────────────────────────────────
// Clear / Reset
// ────────────────────────────────────────────────────────────────────────

/**
 * Reset streaming state flags after generation completes.
 * Called from onExecuted cleanup.
 */
export function resetStreamingState(node) {
    node._popupStreaming = false;
}

/**
 * Serialize chat history for the backend hidden widget, with truncation.
 * Limits to the last N turns (each turn = user + assistant = 2 entries)
 * to prevent context overflow on small models.
 *
 * Excludes the most recent entry (the current user message just pushed
 * to _chatHistory before calling this function), since the backend will
 * combine the serialized history with the current message from the text widget.
 *
 * @param {object} node - The EasyLLM node
 * @param {number} maxTurns - Maximum number of conversation turns to send
 * @returns {string} JSON string suitable for the chat_history hidden widget
 */
export function serializeHistoryForBackend(node, maxTurns = 10) {
    const history = node._chatHistory || [];
    if (history.length === 0) return "";

    // Exclude the last entry (current user message just pushed before send).
    // The backend receives the current message via the "text" widget and
    // will combine it with this serialized history.
    let entries = history.slice(0, -1);

    // ── NEW: If the sliced result is empty, return empty string ──
    // Fixes: bypassed nodes with single-entry _chatHistory would return
    // "[]" (truthy), causing the server to create an empty chat session.
    if (entries.length === 0) return "";

    // Limit to maxTurns (each turn = 1 user + 1 assistant = 2 entries)
    if (entries.length > maxTurns * 2) {
        entries = entries.slice(-(maxTurns * 2));
    }

    return JSON.stringify(entries.map(e => {
        const entry = { role: e.role, message: e.message };
        // Use images array with filename (preferred) or base64 data (fallback)
        if (e.images && e.images.length > 0) {
            entry.images = e.images.map(img => {
                const result = { type: img.type };
                if (img.filename) {
                    result.filename = img.filename;  // DB-stored file (preferred)
                } else if (img.data) {
                    result.data = img.data;           // Base64 fallback
                }
                return result;
            });
        }
        return entry;
    }));
}
