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
    return {
        chatHistory: node._chatHistory || [],
        enhancerHistory: node._enhancerHistory || [],
        cachedText: node._cachedText || "",
        popupSettings: node._popupSettings || {},
        uploadedImage: node._uploadedImage || null,
    };
}

/**
 * Restore serialized state onto a node from workflow JSON data.
 * Called from the configure hook in llm_chat.js.
 */
export function restoreSerializedState(node, data) {
    if (data?.llmchat) {
        node._chatHistory = data.llmchat.chatHistory || [];
        node._enhancerHistory = data.llmchat.enhancerHistory || [];
        node._cachedText = data.llmchat.cachedText || "";
        node._popupSettings = data.llmchat.popupSettings || {};
        node._uploadedImage = data.llmchat.uploadedImage || null;
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
 * @param {string}  [image]  - Optional base64 data URI of an attached image
 */
export function pushUserMessage(node, message, image) {
    if (!node._chatHistory) node._chatHistory = [];
    const lastEntry = node._chatHistory[node._chatHistory.length - 1];
    if (!lastEntry || lastEntry.role !== "user" || lastEntry.message !== message) {
        const entry = { role: "user", message, timestamp: Date.now() };
        if (image) entry.image = image;
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
        return;
    }

    // Normal path: no duplicate present — push new entry.
    const entry = { role: "assistant", message, timestamp: Date.now() };
    if (isError) entry.error = true;
    node._chatHistory.push(entry);
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
export function pushEnhancerEntry(node, input, output, systemPromptText, modelName) {
    if (!node._enhancerHistory) node._enhancerHistory = [];
    node._enhancerHistory.push({
        input,
        output,
        timestamp: Date.now(),
        systemPromptText: systemPromptText || "",
        modelName: modelName || "",
    });
    // Cap at 100 entries
    if (node._enhancerHistory.length > 100) {
        node._enhancerHistory = node._enhancerHistory.slice(-100);
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

    // Limit to maxTurns (each turn = 1 user + 1 assistant = 2 entries)
    if (entries.length > maxTurns * 2) {
        entries = entries.slice(-(maxTurns * 2));
    }

    return JSON.stringify(entries.map(e => ({
        role: e.role,
        message: e.message
    })));
}
