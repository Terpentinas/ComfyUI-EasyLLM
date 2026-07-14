/**
 * EasyLLM — Popup barrel module
 *
 * Re-exports all public symbols from the four split sub-modules.
 * External consumers (llm_chat.js, buttons.js, websocket_bridge.js)
 * continue to `import { ... } from "./popup.js"` with zero changes.
 *
 * Original monolithic file (3106 lines) was split into:
 *   popup_utils.js        (349 lines) — pure utility functions
 *   popup_bubble.js       (946 lines) — bubble/export/image/model-name
 *   popup_chat.js         (954 lines) — open/close popup modals
 *   popup_model_browser.js (864 lines) — GGUF model browser
 */

// ── Utilities (pure functions, no project imports) ──
export {
    extractGeneratedText,
    detectError,
    formatErrorMessage,
    extractRawText,
    renderMarkdown,
    updateScrollState,
    scrollToBottom,
    autoScrollIfNeeded,
    formatTimestamp,
    formatTimingBadge,
    formatTimingTooltip,
    parseThinkBlocks,
    parseAttachedTextBlocks,
    estimateTokens,
    estimateContextTokens,
} from "./popup_utils.js";

// ── Bubble/export/image/model-name (requires app, api, websocket_bridge, history_store, ui_utils) ──
export {
    exportChat,
    exportEnhancerHistory,
    downloadExport,
    abortStreaming,
    openEnhancerExportDialog,
    createBubbleElement,
    renderPopupHistory,
    pasteToInput,
    deleteMessage,
    retryMessage,
    updateContextIndicator,
    syncPopupSettingsToCanvas,
    handlePopupSend,
    showTypingIndicator,
    hideTypingIndicator,
    getModelName,
    openImagePicker,
    handleDroppedFile,
    removeAttachedImage,
    openLightbox,
    closeLightbox,
    buildImagesSection,
} from "./popup_bubble.js";

// ── Popup modals (open/close chat, enhancer history) ──
export {
    openChatPopup,
    closeChatPopup,
    openOutputHistoryPopup,
    openSettingsPopup,
    closeSettingsPopup,
} from "./popup_chat.js";

// ── Model browser popup ──
export {
    openModelBrowserPopup,
    closeModelBrowserPopup,
} from "./popup_model_browser.js";

// ── Database Manager popup ──
export {
    openDatabaseManagerPopup,
} from "./db_manager.js";
