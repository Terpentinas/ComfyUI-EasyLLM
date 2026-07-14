/**
 * EasyLLM — Popup modal module (chat + enhancer history)
 *
 * Contains: openChatPopup, closeChatPopup, openOutputHistoryPopup.
 * These are the public entry points for opening/closing popup modals.
 */

import { createOverlayModal, showToast } from "./ui_utils.js";
import { getActivePopupNode, setActivePopupNode, clearActivePopupNode } from "./constants.js";
import { openPromptManagerDialog } from "./editor.js";
import { startProgressTracking, stopProgressTracking, stopCanvasProgressTracking } from "./websocket_bridge.js";
import { getModelName, abortStreaming, renderPopupHistory, syncPopupSettingsToCanvas, createBubbleElement, createEnhancerCardElement, hideTypingIndicator, exportChat, exportEnhancerHistory, downloadExport, handlePopupSend, openImagePicker, handleDroppedFile, removeAttachedImage, openEnhancerExportDialog, rebuildEnhancerActions } from "./popup_bubble.js";
import { scrollToBottom, estimateContextTokens, autoScrollIfNeeded, renderMarkdown, parseThinkBlocks, parseAttachedTextBlocks } from "./popup_utils.js";
import { openModelBrowserPopup, closeModelBrowserPopup } from "./popup_model_browser.js";
import { createSettingsPanel } from "./popup_settings.js";
import { fetchPrompts } from "./api.js";

// ── Helper: Load persisted history from disk on popup open (Phase 2) ──

const _PAGE_SIZE = 200;

async function _loadHistoryFromDisk(node) {
    try {
        const resp = await fetch(
            `/easyllm/db/history/${node.id}?limit=${_PAGE_SIZE}`
        );
        if (!resp.ok) return;
        const data = await resp.json();

        // Handle empty result: clear stale cache (server has no data for this node)
        if (!data.available || !data.entries || data.entries.length === 0) {
            node._chatHistory = [];
            node._historyPagination = null;
            renderPopupHistory(node);
            return;
        }

        // Server is the single source of truth; replace local cache unconditionally.
        node._chatHistory = data.entries;
        node._historyPagination = {
            totalCount: data.total_count || data.entries.length,
            loadedCount: data.entries.length,
            allLoaded: (data.entries.length >= (data.total_count || data.entries.length)),
            isLoading: false,
        };
        renderPopupHistory(node);
        console.debug(
            `[LLM Chat DB] Loaded ${data.entries.length}/${data.total_count || data.entries.length} entries from disk for node ${node.id}`
        );
    } catch (e) {
        // DB unavailable — silently keep workflow JSON data
        console.debug(`[LLM Chat DB] Failed to load history from disk: ${e}`);
    }
}

/**
 * Fetch the next page of older entries when user scrolls to the top.
 * Prepends them to node._chatHistory and re-renders.
 */
async function _loadNextHistoryPage(node) {
    const pag = node._historyPagination;
    if (!pag || pag.allLoaded || pag.isLoading) return;
    pag.isLoading = true;

    // Show loading indicator at the top of the history container
    const container = node._popupHistoryEl;
    let loadIndicator = container ? container.querySelector(".llm-history-load-more") : null;
    if (loadIndicator) {
        loadIndicator.textContent = "⏳ Loading earlier messages...";
    }

    try {
        const resp = await fetch(
            `/easyllm/db/history/${node.id}?offset=${pag.loadedCount}&limit=${_PAGE_SIZE}`
        );
        if (!resp.ok) {
            pag.allLoaded = true;
            return;
        }
        const data = await resp.json();
        if (!data.entries || data.entries.length === 0) {
            pag.allLoaded = true;
            if (loadIndicator) {
                loadIndicator.textContent = "✓ All messages loaded";
                setTimeout(() => { if (loadIndicator?.parentNode) loadIndicator.remove(); }, 1500);
            }
            return;
        }

        // Save current scroll height for position restoration after prepend
        const prevScrollHeight = container ? container.scrollHeight : 0;

        // Prepend older entries
        node._chatHistory = [...data.entries, ...node._chatHistory];
        pag.loadedCount += data.entries.length;
        if (pag.loadedCount >= pag.totalCount) {
            pag.allLoaded = true;
        }

        // Re-render
        renderPopupHistory(node);

        // Restore scroll position: new content was prepended above, so scroll down
        // by the difference in scrollHeight to keep the same visual position.
        if (container) {
            const newScrollHeight = container.scrollHeight;
            container.scrollTop = newScrollHeight - prevScrollHeight;
        }

        // If all loaded, update the indicator
        if (pag.allLoaded && loadIndicator) {
            loadIndicator.textContent = "✓ All messages loaded";
            setTimeout(() => { if (loadIndicator?.parentNode) loadIndicator.remove(); }, 1500);
        }
    } catch (e) {
        console.debug(`[LLM Chat DB] Failed to load next page: ${e}`);
    } finally {
        pag.isLoading = false;
    }
}

/**
 * Load ALL remaining history entries (used before save-on-close).
 * Idempotent if all entries are already loaded.
 */
async function _loadRemainingHistory(node) {
    const pag = node._historyPagination;
    if (!pag || pag.allLoaded || pag.isLoading) return;
    pag.isLoading = true;
    try {
        while (!pag.allLoaded) {
            const resp = await fetch(
                `/easyllm/db/history/${node.id}?offset=${pag.loadedCount}&limit=${_PAGE_SIZE}`
            );
            if (!resp.ok) break;
            const data = await resp.json();
            if (!data.entries || data.entries.length === 0) break;
            node._chatHistory = [...data.entries, ...node._chatHistory];
            pag.loadedCount += data.entries.length;
            if (pag.loadedCount >= pag.totalCount) {
                pag.allLoaded = true;
            }
        }
    } catch (e) {
        console.debug(`[LLM Chat DB] Failed to load remaining history: ${e}`);
    } finally {
        pag.isLoading = false;
    }
}

// ── Helper: Load enhancer history from disk on popup open ──
// Server is the single source of truth — ImageCapture now persists images
// via /append-images (server-first), so local enrichments are never ahead of disk.

async function _loadEnhancerHistoryFromDisk(node) {
    // ── Guard: prevent re-entrant close/re-open infinite loop ──
    if (node._loadingEnhancerHistory) return;
    node._loadingEnhancerHistory = true;
    try {
        const resp = await fetch(
            `/easyllm/db/history/${node.id}?type=enhancer`
        );
        if (!resp.ok) return;
        const data = await resp.json();

        // Handle empty result: clear stale cache (server has no enhancer data for this node)
        if (!data.available || !data.entries || data.entries.length === 0) {
            node._enhancerHistory = [];
            return;
        }

        // ── Staleness check: prevent overwriting newer in-memory data ──
        // The in-memory _enhancerHistory is written by onExecuted → pushEnhancerEntry()
        // before the popup even opens. Disk data can lag behind if:
        //   • The DB write from onExecuted hasn't completed yet
        //   • The user opened the popup very quickly after queueing
        // Unconditionally replacing with disk data would lose the latest
        // entry's raw text (which contains <think> blocks for the "Thought:" row).
        // Only replace when disk has MORE entries than in-memory.
        const inMemoryCount = (node._enhancerHistory || []).length;
        const diskCount = data.entries.length;
        if (diskCount <= inMemoryCount) {
            // Disk has same or fewer entries — in-memory is already current.
            return;
        }

        // Disk has more entries — merge: keep in-memory entries that disk doesn't have
        // (avoids replacing rich in-memory objects with serialized disk copies)
        if (inMemoryCount > 0 && diskCount > inMemoryCount) {
            // Only append the new disk entries that in-memory doesn't have
            const newDiskEntries = data.entries.slice(0, diskCount - inMemoryCount);
            node._enhancerHistory = [...newDiskEntries, ...node._enhancerHistory];
        } else {
            // No in-memory data — use disk data as-is
            node._enhancerHistory = data.entries;
        }

        // Re-open the popup to re-render if overlay is already open
        if (node._popupOverlay) {
            await closeChatPopup(node);
            openOutputHistoryPopup(node);
        }
    } catch (e) {
        console.debug(`[LLM Chat DB] Failed to load enhancer history from disk: ${e}`);
    } finally {
        node._loadingEnhancerHistory = false;
    }
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Open the chat popup modal for a given node
// ────────────────────────────────────────────────────────────────────────

export function openChatPopup(node) {
    if (node._popupOverlay) return; // Already open

    // Close any other open popup first (only one popup at a time)
    if (getActivePopupNode() && getActivePopupNode() !== node) {
        closeChatPopup(getActivePopupNode());
    }

    const nodeLabel = node.title || `Node #${node.id}`;

    // Get model name for header display
    const modelName = getModelName(node);

    const headerTitle = `🤖 EasyLLM${modelName ? ` | ${modelName}` : ""}`;

    const { overlay, panel, body, footer, header } = createOverlayModal(
        "llm-popup",
        headerTitle,
        () => closeChatPopup(node),
        { hasFooter: true }
    );

    // Store panel reference for resize persistence
    node._popupPanel = panel;

    // Restore saved popup dimensions (from CSS resize)
    const savedW = node._popupSettings?.popupWidth;
    const savedH = node._popupSettings?.popupHeight;
    if (savedW) {
        panel.style.width = savedW;
        console.debug(`[LLM Chat] Restored popup width: ${savedW}`);
    }
    if (savedH) {
        panel.style.height = savedH;
        console.debug(`[LLM Chat] Restored popup height: ${savedH}`);
    }

    // ── Camera badge: visible when IMAGE socket is connected
    //    OR image uploaded via popup ──
    const cameraBadge = document.createElement("span");
    cameraBadge.className = "llm-popup-camera-badge";
    cameraBadge.textContent = "📷";
    cameraBadge.title = "Image input connected";
    // Check if IMAGE socket is connected, or image uploaded
    const imageInput = node.inputs?.find(i => i.name === "image");
    const hasWiredImage = imageInput && imageInput.link != null;
    const hasUploadedImage = !!node._uploadedImage;
    cameraBadge.style.display = (hasWiredImage || hasUploadedImage) ? "inline-block" : "none";
    // Insert before the close button so it stays left of ✕
    let closeBtn = header.querySelector(".llm-popup-close-btn");
    if (closeBtn) {
        header.insertBefore(cameraBadge, closeBtn);
    } else {
        header.appendChild(cameraBadge);
    }

    // ── Header action buttons (clear, export, stop) ──
    const headerActions = document.createElement("div");
    headerActions.className = "llm-popup-header-actions";

    // Clear button
    const clearBtn = document.createElement("button");
    clearBtn.className = "llm-popup-header-btn llm-popup-clear-btn";
    clearBtn.textContent = "🗑 Clear";
    clearBtn.title = "Clear conversation";
    clearBtn.onclick = async () => {
        if (confirm("Clear the entire conversation?")) {
            node._chatHistory = [];
            // Also clear any uploaded image (conversation context is gone)
            if (node._uploadedImage) {
                removeAttachedImage(node);
            }
            renderPopupHistory(node);

            try {
                const resp = await fetch(`/easyllm/db/history/${node.id}`, { method: "DELETE" });
                const result = await resp.json();
                if (result.success) {
                    showToast("Chat cleared", "success", 2000);
                } else {
                    showToast(`Failed to clear on server: ${result.error}`, "error", 3000);
                }
            } catch (e) {
                showToast(`Failed to clear on server: ${e.message}`, "error", 3000);
            }
        }
    };
    headerActions.appendChild(clearBtn);

    // Stop button (hidden by default, shown during generation)
    const stopBtn = document.createElement("button");
    stopBtn.className = "llm-popup-header-btn llm-popup-stop-btn";
    stopBtn.textContent = "■ Stop";
    stopBtn.title = "Stop generation";
    stopBtn.style.display = "none";
    stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping...";
        await abortStreaming(node);
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.textContent = "■ Stop";
        hideTypingIndicator(node);
        showToast("Generation stopped", "info", 2000);
    };
    node._popupStopBtn = stopBtn;
    headerActions.appendChild(stopBtn);

    // Insert header actions before the close button
    closeBtn = header.querySelector(".llm-popup-close-btn");
    if (closeBtn) {
        header.insertBefore(headerActions, closeBtn);
    }

    // ── Progress bar (thin strip below header, hidden by default) ──
    const progressBar = document.createElement("div");
    progressBar.className = "llm-popup-progress";
    progressBar.style.display = "none";
    const progressFill = document.createElement("div");
    progressFill.className = "llm-popup-progress-fill";
    progressBar.appendChild(progressFill);
    panel.insertBefore(progressBar, body);
    node._popupProgressBar = progressBar;
    node._popupProgressFill = progressFill;

    // Start tracking progress events for this node
    startProgressTracking(node);

    // ── Chat history ──
    const historyContainer = document.createElement("div");
    historyContainer.className = "llm-popup-history";
    node._popupHistoryEl = historyContainer;
    body.appendChild(historyContainer);

    // ── Lazy pagination: load older entries when scrolling near the top ──
    historyContainer.addEventListener("scroll", () => {
        const pag = node._historyPagination;
        if (!pag || pag.allLoaded || pag.isLoading) return;
        // User scrolled near the top of the container — load the next page
        // of older entries (threshold: 80px from top).
        if (historyContainer.scrollTop < 80) {
            _loadNextHistoryPage(node);
        }
    });

    // ── Drag-and-drop file upload (images + text files) ──
    const dropZone = document.createElement("div");
    dropZone.className = "llm-popup-drop-zone";
    dropZone.textContent = "📎 Drop image or text file";
    body.appendChild(dropZone);
    node._popupDropZone = dropZone;
    node._popupDragCounter = 0;

    // Drag enter (increment counter — fires for children too)
    body.addEventListener("dragenter", (e) => {
        e.preventDefault();
        e.stopPropagation();
        node._popupDragCounter++;
        if (node._popupDragCounter === 1) {
            body.classList.add("llm-popup-body-dragover");
        }
    });

    // Drag over (must prevent default to allow drop)
    body.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
    });

    // Drag leave (decrement counter — only remove class when all leave)
    body.addEventListener("dragleave", (e) => {
        e.preventDefault();
        e.stopPropagation();
        node._popupDragCounter--;
        if (node._popupDragCounter <= 0) {
            node._popupDragCounter = 0;
            body.classList.remove("llm-popup-body-dragover");
        }
    });

    // Drop
    body.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        node._popupDragCounter = 0;
        body.classList.remove("llm-popup-body-dragover");

        const files = e.dataTransfer?.files;
        if (files?.length) {
            handleDroppedFile(node, files[0]);
        }
    });

    // ── Context indicator ──
    const contextEl = document.createElement("div");
    contextEl.className = "llm-chat-context-indicator";
    contextEl.style.display = "none";
    body.appendChild(contextEl);
    node._popupContextEl = contextEl;

    // Render history (async to let DOM attach)
    requestAnimationFrame(() => {
        renderPopupHistory(node);

        // ── NEW: try loading persisted history from disk (Phase 2) ──
        _loadHistoryFromDisk(node);

        // If streaming was ongoing while popup was closed, pre-fill the
        // assistant bubble with all tokens accumulated so far.
        if (node._streamingAccumulatedText && node._popupStreaming) {
            const bubble = createBubbleElement("assistant", node._streamingAccumulatedText);
            node._popupHistoryEl.appendChild(bubble);
            node._currentStreamBubble = bubble;
            autoScrollIfNeeded(node._popupHistoryEl);
        }
    });

    // ── Input row ──
    const inputRow = document.createElement("div");
    inputRow.className = "llm-popup-input-row";

    // ── File attachment button (image or text) ──
    const attachBtn = document.createElement("button");
    attachBtn.className = "llm-popup-attach-btn";
    attachBtn.textContent = "📎";
    attachBtn.title = "Attach an image or text file";
    attachBtn.onclick = () => {
        // Disable during active generation
        if (node._popupStreaming) {
            showToast("Cannot attach file during generation", "warning", 2000);
            return;
        }
        openImagePicker(node);
    };
    inputRow.appendChild(attachBtn);
    node._popupAttachBtn = attachBtn;

    const textInput = document.createElement("textarea");
    textInput.className = "llm-popup-input";
    textInput.placeholder = "Type a message... (Enter to send, Esc to close)";
    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handlePopupSend(node);
        }
    });
    // Multi-line auto-resize
    textInput.addEventListener("input", () => {
        textInput.style.height = "auto";
        textInput.style.height = Math.min(textInput.scrollHeight, 120) + "px";
    });
    // Pre-fill popup input from canvas text widget
    const canvasTextWidget = node.widgets?.find(w => w.name === "text");
    if (canvasTextWidget?.value) {
        textInput.value = canvasTextWidget.value;
    }
    node._popupInputEl = textInput;
    inputRow.appendChild(textInput);

    const sendBtn = document.createElement("button");
    sendBtn.className = "llm-popup-send-btn";
    sendBtn.textContent = "▶ Send";
    sendBtn.onclick = () => handlePopupSend(node);
    node._popupSendBtn = sendBtn;
    inputRow.appendChild(sendBtn);
    body.appendChild(inputRow);

    // ── Footer (Prompt Library + System Prompt Dropdown + Models + Export + Save & Close) ──
    const manageBtn = document.createElement("button");
    manageBtn.className = "llm-popup-manage-btn";
    manageBtn.textContent = "📚 Prompt Library";
    manageBtn.onclick = () => openPromptManagerDialog();
    footer.appendChild(manageBtn);

    // ── System Prompt Dropdown (only when no chained system prompt is connected) ──
    const sysPromptInput = node.inputs?.find(i => i.name === "system_prompt");
    const hasChainedSystemPrompt = sysPromptInput && sysPromptInput.link != null;

    if (!hasChainedSystemPrompt) {
        const sysPromptSelect = document.createElement("select");
        sysPromptSelect.className = "llm-popup-sysprompt-select";
        sysPromptSelect.title = "Select a system prompt from the library";

        // Default "none" option
        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "— No System Prompt —";
        sysPromptSelect.appendChild(defaultOpt);

        // Populate dropdown from prompt library; refreshes on "llm-prompts-updated" event
        async function refreshSystemPromptDropdown() {
            // Read the persisted selection FIRST (before async fetch can race with it)
            const currentValue = node._popupSystemPrompt || sysPromptSelect.value;
            // Clear all options except the default
            while (sysPromptSelect.options.length > 1) {
                sysPromptSelect.remove(1);
            }
            try {
                const { prompts } = await fetchPrompts();
                // Deduplicate by name
                const seen = new Set();
                for (const p of prompts) {
                    if (!p.name || seen.has(p.name)) continue;
                    seen.add(p.name);
                    const opt = document.createElement("option");
                    opt.value = p.prompt;
                    opt.textContent = `${p.name} (${p.category || "General"})`;
                    sysPromptSelect.appendChild(opt);
                }
            } catch (_e) {
                console.debug("[LLM Chat] Failed to load prompts for dropdown:", _e);
            }
            // Restore selection if the value still exists; otherwise keep default
            sysPromptSelect.value = currentValue;
            node._popupSystemPrompt = sysPromptSelect.value;
        }

        // Initial population (restoration handled inside the function via currentValue)
        refreshSystemPromptDropdown();

        sysPromptSelect.onchange = () => {
            node._popupSystemPrompt = sysPromptSelect.value;
        };

        // Listen for prompt library updates (from api.js after save/import/delete)
        const promptsUpdatedHandler = () => {
            if (sysPromptSelect.isConnected) {
                refreshSystemPromptDropdown();
            }
        };
        document.addEventListener("llm-prompts-updated", promptsUpdatedHandler);
        node._promptsUpdatedHandler = promptsUpdatedHandler;

        footer.appendChild(sysPromptSelect);
    }

    // 📁 Models button (GGUF-only — opens model browser popup)
    if (node.isEasyLLMGGUF) {
        const modelBtn = document.createElement("button");
        modelBtn.className = "llm-popup-header-btn";
        modelBtn.style.marginLeft = "auto";
        modelBtn.textContent = "📁 Models";
        modelBtn.title = "Browse and select GGUF model";
        modelBtn.onclick = () => openModelBrowserPopup(node);
        footer.appendChild(modelBtn);
    }

    // 📥 Export button with dropdown
    const exportBtn = document.createElement("button");
    exportBtn.className = "llm-popup-header-btn llm-popup-export-btn llm-popup-export-btn--footer";
    if (!node.isEasyLLMGGUF) {
        exportBtn.style.marginLeft = "auto";
    }
    exportBtn.textContent = "📥 Export";
    exportBtn.title = "Export conversation";
    exportBtn.onclick = (e) => {
        e.stopPropagation();
        const existing = exportBtn.querySelector(".llm-popup-export-dropdown");
        if (existing) {
            existing.remove();
            return;
        }
        const dropdown = document.createElement("div");
        dropdown.className = "llm-popup-export-dropdown";
        dropdown.onclick = (ev) => ev.stopPropagation();

        const mdOption = document.createElement("button");
        mdOption.textContent = "Markdown (.md)";
        mdOption.onclick = () => {
            const result = exportChat(node._chatHistory, "md", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Markdown", "success", 2000);
            } else {
                showToast("No messages to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(mdOption);

        const txtOption = document.createElement("button");
        txtOption.textContent = "Plain Text (.txt)";
        txtOption.onclick = () => {
            const result = exportChat(node._chatHistory, "txt", nodeLabel);
            if (result) {
                downloadExport(result);
                showToast("✅ Exported as Plain Text", "success", 2000);
            } else {
                showToast("No messages to export", "error", 2000);
            }
            dropdown.remove();
        };
        dropdown.appendChild(txtOption);

        exportBtn.appendChild(dropdown);
        setTimeout(() => {
            document.addEventListener("click", () => { if (dropdown.parentNode) dropdown.remove(); }, { once: true });
        }, 0);
    };
    footer.appendChild(exportBtn);

    const saveBtn = document.createElement("button");
    saveBtn.className = "llm-popup-save-btn";
    saveBtn.textContent = "✕ Close";
    saveBtn.onclick = () => {
        // Capture popup panel dimensions for persistence
        if (node._popupPanel) {
            const w = node._popupPanel.offsetWidth;
            const h = node._popupPanel.offsetHeight;
            if (!node._popupSettings) node._popupSettings = {};
            node._popupSettings.popupWidth = w + "px";
            node._popupSettings.popupHeight = h + "px";
        }
        closeChatPopup(node);
    };
    footer.appendChild(saveBtn);

    // ── Global keyboard shortcuts ──
    const globalKeyHandler = (e) => {
        // Escape to close popup
        if (e.key === "Escape") {
            closeChatPopup(node);
        }
        // Ctrl+L or Cmd+K to clear chat
        if ((e.ctrlKey || e.metaKey) && (e.key === "l" || e.key === "k")) {
            if (node._chatHistory && node._chatHistory.length > 0) {
                e.preventDefault();
                if (confirm("Clear the entire conversation?")) {
                    // ── Server-first clear (DELETE on server, then clear cache) ──
                    (async () => {
                        try {
                            const resp = await fetch(`/easyllm/db/history/${node.id}?type=chat`, {
                                method: "DELETE",
                            });
                            if (!resp.ok) {
                                console.warn("[LLM Chat DB] Server DELETE returned non-ok:", resp.status);
                                showToast("Server delete failed, cleared locally only", "warning", 3000);
                            }
                        } catch (_err) {
                            console.warn("[LLM Chat DB] Server DELETE failed:", _err);
                            showToast("Server delete failed, cleared locally only", "warning", 3000);
                        }
                        // Always clear local cache regardless of server result (best-effort)
                        node._chatHistory = [];
                        // Also clear any uploaded image (conversation context is gone)
                        if (node._uploadedImage) {
                            removeAttachedImage(node);
                        }
                        renderPopupHistory(node);
                        showToast("Chat cleared", "success", 2000);
                    })();
                }
            }
        }
    };
    document.addEventListener("keydown", globalKeyHandler);
    node._popupKeyHandler = globalKeyHandler;

    // Store references and append
    node._popupOverlay = overlay;
    setActivePopupNode(node);
    document.body.appendChild(overlay);

    // Focus the input
    setTimeout(() => textInput.focus(), 100);
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Close the chat popup
// ────────────────────────────────────────────────────────────────────────

export async function closeChatPopup(node) {
    // ── If history was lazily loaded (paginated), load remaining
    //    entries first so the in-memory cache is complete ──
    if (node._historyPagination && !node._historyPagination.allLoaded) {
        await _loadRemainingHistory(node);
    }

    // NOTE: No explicit persist-to-disk is needed here.
    // Every entry was already persisted incrementally during creation:
    //   - User messages: handlePopupSend() calls /append
    //   - Assistant messages: WebSocket done handler calls /append
    //   - Enhancer entries: onExecuted() calls /append
    // See: plans/server-db-cleanup-remaining-code.md

    // Remove global keyboard handler
    if (node._popupKeyHandler) {
        document.removeEventListener("keydown", node._popupKeyHandler);
        node._popupKeyHandler = null;
    }

    // Streaming: do NOT abort — tokens accumulate in _streamingAccumulatedText
    // for display when popup reopens.
    try {
        await fetch(`/easyllm/popup_inactive/${node.id}`, { method: "POST" });
    } catch (_e) {
        // Non-critical
    }
    // Reset streaming bubble reference (DOM element is destroyed below)
    node._currentStreamBubble = null;

    // Stop progress tracking (popup bar + canvas green bar)
    stopProgressTracking();
    stopCanvasProgressTracking(node);

    // Close model browser popup if open
    closeModelBrowserPopup(node);

    // Remove prompts-updated listener
    if (node._promptsUpdatedHandler) {
        document.removeEventListener("llm-prompts-updated", node._promptsUpdatedHandler);
        node._promptsUpdatedHandler = null;
    }

    if (node._popupOverlay && node._popupOverlay.parentElement) {
        node._popupOverlay.parentElement.removeChild(node._popupOverlay);
    }
    node._popupOverlay = null;
    node._popupPanel = null;
    node._popupHistoryEl = null;
    node._popupInputEl = null;
    node._popupProgressBar = null;
    node._popupProgressFill = null;
    node._popupContextEl = null;
    node._popupEnhancerContextEl = null;
    node._popupStopBtn = null;
    node._popupSendBtn = null;
    node._popupAttachBtn = null;
    node._popupImagePreview = null;
    node._popupDropZone = null;
    node._popupDragCounter = null;
    if (getActivePopupNode() === node) {
        clearActivePopupNode();
    }
}

// ────────────────────────────────────────────────────────────────────────
// Enhancer Mode: Context indicator for output history popup
// ────────────────────────────────────────────────────────────────────────

/**
 * Update the context token indicator in the enhancer history popup.
 * Normalizes enhancer entries (input → user, output → assistant) and
 * estimates context using the existing estimateContextTokens() pipeline.
 * Uses per-entry systemPromptText snapshots where available.
 */
function updateEnhancerContextIndicator(node) {
    const el = node._popupEnhancerContextEl;
    if (!el) return;

    const history = node._enhancerHistory || [];
    if (history.length === 0) {
        el.textContent = "";
        el.style.display = "none";
        el.title = "";
        return;
    }

    // Normalize enhancer entries to {message, role} pairs for estimateContextTokens
    const normalizedHistory = [];
    let systemPrompt = "";
    for (const entry of history) {
        normalizedHistory.push({ message: entry.input, role: "user" });
        normalizedHistory.push({ message: entry.output, role: "assistant" });
        // Use systemPromptText from the most recent entry that has one
        if (entry.systemPromptText) {
            systemPrompt = entry.systemPromptText;
        }
    }

    // Determine chat template (same logic as getChatTemplate in popup_bubble.js).
    // "auto" is resolved server-side; use "llama" as default for token estimation.
    let chatTemplate = "llama";
    if (node.isEasyLLMGGUF) {
        const ctW = node.widgets?.find(w => w.name === "chat_template");
        if (ctW?.value && ctW.value !== "auto") chatTemplate = ctW.value;
    }

    const result = estimateContextTokens(normalizedHistory, {
        systemPrompt,
        chatTemplate,
        countImages: false,  // Enhancer entries don't carry images
    });
    const { total: used, system: sysTokens, history: historyTokens } = result;

    // Get max context: GGUF reads from n_ctx widget, CLIP uses _maxContextTokens or default 2048
    let maxCtx = node._maxContextTokens;
    if (!maxCtx && node.isEasyLLMGGUF) {
        const nCtxWidget = node.widgets?.find(w => w.name === "n_ctx");
        maxCtx = nCtxWidget ? parseInt(nCtxWidget.value) : 4096;
    }
    if (!maxCtx) maxCtx = 2048;

    el.textContent = `Context: ~${used} / ${maxCtx}`;
    el.style.display = "block";

    // Breakdown tooltip
    const tooltipParts = [];
    if (sysTokens > 0) tooltipParts.push(`System: ~${sysTokens}`);
    tooltipParts.push(`History: ~${historyTokens}`);
    el.title = `Context breakdown — ${tooltipParts.join(" | ")}`;

    // Color coding: warn at 70%, danger at 90%
    const ratio = used / maxCtx;
    if (ratio > 0.9) {
        el.style.color = "#c0392b"; // Red — critical
    } else if (ratio > 0.7) {
        el.style.color = "#d8a050"; // Amber — warning
    } else {
        el.style.color = "#94a3b8"; // Gray — normal
    }
}

// ────────────────────────────────────────────────────────────────────────
// Enhancer Mode: Open Output History Popup (read-only history log)
// ────────────────────────────────────────────────────────────────────────

/**
 * Open a read-only output history popup for enhancer mode.
 * Shows input/output pairs with scroll, progress bar, export, and stop controls.
 */
export async function openOutputHistoryPopup(node) {
    if (node._popupOverlay) return; // Already open
    if (getActivePopupNode() && getActivePopupNode() !== node) {
        await closeChatPopup(getActivePopupNode());
    }

    const nodeLabel = node.title || `Node #${node.id}`;
    const { overlay, panel, body, footer, header } = createOverlayModal(
        "llm-popup",
        `${nodeLabel} | 📋 Output History`,
        () => closeChatPopup(node),
        { hasFooter: true }
    );

    // ── Header action buttons (stop) ──
    const headerActions = document.createElement("div");
    headerActions.className = "llm-popup-header-actions";

    // Stop button (hidden by default, shown during generation)
    const stopBtn = document.createElement("button");
    stopBtn.className = "llm-popup-header-btn llm-popup-stop-btn";
    stopBtn.textContent = "■ Stop";
    stopBtn.title = "Stop generation";
    stopBtn.style.display = "none";
    stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping...";
        await abortStreaming(node);
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        stopBtn.textContent = "■ Stop";
        showToast("Generation stopped", "info", 2000);
    };
    node._popupStopBtn = stopBtn;
    headerActions.appendChild(stopBtn);

    // Insert header actions before the close button
    const closeBtn = header.querySelector(".llm-popup-close-btn");
    if (closeBtn) {
        header.insertBefore(headerActions, closeBtn);
    }

    // ── Progress bar (thin strip below header, hidden by default) ──
    const progressBar = document.createElement("div");
    progressBar.className = "llm-popup-progress";
    progressBar.style.display = "none";
    const progressFill = document.createElement("div");
    progressFill.className = "llm-popup-progress-fill";
    progressBar.appendChild(progressFill);
    panel.insertBefore(progressBar, body);
    node._popupProgressBar = progressBar;
    node._popupProgressFill = progressFill;
    startProgressTracking(node);

    // Store panel reference for resize / cleanup
    node._popupPanel = panel;

    // ── Scrollable history container ──
    const historyContainer = document.createElement("div");
    historyContainer.className = "llm-popup-history";
    node._popupHistoryEl = historyContainer;
    body.appendChild(historyContainer);

    const history = node._enhancerHistory || [];

    // ── Selection state (hoisted for updateFooterState) ──
    const selectedEntries = new Set();

    if (history.length === 0) {
        const empty = document.createElement("div");
        empty.className = "llm-popup-history-empty";
        empty.textContent = "🔍 Queue Prompt to generate enhanced prompts. Results appear here.";
        historyContainer.appendChild(empty);
    } else {
        /**
         * Estimate token count for a single enhancer entry.
         * @param {Object} entry
         * @returns {number} Estimated token count
         */
        function estimateEntryTokens(entry) {
            try {
                const normalized = [
                    { message: entry.input, role: "user" },
                    { message: entry.output, role: "assistant" },
                ];
                let systemPrompt = entry.systemPromptText || "";
                const result = estimateContextTokens(normalized, {
                    systemPrompt,
                    chatTemplate: "llama",
                    countImages: false,
                });
                return result.total || 0;
            } catch {
                return 0;
            }
        }

        // Paste-to-input handler
        const pasteToInputHandler = (text) => {
            const textW = node.widgets?.find(w => w.name === "text");
            if (textW) {
                textW.value = text;
                showToast("✏️ Pasted to text widget", "info", 2000);
            }
        };

        // Show oldest entries first (chronological order), with newest at bottom
        for (const entry of history) {
            const tokenCount = estimateEntryTokens(entry);

            const card = createEnhancerCardElement(entry, {
                isSelected: selectedEntries.has(entry),
                tokenCount,
                onPasteToInput: pasteToInputHandler,
                onSelect: (ent, isSelected) => {
                    if (isSelected) {
                        selectedEntries.add(ent);
                    } else {
                        selectedEntries.delete(ent);
                    }
                    updateFooterState();
                },
                onEdit: function saveOnEdit(newText) {
                    // 1. Update in-memory entry
                    entry.output = newText;
                    // 2. In-place DOM update on the card element (no full re-render)
                    const outputTextEl = card.querySelector(".enhancer-card-output-text");
                    if (outputTextEl) {
                        const parsed = parseThinkBlocks(newText);
                        const displayText2 = parsed.response || newText;
                        const attachParsed2 = parseAttachedTextBlocks(displayText2);
                        outputTextEl.innerHTML = renderMarkdown(attachParsed2.displayText);
                    }
                    // Restore action buttons (edit + copy) via shared helper
                    const outputContainer = card.querySelector(".enhancer-card-output");
                    if (outputContainer) {
                        rebuildEnhancerActions(outputContainer, entry, { onEdit: saveOnEdit }, newText);
                    }
                    // 3. Persist to disk via incremental update (not full-replace)
                    const enhancerIndex = node._enhancerHistory.indexOf(entry);
                    if (enhancerIndex >= 0) {
                        fetch(`/easyllm/db/history/${node.id}/entry`, {
                            method: "PUT",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                index: enhancerIndex,
                                entry: entry,
                                type: "enhancer",
                            }),
                        }).catch(() => {});
                    }
                    // 4. Show feedback
                    showToast("✅ Output updated", "success", 2000);
                },
            });
            historyContainer.appendChild(card);
        }

        // Initial scroll to bottom (instant, no smooth animation — ensures
        // the <summary>📋 Details</summary> of the last card is fully visible)
        requestAnimationFrame(() => {
            autoScrollIfNeeded(historyContainer);
        });
    }

    // ── Context indicator (below history) ──
    const contextEl = document.createElement("div");
    contextEl.className = "llm-chat-context-indicator";
    contextEl.style.display = "none";
    body.appendChild(contextEl);
    node._popupEnhancerContextEl = contextEl;

    // Show context indicator after DOM is ready
    requestAnimationFrame(() => {
        updateEnhancerContextIndicator(node);
    });

    // ── Footer (enhancer-card-footer bar) ──
    const footerBar = document.createElement("div");
    footerBar.className = "enhancer-card-footer";

    // Function to update footer button states based on selection
    function updateFooterState() {
        const count = selectedEntries.size;
        const total = history.length;

        // Update select all button text
        if (count === total && total > 0) {
            selectAllBtn.textContent = "❌ Deselect All";
        } else {
            selectAllBtn.textContent = "✅ Select All";
        }

        // Update export selected button
        if (count > 0) {
            exportSelectedBtn.textContent = `📥 Export Selected (${count})`;
            exportSelectedBtn.disabled = false;
        } else {
            exportSelectedBtn.textContent = "📥 Export Selected";
            exportSelectedBtn.disabled = true;
        }

        // Update selection count label
        if (count > 0) {
            selectionLabel.textContent = `${count} of ${total} selected`;
            selectionLabel.style.display = "block";
        } else {
            selectionLabel.textContent = "";
            selectionLabel.style.display = "none";
        }

        // Update remove button text based on selection
        if (count > 0) {
            removeAllBtn.textContent = `🗑 Remove Selected (${count})`;
            removeAllBtn.title = "Remove selected enhancer history entries";
        } else {
            removeAllBtn.textContent = "🗑 Remove All";
            removeAllBtn.title = "Clear all enhancer history";
        }
    }

    // Select All / Deselect All button
    const selectAllBtn = document.createElement("button");
    selectAllBtn.className = "enhancer-card-footer-btn";
    selectAllBtn.textContent = "✅ Select All";
    selectAllBtn.title = "Toggle selection of all entries";
    selectAllBtn.onclick = () => {
        const allSelected = selectedEntries.size === history.length;
        const cards = historyContainer.querySelectorAll(".enhancer-card");
        cards.forEach((card, idx) => {
            const entry = history[idx];
            if (allSelected) {
                card.classList.remove("enhancer-card-selected");
                selectedEntries.delete(entry);
            } else {
                card.classList.add("enhancer-card-selected");
                selectedEntries.add(entry);
            }
        });
        updateFooterState();
    };
    footerBar.appendChild(selectAllBtn);

    // Export Selected button — opens enhanced export dialog
    const exportSelectedBtn = document.createElement("button");
    exportSelectedBtn.className = "enhancer-card-footer-btn";
    exportSelectedBtn.textContent = "📥 Export Selected";
    exportSelectedBtn.title = "Export selected entries with options";
    exportSelectedBtn.disabled = true;
    exportSelectedBtn.onclick = () => {
        openEnhancerExportDialog(node, selectedEntries, nodeLabel);
    };
    footerBar.appendChild(exportSelectedBtn);

    // Selection count label (right-aligned)
    const selectionLabel = document.createElement("span");
    selectionLabel.className = "enhancer-card-footer-label";
    selectionLabel.style.display = "none";
    footerBar.appendChild(selectionLabel);

    // Remove button (danger style — matches Reset All from model browser)
    const removeAllBtn = document.createElement("button");
    removeAllBtn.className = "llm-model-browser-btn-danger";
    removeAllBtn.textContent = "🗑 Remove All";
    removeAllBtn.title = "Clear all enhancer history";
    removeAllBtn.onclick = async () => {
        const count = selectedEntries.size;
        if (count > 0) {
            // Remove selected entries only
            if (!confirm(`🗑 Remove ${count} selected enhancer history entr${count === 1 ? "y" : "ies"}?`)) return;
            node._enhancerHistory = history.filter(entry => !selectedEntries.has(entry));
        } else {
            // Remove all entries
            if (!confirm("🗑 Are you sure you want to remove all enhancer history entries?")) return;
            node._enhancerHistory = [];
        }
        // Persist remaining entries to disk (overwrite with filtered list)
        try {
            if (node._enhancerHistory.length === 0) {
                // All entries removed — use DELETE endpoint which calls clear_history()
                const resp = await fetch(`/easyllm/db/history/${node.id}?type=enhancer`, {
                    method: "DELETE",
                });
                const result = await resp.json();
                if (!result.success) {
                    showToast(`Failed to persist on server: ${result.error || "Unknown error"}`, "error", 3000);
                }
            } else {
                const resp = await fetch(`/easyllm/db/history/${node.id}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ entries: node._enhancerHistory, type: "enhancer" }),
                });
                const result = await resp.json();
                if (!result.success) {
                    showToast(`Failed to persist on server: ${result.error || "Unknown error"}`, "error", 3000);
                }
            }
        } catch (e) {
            showToast(`Failed to persist on server: ${e.message}`, "error", 3000);
        }
        // Re-open popup to reflect changes
        if (node._popupOverlay) {
            await closeChatPopup(node);
            openOutputHistoryPopup(node);
        }
    };
    footerBar.appendChild(removeAllBtn);

    footer.appendChild(footerBar);

    // ── Phase 3: Fire-and-forget load enhancer history from DB ──
    // This will update _enhancerHistory and re-render if disk has more data.
    _loadEnhancerHistoryFromDisk(node);

    // Store references and append
    node._popupOverlay = overlay;
    setActivePopupNode(node);
    document.body.appendChild(overlay);
}

// ────────────────────────────────────────────────────────────────────────
// Settings-Only Popup (accessed via gear icon on canvas)
// ────────────────────────────────────────────────────────────────────────

/**
 * Open a lightweight popup showing only settings. Used by the gear icon button.
 * @param {object} node — The LiteGraph node
 */
export function openSettingsPopup(node) {
    if (node._settingsPopupOverlay) return;

    const nodeLabel = node.title || `Node #${node.id}`;
    const { overlay, panel, body, header, footer } = createOverlayModal(
        "llm-settings-popup",
        `⚙️ Settings | ${nodeLabel}`,
        () => closeSettingsPopup(node),
        { hasFooter: true }
    );

    node._settingsPopupOverlay = overlay;

    const settingsPanel = createSettingsPanel(node, {
        showHeader: false,       // No extra header inside the popup
        showApplyButton: false,  // We have our own Apply & Close button
        compact: false,
    });
    body.appendChild(settingsPanel);

    // Footer: Apply & Close button
    const applyBtn = document.createElement("button");
    applyBtn.className = "llm-popup-save-btn";
    applyBtn.textContent = "✅ Apply & Close";
    applyBtn.style.width = "100%";
    applyBtn.onclick = () => {
        syncPopupSettingsToCanvas(node);
        closeSettingsPopup(node);
        showToast("Settings applied", "success", 1500);
    };
    footer?.appendChild(applyBtn);

    // Also sync settings on close via overlay close
    overlay._onClose = () => syncPopupSettingsToCanvas(node);

    document.body.appendChild(overlay);
}

/**
 * Close the settings-only popup.
 * @param {object} node — The LiteGraph node
 */
export function closeSettingsPopup(node) {
    if (node._settingsPopupOverlay) {
        if (node._settingsPopupOverlay._onClose) {
            node._settingsPopupOverlay._onClose();
        }
        node._settingsPopupOverlay.remove();
        node._settingsPopupOverlay = null;
    }
}
