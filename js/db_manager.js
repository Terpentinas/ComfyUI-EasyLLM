/**
 * EasyLLM — Database Manager Popup Module
 *
 * Provides a full-featured session browser, search/sort controls,
 * and action buttons for managing the JSON database.
 *
 * Exports:
 *   - openDatabaseManagerPopup(node)  ← Entry point, replaces openDatabaseDashboard()
 *
 * Dependencies:
 *   - ui_utils.js       (createOverlayModal, showToast)
 *   - popup_chat.js     (openChatPopup, openOutputHistoryPopup)
 *   - popup_utils.js    (formatTimestamp)
 */

import { createOverlayModal, showToast } from "./ui_utils.js";
import { openChatPopup, openOutputHistoryPopup } from "./popup_chat.js";
import { createBubbleElement, createEnhancerCardElement, exportChat, downloadExport, openEnhancerExportDialog } from "./popup_bubble.js";
import { scrollToBottom, autoScrollIfNeeded } from "./popup_utils.js";
// ── Module-level state ──────────────────────────────────────────────
let _managerState = null; // { overlay, panel, body, stats, sessions, filteredSessions, searchText, sortCriterion, categories }

// ── Public Entry Point ──────────────────────────────────────────────

/**
 * Open the Database Manager popup modal.
 * If already open, returns without creating a duplicate.
 *
 * @param {object} node - The ComfyUI node object (used for Open action context).
 */
export function openDatabaseManagerPopup(node) {
    // Guard: if already open, focus it
    if (_managerState && _managerState.overlay && document.body.contains(_managerState.overlay)) {
        return;
    }

    // ── Create modal using shared utility ──
    const { overlay, panel, header, body } = createOverlayModal(
        "llm-db",
        "🗄️ Database Manager",
        () => closeManager(),
        { hasFooter: false }
    );

    // Insert refresh button next to title
    const titleEl = header.querySelector(".llm-db-header-title");
    if (titleEl) {
        const refreshBtn = document.createElement("button");
        refreshBtn.className = "llm-db-header-btn llm-db-refresh-btn";
        refreshBtn.textContent = "🔄";
        refreshBtn.title = "Refresh database info";
        refreshBtn.onclick = (e) => {
            e.stopPropagation();
            refreshAll();
        };
        titleEl.parentNode.insertBefore(refreshBtn, titleEl.nextSibling);
    }

    // ── Override close to clean up state ──
    const closeBtn = header.querySelector(".llm-db-close-btn");
    if (closeBtn) {
        closeBtn.onclick = () => closeManager();
    }

    // ── Store state ──
    _managerState = {
        overlay,
        panel,
        body,
        node,
        stats: null,
        sessions: [],
        filteredSessions: [],
        searchText: "",
        searchActive: false,
        sortCriterion: "updated_desc",
        categories: {},
    };
    // Expose on window so inline event handlers in rendered HTML can reference it
    window.__llmDbState = _managerState;

    // ── Show loading state ──
    body.innerHTML = `<div class="llm-db-loading">Loading database info...</div>`;

    // ── Append to document ──
    document.body.appendChild(overlay);

    // ── Fetch data ──
    refreshAll();
}

// ── Close / Cleanup ────────────────────────────────────────────────

function closeManager() {
    if (_managerState && _managerState.overlay) {
        _managerState.overlay.remove();
    }
    _managerState = null;
}

// ── Data Fetching ──────────────────────────────────────────────────

async function refreshAll() {
    if (!_managerState) return;
    const { body } = _managerState;

    body.innerHTML = `<div class="llm-db-loading">Loading database info...</div>`;

    try {
        const [statsResp, sessionsResp] = await Promise.all([
            fetch("/easyllm/db/stats"),
            fetch("/easyllm/db/sessions"),
        ]);

        const stats = await statsResp.json();
        const sessionsData = await sessionsResp.json();

        if (!stats.available) {
            body.innerHTML = `<div class="llm-db-error">Database not available. Enable it in the config or check the server logs.</div>`;
            return;
        }

        _managerState.stats = stats;
        _managerState.sessions = sessionsData.sessions || [];
        _managerState.filteredSessions = [..._managerState.sessions];

        // Apply current sort
        _managerState.filteredSessions = _sortSessions(
            _managerState.filteredSessions,
            _managerState.sortCriterion
        );

        // Render all sections
        renderAll();
    } catch (e) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "llm-db-error";
        errorDiv.textContent = `Failed to load database info: ${e.message}`;

        const retryBtn = document.createElement("button");
        retryBtn.className = "llm-popup-header-btn";
        retryBtn.textContent = "🔄 Retry";
        retryBtn.style.marginTop = "12px";
        retryBtn.onclick = () => refreshAll();
        errorDiv.appendChild(retryBtn);

        body.innerHTML = "";
        body.appendChild(errorDiv);
        showToast(`Failed to load DB info: ${e.message}`, "error", 3000);
    }
}

// ── Full Render ────────────────────────────────────────────────────

function renderAll() {
    if (!_managerState) return;
    const { body, stats, filteredSessions } = _managerState;

    body.innerHTML = "";

    // Section 1: Stats
    body.appendChild(_renderStatsSection(stats));

    // Section 2: Settings (collapsible)
    body.appendChild(_renderSettingsSection());

    // Section 3: Search + Sort Bar
    body.appendChild(_renderSearchSortBar());

    // Section 4: Session Explorer
    body.appendChild(_renderSessionExplorer(filteredSessions));
}

// ═══════════════════════════════════════════════════════════════════
// Section 1: Stats + Global Buttons
// ═══════════════════════════════════════════════════════════════════

function _renderStatsSection(stats) {
    const section = document.createElement("div");
    section.className = "llm-db-stats-section";

    // ── Stats Grid ──
    const grid = document.createElement("div");
    grid.className = "llm-db-stats-grid";

    const statItems = [
        { label: "Nodes", value: stats.total_nodes ?? 0 },
        { label: "Sessions", value: stats.total_sessions ?? 0 },
        { label: "Images", value: stats.total_images ?? 0 },
        { label: "Disk Size", value: stats.disk_size_mb != null ? `${stats.disk_size_mb} MB` : "—" },
    ];

    statItems.forEach((item) => {
        const statEl = document.createElement("div");
        statEl.className = "llm-db-stat-item";

        const label = document.createElement("div");
        label.className = "llm-db-stat-label";
        label.textContent = item.label;

        const value = document.createElement("div");
        value.className = "llm-db-stat-value";
        value.textContent = item.value;

        if (item.truncate && item.value.length > 50) {
            value.title = item.value; // Full path in tooltip
            value.textContent = item.value.slice(0, 50) + "…";
        }

        statEl.appendChild(label);
        statEl.appendChild(value);
        grid.appendChild(statEl);
    });

    section.appendChild(grid);

    // ── Actions Row ──
    const actionsRow = document.createElement("div");
    actionsRow.className = "llm-db-actions-row";

    // Scan button (replaces old single cleanup button)
    const scanBtn = document.createElement("button");
    scanBtn.className = "llm-popup-header-btn llm-db-action-btn";
    scanBtn.textContent = "🔍 Scan Orphaned Images";
    scanBtn.title = "Count orphaned images without deleting anything";
    actionsRow.appendChild(scanBtn);

    // Delete button (hidden until scan completes)
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "llm-popup-header-btn llm-db-danger-btn";
    deleteBtn.style.display = "none";
    deleteBtn.title = "Permanently delete orphaned images";
    actionsRow.appendChild(deleteBtn);

    // Wire scan click after both buttons exist
    scanBtn.onclick = () => _handleScanOrphans(scanBtn, deleteBtn, feedback);

    // Copy Path button
    const pathBtn = document.createElement("button");
    pathBtn.className = "llm-popup-header-btn llm-db-action-btn";
    pathBtn.textContent = "📋 Copy Path";
    pathBtn.title = "Copy database path to clipboard";
    pathBtn.onclick = () => _handleCopyPath(stats.db_path || "");
    actionsRow.appendChild(pathBtn);

    section.appendChild(actionsRow);

    // ── Live feedback area (hidden initially) ──
    const feedback = document.createElement("div");
    feedback.className = "llm-db-live-feedback";
    feedback.style.display = "none";
    feedback.id = "llm-db-feedback";
    section.appendChild(feedback);

    return section;
}

// ═══════════════════════════════════════════════════════════════════
// Section 2: Search + Sort Bar
// ═══════════════════════════════════════════════════════════════════

function _renderSearchSortBar() {
    const bar = document.createElement("div");
    bar.className = "llm-db-control-bar";

    // ── Search input ──
    const searchInput = document.createElement("input");
    searchInput.className = "llm-db-search-input";
    searchInput.type = "text";
    searchInput.placeholder = "🔍 Full-text search across all entries...";
    searchInput.value = _managerState.searchText || "";
    searchInput.onkeydown = (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            _managerState.searchText = searchInput.value;
            _applyFilterAndSort();
        }
    };
    bar.appendChild(searchInput);

    // ── Search button ──
    const searchBtn = document.createElement("button");
    searchBtn.className = "llm-db-search-btn";
    searchBtn.textContent = "🔍 Search";
    searchBtn.title = "Search across all entries (Enter)";
    searchBtn.onclick = () => {
        _managerState.searchText = searchInput.value;
        _applyFilterAndSort();
    };
    bar.appendChild(searchBtn);

    // ── Clear button (hidden when no active search) ──
    const clearBtn = document.createElement("button");
    clearBtn.className = "llm-model-browser-btn-danger";
    clearBtn.textContent = "✖ Clear";
    clearBtn.title = "Clear search and restore normal view";
    clearBtn.style.display = _managerState.searchActive ? "" : "none";
    clearBtn.onclick = () => {
        _managerState.searchText = "";
        _managerState.searchActive = false;
        searchInput.value = "";
        clearBtn.style.display = "none";
        _applyFilterAndSort();
    };
    bar.appendChild(clearBtn);

    // ── Sort dropdown ──
    const sortSelect = document.createElement("select");
    sortSelect.className = "llm-db-sort-select";
    const sortOptions = [
        { value: "updated_desc", label: "Recent Updated" },
        { value: "created_desc", label: "Start Date (Newest)" },
        { value: "created_asc", label: "Start Date (Oldest)" },
        { value: "size_desc", label: "Size (Largest)" },
        { value: "entries_desc", label: "Turns / Entries" },
        { value: "images_desc", label: "Image Count" },
    ];
    sortOptions.forEach((opt) => {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === _managerState.sortCriterion) {
            option.selected = true;
        }
        sortSelect.appendChild(option);
    });
    sortSelect.onchange = (e) => {
        _managerState.sortCriterion = e.target.value;
        _applyFilterAndSort();
    };
    bar.appendChild(sortSelect);

    return bar;
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: Session Explorer
// ═══════════════════════════════════════════════════════════════════

function _renderSessionExplorer(sessions) {
    const explorer = document.createElement("div");
    explorer.className = "llm-db-session-explorer";

    // ── NEW: Filter out empty sessions (entry_count === 0) ──
    // These can occur when all messages were individually deleted and the
    // session wasn't auto-cleaned (pre-fix) or from buggy history writes.
    const validSessions = sessions.filter((s) => s.entry_count > 0);

    // Split sessions by hist_type
    const chatSessions = validSessions.filter((s) => s.hist_type === "chat");
    const enhancerSessions = validSessions.filter((s) => s.hist_type === "enhancer");

    // Always show both categories — they handle empty/filtered-out states internally
    explorer.appendChild(_createCategory("chat", "💬 Chat Nodes", chatSessions));
    explorer.appendChild(_createCategory("enhancer", "✨ Enhancer Nodes", enhancerSessions));

    return explorer;
}

function _createCategory(type, title, sessions) {
    const category = document.createElement("div");
    category.className = `llm-db-category llm-db-category-${type}`;

    // ── Header (collapsible) ──
    const header = document.createElement("div");
    header.className = "llm-db-category-header";

    const headerLeft = document.createElement("div");
    headerLeft.style.display = "flex";
    headerLeft.style.alignItems = "center";
    headerLeft.style.gap = "8px";

    const chevron = document.createElement("span");
    chevron.className = "llm-db-category-chevron";
    chevron.textContent = "▶"; // Collapsed by default

    const titleSpan = document.createElement("span");
    titleSpan.textContent = title;

    const countBadge = document.createElement("span");
    countBadge.className = "llm-db-category-count";
    countBadge.textContent = `(${sessions.length})`;

    headerLeft.appendChild(chevron);
    headerLeft.appendChild(titleSpan);
    headerLeft.appendChild(countBadge);
    header.appendChild(headerLeft);

    // ── Body (hidden by default) ──
    const body = document.createElement("div");
    body.className = "llm-db-category-body";
    body.style.display = "none"; // Collapsed

    if (sessions.length === 0) {
        const emptyMsg = document.createElement("div");
        emptyMsg.className = "llm-db-empty-message";
        emptyMsg.textContent =
            type === "chat"
                ? "No chat sessions yet"
                : "No enhancer sessions yet";
        body.appendChild(emptyMsg);
    } else {
        sessions.forEach((session) => {
            const row = _createSessionRow(session);
            body.appendChild(row);
        });
    }

    // ── Toggle collapse on header click ──
    header.onclick = (e) => {
        e.stopPropagation();
        const isHidden = body.style.display === "none";
        body.style.display = isHidden ? "block" : "none";
        chevron.textContent = isHidden ? "▼" : "▶";
    };

    category.appendChild(header);
    category.appendChild(body);

    return category;
}

function _createSessionRow(session) {
    const row = document.createElement("div");
    row.className = "llm-db-session-row";
    row.dataset.nodeId = session.node_id || "";

    // ── Info: name (title) + meta row ──
    const info = document.createElement("div");
    info.className = "llm-db-session-info";

    // Title: preview text (or fallback)
    const nameSpan = document.createElement("span");
    nameSpan.className = "llm-db-session-name";
    nameSpan.textContent = _getSessionDisplayName(session);
    info.appendChild(nameSpan);

    // Meta row: relative date + metrics
    const metaRow = document.createElement("div");
    metaRow.className = "llm-db-session-meta";

    const dateSpan = document.createElement("span");
    dateSpan.className = "llm-db-session-date";
    const ts = session.created_at || 0;
    dateSpan.textContent = `📅 ${_formatRelativeDate(ts)}`;
    metaRow.appendChild(dateSpan);

    const turnsSpan = document.createElement("span");
    turnsSpan.className = "llm-db-session-metric";
    turnsSpan.textContent = `🔢 ${session.entry_count ?? 0} turns`;
    metaRow.appendChild(turnsSpan);

    const imgCount = _getImageCount(session);
    const imgSpan = document.createElement("span");
    imgSpan.className = "llm-db-session-metric";
    imgSpan.textContent = `🖼️ ${imgCount} imgs`;
    metaRow.appendChild(imgSpan);

    const sizeSpan = document.createElement("span");
    sizeSpan.className = "llm-db-session-metric";
    sizeSpan.textContent = `📄 ${_formatSize(session.size_bytes ?? 0)}`;
    metaRow.appendChild(sizeSpan);

    info.appendChild(metaRow);
    row.appendChild(info);

    // ── Actions: Open + Delete ──
    const actions = document.createElement("div");
    actions.className = "llm-db-session-actions";

    const openBtn = document.createElement("button");
    openBtn.className = "llm-popup-header-btn llm-db-open-btn";
    openBtn.textContent = "👁️ Open";
    openBtn.title = "Open this session";
    openBtn.onclick = (e) => {
        e.stopPropagation();
        _handleOpenSession(session);
    };
    actions.appendChild(openBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "llm-popup-header-btn llm-db-delete-btn";
    deleteBtn.textContent = "🗑️ Delete";
    deleteBtn.title = "Delete this session and its images";
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        _handleDeleteSession(session);
    };
    actions.appendChild(deleteBtn);

    row.appendChild(actions);

    return row;
}

// ═══════════════════════════════════════════════════════════════════
// Action Handlers
// ═══════════════════════════════════════════════════════════════════

async function _handleCleanup(btn) {
    btn.disabled = true;
    btn.textContent = "🧹 Cleaning...";
    const feedback = document.getElementById("llm-db-feedback");
    if (feedback) {
        feedback.style.display = "block";
        feedback.textContent = "Running cleanup...";
    }

    try {
        const r = await fetch("/easyllm/db/cleanup", { method: "POST" });
        const result = await r.json();
        if (result.status === "ok") {
            const stats = result.stats || {};
            const imagesRemoved = stats.removed_images ?? 0;
            const bytesFreed = stats.bytes_freed ?? 0;
            const msg = `✅ Success: Removed ${imagesRemoved} images, ${(bytesFreed / 1024).toFixed(1)} KB freed.`;
            if (feedback) {
                feedback.textContent = msg;
                feedback.className = "llm-db-live-feedback llm-db-live-success";
            }
            showToast(msg, "success", 3000);
        } else {
            throw new Error(result.error || "Cleanup failed");
        }
    } catch (e) {
        if (feedback) {
            feedback.textContent = `❌ Cleanup failed: ${e.message}`;
            feedback.className = "llm-db-live-feedback llm-db-live-error";
        }
        showToast(`Cleanup failed: ${e.message}`, "error", 3000);
    } finally {
        btn.disabled = false;
        btn.textContent = "🧹 Run Cleanup";
        // Re-fetch data
        refreshAll();
    }
}

// ── Scan orphans (read-only) ────────────────────────────────────────

async function _handleScanOrphans(scanBtn, deleteBtn, feedback) {
    scanBtn.disabled = true;
    scanBtn.textContent = "🔍 Scanning...";
    feedback.style.display = "block";
    feedback.textContent = "Scanning for orphaned images...";
    feedback.className = "llm-db-live-feedback";

    try {
        const r = await fetch("/easyllm/db/scan-orphans", { method: "POST" });
        const result = await r.json();

        if (result.status === "ok") {
            const orphans = result.orphans || {};
            const count = orphans.count ?? 0;
            const bytes = orphans.bytes ?? 0;
            const sizeMb = (bytes / (1024 * 1024)).toFixed(1);

            if (count === 0) {
                feedback.textContent = "✅ No orphaned images found. All images are linked to active sessions.";
                feedback.className = "llm-db-live-feedback llm-db-live-success";
                deleteBtn.style.display = "none";
            } else {
                feedback.textContent = `⚠️ Found ${count} orphaned image(s) (${sizeMb} MB) not linked to any session.`;
                feedback.className = "llm-db-live-feedback llm-db-live-warning";

                // Show delete button with count
                const label = count === 1 ? "Orphaned Image" : "Orphaned Images";
                deleteBtn.textContent = `🗑️ Delete ${count} ${label}`;
                deleteBtn.style.display = "";
                deleteBtn.onclick = () => _handleConfirmDelete(deleteBtn, feedback, scanBtn);
            }
        } else {
            throw new Error(result.error || "Scan failed");
        }
    } catch (e) {
        feedback.textContent = `❌ Scan failed: ${e.message}`;
        feedback.className = "llm-db-live-feedback llm-db-live-error";
    } finally {
        scanBtn.disabled = false;
        scanBtn.textContent = "🔍 Scan Orphaned Images";
    }
}

// ── Confirm and execute deletion ────────────────────────────────────

async function _handleConfirmDelete(deleteBtn, feedback, scanBtn) {
    const count = parseInt(deleteBtn.textContent.match(/\d+/)?.[0] ?? "0", 10);
    const label = count === 1 ? "this orphaned image" : `these ${count} orphaned images`;
    if (!confirm(`⚠️ Permanently delete ${label}? This cannot be undone.`)) {
        return;
    }

    deleteBtn.disabled = true;
    deleteBtn.textContent = "🗑️ Deleting...";
    feedback.textContent = "Running cleanup...";
    feedback.className = "llm-db-live-feedback";

    try {
        const r = await fetch("/easyllm/db/cleanup", { method: "POST" });
        const result = await r.json();

        if (result.status === "ok") {
            const stats = result.stats || {};
            const imagesRemoved = stats.removed_images ?? 0;
            const bytesFreed = stats.bytes_freed ?? 0;
            const msg = `✅ Deleted ${imagesRemoved} images, ${(bytesFreed / 1024).toFixed(1)} KB freed.`;
            feedback.textContent = msg;
            feedback.className = "llm-db-live-feedback llm-db-live-success";
            showToast(msg, "success", 3000);
            deleteBtn.style.display = "none";
            refreshAll();
        } else {
            throw new Error(result.error || "Cleanup failed");
        }
    } catch (e) {
        feedback.textContent = `❌ Deletion failed: ${e.message}`;
        feedback.className = "llm-db-live-feedback llm-db-live-error";
    } finally {
        deleteBtn.disabled = false;
        deleteBtn.textContent = "🗑️ Delete";
    }
}

function _handleCopyPath(dbPath) {
    if (!dbPath) {
        showToast("No database path available", "error", 1500);
        return;
    }
    navigator.clipboard.writeText(dbPath).then(() => {
        showToast("Path copied to clipboard", "success", 1500);
    }).catch(() => {
        showToast("Failed to copy path", "error", 1500);
    });
}

async function _handleOpenSession(session) {
    const nodeId = session.node_id;
    if (!nodeId) {
        showToast("Session has no node ID", "error", 2000);
        return;
    }

    // Search the ComfyUI graph for a node with matching ID
    const foundNode = _findNodeInGraph(nodeId);

    if (foundNode) {
        // Node exists — open appropriate popup
        closeManager();
        try {
            if (session.hist_type === "chat") {
                await openChatPopup(foundNode);
            } else if (session.hist_type === "enhancer") {
                await openOutputHistoryPopup(foundNode);
            } else {
                // Fallback for unknown hist_type — default to chat with a warning
                showToast(
                    `Unknown session type "${session.hist_type}", opening as chat`,
                    "warning", 3000
                );
                await openChatPopup(foundNode);
            }
        } catch (e) {
            showToast(`Failed to open popup: ${e.message}`, "error", 3000);
        }
    } else {
        // Node was deleted from graph — show read-only fallback
        _showReadOnlyViewer(session);
    }
}

async function _handleDeleteSession(session) {
    const nodeId = session.node_id;
    const histType = session.hist_type || "chat";
    const displayName = _getSessionDisplayName(session);

    if (!confirm(`Delete session "${displayName}"? This will remove all its data and referenced images.`)) {
        return;
    }

    try {
        const r = await fetch(`/easyllm/db/history/${nodeId}?type=${histType}`, { method: "DELETE" });
        const result = await r.json();
        if (result.success) {
            showToast("Session deleted. Running cleanup...", "success", 2000);

            // Clear in-memory cache on the graph node (fix stale data after delete)
            const graphNode = _findNodeInGraph(session.node_id);
            if (graphNode) {
                graphNode._chatHistory = [];
                graphNode._enhancerHistory = [];
            }

            // Fire-and-forget cleanup in background (best-effort maintenance)
            fetch("/easyllm/db/cleanup", { method: "POST" })
                .catch((e) => console.warn("Cleanup POST failed (non-critical):", e));
            // Re-fetch and re-render
            refreshAll();
        } else {
            throw new Error(result.error || "Delete failed");
        }
    } catch (e) {
        showToast(`Failed to delete session: ${e.message}`, "error", 3000);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Fallback Read-Only Viewer
// ═══════════════════════════════════════════════════════════════════

async function _showReadOnlyViewer(session) {
    const nodeId = session.node_id;
    const histType = session.hist_type || "chat";
    const displayName = _getSessionDisplayName(session);

    // Close the manager
    closeManager();

    // Create a modal popup styled like the real chat/history popup
    const { overlay, panel, body, footer } = createOverlayModal(
        "llm-popup",
        `📋 ${displayName}`,
        () => overlay.remove(),
        { hasFooter: true }
    );
    document.body.appendChild(overlay);

    body.innerHTML = `<div class="llm-db-loading">Loading session data...</div>`;

    try {
        const url = `/easyllm/db/history/${nodeId}?type=${histType}`;
        const resp = await fetch(url);
        const data = await resp.json();
        const entries = data.entries || [];

        if (entries.length === 0) {
            body.innerHTML = `<div class="llm-popup-history-empty">No entries in this session.</div>`;
            return;
        }

        body.innerHTML = "";

        // ── Scrollable history container (same class & behaviour as real popup) ──
        const historyContainer = document.createElement("div");
        historyContainer.className = "llm-popup-history";
        body.appendChild(historyContainer);

        // Track mutable entries for edit/delete operations
        const mutableEntries = entries.map(e => ({...e}));

        // ── Persistence helpers ──
        async function persistChatEntries(updatedEntries) {
            try {
                const resp = await fetch(`/easyllm/db/history/${nodeId}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ entries: updatedEntries, type: "chat" }),
                });
                const result = await resp.json();
                if (result.success) {
                    showToast("✅ Saved", "success", 2000);
                } else {
                    showToast("❌ Save failed: " + (result.error || "unknown"), "error", 3000);
                }
            } catch (e) {
                showToast("❌ Save failed: " + e.message, "error", 3000);
            }
        }

        async function persistEnhancerEntries(updatedEntries) {
            try {
                const resp = await fetch(`/easyllm/db/history/${nodeId}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ entries: updatedEntries, type: "enhancer" }),
                });
                const result = await resp.json();
                if (result.success) {
                    showToast("✅ Saved", "success", 2000);
                } else {
                    showToast("❌ Save failed: " + (result.error || "unknown"), "error", 3000);
                }
            } catch (e) {
                showToast("❌ Save failed: " + e.message, "error", 3000);
            }
        }

        // ── Re-render helper (used after delete) ──
        // Hoisted variables set by each branch below
        let selectedEntries = null;
        let updateFooterState = () => {};

        function reRenderView() {
            historyContainer.innerHTML = "";
            if (histType === "chat") {
                for (let i = 0; i < mutableEntries.length; i++) {
                    const entry = mutableEntries[i];
                    const role = entry.role || "user";
                    const message = entry.message || "";
                    const bubble = createBubbleElement(role, message, {
                        timestamp: entry.timestamp,
                        images: entry.images || null,
                        onEdit: (newText) => {
                            mutableEntries[i].message = newText;
                            persistChatEntries(mutableEntries);
                        },
                        onDelete: () => {
                            if (!confirm("Delete this message?")) return;
                            mutableEntries.splice(i, 1);
                            persistChatEntries(mutableEntries).then(() => reRenderView());
                        },
                    });
                    historyContainer.appendChild(bubble);
                }
            } else if (selectedEntries) {
                // Re-render enhancer cards preserving selection
                for (let i = 0; i < mutableEntries.length; i++) {
                    const entry = mutableEntries[i];
                    const card = createEnhancerCardElement(entry, {
                        isSelected: selectedEntries.has(entry),
                        onSelect: (ent, isSelected) => {
                            if (isSelected) selectedEntries.add(ent);
                            else selectedEntries.delete(ent);
                            updateFooterState();
                        },
                        onEdit: (newText) => {
                            mutableEntries[i].output = newText;
                            persistEnhancerEntries(mutableEntries);
                        },
                    });
                    historyContainer.appendChild(card);
                }
            }
            requestAnimationFrame(() => {
                scrollToBottom(historyContainer);
            });
        }

        // ── Render based on type ──
        if (histType === "chat") {
            for (let idx = 0; idx < mutableEntries.length; idx++) {
                const entry = mutableEntries[idx];
                const role = entry.role || "user";
                const message = entry.message || "";
                const bubble = createBubbleElement(role, message, {
                    timestamp: entry.timestamp,
                    images: entry.images || null,
                    onEdit: (newText) => {
                        mutableEntries[idx].message = newText;
                        persistChatEntries(mutableEntries);
                    },
                    onDelete: () => {
                        if (!confirm("Delete this message?")) return;
                        mutableEntries.splice(idx, 1);
                        persistChatEntries(mutableEntries).then(() => reRenderView());
                    },
                });
                historyContainer.appendChild(bubble);
            }
        } else {
            // ── Enhancer type: render cards with selection + edit support ──
            selectedEntries = new Set();

            for (let idx = 0; idx < mutableEntries.length; idx++) {
                const entry = mutableEntries[idx];
                const card = createEnhancerCardElement(entry, {
                    isSelected: selectedEntries.has(entry),
                    onSelect: (ent, isSelected) => {
                        if (isSelected) selectedEntries.add(ent);
                        else selectedEntries.delete(ent);
                        updateFooterState();
                    },
                    onEdit: (newText) => {
                        mutableEntries[idx].output = newText;
                        persistEnhancerEntries(mutableEntries);
                    },
                });
                historyContainer.appendChild(card);
            }

            // ── Enhancer footer bar (inside the modal footer) ──
            if (footer) {
                const footerBar = document.createElement("div");
                footerBar.className = "enhancer-card-footer";

                const selectAllBtn = document.createElement("button");
                selectAllBtn.className = "enhancer-card-footer-btn";
                selectAllBtn.textContent = "✅ Select All";
                selectAllBtn.title = "Toggle selection of all entries";

                const exportSelectedBtn = document.createElement("button");
                exportSelectedBtn.className = "enhancer-card-footer-btn";
                exportSelectedBtn.textContent = "📥 Export Selected";
                exportSelectedBtn.title = "Export selected entries with options";
                exportSelectedBtn.disabled = true;

                const selectionLabel = document.createElement("span");
                selectionLabel.className = "enhancer-card-footer-label";
                selectionLabel.style.display = "none";

                updateFooterState = () => {
                    const count = selectedEntries.size;
                    const total = mutableEntries.length;

                    selectAllBtn.textContent = (count === total && total > 0)
                        ? "❌ Deselect All"
                        : "✅ Select All";

                    if (count > 0) {
                        exportSelectedBtn.textContent = `📥 Export Selected (${count})`;
                        exportSelectedBtn.disabled = false;
                    } else {
                        exportSelectedBtn.textContent = "📥 Export Selected";
                        exportSelectedBtn.disabled = true;
                    }

                    if (count > 0) {
                        selectionLabel.textContent = `${count} of ${total} selected`;
                        selectionLabel.style.display = "block";
                    } else {
                        selectionLabel.textContent = "";
                        selectionLabel.style.display = "none";
                    }

                    // Update remove button label based on selection state
                    if (count > 0) {
                        removeBtn.textContent = `🗑 Remove Selected (${count})`;
                        removeBtn.disabled = false;
                    } else {
                        removeBtn.textContent = "🗑 Remove All";
                        removeBtn.disabled = (total === 0);
                    }
                };

                selectAllBtn.onclick = () => {
                    const allSelected = selectedEntries.size === mutableEntries.length;
                    const cards = historyContainer.querySelectorAll(".enhancer-card");
                    cards.forEach((card, idx) => {
                        const entry = mutableEntries[idx];
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

                exportSelectedBtn.onclick = () => {
                    openEnhancerExportDialog(null, selectedEntries, displayName);
                };
                footerBar.appendChild(exportSelectedBtn);

                // ── Remove button (danger style) ──
                const removeBtn = document.createElement("button");
                removeBtn.className = "llm-model-browser-btn-danger";
                removeBtn.textContent = "🗑 Remove All";
                removeBtn.title = "Remove all entries or only selected entries";
                removeBtn.onclick = async () => {
                    const count = selectedEntries.size;
                    if (count > 0) {
                        const msg = `🗑 Remove ${count} selected enhancer entr${count === 1 ? "y" : "ies"}?`;
                        if (!confirm(msg)) return;
                        // Filter out selected entries (iterate backward to preserve indices)
                        const toRemove = new Set(selectedEntries);
                        for (let i = mutableEntries.length - 1; i >= 0; i--) {
                            if (toRemove.has(mutableEntries[i])) {
                                mutableEntries.splice(i, 1);
                            }
                        }
                        selectedEntries.clear();
                    } else {
                        if (!confirm("🗑 Remove ALL enhancer entries? This cannot be undone.")) return;
                        mutableEntries.length = 0; // Clear all
                    }
                    await persistEnhancerEntries(mutableEntries);
                    reRenderView();
                };
                footerBar.appendChild(removeBtn);

                footerBar.appendChild(selectionLabel);
                footer.appendChild(footerBar);
            }
        }

        // Auto-scroll to bottom after render
        requestAnimationFrame(() => {
            scrollToBottom(historyContainer);
        });

        // ── Footer: Chat export dropdown ──
        if (histType === "chat" && footer) {
            // 📥 Export dropdown — same pattern as popup_chat.js:860-912
            const exportBtn = document.createElement("button");
            exportBtn.className = "llm-popup-header-btn llm-popup-export-btn llm-popup-export-btn--footer";
            exportBtn.style.marginLeft = "auto";
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
                    const result = exportChat(entries, "md", displayName);
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
                    const result = exportChat(entries, "txt", displayName);
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
        }

    } catch (e) {
        body.innerHTML = `<div class="llm-db-error">Failed to load session data: ${e.message}</div>`;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Filter & Sort
// ═══════════════════════════════════════════════════════════════════

function _applyFilterAndSort() {
    if (!_managerState) return;

    const searchText = (_managerState.searchText || "").trim();

    if (searchText) {
        _managerState.searchActive = true;
        _performFullTextSearch(searchText);
    } else {
        _managerState.searchActive = false;
        // No query — show all sessions as normal
        _renderNormalView(_getAllSessions());
    }
}

function _renderNormalView(sessions) {
    if (!_managerState) return;
    sessions = _sortSessions(sessions, _managerState.sortCriterion);
    _managerState.filteredSessions = sessions;

    // Replace explorer (or search results if present)
    const explorer = _managerState.body.querySelector(".llm-db-session-explorer");
    if (explorer) {
        explorer.replaceWith(_renderSessionExplorer(sessions));
    } else {
        // Explorer was removed during search — append a new one
        _managerState.body.appendChild(_renderSessionExplorer(sessions));
    }
    // Remove any search results panel
    const resultsPanel = _managerState.body.querySelector(".llm-db-search-results");
    if (resultsPanel) {
        resultsPanel.remove();
    }
    // Hide clear button
    const clearBtn = _managerState.body.querySelector(".llm-db-clear-btn");
    if (clearBtn) {
        clearBtn.style.display = "none";
    }
}

// ═══════════════════════════════════════════════════════════════════
// Full-Text Search
// ═══════════════════════════════════════════════════════════════════

async function _performFullTextSearch(query) {
    if (!_managerState) return;

    const body = _managerState.body;

    // Show loading indicator (remove existing results/explorer)
    let resultsContainer = body.querySelector(".llm-db-search-results");
    if (!resultsContainer) {
        const explorer = body.querySelector(".llm-db-session-explorer");
        if (explorer) explorer.remove();
        resultsContainer = document.createElement("div");
        resultsContainer.className = "llm-db-search-results";
        body.appendChild(resultsContainer);
    }
    resultsContainer.innerHTML = `<div class="llm-db-loading">🔍 Searching across all sessions...</div>`;

    try {
        const resp = await fetch(`/easyllm/db/search?q=${encodeURIComponent(query)}&max=200`);
        const data = await resp.json();

        if (!data.available) {
            resultsContainer.innerHTML = `<div class="llm-db-error">Database not available.</div>`;
            return;
        }

        const results = data.results || [];

        if (results.length === 0) {
            resultsContainer.innerHTML = `<div class="llm-db-empty-message">No entries found matching "<strong>${_escapeHtml(query)}</strong>".</div>`;
            return;
        }

        _renderSearchResults(resultsContainer, results, query);

        // Show clear button
        const clearBtn = body.querySelector(".llm-db-clear-btn");
        if (clearBtn) {
            clearBtn.style.display = "";
        }
    } catch (e) {
        resultsContainer.innerHTML = `<div class="llm-db-error">Full-text search failed: ${e.message}</div>`;
    }
}

function _renderSearchResults(container, results, query) {
    // Group results by node_id
    const grouped = {};
    const sessionMap = {}; // nodeId → session data from _getAllSessions
    for (const s of _getAllSessions()) {
        sessionMap[s.node_id] = s;
    }

    for (const r of results) {
        const nid = r.node_id || "unknown";
        if (!grouped[nid]) {
            grouped[nid] = {
                node_id: nid,
                hist_type: r.hist_type || "",
                entries: [],
            };
        }
        grouped[nid].entries.push(r);
    }

    const nodeIds = Object.keys(grouped);
    container.innerHTML = "";

    // ── Summary header ──
    const summary = document.createElement("div");
    summary.className = "llm-db-search-header";
    summary.innerHTML = `🔍 Found <strong>${results.length}</strong> match(es) in <strong>${nodeIds.length}</strong> node(s) for "<strong>${_escapeHtml(query)}</strong>"`;
    container.appendChild(summary);

    // ── Render each matching node as a row with expandable match details ──
    for (const nid of nodeIds) {
        const group = grouped[nid];
        const sessionData = sessionMap[nid] || {
            node_id: nid,
            hist_type: group.hist_type,
            created_at: 0,
            updated_at: 0,
            entry_count: 0,
            size_bytes: 0,
            image_count: 0,
            model_name: "",
            preview: "",
        };

        // Create a wrapper for this session result
        const sessionWrapper = document.createElement("div");
        sessionWrapper.className = "llm-db-search-session";

        // Session row (reuse _createSessionRow visual but add match badge)
        const row = _createSearchSessionRow(sessionData, group.entries.length, query);
        sessionWrapper.appendChild(row);

        // Match entries container (collapsible, starts expanded)
        const matchesBody = document.createElement("div");
        matchesBody.className = "llm-db-search-matches";
        for (const entry of group.entries) {
            const matchEl = _createMatchEntry(entry, query);
            matchesBody.appendChild(matchEl);
        }
        sessionWrapper.appendChild(matchesBody);

        // Toggle matches on row click
        row._toggleMatches = () => {
            const isHidden = matchesBody.style.display === "none";
            matchesBody.style.display = isHidden ? "block" : "none";
            // Toggle a visual indicator on the row
            row.classList.toggle("llm-db-search-session-collapsed", !isHidden);
        };
        row.addEventListener("click", (e) => {
            // Don't toggle if clicking on action buttons
            if (e.target.closest(".llm-db-session-actions")) return;
            row._toggleMatches();
        });

        container.appendChild(sessionWrapper);
    }
}

function _createSearchSessionRow(session, matchCount, query) {
    const row = document.createElement("div");
    row.className = "llm-db-session-row llm-db-search-session-row";
    row.dataset.nodeId = session.node_id || "";
    row.style.cursor = "pointer";

    // ── Info: name (title) + meta row + match badge ──
    const info = document.createElement("div");
    info.className = "llm-db-session-info";

    const nameRow = document.createElement("div");
    nameRow.style.display = "flex";
    nameRow.style.alignItems = "center";
    nameRow.style.gap = "8px";

    const nameSpan = document.createElement("span");
    nameSpan.className = "llm-db-session-name";
    nameSpan.textContent = _getSessionDisplayName(session);
    nameRow.appendChild(nameSpan);

    // Match count badge
    const badge = document.createElement("span");
    badge.className = "llm-db-match-badge";
    badge.textContent = `${matchCount} match(es)`;
    nameRow.appendChild(badge);

    info.appendChild(nameRow);

    // Meta row: relative date + metrics
    const metaRow = document.createElement("div");
    metaRow.className = "llm-db-session-meta";

    const ts = session.created_at || 0;
    const dateSpan = document.createElement("span");
    dateSpan.className = "llm-db-session-date";
    dateSpan.textContent = `📅 ${_formatRelativeDate(ts)}`;
    metaRow.appendChild(dateSpan);

    const turnsSpan = document.createElement("span");
    turnsSpan.className = "llm-db-session-metric";
    turnsSpan.textContent = `🔢 ${session.entry_count ?? 0} turns`;
    metaRow.appendChild(turnsSpan);

    const imgCount = _getImageCount(session);
    const imgSpan = document.createElement("span");
    imgSpan.className = "llm-db-session-metric";
    imgSpan.textContent = `🖼️ ${imgCount} imgs`;
    metaRow.appendChild(imgSpan);

    const sizeSpan = document.createElement("span");
    sizeSpan.className = "llm-db-session-metric";
    sizeSpan.textContent = `📄 ${_formatSize(session.size_bytes ?? 0)}`;
    metaRow.appendChild(sizeSpan);

    info.appendChild(metaRow);
    row.appendChild(info);

    // ── Actions: Open + Delete ──
    const actions = document.createElement("div");
    actions.className = "llm-db-session-actions";

    const openBtn = document.createElement("button");
    openBtn.className = "llm-popup-header-btn llm-db-open-btn";
    openBtn.textContent = "👁️ Open";
    openBtn.title = "Open this session";
    openBtn.onclick = (e) => {
        e.stopPropagation();
        _handleOpenSession(session);
    };
    actions.appendChild(openBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "llm-popup-header-btn llm-db-delete-btn";
    deleteBtn.textContent = "🗑️ Delete";
    deleteBtn.title = "Delete this session and its images";
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        _handleDeleteSession(session);
    };
    actions.appendChild(deleteBtn);

    row.appendChild(actions);

    return row;
}

function _createMatchEntry(entry, query) {
    const el = document.createElement("div");
    el.className = "llm-db-match-entry";

    // Meta line: role + timestamp
    const meta = document.createElement("div");
    meta.className = "llm-db-match-entry-meta";
    const role = entry.role || "entry";
    const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : "";
    meta.textContent = `#${role}${ts ? " — " + ts : ""}`;
    el.appendChild(meta);

    // Text snippet with highlighting
    const text = entry.message || entry.input || entry.output || "";
    const snippet = text.length > 300 ? text.slice(0, 300) + "…" : text;
    const textEl = document.createElement("div");
    textEl.className = "llm-db-match-entry-text";
    textEl.innerHTML = _highlightMatch(_escapeHtml(snippet), query);
    el.appendChild(textEl);

    return el;
}

function _highlightMatch(text, query) {
    // Case-insensitive highlight of the query in the text
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escaped})`, "gi");
    return text.replace(regex, '<mark class="llm-db-highlight">$1</mark>');
}

function _escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function _sortSessions(sessions, criterion) {
    const sorted = [...sessions];
    switch (criterion) {
        case "updated_desc":
            sorted.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
            break;
        case "created_desc":
            sorted.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
            break;
        case "created_asc":
            sorted.sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
            break;
        case "size_desc":
            sorted.sort((a, b) => (b.size_bytes || 0) - (a.size_bytes || 0));
            break;
        case "entries_desc":
            sorted.sort((a, b) => (b.entry_count || 0) - (a.entry_count || 0));
            break;
        case "images_desc":
            sorted.sort((a, b) => (b.image_count || 0) - (a.image_count || 0));
            break;
        default:
            sorted.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
            break;
    }
    return sorted;
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

function _getAllSessions() {
    return _managerState ? _managerState.sessions || [] : [];
}

function _getSessionDisplayName(session) {
    if (!session) return "Unknown";
    // Primary: preview text (first user message or latest output)
    if (session.preview) {
        const preview = session.preview.trim();
        if (preview.length > 0) {
            return preview.length > 35 ? preview.slice(0, 35) + "…" : preview;
        }
    }
    // Fallback: model_name
    if (session.model_name) {
        const name = session.model_name;
        return name.length > 40 ? name.slice(0, 40) + "…" : name;
    }
    // Fallback: type-based label
    if (session.hist_type === "enhancer") {
        return "Enhancer Run";
    }
    return "Chat Session";
}

function _formatRelativeDate(ts) {
    if (!ts) return "N/A";
    try {
        const d = new Date(ts);
        const now = new Date();
        const diffMs = now - d;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

        if (diffDays === 0) return `Today ${time}`;
        if (diffDays === 1) return `Yesterday ${time}`;
        if (diffDays < 7) return `${diffDays} days ago ${time}`;
        // Fall back to absolute date
        return d.toLocaleString(undefined, {
            month: "2-digit",
            day: "2-digit",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    } catch {
        return String(ts);
    }
}

function _formatSize(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const mb = bytes / (1024 * 1024);
    if (mb >= 1) return `${mb.toFixed(2)} MB`;
    const kb = bytes / 1024;
    if (kb >= 1) return `${kb.toFixed(1)} KB`;
    return `${bytes} B`;
}

function _estimateImageCount(session) {
    // Estimate from entry preview content — count image references
    // Since the backend may not expose image_count directly, we estimate
    // from the session's size as a rough heuristic.
    // A better approach would be to count images in the preview metadata,
    // but for now we use a simple heuristic based on entry_count and size.
    const entryCount = session.entry_count || 0;
    const sizeBytes = session.size_bytes || 0;
    // Rough estimate: images are ~100KB+ each, if size >> text
    if (entryCount === 0) return 0;
    const textBytesPerEntry = 500; // rough estimate of text per entry
    const textEstimate = entryCount * textBytesPerEntry;
    const imageBytes = Math.max(0, sizeBytes - textEstimate);
    return Math.round(imageBytes / (100 * 1024)); // rough image count
}

function _getImageCount(session) {
    // Prefer the backend-provided exact image_count (stored in _meta.json).
    // Fall back to the heuristic for legacy sessions that lack this field.
    if (session.image_count !== undefined && session.image_count !== null) {
        return session.image_count;
    }
    return _estimateImageCount(session);
}

function _findNodeInGraph(nodeId) {
    try {
        // Access ComfyUI's graph via window.app.graph
        const graph = window.app?.graph;
        if (!graph) return null;
        return graph._nodes.find((n) => String(n.id) === String(nodeId));
    } catch {
        return null;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Settings Section (Collapsible)
// ═══════════════════════════════════════════════════════════════════════

function _renderSettingsSection() {
    const section = document.createElement("div");
    section.className = "llm-db-settings-section";

    // ── Collapsible Header ──
    const header = document.createElement("div");
    header.className = "llm-db-settings-header";

    const chevron = document.createElement("span");
    chevron.className = "llm-db-settings-chevron";
    chevron.textContent = "▶";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = "⚙️ Settings";

    header.appendChild(chevron);
    header.appendChild(titleSpan);

    // ── Body (hidden by default) ──
    const body = document.createElement("div");
    body.className = "llm-db-settings-body";
    body.style.display = "none";

    // Toggle collapse on header click
    header.onclick = (e) => {
        e.stopPropagation();
        const isHidden = body.style.display === "none";
        body.style.display = isHidden ? "block" : "none";
        chevron.textContent = isHidden ? "▼" : "▶";
        // Fetch settings when first expanded
        if (isHidden && !body.dataset.loaded) {
            _loadSettingsIntoBody(body);
        }
    };

    section.appendChild(header);
    section.appendChild(body);
    return section;
}

async function _loadSettingsIntoBody(body) {
    body.innerHTML = `<div class="llm-db-loading">Loading settings...</div>`;
    body.dataset.loaded = "true";

    try {
        const r = await fetch("/easyllm/db/settings");
        const data = await r.json();

        if (!data.available) {
            body.innerHTML = `<div class="llm-db-error">Database not available</div>`;
            return;
        }

        // Clear loading message before appending settings content
        body.innerHTML = "";

        // ── DB Path (read-only) ──
        const pathRow = document.createElement("div");
        pathRow.className = "llm-db-settings-row";
        pathRow.innerHTML = `
            <span class="llm-db-settings-label">Database Path</span>
            <code class="llm-db-settings-path">${_escapeHtml(data.db_path || "")}</code>
        `;
        body.appendChild(pathRow);

        // ── Auto-Cleanup Toggle ──
        const toggleRow = document.createElement("div");
        toggleRow.className = "llm-db-settings-row";
        toggleRow.innerHTML = `
            <span class="llm-db-settings-label">Auto-Cleanup</span>
            <label class="llm-toggle-switch">
                <input type="checkbox" id="llm-setting-auto-cleanup"
                       ${data.auto_cleanup_enabled ? "checked" : ""}>
                <span class="llm-toggle-slider"></span>
            </label>
        `;
        body.appendChild(toggleRow);

        // ── Cleanup Interval ──
        body.appendChild(_createSettingInput(
            "Interval (seconds)",
            "llm-setting-interval",
            "number",
            data.auto_cleanup_interval_sec,
            "Min 60"
        ));

        // ── Max Age ──
        body.appendChild(_createSettingInput(
            "Max Age (days)",
            "llm-setting-max-age",
            "number",
            data.max_age_days,
            "0 = never delete"
        ));

        // ── Immediate Write Toggle ──
        const writeRow = document.createElement("div");
        writeRow.className = "llm-db-settings-row";
        writeRow.innerHTML = `
            <span class="llm-db-settings-label">Immediate Write</span>
            <label class="llm-toggle-switch">
                <input type="checkbox" id="llm-setting-immediate-write"
                       ${data.immediate_write ? "checked" : ""}>
                <span class="llm-toggle-slider"></span>
            </label>
        `;
        body.appendChild(writeRow);

        // ── Save Button ──
        const saveBtn = document.createElement("button");
        saveBtn.className = "llm-popup-header-btn llm-db-settings-save";
        saveBtn.textContent = "💾 Save Settings";
        saveBtn.onclick = () => _handleSaveSettings(body);
        body.appendChild(saveBtn);

        // ── Feedback area ──
        const feedback = document.createElement("div");
        feedback.className = "llm-db-live-feedback";
        feedback.style.display = "none";
        feedback.id = "llm-db-settings-feedback";
        body.appendChild(feedback);

        // ── Danger Zone ──
        const dangerDivider = document.createElement("hr");
        dangerDivider.className = "llm-db-settings-divider llm-db-danger-divider";
        body.appendChild(dangerDivider);

        const dangerLabel = document.createElement("div");
        dangerLabel.className = "llm-db-settings-subtitle llm-db-danger-title";
        dangerLabel.textContent = "⚠️ Danger Zone";
        body.appendChild(dangerLabel);

        const destroyBtn = document.createElement("button");
        destroyBtn.className = "llm-popup-header-btn llm-db-destroy-btn";
        destroyBtn.textContent = "💥 Destroy & Recreate Database";
        destroyBtn.title = "Completely remove all data and create a fresh database";
        destroyBtn.onclick = () => _handleDestroyDatabase(destroyBtn);
        body.appendChild(destroyBtn);

    } catch (e) {
        body.innerHTML = `<div class="llm-db-error">Failed to load settings: ${_escapeHtml(e.message)}</div>`;
    }
}

// ── Settings Helpers ──────────────────────────────────────────────────

function _createSettingInput(label, id, type, value, placeholder) {
    const row = document.createElement("div");
    row.className = "llm-db-settings-row";
    const inputHtml = type === "number"
        ? `<input type="number" id="${id}" class="llm-db-settings-input"
                  value="${Number(value) ?? 0}" placeholder="${_escapeHtml(placeholder || "")}"
                  min="0">`
        : `<input type="${type}" id="${id}" class="llm-db-settings-input"
                  value="${_escapeHtml(String(value))}">`;
    row.innerHTML = `
        <span class="llm-db-settings-label">${_escapeHtml(label)}</span>
        ${inputHtml}
    `;
    return row;
}

// ── Save Handler ──────────────────────────────────────────────────────

async function _handleSaveSettings(body) {
    const enabled = document.getElementById("llm-setting-auto-cleanup")?.checked ?? true;
    const interval = parseInt(document.getElementById("llm-setting-interval")?.value ?? "3600", 10);
    const maxAge = parseInt(document.getElementById("llm-setting-max-age")?.value ?? "0", 10);
    const immediateWrite = document.getElementById("llm-setting-immediate-write")?.checked ?? true;

    const payload = {
        auto_cleanup_enabled: enabled,
        auto_cleanup_interval_sec: interval,
        max_age_days: maxAge,
        immediate_write: immediateWrite,
    };

    const feedback = document.getElementById("llm-db-settings-feedback");
    if (feedback) {
        feedback.style.display = "block";
        feedback.textContent = "Saving settings...";
        feedback.className = "llm-db-live-feedback";
    }

    try {
        const r = await fetch("/easyllm/db/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const result = await r.json();
        if (result.status === "ok") {
            const msg = "✅ Settings saved and persisted";
            if (feedback) {
                feedback.textContent = msg;
                feedback.className = "llm-db-live-feedback llm-db-live-success";
            }
            showToast(msg, "success", 3000);
        } else {
            throw new Error(result.error || "Save failed");
        }
    } catch (e) {
        if (feedback) {
            feedback.textContent = `❌ Failed to save: ${e.message}`;
            feedback.className = "llm-db-live-feedback llm-db-live-error";
        }
        showToast(`Settings save failed: ${e.message}`, "error", 3000);
    }
}

// ── Destroy Handler (double confirmation) ────────────────────────────

async function _handleDestroyDatabase(btn) {
    // Step 1: Warning dialog
    if (!confirm(
        "⚠️ DESTROY DATABASE\n\n" +
        "This will permanently delete ALL:\n" +
        "• Chat history\n" +
        "• Enhancer output logs\n" +
        "• All stored images\n\n" +
        "This action CANNOT be undone!\n\n" +
        "Click OK only if you are absolutely sure."
    )) {
        return;
    }

    // Step 2: Type confirmation
    const typeConfirm = prompt(
        'Type "DESTROY" to confirm:'
    );
    if (typeConfirm !== "DESTROY") {
        showToast("Destroy cancelled", "info", 2000);
        return;
    }

    btn.disabled = true;
    btn.textContent = "💥 Destroying...";

    try {
        const r = await fetch("/easyllm/db/destroy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ confirm: true }),
        });
        const result = await r.json();
        if (result.status === "ok") {
            showToast("✅ Database destroyed and recreated", "success", 3000);

            // Clear in-memory cache on ALL graph nodes (stale data no longer valid)
            try {
                const graph = window.app?.graph;
                if (graph && graph._nodes) {
                    graph._nodes.forEach((n) => {
                        n._chatHistory = [];
                        n._enhancerHistory = [];
                    });
                }
            } catch (_) {
                // Silently ignore if graph is not accessible
            }

            refreshAll();
        } else {
            throw new Error(result.error || "Destroy failed");
        }
    } catch (e) {
        showToast(`❌ Destroy failed: ${e.message}`, "error", 3000);
    } finally {
        btn.disabled = false;
        btn.textContent = "💥 Destroy & Recreate Database";
    }
}
