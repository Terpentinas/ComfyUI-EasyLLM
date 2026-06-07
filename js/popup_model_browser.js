/**
 * EasyLLM — GGUF Model Browser Popup
 *
 * New Layout Architecture:
 *   Main View (always visible):
 *     - Search bar with (X) clear button
 *     - Filter toggles: All / Text Only / Vision 🖼️
 *     - Sort buttons: Recent / Name A-Z / Family / Size ⬇️
 *     - Scrollable model card list with badges
 *     - Selected model info panel
 *   Foldable Settings Panel (hidden behind ⚙️):
 *     - Default ComfyUI Dirs (collapsible)
 *     - Custom Directories input + list
 *     - Multimodal Vision Path with Browse
 *     - Remember paths checkbox
 *   Footer: [Reset All] [⚙️ Folder Settings]   [Cancel] [Apply]
 *
 * Preserved helpers: directory management, mmproj auto-detect,
 * Apply/Cancel/Reset logic, recently-used tracking on Apply.
 */

import { createOverlayModal, showToast } from "./ui_utils.js";
import { syncPopupSettingsToCanvas, getModelName } from "./popup_bubble.js";
import { extractQuantization, extractModelFamily, formatFileSize } from "./popup_utils.js";

// ────────────────────────────────────────────────────────────────────────
// Model Browser Popup (GGUF model + mmproj selection)
// ────────────────────────────────────────────────────────────────────────

/**
 * Open GGUF model + mmproj selection popup with the new card-based layout.
 *
 * @param {object} node - The EasyLLMGGUF LiteGraph node instance
 */
export function openModelBrowserPopup(node) {
    if (node._modelBrowserOverlay) return; // Already open

    const { overlay, panel, body, footer } = createOverlayModal(
        "llm-model-browser",        // CSS prefix
        "📁 GGUF Model Browser",   // Title
        () => closeModelBrowserPopup(node),
        { hasFooter: true }
    );

    node._modelBrowserOverlay = overlay;
    document.body.appendChild(overlay);

    // Restore saved popup dimensions
    const savedW = node._popupSettings?.modelBrowserWidth;
    const savedH = node._popupSettings?.modelBrowserHeight;
    if (savedW) panel.style.width = savedW;
    if (savedH) panel.style.height = savedH;

    const currentModelPath = node._popupSettings?.model_path ||
        node.widgets?.find(w => w.name === "model_path")?.value || "";
    const currentMmprojPath = node._popupSettings?.mmproj_path ||
        node.widgets?.find(w => w.name === "mmproj_path")?.value || "";

    // ── State ──────────────────────────────────────────────────────────
    let allFiles = [];
    let activeFilter = "all";       // "all" | "text" | "vision"
    let activeSort = "recent";      // "recent" | "name" | "family" | "size"
    let sortAscending = true;       // Direction for current activeSort
    let settingsExpanded = false;
    let selectedFilePath = "";      // Currently selected model path
    let allDirsExpanded = false;    // For default dirs collapsible

    // ═══════════════════════════════════════════════════════════════════
    // MAIN VIEW: Search → Filter/Sort → Card List → Info Panel
    // ═══════════════════════════════════════════════════════════════════

    // ── 1. Search wrapper with clear button ────────────────────────────
    const searchWrapper = document.createElement("div");
    searchWrapper.className = "llm-model-search-wrapper";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "llm-popup-settings-input";
    searchInput.placeholder = "🔍 Search models...";
    searchInput.value = "";
    searchWrapper.appendChild(searchInput);

    const searchClear = document.createElement("button");
    searchClear.className = "llm-model-search-clear";
    searchClear.textContent = "✕";
    searchClear.title = "Clear search";
    searchClear.onclick = () => clearSearch();
    searchWrapper.appendChild(searchClear);
    body.appendChild(searchWrapper);

    // ── 2. Filter + Sort bar ───────────────────────────────────────────
    const filterBar = document.createElement("div");
    filterBar.className = "llm-model-filter-bar";

    const filterLabel = document.createElement("span");
    filterLabel.className = "llm-model-filter-label";
    filterLabel.textContent = "Filter:";
    filterBar.appendChild(filterLabel);

    const filterGroup = document.createElement("div");
    filterGroup.className = "llm-model-filter-group";

    const filterTypes = [
        { key: "all", label: "All" },
        { key: "text", label: "Text Only" },
        { key: "vision", label: "Vision 🖼️" },
    ];
    const filterBtns = {};
    filterTypes.forEach(ft => {
        const btn = document.createElement("button");
        btn.className = "llm-model-filter-btn" + (ft.key === activeFilter ? " active" : "");
        btn.textContent = ft.label;
        btn.dataset.filter = ft.key;
        btn.onclick = () => handleFilterChange(ft.key);
        filterGroup.appendChild(btn);
        filterBtns[ft.key] = btn;
    });
    filterBar.appendChild(filterGroup);

    const sep = document.createElement("div");
    sep.className = "llm-model-filter-sep";
    filterBar.appendChild(sep);

    const sortLabel = document.createElement("span");
    sortLabel.className = "llm-model-filter-label";
    sortLabel.textContent = "Sort:";
    filterBar.appendChild(sortLabel);

    const sortGroup = document.createElement("div");
    sortGroup.className = "llm-model-filter-group";

    const sortTypes = [
        { key: "recent", label: "Recent",       defaultAscending: true  },
        { key: "name",   label: "Name A–Z",     defaultAscending: true  },
        { key: "family", label: "Family",       defaultAscending: true  },
        { key: "size",   label: "Size ⬇️",     defaultAscending: false },
    ];
    const sortBtns = {};
    sortTypes.forEach(st => {
        const btn = document.createElement("button");
        btn.className = "llm-model-sort-btn" + (st.key === activeSort ? " active" : "");
        btn.textContent = st.label;
        btn.dataset.sort = st.key;
        btn.onclick = () => handleSortChange(st.key);
        sortGroup.appendChild(btn);
        sortBtns[st.key] = btn;
    });
    filterBar.appendChild(sortGroup);
    body.appendChild(filterBar);

    // ── 3. Cache refresh row ──────────────────────────────────────────
    const cacheRow = document.createElement("div");
    cacheRow.className = "llm-model-cache-row";

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 Refresh";
    refreshBtn.title = "Re-scan directories and rebuild model cache";
    refreshBtn.className = "llm-model-browser-btn";
    refreshBtn.onclick = async () => {
        refreshBtn.textContent = "⏳ Refreshing...";
        refreshBtn.disabled = true;
        try {
            const resp = await fetch("/easyllm/refresh_model_index", { method: "POST" });
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            await fetchGgufFiles();
            showToast("Model list refreshed", "success");
        } catch (_e) {
            showToast("Failed to refresh model list", "error");
        }
        refreshBtn.textContent = "🔄 Refresh";
        refreshBtn.disabled = false;
    };
    cacheRow.appendChild(refreshBtn);

    const cacheLabel = document.createElement("span");
    cacheLabel.id = "llm-model-cache-label";
    cacheLabel.textContent = "";
    cacheRow.appendChild(cacheLabel);
    body.appendChild(cacheRow);

    // ── 4. Model card list ─────────────────────────────────────────────
    const cardList = document.createElement("div");
    cardList.className = "llm-model-card-list";
    body.appendChild(cardList);

    // ── 5. Metadata info line (hidden until selection) ─────────────────
    // Shows a clean, minimal text block with extra metadata not already
    // visible in the card (full file path, quantization, architecture source).
    const infoPanel = document.createElement("div");
    infoPanel.className = "llm-model-info-panel";
    infoPanel.style.display = "none";
    infoPanel.innerHTML = `<span class="llm-model-info-text"></span>`;
    body.appendChild(infoPanel);

    // ═══════════════════════════════════════════════════════════════════
    // FOLDABLE SETTINGS PANEL
    // ═══════════════════════════════════════════════════════════════════

    const settingsPanel = document.createElement("div");
    settingsPanel.className = "llm-model-settings-panel";

    const settingsToggle = document.createElement("button");
    settingsToggle.className = "llm-model-settings-toggle";
    settingsToggle.innerHTML = "▶ ⚙️ Folder Settings";
    settingsToggle.onclick = () => toggleSettingsPanel(!settingsExpanded);
    settingsPanel.appendChild(settingsToggle);

    const settingsContent = document.createElement("div");
    settingsContent.className = "llm-model-settings-content";

    // ── Default ComfyUI Dirs (collapsible) ────────────────────────────
    const allDirsToggle = document.createElement("div");
    allDirsToggle.className = "llm-model-browser-dir-toggle";
    allDirsToggle.title = "Click to toggle visibility of all search directories";
    const expandIcon = document.createElement("span");
    expandIcon.textContent = "▶";
    expandIcon.className = "llm-model-browser-expand-icon";
    const allDirsLabel = document.createElement("span");
    allDirsLabel.textContent = "Default ComfyUI Directories (click to expand)";
    allDirsToggle.appendChild(expandIcon);
    allDirsToggle.appendChild(allDirsLabel);
    settingsContent.appendChild(allDirsToggle);

    const allDirsContainer = document.createElement("div");
    allDirsContainer.id = "llm-model-browser-all-dirs";
    allDirsContainer.className = "llm-model-browser-dir-container";
    settingsContent.appendChild(allDirsContainer);

    allDirsToggle.onclick = async () => {
        allDirsExpanded = !allDirsExpanded;
        expandIcon.textContent = allDirsExpanded ? "▼" : "▶";
        if (allDirsExpanded) {
            await refreshAllDirsList();
            allDirsContainer.style.display = "block";
            allDirsLabel.textContent = "Default ComfyUI Directories";
        } else {
            allDirsContainer.style.display = "none";
            allDirsLabel.textContent = "Default ComfyUI Directories (click to expand)";
        }
    };

    // ── Custom Directories ────────────────────────────────────────────
    const customDirLabel = document.createElement("div");
    customDirLabel.className = "llm-model-browser-section-label";
    const customDirLabelText = document.createElement("span");
    customDirLabelText.textContent = "📁 Custom Model Folders:";
    customDirLabel.appendChild(customDirLabelText);

    // "Clear All" button for custom dirs (hidden until dirs exist)
    const clearAllDirsBtn = document.createElement("button");
    clearAllDirsBtn.textContent = "× Clear All";
    clearAllDirsBtn.title = "Remove all custom directories";
    clearAllDirsBtn.className = "llm-model-browser-btn-danger-sm";
    clearAllDirsBtn.style.display = "none";
    clearAllDirsBtn.onclick = async () => {
        try {
            const resp = await fetch("/easyllm/clear_all_gguf_state", { method: "POST" });
            const data = await resp.json();
            if (data.success) {
                await refreshDirList();
                if (typeof refreshAllDirsList === "function") await refreshAllDirsList();
                showToast("All custom directories cleared", "success");
            }
        } catch (_e) {
            showToast("Failed to clear directories", "error");
        }
    };
    customDirLabel.appendChild(clearAllDirsBtn);
    settingsContent.appendChild(customDirLabel);

    const dirInputRow = document.createElement("div");
    dirInputRow.className = "llm-model-browser-row";

    const dirInput = document.createElement("input");
    dirInput.type = "text";
    dirInput.className = "llm-popup-settings-input";
    dirInput.style.flex = "1";
    dirInput.placeholder = "Paste folder path e.g. C:\\llm-studio\\models";
    dirInputRow.appendChild(dirInput);

    const addDirBtn = document.createElement("button");
    addDirBtn.textContent = "Add";
    addDirBtn.title = "Add this directory to the search path";
    addDirBtn.className = "llm-model-browser-btn-primary";
    addDirBtn.onclick = async () => {
        const path = dirInput.value.trim();
        if (path) await addDirectory(path);
    };
    dirInputRow.appendChild(addDirBtn);
    settingsContent.appendChild(dirInputRow);

    // Directory list container for custom dirs
    const dirList = document.createElement("div");
    dirList.id = "llm-model-browser-dir-list";
    dirList.className = "llm-model-browser-dir-list";
    settingsContent.appendChild(dirList);

    // ── Multimodal Vision Path ─────────────────────────────────────────
    const mmprojLabel = document.createElement("div");
    mmprojLabel.textContent = "🖼️ Multimodal Vision Path (Auto-detected if blank)";
    mmprojLabel.className = "llm-model-browser-section-label";
    mmprojLabel.style.marginTop = "10px";
    settingsContent.appendChild(mmprojLabel);

    const mmprojRow = document.createElement("div");
    mmprojRow.className = "llm-model-browser-row";
    const mmprojInput = document.createElement("input");
    mmprojInput.type = "text";
    mmprojInput.className = "llm-popup-settings-input";
    mmprojInput.style.flex = "1";
    mmprojInput.value = currentMmprojPath;
    mmprojInput.placeholder = "path/to/mmproj.gguf (optional)";
    node._popupMmprojInput = mmprojInput;
    mmprojRow.appendChild(mmprojInput);

    const autoDetectBtn = document.createElement("button");
    autoDetectBtn.textContent = "🔄 Auto-detect";
    autoDetectBtn.title = "Search for companion mmproj file";
    autoDetectBtn.className = "llm-model-browser-btn";
    autoDetectBtn.onclick = async () => {
        await autoDetectMmproj(node, selectedFilePath, mmprojInput);
    };
    mmprojRow.appendChild(autoDetectBtn);

    settingsContent.appendChild(mmprojRow);

    // ── mmproj hint ──
    const mmprojHint = document.createElement("div");
    mmprojHint.textContent = "💡 Auto-detected from the same folder as the selected model. You can also type the path manually.";
    mmprojHint.className = "llm-model-browser-hint";
    settingsContent.appendChild(mmprojHint);

    // ── Remember checkbox ──
    const rememberRow = document.createElement("label");
    rememberRow.style.cssText = "display: block; margin-top: 10px; font-size: 12px;";
    rememberRow.style.color = "var(--llm-text-secondary)";
    const rememberCb = document.createElement("input");
    rememberCb.type = "checkbox";
    rememberCb.checked = true;
    rememberRow.appendChild(rememberCb);
    rememberRow.appendChild(document.createTextNode(" Remember paths for future workflow sessions"));
    settingsContent.appendChild(rememberRow);

    settingsPanel.appendChild(settingsContent);
    body.appendChild(settingsPanel);

    // ═══════════════════════════════════════════════════════════════════
    // RENDERING HELPERS
    // ═══════════════════════════════════════════════════════════════════

    /** Update the info panel with structured metadata not already visible on the card. */
    function updateInfoPanel(file) {
        if (!file) {
            infoPanel.style.display = "none";
            return;
        }

        const fullPath = file.path || "";
        const displayPath = fullPath.length > 72
            ? "..." + fullPath.slice(-72)
            : fullPath;
        const quant = extractQuantization(file.name);

        // ── Rebuild info panel DOM structure ──────────────────────────
        infoPanel.innerHTML = "";

        // Row 1: File path + copy-to-clipboard button
        const pathRow = document.createElement("div");
        pathRow.className = "llm-model-info-path-row";

        const pathText = document.createElement("span");
        pathText.className = "llm-model-info-path-text";
        pathText.textContent = `📁 ${displayPath}`;
        pathRow.appendChild(pathText);

        const copyBtn = document.createElement("button");
        copyBtn.className = "llm-model-info-copy-btn";
        copyBtn.textContent = "📋 Copy";
        copyBtn.title = "Copy full path to clipboard";
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(fullPath).then(() => {
                copyBtn.textContent = "✅ Copied!";
                setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 2000);
            }).catch(() => {
                // Fallback for older browsers
                const ta = document.createElement("textarea");
                ta.value = fullPath;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
                copyBtn.textContent = "✅ Copied!";
                setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 2000);
            });
        };
        pathRow.appendChild(copyBtn);
        infoPanel.appendChild(pathRow);

        // Row 2: Source + quant badges
        const badgesRow = document.createElement("div");
        badgesRow.className = "llm-model-info-badges";

        // Source badge (metadata quality indicator)
        if (file.architecture_source) {
            const sourceBadge = document.createElement("span");
            sourceBadge.className = "llm-pill";
            sourceBadge.textContent = file.architecture_source === "metadata"
                ? "✅ metadata"
                : "🔤 filename";
            badgesRow.appendChild(sourceBadge);

            // Visual separator
            const sep = document.createElement("span");
            sep.className = "llm-model-info-sep";
            sep.textContent = "│";
            badgesRow.appendChild(sep);
        }

        // Quantization badge
        if (quant) {
            const quantBadge = document.createElement("span");
            quantBadge.className = "llm-pill llm-pill-size";
            quantBadge.textContent = quant;
            badgesRow.appendChild(quantBadge);
        }

        infoPanel.appendChild(badgesRow);
        infoPanel.style.display = "";
    }

    /**
     * Apply filter + sort to the file list.
     * @returns {Array} Filtered and sorted file entries.
     */
    function applyFilterAndSort() {
        let result = [...allFiles];

        // Apply search text filter
        const searchText = searchInput.value.toLowerCase().trim();
        if (searchText) {
            result = result.filter(f => f.name.toLowerCase().includes(searchText));
        }

        // Apply type filter
        if (activeFilter === "vision") {
            result = result.filter(f => f.has_mmproj);
        } else if (activeFilter === "text") {
            result = result.filter(f => !f.has_mmproj);
        }

        // Apply sort with direction support
        if (activeSort === "name") {
            const dir = sortAscending ? 1 : -1;
            result.sort((a, b) => dir * (a.name || "").localeCompare(b.name || ""));
        } else if (activeSort === "family") {
            const dir = sortAscending ? 1 : -1;
            result.sort((a, b) => {
                const fa = (extractModelFamily(a.name) || a.architecture || "").toLowerCase();
                const fb = (extractModelFamily(b.name) || b.architecture || "").toLowerCase();
                return dir * (fa.localeCompare(fb) || (a.name || "").localeCompare(b.name || ""));
            });
        } else if (activeSort === "size") {
            if (sortAscending) {
                result.sort((a, b) => (a.file_size || 0) - (b.file_size || 0));
            } else {
                result.sort((a, b) => (b.file_size || 0) - (a.file_size || 0));
            }
        }
        // "recent" — keep server order (default)

        return result;
    }

    /** Render the model card list based on current filter + sort state. */
    function renderModelCardList() {
        cardList.innerHTML = "";

        if (allFiles.length === 0) {
            // No files loaded — show empty state placeholder
            return;
        }

        const filtered = applyFilterAndSort();

        if (filtered.length === 0) {
            const empty = document.createElement("div");
            empty.className = "llm-model-card-list-empty";
            empty.textContent = searchInput.value.trim()
                ? "(No matching models)"
                : activeFilter !== "all"
                    ? `(No ${activeFilter === "vision" ? "vision" : "text-only"} models found)`
                    : "(No .gguf files found in search directories)";
            cardList.appendChild(empty);
            return;
        }

        filtered.forEach(file => {
            const card = createModelCardElement(file);
            cardList.appendChild(card);
        });
    }

    /** Create a single model card DOM element. */
    function createModelCardElement(file) {
        const card = document.createElement("div");
        card.className = "llm-model-card";
        // Store file name in data attribute for selection
        card.dataset.path = file.name || file.path.split(/[\\/]/).pop() || "";
        if (file.path === selectedFilePath) {
            card.classList.add("selected");
        }

        // Model name
        const nameEl = document.createElement("div");
        nameEl.className = "llm-model-card-name";
        nameEl.textContent = file.name || file.path.split(/[\\/]/).pop() || "";
        card.appendChild(nameEl);

        // Badges row
        const badgesEl = document.createElement("div");
        badgesEl.className = "llm-model-card-badges";

        if (file.architecture) {
            const pill = document.createElement("span");
            pill.className = "llm-arch-pill";
            pill.textContent = file.architecture;
            badgesEl.appendChild(pill);
        }
        if (file.has_mmproj) {
            const badge = document.createElement("span");
            badge.className = "llm-pill llm-pill-mmproj";
            badge.textContent = "🖼️ mmproj";
            badge.title = "Vision/multimodal model — requires an mmproj projection file";
            badgesEl.appendChild(badge);
        }
        if (file.file_size != null && file.file_size > 0) {
            const sizeBadge = document.createElement("span");
            sizeBadge.className = "llm-pill llm-pill-size";
            sizeBadge.textContent = `⚡ ${formatFileSize(file.file_size)}`;
            badgesEl.appendChild(sizeBadge);
        }
        card.appendChild(badgesEl);

        card.onclick = () => selectModel(file);
        return card;
    }

    /** Handle selecting a model from the card list. */
    async function selectModel(file) {
        selectedFilePath = file.path;

        // Update visual selection state
        const cards = cardList.querySelectorAll(".llm-model-card");
        cards.forEach(c => c.classList.remove("selected"));
        const targetName = file.name || file.path.split(/[\\/]/).pop() || "";
        cards.forEach(c => {
            if (c.dataset.path === targetName) {
                c.classList.add("selected");
            }
        });

        if (!node._popupSettings) node._popupSettings = {};
        node._popupSettings.model_path = file.path;

        // Show model info panel instantly from cache
        updateInfoPanel(file);

        // Auto-detect companion mmproj
        await autoDetectMmproj(node, file.path, mmprojInput);

        // Re-render info panel if auto-detect found an mmproj
        if (mmprojInput.value) {
            file.has_mmproj = true;
            file.mmproj_path = mmprojInput.value;
            updateInfoPanel(file);
        }
    }

    /** Clear search input and re-render. */
    function clearSearch() {
        searchInput.value = "";
        searchClear.classList.remove("visible");
        renderModelCardList();
        searchInput.focus();
    }

    /** Handle filter type change. */
    function handleFilterChange(filterType) {
        activeFilter = filterType;
        Object.keys(filterBtns).forEach(k => {
            filterBtns[k].classList.toggle("active", k === filterType);
        });
        renderModelCardList();
    }

    /** Handle sort type change with bidirectional toggle. */
    function handleSortChange(sortType) {
        const sortDef = sortTypes.find(st => st.key === sortType);
        if (!sortDef) return;

        if (sortType === activeSort && sortType !== "recent") {
            // Same button clicked — toggle direction
            sortAscending = !sortAscending;
        } else {
            // Different button — switch mode and reset to default direction
            activeSort = sortType;
            sortAscending = sortDef.defaultAscending;
        }

        // Update button active states
        Object.keys(sortBtns).forEach(k => {
            sortBtns[k].classList.toggle("active", k === sortType);
        });

        // Update button labels to reflect current direction
        updateSortButtonLabels();

        renderModelCardList();
    }

    /** Update sort button labels to show current direction. */
    function updateSortButtonLabels() {
        sortTypes.forEach(st => {
            const btn = sortBtns[st.key];
            if (!btn) return;
            let label;
            if (st.key === "name") {
                label = sortAscending ? "Name A–Z" : "Name Z–A";
            } else if (st.key === "size") {
                label = sortAscending ? "Size ⬆️" : "Size ⬇️";
            } else {
                label = st.label; // "Recent" or "Family" — no direction indicator needed
            }
            btn.textContent = label;
        });
    }

    /** Toggle the foldable settings panel. */
    function toggleSettingsPanel(show) {
        settingsExpanded = (show !== undefined) ? show : !settingsExpanded;
        if (settingsExpanded) {
            settingsPanel.classList.add("llm-model-settings-expanded");
            settingsToggle.innerHTML = "▼ Hide Folder Settings";
        } else {
            settingsPanel.classList.remove("llm-model-settings-expanded");
            settingsToggle.innerHTML = "▶ ⚙️ Folder Settings";
        }
    }

    // ── Search input events ──────────────────────────────────────────
    searchInput.addEventListener("input", () => {
        searchClear.classList.toggle("visible", searchInput.value.length > 0);
        // Debounced re-render
        clearTimeout(searchInput._debounce);
        searchInput._debounce = setTimeout(() => {
            renderModelCardList();
        }, 200);
    });

    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            clearSearch();
        }
    });

    // ═══════════════════════════════════════════════════════════════════
    // SERVER COMMUNICATION HELPERS
    // ═══════════════════════════════════════════════════════════════════

    /** Fetch the GGUF file list from server. */
    async function fetchGgufFiles(staleOk) {
        // Show loading state in card list
        cardList.innerHTML = `
            <div class="llm-model-card-list-loading">
                <span class="llm-spinner"></span>
                Loading model list...
            </div>`;

        /** Update the cache freshness label. */
        function updateCacheLabel(text, title) {
            const el = document.getElementById("llm-model-cache-label");
            if (el) {
                el.textContent = text || "";
                if (title) el.title = title;
            }
        }

        try {
            const url = staleOk
                ? "/easyllm/list_gguf_files?stale_ok=true"
                : "/easyllm/list_gguf_files";
            const resp = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
            });
            if (resp.ok) {
                const data = await resp.json();
                allFiles = data.files || [];

                // ── Update cache freshness timestamp ──
                if (data.cached_at) {
                    const now = Date.now();
                    const cacheTime = new Date(data.cached_at).getTime();
                    const ageMs = now - cacheTime;
                    const ageMin = Math.floor(ageMs / 60000);
                    const ageHr = Math.floor(ageMs / 3600000);

                    let ageText;
                    if (ageMin < 1) {
                        ageText = "just now";
                    } else if (ageMin < 60) {
                        ageText = `${ageMin} min ago`;
                    } else if (ageHr < 24) {
                        ageText = `${ageHr} hr ${ageMin % 60} min ago`;
                    } else {
                        ageText = `${Math.floor(ageHr / 24)} days ago`;
                    }
                    updateCacheLabel(`Updated ${ageText}`, `Cache built at: ${data.cached_at}`);
                } else {
                    updateCacheLabel("(scanning...)");
                }
            } else {
                allFiles = [];
                updateCacheLabel("(scan failed)");
            }
        } catch (_e) {
            allFiles = [];
            updateCacheLabel("(scan failed)");
        }
        // Always render after fetch
        renderModelCardList();
    }

    // ── Directory management helpers ───────────────────────────────────
    async function addDirectory(path) {
        try {
            const resp = await fetch("/easyllm/register_gguf_dir", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ directory: path }),
            });
            const data = await resp.json();
            if (data.success) {
                dirInput.value = "";
                await refreshDirList();
                if (allDirsExpanded) await refreshAllDirsList();
                showToast(`Added search directory`, "success");
            } else {
                showToast(data.error || "Failed to add directory", "error");
            }
        } catch (_e) {
            showToast("Failed to add directory", "error");
        }
    }

    async function removeDirectory(path) {
        try {
            const resp = await fetch("/easyllm/unregister_gguf_dir", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ directory: path }),
            });
            const data = await resp.json();
            if (data.success) {
                await refreshDirList();
                if (allDirsExpanded) await refreshAllDirsList();
            }
        } catch (_e) {}
    }

    async function toggleExcludeDirectory(path) {
        try {
            const exclResp = await fetch("/easyllm/list_excluded_dirs");
            const exclData = await exclResp.json();
            const excluded = exclData.directories || [];
            const isExcluded = excluded.includes(path);

            const endpoint = isExcluded ? "/easyllm/unexclude_gguf_dir" : "/easyllm/exclude_gguf_dir";
            const resp = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ directory: path }),
            });
            const data = await resp.json();
            if (data.success) {
                await refreshAllDirsList();
                await fetchGgufFiles(true);
                showToast(
                    isExcluded ? `Directory re-enabled: ${path}` : `Directory excluded: ${path}`,
                    "success"
                );
            }
        } catch (_e) {}
    }

    async function refreshAllDirsList() {
        try {
            const resp = await fetch("/easyllm/list_all_search_dirs");
            const data = await resp.json();
            const allDirs = data.directories || [];
            const excluded = new Set(data.excluded || []);
            const custom = new Set(data.custom || []);
            allDirsContainer.innerHTML = "";

            if (allDirs.length === 0) {
                const empty = document.createElement("div");
                empty.textContent = "(no search directories available)";
                empty.className = "llm-model-browser-empty-text";
                allDirsContainer.appendChild(empty);
            } else {
                allDirs.forEach(d => {
                    const row = document.createElement("div");
                    row.className = "llm-model-browser-dir-row";

                    const toggleBtn = document.createElement("button");
                    const isExcluded = excluded.has(d);
                    toggleBtn.textContent = isExcluded ? "⬜" : "✅";
                    toggleBtn.title = isExcluded
                        ? "Click to include this directory in search"
                        : "Click to exclude this directory from search";
                    toggleBtn.className = "llm-model-browser-btn-icon";
                    toggleBtn.onclick = () => toggleExcludeDirectory(d);
                    row.appendChild(toggleBtn);

                    const label = document.createElement("span");
                    label.textContent = d;
                    label.className = isExcluded ? "llm-model-browser-dir-label-excluded" : "llm-model-browser-dir-label";
                    row.appendChild(label);

                    if (custom.has(d)) {
                        const badge = document.createElement("span");
                        badge.textContent = "📁";
                        badge.title = "Custom directory (added by you)";
                        badge.className = "llm-model-browser-custom-badge";
                        row.appendChild(badge);
                    }

                    allDirsContainer.appendChild(row);
                });
            }
        } catch (_e) {
            allDirsContainer.innerHTML = "<div class='llm-model-browser-error-text'>Failed to load directories</div>";
        }
    }

    async function refreshDirList() {
        try {
            const resp = await fetch("/easyllm/list_browsed_dirs");
            const data = await resp.json();
            const dirs = data.directories || [];
            dirList.innerHTML = "";
            if (clearAllDirsBtn) {
                clearAllDirsBtn.style.display = dirs.length > 0 ? "" : "none";
            }
            if (dirs.length === 0) {
                const empty = document.createElement("div");
                empty.textContent = "(no custom directories added)";
                empty.className = "llm-model-browser-empty-text";
                dirList.appendChild(empty);
            } else {
                dirs.forEach(d => {
                    const row = document.createElement("div");
                    row.className = "llm-model-browser-dir-row";
                    const dot = document.createElement("span");
                    dot.textContent = "•";
                    dot.className = "llm-model-browser-dot";
                    row.appendChild(dot);
                    const label = document.createElement("span");
                    label.textContent = d;
                    label.className = "llm-model-browser-dir-label";
                    row.appendChild(label);
                    const removeBtn = document.createElement("button");
                    removeBtn.textContent = "✕";
                    removeBtn.title = "Remove this directory from search";
                    removeBtn.className = "llm-model-browser-btn-danger-sm";
                    removeBtn.onclick = () => removeDirectory(d);
                    row.appendChild(removeBtn);
                    dirList.appendChild(row);
                });
            }
            // Re-fetch GGUF file list after directory change
            await fetchGgufFiles();
        } catch (_e) {
            dirList.innerHTML = "<div class='llm-model-browser-error-text'>Failed to load directories</div>";
        }
    }

    // ── Initial load ───────────────────────────────────────────────────
    refreshDirList();

    // ═══════════════════════════════════════════════════════════════════
    // FOOTER BUTTONS
    // ═══════════════════════════════════════════════════════════════════

    // Left side: Reset All + ⚙️ Folder Settings
    const footerLeft = document.createElement("div");
    footerLeft.className = "llm-model-browser-footer-left";

    // Reset All button
    const resetAllBtn = document.createElement("button");
    resetAllBtn.textContent = "Reset All";
    resetAllBtn.title = "Clear all custom directories, exclusions, and recently used models";
    resetAllBtn.className = "llm-model-browser-btn-danger";
    resetAllBtn.onclick = async () => {
        if (!confirm("Reset all GGUF state? This will clear your custom directories, exclusions, and recent model list.")) {
            return;
        }
        try {
            const resp = await fetch("/easyllm/clear_all_gguf_state", { method: "POST" });
            const data = await resp.json();
            if (!data.success) {
                showToast("Failed to reset state: " + (data.error || "unknown error"), "error");
                return;
            }
            localStorage.removeItem("llm_chat_recent_gguf_pairs");
            // Refresh directory and exclusion views
            await refreshDirList();
            if (typeof refreshAllDirsList === "function") await refreshAllDirsList();
            showToast("All GGUF state reset to defaults", "success");
        } catch (_e) {
            showToast("Failed to reset state", "error");
        }
    };
    footerLeft.appendChild(resetAllBtn);

    // ⚙️ Folder Settings toggle
    const settingsFooterBtn = document.createElement("button");
    settingsFooterBtn.textContent = "⚙️ Folder Settings";
    settingsFooterBtn.title = "Show or hide folder configuration panel";
    settingsFooterBtn.className = "llm-model-browser-btn-ghost";
    settingsFooterBtn.onclick = () => {
        toggleSettingsPanel();
        // Sync toggle text with the settings panel header
        settingsToggle.innerHTML = settingsExpanded ? "▼ Hide Folder Settings" : "▶ ⚙️ Folder Settings";
    };
    footerLeft.appendChild(settingsFooterBtn);
    footer.appendChild(footerLeft);

    // Right side: Cancel + Apply
    const footerRight = document.createElement("div");
    footerRight.className = "llm-model-browser-footer-right";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.className = "llm-model-browser-btn-cancel";
    cancelBtn.onclick = () => closeModelBrowserPopup(node);
    footerRight.appendChild(cancelBtn);

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.className = "llm-model-browser-btn-apply";
    applyBtn.onclick = async () => {
        // Don't allow Apply while auto-detect is in-flight
        if (node._mmprojDetecting) {
            showToast("Auto-detecting mmproj... wait a moment", "warning");
            return;
        }
        const selectedPath = selectedFilePath.trim();
        if (!selectedPath) { showToast("No model selected", "error"); return; }
        const selectedMmproj = mmprojInput.value.trim();

        // Optionally validate on server (and trigger browse-to-learn)
        if (rememberCb.checked) {
            try {
                await fetch("/easyllm/validate_gguf_path", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: selectedPath }),
                });
                if (selectedMmproj) {
                    await fetch("/easyllm/validate_gguf_path", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ path: selectedMmproj }),
                    });
                }
            } catch (_) {}
        }

        // Update node state
        if (!node._popupSettings) node._popupSettings = {};
        node._popupSettings.model_path = selectedPath;
        node._popupSettings.mmproj_path = selectedMmproj;

        // Directly update canvas widget values (syncPopupSettingsToCanvas reads
        // from _popupModelPathInput which is only set by the chat popup, not the
        // model browser — without this, the canvas node never receives the update).
        const mpW = node.widgets?.find(w => w.name === "model_path");
        if (mpW) mpW.value = selectedPath;
        const mmW = node.widgets?.find(w => w.name === "mmproj_path");
        if (mmW) mmW.value = selectedMmproj;

        syncPopupSettingsToCanvas(node);

        // ── Refresh the chat popup header title with new model name ──
        // Without this, the model name stays stale until the user closes
        // and reopens the chat popup.
        const chatHeaderTitle = node._popupPanel?.querySelector(".llm-popup-header-title");
        if (chatHeaderTitle) {
            const newName = getModelName(node);
            chatHeaderTitle.textContent = `🤖 EasyLLM${newName ? ` | ${newName}` : ""}`;
        }

        // ── Pre-load model only if vram_mode is keep_loaded ──
        const vramW = node.widgets?.find(w => w.name === "vram_mode");
        const vramMode = vramW ? vramW.value : "unload";
        if (vramMode === "keep_loaded") {
            const nGpuW = node.widgets?.find(w => w.name === "n_gpu_layers");
            const nCtxW = node.widgets?.find(w => w.name === "n_ctx");
            const mlockW = node.widgets?.find(w => w.name === "use_mlock");
            fetch("/easyllm/preload_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_path: selectedPath,
                    mmproj_path: selectedMmproj,
                    n_gpu_layers: nGpuW ? nGpuW.value : -1,
                    use_mlock: mlockW ? mlockW.value : true,
                    n_ctx: nCtxW ? nCtxW.value : 4096,
                }),
            }).catch(() => {}); // fire-and-forget — ignore errors
        }

        // Save paired entry to recent list
        const recent = JSON.parse(
            localStorage.getItem("llm_chat_recent_gguf_pairs") || "[]"
        );
        const entry = { model: selectedPath, mmproj: selectedMmproj || "" };
        const updated = [entry, ...recent.filter(p => p.model !== selectedPath)].slice(0, 10);
        localStorage.setItem("llm_chat_recent_gguf_pairs", JSON.stringify(updated));

        // Save popup dimensions for next open
        const panelW = panel.offsetWidth;
        const panelH = panel.offsetHeight;
        if (!node._popupSettings) node._popupSettings = {};
        node._popupSettings.modelBrowserWidth = panelW + "px";
        node._popupSettings.modelBrowserHeight = panelH + "px";

        const displayName = selectedPath.split(/[\\/]/).pop() || selectedPath;
        showToast(`Model set to: ${displayName}`, "success");
        closeModelBrowserPopup(node);
    };
    footerRight.appendChild(applyBtn);
    footer.appendChild(footerRight);
}

// ═══════════════════════════════════════════════════════════════════════
// AUTO-DETECT MMPROJ (preserved helper)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Auto-detect companion mmproj file via server API.
 *
 * @param {object} node - The EasyLLMGGUF node
 * @param {string} modelPath - The resolved model path
 * @param {HTMLInputElement} mmprojInput - The mmproj input element to fill
 */
async function autoDetectMmproj(node, modelPath, mmprojInput) {
    if (!modelPath || !modelPath.trim()) return;
    node._mmprojDetecting = true;
    try {
        const resp = await fetch("/easyllm/auto_detect_mmproj", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_path: modelPath }),
        });
        if (resp.ok) {
            const data = await resp.json();
            if (data.mmproj_path) {
                mmprojInput.value = data.mmproj_path;
                const displayName = data.mmproj_path.split(/[\\/]/).pop() || data.mmproj_path;
                showToast(`Auto-detected mmproj: ${displayName}`, "success");
            } else {
                // No companion found — clear old value and notify user
                mmprojInput.value = "";
                showToast("No companion mmproj file found", "warning");
            }
        } else {
            mmprojInput.value = "";
            showToast("Auto-detect mmproj failed", "warning");
        }
    } catch (_e) {
        mmprojInput.value = "";
        showToast("Auto-detect mmproj error", "warning");
    } finally {
        node._mmprojDetecting = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CLOSE POPUP (preserved helper)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Close the model browser popup and clean up references.
 *
 * @param {object} node - The EasyLLMGGUF LiteGraph node instance
 */
export function closeModelBrowserPopup(node) {
    if (node._modelBrowserOverlay) {
        node._modelBrowserOverlay.remove();
        node._modelBrowserOverlay = null;
    }
    node._popupModelPathInput = null;
    node._popupMmprojInput = null;
}
