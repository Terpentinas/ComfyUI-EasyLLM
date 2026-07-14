/**
 * EasyLLM — Prompt management dialog (add/edit/delete/import/export system prompts)
 *
 * v3: Flat prompts model, "System Prompts" locked at bottom, no default/hardcoded prompts.
 * All prompts are editable, movable, and deletable.
 * Uses CSS classes (.llm-manager-*) defined in llm_chat.css and constants.js.
 */

import {
    fetchPrompts,
    savePromptsToBackend,
    importPromptsFromFile,
    importTextFilesAsPrompts,
    saveCategories as saveCategoriesApi,
    deleteCategory as deleteCategoryApi,
    exportAllPrompts as exportAllPromptsApi,
} from "./api.js";
import { createOverlayModal } from "./ui_utils.js";

// ────────────────────────────────────────────────────────────────────────
// Management Dialog: Open the prompt manager modal
// ────────────────────────────────────────────────────────────────────────

export function openPromptManagerDialog() {
    const onClose = () => {
        // Save popup dimensions before closing
        const w = panel.offsetWidth;
        const h = panel.offsetHeight;
        try {
            localStorage.setItem("llm_manager_panel_width", w + "px");
            localStorage.setItem("llm_manager_panel_height", h + "px");
        } catch (_e) { /* localStorage may be unavailable */ }
        document.body.removeChild(overlay);
    };

    const { overlay, panel, body, footer } = createOverlayModal(
        "llm-manager",
        "📚 Prompt Library",
        onClose,
        {
            hasFooter: true,
            closeBtnClass: "llm-manager-btn llm-manager-btn-danger llm-manager-btn-small",
        }
    );

    document.body.appendChild(overlay);

    // Restore saved popup dimensions
    try {
        const savedW = localStorage.getItem("llm_manager_panel_width");
        const savedH = localStorage.getItem("llm_manager_panel_height");
        if (savedW) panel.style.width = savedW;
        if (savedH) panel.style.height = savedH;
    } catch (_e) { /* localStorage may be unavailable */ }

    renderPromptManager(body, footer);
}

// ────────────────────────────────────────────────────────────────────────
// Confirmation Dialog (Promise-based styled modal)
// ────────────────────────────────────────────────────────────────────────

function showConfirmDialog(title, message, okText = "Delete", okClass = "llm-manager-btn-danger") {
    return new Promise((resolve) => {
        const confirmOverlay = document.createElement("div");
        confirmOverlay.className = "llm-manager-confirm-overlay";

        const confirmPanel = document.createElement("div");
        confirmPanel.className = "llm-manager-confirm-panel";

        const titleEl = document.createElement("div");
        titleEl.className = "llm-manager-confirm-title";
        titleEl.textContent = title;
        confirmPanel.appendChild(titleEl);

        const msgEl = document.createElement("div");
        msgEl.className = "llm-manager-confirm-message";
        msgEl.textContent = message;
        confirmPanel.appendChild(msgEl);

        const btnRow = document.createElement("div");
        btnRow.className = "llm-manager-confirm-btns";

        const cancelBtn = document.createElement("button");
        cancelBtn.className = "llm-manager-btn";
        cancelBtn.textContent = "Cancel";
        cancelBtn.onclick = () => {
            document.body.removeChild(confirmOverlay);
            resolve(false);
        };
        btnRow.appendChild(cancelBtn);

        const okBtn = document.createElement("button");
        okBtn.className = `llm-manager-btn ${okClass}`;
        okBtn.textContent = okText;
        okBtn.onclick = () => {
            document.body.removeChild(confirmOverlay);
            resolve(true);
        };
        btnRow.appendChild(okBtn);

        confirmPanel.appendChild(btnRow);
        confirmOverlay.appendChild(confirmPanel);
        document.body.appendChild(confirmOverlay);
    });
}

// ────────────────────────────────────────────────────────────────────────
// Simple toast feedback (auto-fades after 3s)
// ────────────────────────────────────────────────────────────────────────

function showFeedback(container, message, type = "success") {
    const el = document.createElement("div");
    el.className = `llm-manager-feedback llm-manager-feedback-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
        if (el.parentElement) el.parentElement.removeChild(el);
    }, 3000);
}

// ────────────────────────────────────────────────────────────────────────
// State (per render cycle)
// ────────────────────────────────────────────────────────────────────────

let _selectedForExport = new Set(); // tracks which prompt indices are checked
let _currentFilter = "All Categories"; // current category filter value

const LOCKED_CATEGORY = "System Prompts";

// ────────────────────────────────────────────────────────────────────────
// Render: Main prompt manager UI
// ────────────────────────────────────────────────────────────────────────

export async function renderPromptManager(body, footer) {
    const struct = await fetchPrompts();
    body.innerHTML = "";
    footer.innerHTML = "";
    _selectedForExport = new Set();
    _currentFilter = "All Categories";

    // ── Helper: get flat array of all prompts (no _isDefault flags) ──
    function getFlatPrompts() {
        return struct.prompts || [];
    }

    // ── Row 1: Toolbar (search + import + export all) ──
    const toolbar = document.createElement("div");
    toolbar.className = "llm-manager-toolbar";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "llm-manager-search";
    searchInput.placeholder = "🔍 Search prompts by name...";
    toolbar.appendChild(searchInput);

    // Auto-focus the search input when dialog opens
    setTimeout(() => searchInput.focus(), 100);

    const importBtn = document.createElement("button");
    importBtn.className = "llm-manager-btn";
    importBtn.textContent = "📥 Import";
    importBtn.title = "Import prompts from a JSON file";
    importBtn.onclick = () => showImportDialog(body, footer, struct);
    toolbar.appendChild(importBtn);

    const exportAllBtn = document.createElement("button");
    exportAllBtn.className = "llm-manager-btn";
    exportAllBtn.textContent = "📤 Export All";
    exportAllBtn.title = "Download all prompts with categories as JSON";
    exportAllBtn.onclick = () => doExportAll();
    toolbar.appendChild(exportAllBtn);

    body.appendChild(toolbar);

    // ── Row 2: Add New Prompt (full-width) ──
    const addBtn = document.createElement("button");
    addBtn.className = "llm-manager-btn llm-manager-btn-primary llm-manager-add-row";
    addBtn.textContent = "+ Add New Prompt";
    addBtn.onclick = () => {
        showPromptEditor(body, footer, struct, null, true);
    };
    body.appendChild(addBtn);

    // ── Row 3: Category filter + manage ──
    const filterRow = document.createElement("div");
    filterRow.className = "llm-manager-filter-row";

    const filterLabel = document.createElement("span");
    filterLabel.className = "llm-manager-filter-label";
    filterLabel.textContent = "Category:";
    filterRow.appendChild(filterLabel);

    const filterSelect = document.createElement("select");
    filterSelect.className = "llm-manager-filter-select";
    body.appendChild(filterRow);

    const manageCatBtn = document.createElement("button");
    manageCatBtn.className = "llm-manager-btn llm-manager-btn-small";
    manageCatBtn.textContent = "⚙ Manage";
    manageCatBtn.title = "Add, rename, reorder, or delete categories";
    manageCatBtn.onclick = () => showCategoryManager(body, footer, struct);

    // Build filter dropdown options
    function buildFilterOptions() {
        filterSelect.innerHTML = "";
        const cats = struct.categories || [LOCKED_CATEGORY];

        const allOpt = document.createElement("option");
        allOpt.value = "All Categories";
        allOpt.textContent = "All Categories";
        filterSelect.appendChild(allOpt);

        for (const cat of cats) {
            const opt = document.createElement("option");
            opt.value = cat;
            opt.textContent = cat;
            filterSelect.appendChild(opt);
        }
        // No separate "System Prompts" option — it's already in the categories list
    }
    buildFilterOptions();
    filterSelect.value = _currentFilter;

    filterRow.appendChild(filterSelect);
    filterRow.appendChild(manageCatBtn);

    // ── Batch operations toolbar ──
    const batchBar = document.createElement("div");
    batchBar.className = "llm-manager-batch-bar";

    // Left group: checkbox + "Select all"
    const batchLeft = document.createElement("div");
    batchLeft.className = "llm-manager-batch-left";

    const selectAllCheckbox = document.createElement("input");
    selectAllCheckbox.type = "checkbox";
    selectAllCheckbox.className = "llm-manager-checkbox";
    selectAllCheckbox.title = "Select all visible";
    batchLeft.appendChild(selectAllCheckbox);

    const selectAllLabel = document.createElement("span");
    selectAllLabel.className = "llm-manager-batch-label";
    selectAllLabel.textContent = "Select all";
    batchLeft.appendChild(selectAllLabel);

    batchBar.appendChild(batchLeft);

    // Right group: "Move to:" + dropdown + Apply + Remove
    const batchRight = document.createElement("div");
    batchRight.className = "llm-manager-batch-right";

    const moveLabel = document.createElement("span");
    moveLabel.className = "llm-manager-batch-label";
    moveLabel.textContent = "Move to:";
    batchRight.appendChild(moveLabel);

    const moveSelect = document.createElement("select");
    moveSelect.className = "llm-manager-select llm-manager-select-small";
    batchRight.appendChild(moveSelect);

    const moveBtn = document.createElement("button");
    moveBtn.className = "llm-manager-btn llm-manager-btn-small";
    moveBtn.textContent = "Apply";
    moveBtn.onclick = () => doBatchMove();
    batchRight.appendChild(moveBtn);

    const removeBtn = document.createElement("button");
    removeBtn.className = "llm-manager-btn llm-manager-btn-danger llm-manager-btn-small";
    removeBtn.textContent = "🗑 Remove Selected";
    removeBtn.onclick = () => doBatchRemove();
    batchRight.appendChild(removeBtn);

    batchBar.appendChild(batchRight);

    // ── Grid container ──
    const grid = document.createElement("div");
    grid.className = "llm-manager-grid";

    // ── Render function ──
    function renderGrid() {
        grid.innerHTML = "";
        const allFlat = getFlatPrompts();
        const lowerSearch = (searchInput.value || "").toLowerCase();

        // Filter by category + search
        let visible = allFlat.filter((p) => {
            // Category filter (no _isDefault checks — all prompts are equal)
            if (_currentFilter !== "All Categories") {
                const pCat = p.category || "Favorites";
                if (pCat !== _currentFilter) return false;
            }

            // Search filter
            if (lowerSearch && !(p.name || "").toLowerCase().includes(lowerSearch)) {
                return false;
            }

            return true;
        });

        // Update select-all checkbox
        const allVisibleSelected = visible.length > 0 && visible.every((p) => {
            const flatIdx = allFlat.indexOf(p);
            return _selectedForExport.has(flatIdx);
        });
        selectAllCheckbox.checked = allVisibleSelected;
        selectAllLabel.textContent = allVisibleSelected ? "Deselect all" : "Select all";

        // Build move-to options
        function buildMoveSelect(sel) {
            sel.innerHTML = "";
            const cats = struct.categories || [LOCKED_CATEGORY];
            for (const cat of cats) {
                const opt = document.createElement("option");
                opt.value = cat;
                opt.textContent = cat;
                sel.appendChild(opt);
            }
        }
        buildMoveSelect(moveSelect);

        if (visible.length === 0) {
            const empty = document.createElement("div");
            empty.className = "llm-manager-empty";
            empty.textContent = lowerSearch
                ? "No prompts match your search."
                : _currentFilter === "All Categories"
                    ? "No prompts yet. Click '+ Add New Prompt' to create one."
                    : `No prompts in "${_currentFilter}".`;
            grid.appendChild(empty);

            // Update footer
            updateFooter(allFlat, visible);
            return;
        }

        // Render cards (no isSystemView — all cards are equal)
        for (const p of visible) {
            const flatIdx = allFlat.indexOf(p);
            const card = createCard(p, flatIdx, struct, () => renderGrid());
            grid.appendChild(card);
        }

        // Update footer
        updateFooter(allFlat, visible);
    }

    // ── Footer ──
    function updateFooter(allFlat, visible) {
        footer.innerHTML = "";
        const countLabel = document.createElement("span");
        countLabel.className = "llm-manager-count";
        const selectedCount = _selectedForExport.size;
        let text = `${visible.length} shown · ${allFlat.length} total`;
        if (selectedCount > 0) {
            text += ` · ${selectedCount} selected for export`;
        }
        countLabel.textContent = text;
        footer.appendChild(countLabel);

        if (selectedCount > 0) {
            const exportSelBtn = document.createElement("button");
            exportSelBtn.className = "llm-manager-btn llm-manager-btn-small";
            exportSelBtn.textContent = "📤 Export Selected";
            exportSelBtn.onclick = () => doExportSelected(allFlat);
            footer.appendChild(exportSelBtn);
        }
    }

    // ── Event handlers ──

    // Select all / deselect all
    selectAllCheckbox.onchange = () => {
        const allFlat = getFlatPrompts();
        const lowerSearch = (searchInput.value || "").toLowerCase();
        const checked = selectAllCheckbox.checked;
        for (let i = 0; i < allFlat.length; i++) {
            const p = allFlat[i];
            // Match same filter logic (no _isDefault checks)
            if (_currentFilter !== "All Categories") {
                const pCat = p.category || "Favorites";
                if (pCat !== _currentFilter) continue;
            }
            if (lowerSearch && !(p.name || "").toLowerCase().includes(lowerSearch)) continue;
            if (checked) {
                _selectedForExport.add(i);
            } else {
                _selectedForExport.delete(i);
            }
        }
        renderGrid();
    };

    // Search handler
    searchInput.addEventListener("input", () => renderGrid());

    // Filter change (batch bar always visible — all categories support batch ops)
    filterSelect.addEventListener("change", () => {
        _currentFilter = filterSelect.value;
        renderGrid();
    });

    // Batch move
    async function doBatchMove() {
        const allFlat = getFlatPrompts();
        const targetCat = moveSelect.value;
        const toMove = [];
        for (const idx of _selectedForExport) {
            if (allFlat[idx]) {
                toMove.push(allFlat[idx]);
            }
        }
        if (toMove.length === 0) {
            showFeedback(body, "No selected prompts to move.", "error");
            return;
        }
        for (const p of toMove) {
            p.category = targetCat;
        }
        // Ensure target category exists in categories list
        if (!struct.categories.includes(targetCat)) {
            struct.categories.push(targetCat);
        }
        const result = await savePromptsToBackend({
            categories: struct.categories,
            prompts: struct.prompts,
        });
        if (result && result.success !== false) {
            _selectedForExport = new Set();
            renderGrid();
            showFeedback(body, `✅ Moved ${toMove.length} prompt${toMove.length > 1 ? "s" : ""} to "${targetCat}".`);
        } else {
            showFeedback(body, "❌ Failed to save. Check console.", "error");
        }
    }

    // Batch remove
    async function doBatchRemove() {
        const allFlat = getFlatPrompts();
        const toRemove = [];
        for (const idx of _selectedForExport) {
            if (allFlat[idx]) {
                toRemove.push(allFlat[idx]);
            }
        }
        if (toRemove.length === 0) {
            showFeedback(body, "No selected prompts to remove.", "error");
            return;
        }
        const names = toRemove.map(p => `"${p.name}"`).join(", ");
        const confirmed = await showConfirmDialog(
            "Remove Selected",
            `Are you sure you want to permanently delete ${toRemove.length} prompt${toRemove.length > 1 ? "s" : ""}?`,
            "Remove",
            "llm-manager-btn-danger"
        );
        if (!confirmed) return;

        // Remove from prompts array
        struct.prompts = struct.prompts.filter(p =>
            !toRemove.some(r => r.name === p.name && r.prompt === p.prompt)
        );
        const result = await savePromptsToBackend({
            categories: struct.categories,
            prompts: struct.prompts,
        });
        if (result && result.success !== false) {
            _selectedForExport = new Set();
            renderGrid();
            showFeedback(body, `✅ Removed ${toRemove.length} prompt${toRemove.length > 1 ? "s" : ""}.`);
        } else {
            showFeedback(body, "❌ Failed to save. Check console.", "error");
        }
    }

    // Export All
    async function doExportAll() {
        const exportData = await exportAllPromptsApi();
        if (!exportData) {
            showFeedback(body, "❌ Failed to export prompts.", "error");
            return;
        }
        const blob = new Blob(
            [JSON.stringify(exportData, null, 2)],
            { type: "application/json" }
        );
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `llm_chat_prompts_export_all.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showFeedback(body, `✅ Exported all ${exportData.prompts.length} prompts with categories.`);
    }

    // Export Selected
    function doExportSelected(allFlat) {
        const selected = [];
        for (const idx of _selectedForExport) {
            if (allFlat[idx]) {
                selected.push({
                    name: allFlat[idx].name,
                    prompt: allFlat[idx].prompt,
                    category: allFlat[idx].category || "Favorites",
                });
            }
        }
        if (selected.length === 0) {
            showFeedback(body, "No prompts selected.", "error");
            return;
        }
        const blob = new Blob(
            [JSON.stringify({ prompts: selected }, null, 2)],
            { type: "application/json" }
        );
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `llm_chat_prompts_export_${selected.length}_prompts.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showFeedback(body, `✅ Exported ${selected.length} prompt${selected.length > 1 ? "s" : ""} successfully!`);
    }

    // ── Assemble ──
    body.appendChild(batchBar);
    body.appendChild(grid);

    // Initial render
    renderGrid();
}

// ────────────────────────────────────────────────────────────────────────
// Card: Create a single prompt card element
// ────────────────────────────────────────────────────────────────────────
// ALL cards get full controls: checkbox, edit, category selector, reorder arrows.
// No _isDefault flags, no "🔒 default" badges, no special system view.

function createCard(p, flatIdx, struct, onRefresh) {
    const card = document.createElement("div");
    card.className = "llm-manager-card";

    // ── Action row ──
    const actions = document.createElement("div");
    actions.className = "llm-manager-card-actions";

    // Checkbox (always visible for all prompts)
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "llm-manager-checkbox";
    checkbox.checked = _selectedForExport.has(flatIdx);
    checkbox.onchange = () => {
        if (checkbox.checked) {
            _selectedForExport.add(flatIdx);
        } else {
            _selectedForExport.delete(flatIdx);
        }
        onRefresh();
    };
    actions.appendChild(checkbox);

    const copyBtn = document.createElement("button");
    copyBtn.className = "llm-manager-btn-icon";
    copyBtn.textContent = "📋";
    copyBtn.title = "Copy prompt text";
    copyBtn.onclick = async () => {
        try {
            await navigator.clipboard.writeText(p.prompt || "");
            showFeedback(
                document.querySelector(".llm-manager-body"),
                "✅ Copied to clipboard!",
                "success"
            );
        } catch (_e) {
            // Fallback
            const ta = document.createElement("textarea");
            ta.value = p.prompt || "";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            showFeedback(
                document.querySelector(".llm-manager-body"),
                "✅ Copied to clipboard!",
                "success"
            );
        }
    };
    actions.appendChild(copyBtn);

    // Edit button (always visible)
    const editBtn = document.createElement("button");
    editBtn.className = "llm-manager-btn-ghost";
    editBtn.textContent = "✏️";
    editBtn.title = "Edit";
    editBtn.onclick = () => {
        const body = document.querySelector(".llm-manager-body");
        const footer = document.querySelector(".llm-manager-footer");
        if (body && footer) {
            showPromptEditor(body, footer, struct, flatIdx, false);
        }
    };
    actions.appendChild(editBtn);

    // No "🔒 default" badge — all prompts are editable

    card.appendChild(actions);

    // ── Title ──
    const title = document.createElement("div");
    title.className = "llm-manager-card-title";
    title.textContent = p.name || "(unnamed)";
    card.appendChild(title);

    // ── Text preview (truncated) ──
    const textEl = document.createElement("div");
    textEl.className = "llm-manager-card-text";
    const fullText = p.prompt || "";
    const snippet = fullText.substring(0, 120).replace(/\n/g, " ");
    textEl.textContent = snippet + (fullText.length > 120 ? "..." : "");
    textEl.title = fullText;
    textEl.onclick = () => {
        textEl.classList.toggle("expanded");
        if (textEl.classList.contains("expanded")) {
            textEl.textContent = fullText;
        } else {
            textEl.textContent = snippet + (fullText.length > 120 ? "..." : "");
        }
    };
    card.appendChild(textEl);

    // ── Footer row: category dropdown + char count + arrows ──
    const cardFooter = document.createElement("div");
    cardFooter.className = "llm-manager-card-footer";

    // Category selector (always visible for all prompts)
    const catSelect = document.createElement("select");
    catSelect.className = "llm-manager-card-cat-select";
    const cats = struct.categories || [LOCKED_CATEGORY];
    for (const cat of cats) {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        if (cat === (p.category || "Favorites")) opt.selected = true;
        catSelect.appendChild(opt);
    }
    catSelect.onchange = async () => {
        p.category = catSelect.value;
        // Ensure category exists in struct
        if (!struct.categories.includes(p.category)) {
            struct.categories.push(p.category);
        }
        const result = await savePromptsToBackend({
            categories: struct.categories,
            prompts: struct.prompts,
        });
        if (result && result.success !== false) {
            onRefresh();
        } else {
            showFeedback(
                document.querySelector(".llm-manager-body"),
                "❌ Failed to save category change.",
                "error"
            );
        }
    };
    cardFooter.appendChild(catSelect);

    // Spacer
    const spacer2 = document.createElement("span");
    spacer2.style.flex = "1";
    cardFooter.appendChild(spacer2);

    // Char count
    const chars = document.createElement("span");
    chars.className = "llm-manager-card-chars";
    chars.textContent = `${fullText.length} chars`;
    cardFooter.appendChild(chars);

    // Reorder arrows (always visible for all prompts)
    const arrLeft = document.createElement("button");
    arrLeft.className = "llm-manager-btn-icon";
    arrLeft.textContent = "◀";
    arrLeft.title = "Move earlier in this category";
    arrLeft.onclick = () => reorderPrompt(p, -1);
    cardFooter.appendChild(arrLeft);

    const arrRight = document.createElement("button");
    arrRight.className = "llm-manager-btn-icon";
    arrRight.textContent = "▶";
    arrRight.title = "Move later in this category";
    arrRight.onclick = () => reorderPrompt(p, 1);
    cardFooter.appendChild(arrRight);

    card.appendChild(cardFooter);

    return card;

    // ── Reorder helper ──
    async function reorderPrompt(targetP, direction) {
        const arr = struct.prompts;
        const idx = arr.findIndex(cp => cp.name === targetP.name && cp.prompt === targetP.prompt);
        if (idx === -1) return;
        const newIdx = idx + direction;
        if (newIdx < 0 || newIdx >= arr.length) return;
        [arr[idx], arr[newIdx]] = [arr[newIdx], arr[idx]];
        const result = await savePromptsToBackend({
            categories: struct.categories,
            prompts: struct.prompts,
        });
        if (result && result.success !== false) {
            onRefresh();
        } else {
            // Revert
            [arr[idx], arr[newIdx]] = [arr[newIdx], arr[idx]];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Category Manager Modal
// ────────────────────────────────────────────────────────────────────────

async function showCategoryManager(body, footer, struct) {
    // Overlay
    const overlay = document.createElement("div");
    overlay.className = "llm-manager-confirm-overlay";

    const panel = document.createElement("div");
    panel.className = "llm-manager-confirm-panel";
    panel.style.width = "400px";
    panel.style.maxHeight = "70vh";
    panel.style.overflow = "auto";

    const titleEl = document.createElement("div");
    titleEl.className = "llm-manager-confirm-title";
    titleEl.textContent = "Manage Categories";
    panel.appendChild(titleEl);

    const listEl = document.createElement("div");
    listEl.style.marginBottom = "12px";

    function renderCatList() {
        listEl.innerHTML = "";
        const cats = struct.categories || [LOCKED_CATEGORY];

        for (let i = 0; i < cats.length; i++) {
            const cat = cats[i];
            const row = document.createElement("div");
            row.className = "llm-manager-cat-row";

            // Reorder buttons
            const upBtn = document.createElement("button");
            upBtn.className = "llm-manager-btn-icon";
            upBtn.textContent = "↑";
            upBtn.disabled = i === 0 || cat === LOCKED_CATEGORY;
            upBtn.onclick = () => moveCat(i, -1);
            row.appendChild(upBtn);

            const downBtn = document.createElement("button");
            downBtn.className = "llm-manager-btn-icon";
            downBtn.textContent = "↓";
            downBtn.disabled = i === cats.length - 1 || cat === LOCKED_CATEGORY;
            downBtn.onclick = () => moveCat(i, 1);
            row.appendChild(downBtn);

            // Name
            const nameSpan = document.createElement("span");
            nameSpan.textContent = cat;
            nameSpan.style.flex = "1";
            nameSpan.style.padding = "0 6px";
            row.appendChild(nameSpan);

            // Rename/Delete buttons (not for LOCKED_CATEGORY)
            if (cat !== LOCKED_CATEGORY) {
                const renameBtn = document.createElement("button");
                renameBtn.className = "llm-manager-btn-ghost";
                renameBtn.textContent = "✏️";
                renameBtn.title = "Rename";
                renameBtn.onclick = async () => {
                    const newName = prompt(`Rename "${cat}" to:`, cat);
                    if (!newName || newName.trim() === cat || !newName.trim()) return;
                    const trimmed = newName.trim();
                    cats[i] = trimmed;
                    // Update prompts with old category name
                    for (const p of struct.prompts) {
                        if (p.category === cat) {
                            p.category = trimmed;
                        }
                    }
                    await saveAndRefresh();
                };
                row.appendChild(renameBtn);

                const delBtn = document.createElement("button");
                delBtn.className = "llm-manager-btn-icon";
                delBtn.textContent = "🗑";
                delBtn.title = `Delete "${cat}" (prompts → Favorites)`;
                delBtn.onclick = async () => {
                    const count = struct.prompts.filter(p => p.category === cat).length;
                    const msg = count > 0
                        ? `Delete "${cat}"? ${count} prompt${count > 1 ? "s" : ""} will be moved to "Favorites".`
                        : `Delete empty category "${cat}"?`;
                    const confirmed = await showConfirmDialog(
                        "Delete Category",
                        msg,
                        "Delete Category",
                        "llm-manager-btn-danger"
                    );
                    if (!confirmed) return;
                    const result = await deleteCategoryApi(cat, "Favorites");
                    if (result && result.success !== false) {
                        struct.categories = result.categories || struct.categories.filter(c => c !== cat);
                        struct.prompts = result.prompts || struct.prompts;
                        renderCatList();
                    }
                };
                row.appendChild(delBtn);
            } else {
                // "System Prompts" badge
                const badge = document.createElement("span");
                badge.className = "llm-manager-default-badge";
                badge.textContent = "🔒 fixed";
                row.appendChild(badge);
            }

            listEl.appendChild(row);
        }
    }

    async function moveCat(idx, direction) {
        const cats = struct.categories;
        // Don't allow moving LOCKED_CATEGORY
        if (cats[idx] === LOCKED_CATEGORY) return;
        const newIdx = idx + direction;
        if (newIdx < 0 || newIdx >= cats.length) return;
        // Don't allow moving past LOCKED_CATEGORY (it should stay last)
        if (cats[newIdx] === LOCKED_CATEGORY) return;
        [cats[idx], cats[newIdx]] = [cats[newIdx], cats[idx]];
        await saveAndRefresh();
    }

    async function saveAndRefresh() {
        const result = await saveCategoriesApi(struct.categories);
        if (result && result.success !== false) {
            struct.categories = result.categories || struct.categories;
            struct.prompts = result.prompts || struct.prompts;
            renderCatList();
        }
    }

    renderCatList();
    panel.appendChild(listEl);

    // Add category button
    const addCatBtn = document.createElement("button");
    addCatBtn.className = "llm-manager-btn llm-manager-btn-primary";
    addCatBtn.textContent = "+ Add Category";
    addCatBtn.style.width = "100%";
    addCatBtn.onclick = async () => {
        const name = prompt("New category name:");
        if (!name || !name.trim()) return;
        const trimmed = name.trim();
        if (struct.categories.includes(trimmed)) {
            alert(`Category "${trimmed}" already exists.`);
            return;
        }
        // Insert before LOCKED_CATEGORY to keep it last
        struct.categories.splice(struct.categories.length - 1, 0, trimmed);
        await saveAndRefresh();
    };
    panel.appendChild(addCatBtn);

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.className = "llm-manager-btn";
    closeBtn.textContent = "Close";
    closeBtn.style.marginTop = "12px";
    closeBtn.style.width = "100%";
    closeBtn.onclick = () => {
        document.body.removeChild(overlay);
    };
    panel.appendChild(closeBtn);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
}

// ────────────────────────────────────────────────────────────────────────
// Management Dialog: Show the prompt editor (add/edit form)
// ────────────────────────────────────────────────────────────────────────

export function showPromptEditor(body, footer, struct, flatIdx, isNew) {
    body.innerHTML = "";
    footer.innerHTML = "";

    // Find the prompt being edited (flat prompts array — no default/custom split)
    let editingPrompt = null;
    if (!isNew && flatIdx !== null) {
        const allFlat = struct.prompts || [];
        editingPrompt = allFlat[flatIdx];
    }

    const backBtn = document.createElement("button");
    backBtn.className = "llm-manager-btn";
    backBtn.textContent = "← Back to list";
    backBtn.onclick = () => renderPromptManager(body, footer);
    body.appendChild(backBtn);

    const title = document.createElement("div");
    title.style.cssText = "font-weight: bold; margin-bottom: 12px; font-size: 15px;";
    title.textContent = isNew ? "Add New Prompt" : "Edit Prompt";
    body.appendChild(title);

    const form = document.createElement("div");
    form.className = "llm-manager-form";

    // Name input
    const nameLabel = document.createElement("div");
    nameLabel.className = "llm-manager-label";
    nameLabel.textContent = "Name:";
    form.appendChild(nameLabel);

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "llm-manager-input";
    nameInput.value = editingPrompt?.name || "";
    nameInput.placeholder = "e.g. My Custom Prompt";
    form.appendChild(nameInput);

    // Category selector
    const catLabel = document.createElement("div");
    catLabel.className = "llm-manager-label";
    catLabel.textContent = "Category:";
    form.appendChild(catLabel);

    const catSelect = document.createElement("select");
    catSelect.className = "llm-manager-select";
    catSelect.style.width = "100%";
    const cats = struct.categories || [LOCKED_CATEGORY];
    for (const cat of cats) {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        if (cat === (editingPrompt?.category || "Favorites")) opt.selected = true;
        catSelect.appendChild(opt);
    }
    form.appendChild(catSelect);

    // Prompt textarea
    const promptLabel = document.createElement("div");
    promptLabel.className = "llm-manager-label";
    promptLabel.textContent = "Prompt:";
    form.appendChild(promptLabel);

    const promptInput = document.createElement("textarea");
    promptInput.className = "llm-manager-textarea";
    promptInput.value = editingPrompt?.prompt || "";
    promptInput.placeholder = "Enter your system prompt text here...";
    form.appendChild(promptInput);

    // Validation warning
    const warning = document.createElement("div");
    warning.className = "llm-manager-warning";
    warning.textContent = "Both name and prompt are required.";
    form.appendChild(warning);

    // Save button
    const saveBtn = document.createElement("button");
    saveBtn.className = "llm-manager-btn llm-manager-btn-primary";
    saveBtn.textContent = "💾 Save";
    saveBtn.style.width = "100%";
    saveBtn.style.padding = "8px";
    saveBtn.style.fontWeight = "bold";
    saveBtn.onclick = async () => {
        const newName = nameInput.value.trim();
        const newPrompt = promptInput.value.trim();
        const newCategory = catSelect.value;

        if (!newName || !newPrompt) {
            warning.style.display = "block";
            return;
        }
        warning.style.display = "none";

        if (isNew) {
            // Check for duplicate name in prompts
            const dup = struct.prompts.find(p => p.name === newName);
            if (dup) {
                const confirmed = await showConfirmDialog(
                    "Overwrite?",
                    `A prompt named "${newName}" already exists. Overwrite it?`,
                    "Overwrite",
                    "llm-manager-btn-primary"
                );
                if (!confirmed) return;
                // Remove the duplicate
                struct.prompts = struct.prompts.filter(p => p.name !== newName);
            }
            // Add new prompt
            struct.prompts.push({
                name: newName,
                prompt: newPrompt,
                category: newCategory,
            });
        } else {
            // Update existing prompt (all prompts are directly editable — no default/custom split)
            if (editingPrompt) {
                const idx = struct.prompts.findIndex(p =>
                    p.name === editingPrompt.name && p.prompt === editingPrompt.prompt
                );
                if (idx >= 0) {
                    struct.prompts[idx] = {
                        name: newName,
                        prompt: newPrompt,
                        category: newCategory,
                    };
                }
            }
        }

        // Ensure category exists in categories list
        if (!struct.categories.includes(newCategory)) {
            struct.categories.push(newCategory);
        }

        const result = await savePromptsToBackend({
            categories: struct.categories,
            prompts: struct.prompts,
        });
        if (result && result.success !== false) {
            renderPromptManager(body, footer);
        } else {
            showFeedback(body, "Failed to save. Check console for details.", "error");
        }
    };
    form.appendChild(saveBtn);

    body.appendChild(form);
}

// ────────────────────────────────────────────────────────────────────────
// Import Dialog: File picker + strategy + preview + import
// ────────────────────────────────────────────────────────────────────────

function showImportDialog(body, footer, struct) {
    body.innerHTML = "";
    footer.innerHTML = "";

    const backBtn = document.createElement("button");
    backBtn.className = "llm-manager-btn";
    backBtn.textContent = "← Back to list";
    backBtn.onclick = () => renderPromptManager(body, footer);
    body.appendChild(backBtn);

    const title = document.createElement("div");
    title.style.cssText = "font-weight: bold; margin-bottom: 12px; font-size: 15px;";
    title.textContent = "📥 Import Prompts from File";
    body.appendChild(title);

    const form = document.createElement("div");
    form.className = "llm-manager-form";

    // File picker
    const fileLabel = document.createElement("div");
    fileLabel.className = "llm-manager-label";
    fileLabel.textContent = "Select JSON file:";
    form.appendChild(fileLabel);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.className = "llm-manager-file-input";
    fileInput.accept = ".json";

    const fileBtn = document.createElement("button");
    fileBtn.className = "llm-manager-btn";
    fileBtn.textContent = "📂 Choose File...";
    fileBtn.onclick = () => fileInput.click();
    form.appendChild(fileBtn);

    const fileNameDisplay = document.createElement("span");
    fileNameDisplay.style.cssText = "color: #888; font-size: 11px; margin-left: 8px;";
    fileNameDisplay.textContent = "No file selected";
    form.appendChild(fileNameDisplay);

    // Strategy selector
    const strategyLabel = document.createElement("div");
    strategyLabel.className = "llm-manager-label";
    strategyLabel.style.marginTop = "8px";
    strategyLabel.textContent = "Import strategy:";
    form.appendChild(strategyLabel);

    const strategySelect = document.createElement("select");
    strategySelect.className = "llm-manager-select";
    strategySelect.style.width = "100%";
    strategySelect.innerHTML = `
        <option value="append">Append — add imported prompts to existing list</option>
        <option value="replace">Replace — overwrite all existing prompts</option>
        <option value="skip_duplicates">Skip duplicates — only add new prompt names</option>
    `;
    form.appendChild(strategySelect);

    // Preview area (hidden until file selected)
    const previewContainer = document.createElement("div");
    previewContainer.style.display = "none";
    form.appendChild(previewContainer);

    // Import button (disabled until file selected)
    const importBtn = document.createElement("button");
    importBtn.className = "llm-manager-btn llm-manager-btn-primary";
    importBtn.textContent = "📥 Import";
    importBtn.style.width = "100%";
    importBtn.style.padding = "8px";
    importBtn.style.marginTop = "8px";
    importBtn.disabled = true;
    importBtn.onclick = async () => {
        const file = fileInput.files[0];
        if (!file) return;

        importBtn.disabled = true;
        importBtn.textContent = "Importing...";

        const strategy = strategySelect.value;
        const result = await importPromptsFromFile(file, strategy);

        if (result && result.success) {
            // Update struct from response
            struct.categories = result.categories || struct.categories;
            struct.prompts = result.prompts || struct.prompts;
            showFeedback(body, `✅ Imported ${result.imported_count} prompt${result.imported_count !== 1 ? "s" : ""}` +
                (result.skipped_count > 0 ? ` (${result.skipped_count} skipped)` : ""), "success");
            // Return to list after short delay
            setTimeout(() => renderPromptManager(body, footer), 1500);
        } else {
            importBtn.disabled = false;
            importBtn.textContent = "📥 Import";
            showFeedback(body, "❌ Import failed. Is the file valid JSON with a 'prompts' array?", "error");
        }
    };
    form.appendChild(importBtn);

    body.appendChild(form);

    // ── Divider ──
    const divider = document.createElement("hr");
    divider.style.cssText = "margin: 20px 0; border: none; border-top: 1px solid #444;";
    body.appendChild(divider);

    const orLabel = document.createElement("div");
    orLabel.style.cssText = "text-align: center; color: #888; font-size: 12px; margin: -10px 0 16px;";
    orLabel.textContent = "── or ──";
    body.appendChild(orLabel);

    // ── Section 2: Import from .txt / .md files ──
    const textForm = document.createElement("div");
    textForm.className = "llm-manager-form";

    const textSectionTitle = document.createElement("div");
    textSectionTitle.style.cssText = "font-weight: bold; margin-bottom: 8px; font-size: 14px;";
    textSectionTitle.textContent = "📂 Import from .txt / .md Files";
    textForm.appendChild(textSectionTitle);

    const textDesc = document.createElement("div");
    textDesc.style.cssText = "color: #888; font-size: 11px; margin-bottom: 12px;";
    textDesc.textContent = "Each file becomes a prompt — filename → name, content → prompt text.";
    textForm.appendChild(textDesc);

    // Category selector
    const catLabel = document.createElement("div");
    catLabel.className = "llm-manager-label";
    catLabel.textContent = "Target category:";
    textForm.appendChild(catLabel);

    const catSelect = document.createElement("select");
    catSelect.className = "llm-manager-select";
    catSelect.style.width = "100%";
    // Populate from existing categories
    const cats = struct.categories || ["Favorites"];
    for (const c of cats) {
        if (c === "System Prompts") continue; // skip locked category
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        catSelect.appendChild(opt);
    }
    textForm.appendChild(catSelect);

    // File picker (multiple .txt/.md)
    const textFileInput = document.createElement("input");
    textFileInput.type = "file";
    textFileInput.className = "llm-manager-file-input";
    textFileInput.accept = ".txt,.md";
    textFileInput.multiple = true;

    const textFileBtn = document.createElement("button");
    textFileBtn.className = "llm-manager-btn";
    textFileBtn.textContent = "📂 Choose .txt / .md Files...";
    textFileBtn.style.marginTop = "8px";
    textFileBtn.onclick = () => textFileInput.click();
    textForm.appendChild(textFileBtn);

    const textFileNameDisplay = document.createElement("span");
    textFileNameDisplay.style.cssText = "color: #888; font-size: 11px; margin-left: 8px;";
    textFileNameDisplay.textContent = "No files selected";
    textForm.appendChild(textFileNameDisplay);

    // Preview area (hidden until files selected)
    const textPreviewContainer = document.createElement("div");
    textPreviewContainer.style.display = "none";
    textPreviewContainer.style.marginTop = "8px";
    textForm.appendChild(textPreviewContainer);

    // Import button (disabled until files selected)
    const textImportBtn = document.createElement("button");
    textImportBtn.className = "llm-manager-btn llm-manager-btn-primary";
    textImportBtn.textContent = "📥 Import as Prompts";
    textImportBtn.style.width = "100%";
    textImportBtn.style.padding = "8px";
    textImportBtn.style.marginTop = "8px";
    textImportBtn.disabled = true;
    textImportBtn.onclick = async () => {
        const files = textFileInput.files;
        if (!files || files.length === 0) return;

        textImportBtn.disabled = true;
        textImportBtn.textContent = "Importing...";

        const category = catSelect.value;
        const fileArray = Array.from(files);
        const result = await importTextFilesAsPrompts(fileArray, category);

        if (result && result.success) {
            struct.categories = result.categories || struct.categories;
            struct.prompts = result.prompts || struct.prompts;
            showFeedback(body, `✅ Imported ${result.imported_count} prompt${result.imported_count !== 1 ? "s" : ""} into "${category}"` +
                (result.skipped_count > 0 ? ` (${result.skipped_count} skipped)` : ""), "success");
            setTimeout(() => renderPromptManager(body, footer), 1500);
        } else {
            textImportBtn.disabled = false;
            textImportBtn.textContent = "📥 Import as Prompts";
            showFeedback(body, "❌ Import failed. Check the console for details.", "error");
        }
    };
    textForm.appendChild(textImportBtn);

    body.appendChild(textForm);

    // File selected handler for text files
    textFileInput.addEventListener("change", () => {
        const files = textFileInput.files;
        if (!files || files.length === 0) {
            textFileNameDisplay.textContent = "No files selected";
            textImportBtn.disabled = true;
            textPreviewContainer.style.display = "none";
            return;
        }

        textFileNameDisplay.textContent = `📄 ${files.length} file${files.length !== 1 ? "s" : ""} selected`;
        textImportBtn.disabled = false;

        // Build preview
        textPreviewContainer.style.display = "block";
        let previewHtml = '<div class="llm-manager-label" style="margin-top:8px;">Preview:</div>';
        previewHtml += '<div class="llm-manager-import-preview">';

        const maxPreview = 20;
        const shown = Math.min(files.length, maxPreview);
        for (let i = 0; i < shown; i++) {
            const f = files[i];
            const name = f.name.replace(/\.[^.]+$/, "");
            // Show names first; content snippets fill in asynchronously below
            previewHtml += `• <strong>${name.replace(/</g, "<")}</strong> — <span id="txt-preview-${i}">(reading...)</span><br>`;
        }
        if (files.length > maxPreview) {
            previewHtml += `<br>... and ${files.length - maxPreview} more`;
        }
        previewHtml += "</div>";
        textPreviewContainer.innerHTML = previewHtml;

        // Read each file's first chunk for the preview
        for (let i = 0; i < shown; i++) {
            const f = files[i];
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target.result;
                const snippet = (content || "").substring(0, 60).replace(/</g, "<");
                const span = document.getElementById(`txt-preview-${i}`);
                if (span) span.textContent = snippet + (content.length > 60 ? "..." : "");
            };
            // Read just first 200 chars for speed
            reader.readAsText(f.slice(0, 200));
        }
    });

    // File selected handler
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (!file) {
            fileNameDisplay.textContent = "No file selected";
            importBtn.disabled = true;
            previewContainer.style.display = "none";
            return;
        }

        fileNameDisplay.textContent = `📄 ${file.name}`;
        importBtn.disabled = false;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                const imported = Array.isArray(data) ? data : (data.prompts || []);
                const validCount = imported.filter(p => p && p.name && p.prompt).length;

                // Detect unique categories in imported data
                const uniqueCats = new Set();
                for (const p of imported) {
                    if (p.category) uniqueCats.add(p.category);
                }
                const catInfo = uniqueCats.size > 0
                    ? `<br>Categories detected: ${[...uniqueCats].join(", ")}`
                    : "<br>No category data — prompts will default to 'Favorites'";

                previewContainer.style.display = "block";
                previewContainer.innerHTML = `
                    <div class="llm-manager-label" style="margin-top:8px;">Preview:</div>
                    <div class="llm-manager-import-preview">
                        Found <strong>${imported.length}</strong> prompt${imported.length !== 1 ? "s" : ""} in file
                        (${validCount} valid)${catInfo}
                        <br><br>
                        ${imported.slice(0, 10).map(p =>
                            `• <strong>${(p.name || "?").replace(/</g, "<")}</strong>` +
                            (p.category ? ` [${p.category.replace(/</g, "<")}]` : "") +
                            ` — ${(p.prompt || "").substring(0, 50).replace(/</g, "<")}...`
                        ).join("<br>")}
                        ${imported.length > 10 ? `<br>... and ${imported.length - 10} more` : ""}
                    </div>
                `;
            } catch (_e) {
                previewContainer.style.display = "block";
                previewContainer.innerHTML = `<div class="llm-manager-feedback llm-manager-feedback-error">Invalid JSON file</div>`;
                importBtn.disabled = true;
            }
        };
        reader.readAsText(file);
    });
}
