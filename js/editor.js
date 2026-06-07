/**
 * EasyLLM — Prompt management dialog (add/edit/delete/import/export system prompts)
 *
 * Self-contained modal UI for managing system prompt templates.
 * Uses CSS classes (.llm-manager-*) defined in llm_chat.css and constants.js.
 */

import { fetchPrompts, savePromptsToBackend, refreshAllTemplateWidgets, importPromptsFromFile } from "./api.js";
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
        "⚙ Manage System Prompts",
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

function showConfirmDialog(title, message) {
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
        okBtn.className = "llm-manager-btn llm-manager-btn-danger";
        okBtn.textContent = "Delete";
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
// Management Dialog: Render the prompt list (with search, reorder, export)
// ────────────────────────────────────────────────────────────────────────

let _selectedForExport = new Set(); // tracks which prompt indices are checked for export

export async function renderPromptManager(body, footer) {
    const prompts = await fetchPrompts();
    body.innerHTML = "";
    footer.innerHTML = "";
    _selectedForExport = new Set();

    // ── Toolbar: search + action buttons ──
    const toolbar = document.createElement("div");
    toolbar.className = "llm-manager-toolbar";

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "llm-manager-search";
    searchInput.placeholder = "🔍 Search prompts by name...";
    toolbar.appendChild(searchInput);

    const importBtn = document.createElement("button");
    importBtn.className = "llm-manager-btn";
    importBtn.textContent = "📥 Import...";
    importBtn.title = "Import prompts from a JSON file";
    importBtn.onclick = () => showImportDialog(body, footer, prompts);
    toolbar.appendChild(importBtn);

    const exportBtn = document.createElement("button");
    exportBtn.className = "llm-manager-btn";
    exportBtn.textContent = "📤 Export Selected";
    exportBtn.title = "Download checked prompts as JSON file";
    exportBtn.onclick = () => exportSelectedPrompts(prompts);
    toolbar.appendChild(exportBtn);

    body.appendChild(toolbar);

    // ── Add New button (full width below toolbar) ──
    const addBtn = document.createElement("button");
    addBtn.className = "llm-manager-btn llm-manager-btn-primary";
    addBtn.textContent = "+ Add New Prompt";
    addBtn.style.width = "100%";
    addBtn.style.marginBottom = "4px";
    addBtn.onclick = () => {
        const newPrompts = [...prompts, { name: "", prompt: "" }];
        showPromptEditor(body, footer, newPrompts, newPrompts.length - 1, true);
    };
    body.appendChild(addBtn);

    // ── List container (scrollable) ──
    const listContainer = document.createElement("div");
    listContainer.className = "llm-manager-list";

    function renderList(filterText) {
        listContainer.innerHTML = "";
        const lowerFilter = (filterText || "").toLowerCase();

        const filtered = prompts.filter((p, _i) =>
            !lowerFilter || (p.name || "").toLowerCase().includes(lowerFilter)
        );

        if (filtered.length === 0) {
            const empty = document.createElement("div");
            empty.className = "llm-manager-empty";
            empty.textContent = filterText
                ? "No prompts match your search."
                : "No prompts yet. Click '+ Add New Prompt' to create one.";
            listContainer.appendChild(empty);
            return;
        }

        // Toggle all checkbox state
        const allSelected = filtered.every((_p, i) => {
            const origIdx = prompts.indexOf(filtered[i]);
            return _selectedForExport.has(origIdx);
        });

        // Select all / deselect all header row
        const selectAllRow = document.createElement("div");
        selectAllRow.style.cssText = "display:flex;align-items:center;gap:6px;padding:4px 10px;flex-shrink:0;";
        const selectAllCheckbox = document.createElement("input");
        selectAllCheckbox.type = "checkbox";
        selectAllCheckbox.className = "llm-manager-checkbox";
        selectAllCheckbox.checked = allSelected;
        selectAllCheckbox.title = allSelected ? "Deselect all" : "Select all";
        selectAllCheckbox.onchange = () => {
            const checked = selectAllCheckbox.checked;
            for (const p of filtered) {
                const origIdx = prompts.indexOf(p);
                if (checked) {
                    _selectedForExport.add(origIdx);
                } else {
                    _selectedForExport.delete(origIdx);
                }
            }
            renderList(searchInput.value);
        };
        selectAllRow.appendChild(selectAllCheckbox);
        const selectAllLabel = document.createElement("span");
        selectAllLabel.style.cssText = "color:#888;font-size:11px;";
        selectAllLabel.textContent = allSelected ? "Deselect all" : "Select all";
        selectAllRow.appendChild(selectAllLabel);
        listContainer.appendChild(selectAllRow);

        // Prompt rows
        for (const p of filtered) {
            const origIdx = prompts.indexOf(p);
            const index = prompts.indexOf(p);

            const row = document.createElement("div");
            row.className = "llm-manager-prompt-row";

            // Checkbox for export
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.className = "llm-manager-checkbox";
            checkbox.checked = _selectedForExport.has(origIdx);
            checkbox.onchange = () => {
                if (checkbox.checked) {
                    _selectedForExport.add(origIdx);
                } else {
                    _selectedForExport.delete(origIdx);
                }
                renderList(searchInput.value);
            };
            row.appendChild(checkbox);

            // Name
            const nameSpan = document.createElement("span");
            nameSpan.className = "llm-manager-prompt-name";
            nameSpan.textContent = p.name || "(unnamed)";
            row.appendChild(nameSpan);

            // Preview (clickable to expand/collapse)
            const preview = document.createElement("span");
            preview.className = "llm-manager-prompt-preview";
            const fullText = p.prompt || "";
            const snippet = fullText.substring(0, 80).replace(/\n/g, " ");
            preview.textContent = snippet + (fullText.length > 80 ? "..." : "");
            preview._expanded = false;
            preview._fullText = fullText;
            preview.addEventListener("click", (e) => {
                e.stopPropagation();
                preview._expanded = !preview._expanded;
                if (preview._expanded) {
                    preview.textContent = preview._fullText;
                    preview.classList.add("expanded");
                } else {
                    const s = preview._fullText.substring(0, 80).replace(/\n/g, " ");
                    preview.textContent = s + (preview._fullText.length > 80 ? "..." : "");
                    preview.classList.remove("expanded");
                }
            });
            row.appendChild(preview);

            // Character count badge
            const badge = document.createElement("span");
            badge.className = "llm-manager-badge";
            badge.textContent = `${fullText.length} chars`;
            row.appendChild(badge);

            // Reorder + edit + delete buttons
            const btnGroup = document.createElement("span");
            btnGroup.className = "llm-manager-row-btns";

            // Move up
            const upBtn = document.createElement("button");
            upBtn.className = "llm-manager-btn-icon";
            upBtn.textContent = "↑";
            upBtn.title = "Move up";
            upBtn.disabled = index === 0;
            upBtn.onclick = async (e) => {
                e.stopPropagation();
                if (index <= 0) return;
                [prompts[index - 1], prompts[index]] = [prompts[index], prompts[index - 1]];
                const ok = await savePromptsToBackend(prompts);
                if (ok) {
                    refreshAllTemplateWidgets(prompts);
                    renderList(searchInput.value);
                }
            };
            btnGroup.appendChild(upBtn);

            // Move down
            const downBtn = document.createElement("button");
            downBtn.className = "llm-manager-btn-icon";
            downBtn.textContent = "↓";
            downBtn.title = "Move down";
            downBtn.disabled = index === prompts.length - 1;
            downBtn.onclick = async (e) => {
                e.stopPropagation();
                if (index >= prompts.length - 1) return;
                [prompts[index], prompts[index + 1]] = [prompts[index + 1], prompts[index]];
                const ok = await savePromptsToBackend(prompts);
                if (ok) {
                    refreshAllTemplateWidgets(prompts);
                    renderList(searchInput.value);
                }
            };
            btnGroup.appendChild(downBtn);

            // Edit
            const editBtn = document.createElement("button");
            editBtn.className = "llm-manager-btn-ghost";
            editBtn.textContent = "✎";
            editBtn.title = "Edit";
            editBtn.onclick = () => showPromptEditor(body, footer, prompts, index, false);
            btnGroup.appendChild(editBtn);

            // Delete
            const delBtn = document.createElement("button");
            delBtn.className = "llm-manager-btn llm-manager-btn-danger llm-manager-btn-small";
            delBtn.textContent = "🗑";
            delBtn.title = "Delete";
            delBtn.onclick = async () => {
                const name = p.name || "(unnamed)";
                const confirmed = await showConfirmDialog("Delete Prompt", `Are you sure you want to delete "${name}"?`);
                if (!confirmed) return;
                prompts.splice(index, 1);
                const ok = await savePromptsToBackend(prompts);
                if (ok) {
                    refreshAllTemplateWidgets(prompts);
                    renderList(searchInput.value);
                } else {
                    showFeedback(body, "Failed to save", "error");
                }
            };
            btnGroup.appendChild(delBtn);

            row.appendChild(btnGroup);
            listContainer.appendChild(row);
        }

        // Count in footer
        footer.innerHTML = "";
        const countLabel = document.createElement("span");
        countLabel.className = "llm-manager-count";
        const selectedCount = _selectedForExport.size;
        countLabel.textContent = `${filtered.length} shown · ${prompts.length} total` +
            (selectedCount > 0 ? ` · ${selectedCount} selected for export` : "");
        footer.appendChild(countLabel);
    }

    body.appendChild(listContainer);

    // Search handler
    searchInput.addEventListener("input", () => {
        renderList(searchInput.value);
    });

    // Initial render
    renderList("");
}

// ────────────────────────────────────────────────────────────────────────
// Export: Download selected prompts as JSON file
// ────────────────────────────────────────────────────────────────────────

function exportSelectedPrompts(allPrompts) {
    const selected = [];
    for (const idx of _selectedForExport) {
        if (allPrompts[idx]) {
            selected.push(allPrompts[idx]);
        }
    }

    if (selected.length === 0) {
        // Show feedback on the body
        const body = document.querySelector(".llm-manager-body");
        if (body) showFeedback(body, "No prompts selected. Check the boxes next to prompts you want to export.", "error");
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

    const body = document.querySelector(".llm-manager-body");
    if (body) showFeedback(body, `✅ Exported ${selected.length} prompt${selected.length > 1 ? "s" : ""} successfully!`);
}

// ────────────────────────────────────────────────────────────────────────
// Management Dialog: Show the prompt editor (add/edit form)
// ────────────────────────────────────────────────────────────────────────

export function showPromptEditor(body, footer, prompts, index, isNew) {
    body.innerHTML = "";
    footer.innerHTML = "";

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
    nameInput.value = prompts[index]?.name || "";
    nameInput.placeholder = "e.g. My Custom Prompt";
    form.appendChild(nameInput);

    // Prompt textarea
    const promptLabel = document.createElement("div");
    promptLabel.className = "llm-manager-label";
    promptLabel.textContent = "Prompt:";
    form.appendChild(promptLabel);

    const promptInput = document.createElement("textarea");
    promptInput.className = "llm-manager-textarea";
    promptInput.value = prompts[index]?.prompt || "";
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

        if (!newName || !newPrompt) {
            warning.style.display = "block";
            return;
        }
        warning.style.display = "none";

        const dupIndex = prompts.findIndex((p, i) => i !== index && p.name === newName);
        if (dupIndex >= 0) {
            const confirmed = await showConfirmDialog(
                "Overwrite?",
                `A prompt named "${newName}" already exists. Overwrite it?`
            );
            if (!confirmed) return;
            prompts.splice(dupIndex, 1);
            if (dupIndex < index) index--;
        }

        prompts[index] = { name: newName, prompt: newPrompt };

        const ok = await savePromptsToBackend(prompts);
        if (ok) {
            refreshAllTemplateWidgets(prompts);
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

function showImportDialog(body, footer, currentPrompts) {
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
            refreshAllTemplateWidgets(result.prompts);
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

                previewContainer.style.display = "block";
                previewContainer.innerHTML = `
                    <div class="llm-manager-label" style="margin-top:8px;">Preview:</div>
                    <div class="llm-manager-import-preview">
                        Found <strong>${imported.length}</strong> prompt${imported.length !== 1 ? "s" : ""} in file
                        (${validCount} valid)
                        <br><br>
                        ${imported.slice(0, 10).map(p =>
                            `• <strong>${(p.name || "?").replace(/</g, "<")}</strong> — ` +
                            `${(p.prompt || "").substring(0, 50).replace(/</g, "<")}...`
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
