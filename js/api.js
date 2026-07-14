/**
 * EasyLLM — API layer: load/save prompts
 *
 * v3: flat prompts model
 *   fetchPrompts() returns { categories, prompts }
 */

import { app } from "../../../scripts/app.js";

// ────────────────────────────────────────────────────────────────────────
// Event helper: notify listeners when the prompt library changes
// ────────────────────────────────────────────────────────────────────────

/**
 * Dispatch a custom event to notify listeners that the prompt library
 * has changed (save, import, delete, category rename, etc.).
 * The system prompt dropdown in the chat popup listens for this event
 * to refresh its options.
 */
function notifyPromptsUpdated() {
    document.dispatchEvent(new CustomEvent("llm-prompts-updated"));
}

// ────────────────────────────────────────────────────────────────────────
// API: Load prompts from the backend (v3 structure)
// ────────────────────────────────────────────────────────────────────────

/**
 * Fetch the full prompt structure from the backend.
 *
 * @returns {Object} { categories: string[], prompts: array }
 *                    Returns empty structure on failure.
 */
export async function fetchPrompts() {
    try {
        const resp = await fetch("/easyllm/prompts/load");
        if (!resp.ok) {
            const text = await resp.text();
            console.error(
                `[LLM Chat] Load prompts HTTP ${resp.status}:`,
                (text || "(empty)").slice(0, 300)
            );
            return { categories: ["System Prompts"], prompts: [] };
        }
        const data = await resp.json();
        if (data.success) {
            return {
                categories: data.categories || ["System Prompts"],
                prompts: data.prompts || [],
            };
        }
        return { categories: ["System Prompts"], prompts: [] };
    } catch (e) {
        console.error("[LLM Chat] Failed to load prompts:", e);
        return { categories: ["System Prompts"], prompts: [] };
    }
}

// ────────────────────────────────────────────────────────────────────────
// API: Save prompts to the backend (v3 structure)
// ────────────────────────────────────────────────────────────────────────

/**
 * Save the full v3 prompt structure to the backend.
 *
 * @param {Object} struct - { categories, prompts }
 * @returns {Object|null} The parsed response object, or null on failure
 */
export async function savePromptsToBackend(struct) {
    try {
        const resp = await fetch("/easyllm/prompts/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(struct),
        });
        if (!resp.ok) {
            const text = await resp.text();
            console.error(
                `[LLM Chat] Save prompts HTTP ${resp.status}:`,
                (text || "(empty)").slice(0, 300)
            );
            return null;
        }
        const result = await resp.json();
        if (result && result.success !== false) {
            notifyPromptsUpdated();
        }
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to save prompts:", e);
        return null;
    }
}

// ────────────────────────────────────────────────────────────────────────
// API: Import prompts from a file (read + parse + POST to backend)
// ────────────────────────────────────────────────────────────────────────

/**
 * Import prompts from a JSON file.
 *
 * Reads the file, validates the format, then POSTs to the backend
 * with the chosen merge strategy.
 *
 * Supports formats:
 *   - { prompts: [...] }   (standard, category optional)
 *   - [...]                 (flat array)
 *   - { categories: [...], prompts: [...] }  (full export format)
 *
 * @param {File} file - The .json file selected by the user
 * @param {"append"|"replace"|"skip_duplicates"} strategy - Merge strategy
 * @returns {Object|null} Response with { success, categories, prompts, imported_count, skipped_count }
 */
export async function importPromptsFromFile(file, strategy = "append") {
    try {
        const text = await file.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (_e) {
            console.error("[LLM Chat] Import: invalid JSON file");
            return null;
        }

        // Support both { prompts: [...] }, [...] array, and { categories: [...], prompts: [...] } formats
        let importedPrompts;
        if (Array.isArray(data)) {
            importedPrompts = data;
        } else if (data && Array.isArray(data.prompts)) {
            importedPrompts = data.prompts;
        } else {
            console.error("[LLM Chat] Import: file must contain a prompts array or be a flat array");
            return null;
        }

        // Basic validation — each entry must have a name and prompt
        const valid = importedPrompts.filter(p => p && p.name && p.prompt);
        if (valid.length === 0) {
            console.error("[LLM Chat] Import: no valid prompts found in file");
            return null;
        }

        const resp = await fetch("/easyllm/prompts/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompts: valid, strategy }),
        });

        if (!resp.ok) {
            const errText = await resp.text();
            console.error(
                `[LLM Chat] Import HTTP ${resp.status}:`,
                (errText || "(empty)").slice(0, 300)
            );
            return null;
        }

        const result = await resp.json();
        if (result && result.success !== false) {
            notifyPromptsUpdated();
        }
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to import prompts:", e);
        return null;
    }
}

/**
 * Import multiple .txt / .md files as prompts in a chosen category.
 *
 * Reads each file's text content client-side, strips the file extension
 * to derive the prompt name, then POSTs to the existing
 * /easyllm/prompts/import endpoint with strategy="append".
 *
 * @param {File[]} files - Selected .txt / .md File objects
 * @param {string} category - Target category (default: "Favorites")
 * @returns {Object|null} { success, categories, prompts, imported_count, skipped_count }
 */
export async function importTextFilesAsPrompts(files, category = "Favorites") {
    try {
        const prompts = [];
        for (const file of files) {
            const text = await file.text();
            // Strip last extension to derive prompt name from filename
            const name = file.name.replace(/\.[^.]+$/, "");
            prompts.push({ name, prompt: text, category });
        }

        if (prompts.length === 0) {
            console.error("[LLM Chat] Import files: no files provided");
            return null;
        }

        const resp = await fetch("/easyllm/prompts/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompts, strategy: "append" }),
        });

        if (!resp.ok) {
            const errText = await resp.text();
            console.error(
                `[LLM Chat] Import files HTTP ${resp.status}:`,
                (errText || "(empty)").slice(0, 300)
            );
            return null;
        }

        const result = await resp.json();
        if (result && result.success !== false) {
            notifyPromptsUpdated();
        }
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to import text files:", e);
        return null;
    }
}

// ────────────────────────────────────────────────────────────────────────
// API: Category CRUD
// ────────────────────────────────────────────────────────────────────────

/**
 * Save updated categories list (after add/rename/reorder/delete).
 *
 * @param {string[]} categories - Ordered list of category names
 * @returns {Object|null} Response with { success, categories, prompts }
 */
export async function saveCategories(categories) {
    try {
        const resp = await fetch("/easyllm/prompts/categories/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ categories }),
        });
        if (!resp.ok) {
            const text = await resp.text();
            console.error(`[LLM Chat] Save categories HTTP ${resp.status}:`, (text || "").slice(0, 300));
            return null;
        }
        const result = await resp.json();
        if (result && result.success !== false) {
            notifyPromptsUpdated();
        }
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to save categories:", e);
        return null;
    }
}

/**
 * Delete a category, reassigning its prompts to another category.
 *
 * @param {string} category - The category name to delete
 * @param {string} reassignTo - Target category for reassignment (default: "Favorites")
 * @returns {Object|null} Response with { success, categories, prompts }
 */
export async function deleteCategory(category, reassignTo = "Favorites") {
    try {
        const resp = await fetch("/easyllm/prompts/categories/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ category, reassign_to: reassignTo }),
        });
        if (!resp.ok) {
            const text = await resp.text();
            console.error(`[LLM Chat] Delete category HTTP ${resp.status}:`, (text || "").slice(0, 300));
            return null;
        }
        const result = await resp.json();
        if (result && result.success !== false) {
            notifyPromptsUpdated();
        }
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to delete category:", e);
        return null;
    }
}

// ────────────────────────────────────────────────────────────────────────
// API: Export all prompts
// ────────────────────────────────────────────────────────────────────────

/**
 * Export ALL prompts with categories as a downloadable blob.
 * Returns the export data directly (client can then trigger a download).
 *
 * @returns {Object|null} { categories: string[], prompts: array } or null on failure
 */
export async function exportAllPrompts() {
    try {
        const resp = await fetch("/easyllm/prompts/export");
        if (!resp.ok) {
            const text = await resp.text();
            console.error(`[LLM Chat] Export prompts HTTP ${resp.status}:`, (text || "").slice(0, 300));
            return null;
        }
        const data = await resp.json();
        if (data.success && data.export) {
            return data.export;
        }
        return null;
    } catch (e) {
        console.error("[LLM Chat] Failed to export prompts:", e);
        return null;
    }
}
