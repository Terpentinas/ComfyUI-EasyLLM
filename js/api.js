/**
 * EasyLLM — API layer: load/save prompts and refresh template widgets
 */

import { app } from "../../../scripts/app.js";

// ────────────────────────────────────────────────────────────────────────
// API: Load prompts from the backend
// ────────────────────────────────────────────────────────────────────────

export async function fetchPrompts() {
    try {
        const resp = await fetch("/easyllm/prompts/load");
        if (!resp.ok) {
            const text = await resp.text();
            console.error(
                `[LLM Chat] Load prompts HTTP ${resp.status}:`,
                (text || "(empty)").slice(0, 300)
            );
            return [];
        }
        const data = await resp.json();
        return data.success ? data.prompts : [];
    } catch (e) {
        console.error("[LLM Chat] Failed to load prompts:", e);
        return [];
    }
}

// ────────────────────────────────────────────────────────────────────────
// API: Save prompts to the backend
// ────────────────────────────────────────────────────────────────────────

export async function savePromptsToBackend(prompts) {
    try {
        const resp = await fetch("/easyllm/prompts/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompts }),
        });
        if (!resp.ok) {
            const text = await resp.text();
            console.error(
                `[LLM Chat] Save prompts HTTP ${resp.status}:`,
                (text || "(empty)").slice(0, 300)
            );
            return false;
        }
        const data = await resp.json();
        return data.success;
    } catch (e) {
        console.error("[LLM Chat] Failed to save prompts:", e);
        return false;
    }
}

// ────────────────────────────────────────────────────────────────────────
// Global: Refresh prompt_template dropdowns on ALL nodes in the graph
// ────────────────────────────────────────────────────────────────────────

export function refreshAllTemplateWidgets(prompts) {
    const names = ["Custom", ...prompts.map(p => p.name)];

    if (!app.graph) return;
    for (const node of app.graph._nodes) {
        const tw = node.widgets?.find(w => w.name === "prompt_template");
        if (tw) {
            const currentVal = tw.value;
            tw.options.values = names;
            if (!names.includes(currentVal)) {
                tw.value = "Custom";
            }
        }
    }
    app.graph.setDirtyCanvas(true, false);
}

// ────────────────────────────────────────────────────────────────────────
// API: Import prompts from a file (read + parse + POST to backend)
// ────────────────────────────────────────────────────────────────────────

/**
 * Import prompts from a JSON file.
 *
 * Reads the file, validates the format ({ prompts: [...] } or [...]),
 * then POSTs to the backend with the chosen merge strategy.
 *
 * @param {File} file - The .json file selected by the user
 * @param {"append"|"replace"|"skip_duplicates"} strategy - Merge strategy
 * @returns {Object|null} { success, prompts, imported_count, skipped_count } or null on failure
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

        // Support both { prompts: [...] } and [...] array formats
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
        return result;
    } catch (e) {
        console.error("[LLM Chat] Failed to import prompts:", e);
        return null;
    }
}
