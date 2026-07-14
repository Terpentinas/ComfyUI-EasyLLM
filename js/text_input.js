/**
 * 📝 EasyLLM Text Input — Frontend Extension
 *
 * Handles dimming/enabling the text widget when prompt_selector changes.
 *
 * When a named prompt is selected from the dropdown, the text field is
 * dimmed (opacity 0.4) and disabled so users know the typed content
 * won't be used.  When "Custom" is selected, the text field is fully
 * enabled and interactive.
 */

import { app } from "../../../scripts/app.js";

const NODE_NAME = "LLM_TextInput";

app.registerExtension({
    name: "Comfy.EasyLLM.TextInput",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const selW = this.widgets?.find(w => w.name === "prompt_selector");
            const textW = this.widgets?.find(w => w.name === "text");

            if (!selW || !textW) return result;

            // ── Dim/enable text widget when selector changes ──
            const updateDim = (val) => {
                const isCustom = val === "Custom";
                // ComfyUI widget disabled state
                textW.disabled = !isCustom;
                // DOM element disabled attribute
                const el = textW.element || textW.inputEl;
                if (el) {
                    el.disabled = !isCustom;
                    el.style.opacity = isCustom ? "1" : "0.4";
                    el.style.pointerEvents = isCustom ? "auto" : "none";
                }
                // Also update the canvas textarea if it exists
                if (textW.widgetEl) {
                    textW.widgetEl.style.opacity = isCustom ? "1" : "0.4";
                }
            };

            // Set initial state
            updateDim(selW.value);

            // Wire callback
            const prevCb = selW.callback;
            selW.callback = function (val) {
                if (prevCb) prevCb.call(this, val);
                updateDim(val);
            };

            return result;
        };
    },
});
