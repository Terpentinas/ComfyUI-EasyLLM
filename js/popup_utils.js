/**
 * EasyLLM — Pure utility functions extracted from popup.js
 *
 * Contains: markdown rendering, auto-scroll, timestamp/timing formatting,
 * think-tag parsing, token estimation, text extraction, error detection.
 * All functions are self-contained with no project-internal imports.
 */

// ────────────────────────────────────────────────────────────────────────
// Popup: Extract generated text from execution message
// ────────────────────────────────────────────────────────────────────────

export function extractGeneratedText(message) {
    if (!message) return "";

    if (typeof message === "string") return message;

    if (Array.isArray(message.text)) {
        return message.text[0] || "";
    }
    if (typeof message.text === "string") {
        return message.text;
    }
    if (message.text && typeof message.text === "object") {
        const vals = Object.values(message.text);
        if (vals.length > 0) {
            const v = vals[0];
            if (typeof v === "string") return v;
            if (Array.isArray(v)) return v[0] || "";
        }
    }

    if (message.output?.text) {
        return Array.isArray(message.output.text)
            ? message.output.text[0]
            : message.output.text;
    }

    for (const [key, val] of Object.entries(message)) {
        if (key === "clip") continue;
        if (typeof val === "string") return val;
        if (Array.isArray(val) && val.length > 0 && typeof val[0] === "string") return val[0];
    }

    return "";
}

// ────────────────────────────────────────────────────────────────────────
// Popup: Extract raw (uncleaned) generated text from execution message
// ────────────────────────────────────────────────────────────────────────

/**
 * Detect whether generated text represents an error condition.
 * Checks for error keywords, "Error during generation" prefix, and "ERROR:" prefix.
 */
export function detectError(text) {
    if (!text) return true; // Empty response = error
    const trimmed = text.trim();
    // Python backend wraps errors: "Error during generation: ..."
    if (/^Error during generation:/i.test(trimmed)) return true;
    // Catch "ERROR:" prefixed messages
    if (/^ERROR:/i.test(trimmed)) return true;
    // Heuristic: short text with error keywords
    if (trimmed.length < 200) {
        const errorPatterns = ["error", "exception", "traceback", "failed", "unable"];
        return errorPatterns.some(p => trimmed.toLowerCase().includes(p));
    }
    return false;
}

/**
 * Format an error message for display.
 * Wraps raw Python error text into a user-friendly format.
 */
export function formatErrorMessage(rawText) {
    if (!rawText) return "⚠️ Generation failed: Unknown error";
    const trimmed = rawText.trim();
    if (/^Error during generation:/i.test(trimmed)) {
        const reason = trimmed.replace(/^Error during generation:\s*/i, "");
        return `⚠️ Generation failed: ${reason}`;
    }
    // Already user-friendly — return as-is
    if (/^ERROR:/i.test(trimmed)) {
        return trimmed;
    }
    return `⚠️ Generation failed: ${trimmed}`;
}

export function extractRawText(message) {
    if (!message) return "";

    // raw_text is sent from Python in the ui dict alongside cleaned text
    if (message.raw_text) {
        return Array.isArray(message.raw_text) ? message.raw_text[0] : message.raw_text;
    }

    // Fallback to cleaned text if raw is not available
    return extractGeneratedText(message);
}

// ────────────────────────────────────────────────────────────────────────
// Markdown: Minimal inline markdown-to-HTML renderer
// ────────────────────────────────────────────────────────────────────────

/**
 * Minimal markdown to HTML renderer.
 * Handles: **bold**, *italic*, `inline code`, ```code blocks```, links.
 * Escapes HTML entities first for safety.
 */
export function renderMarkdown(text) {
    if (!text) return "";

    // Escape HTML entities first (safety)
    let html = text
        .replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">");

    // Code blocks (must come before inline code)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
        '<pre><code class="language-$1">$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Convert newlines to <br>
    html = html.replace(/\n/g, '<br>');

    return html;
}

// ────────────────────────────────────────────────────────────────────────
// Auto-Scroll: Track whether user has scrolled up
// ────────────────────────────────────────────────────────────────────────

const SCROLL_THRESHOLD = 40;

/**
 * Check if the user has scrolled up away from the bottom.
 * Attach this as a scroll event listener on the history container.
 */
export function updateScrollState(container) {
    if (!container) return;
    const atBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - SCROLL_THRESHOLD;
    container._isUserScrolledUp = !atBottom;
    // Show/hide the scroll-to-bottom button using CSS class
    const btn = container._scrollToBottomBtn;
    if (btn) {
        btn.classList.toggle("llm-scroll-bottom-visible", !atBottom);
    }
}

/**
 * Scroll to bottom of the history container, resetting scroll state.
 */
export function scrollToBottom(container) {
    if (!container) return;
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    container._isUserScrolledUp = false;
    container._pendingNewMsgCount = 0;
    const btn = container._scrollToBottomBtn;
    if (btn) {
        btn.classList.remove("llm-scroll-bottom-visible");
        const badge = btn.querySelector(".llm-chat-scroll-badge");
        if (badge) badge.textContent = "";
    }
}

/**
 * Auto-scroll the container to bottom only if user hasn't scrolled up.
 */
export function autoScrollIfNeeded(container) {
    if (!container) return;
    if (!container._isUserScrolledUp) {
        container.scrollTop = container.scrollHeight;
    } else {
        // Increment pending new message count for badge
        container._pendingNewMsgCount = (container._pendingNewMsgCount || 0) + 1;
        const btn = container._scrollToBottomBtn;
        if (btn) {
            const badge = btn.querySelector(".llm-chat-scroll-badge");
            if (badge) badge.textContent = String(container._pendingNewMsgCount);
        }
    }
}

/**
 * Create a floating scroll-to-bottom button for a history container.
 */
export function createScrollToBottomBtn(container) {
    // Don't recreate orphaned refs; container.innerHTML = '' detaches them
    if (container._scrollToBottomBtn?.isConnected) return container._scrollToBottomBtn;
    const btn = document.createElement("button");
    btn.className = "llm-chat-scroll-bottom-btn";
    btn.innerHTML = '<span class="llm-chat-scroll-badge"></span>⬇';
    btn.title = "Scroll to latest message";
    btn.onclick = () => scrollToBottom(container);
    container.appendChild(btn);
    container._scrollToBottomBtn = btn;
    container._pendingNewMsgCount = 0;
    return btn;
}

// ────────────────────────────────────────────────────────────────────────
// Timestamp formatting
// ────────────────────────────────────────────────────────────────────────

/**
 * Format a timestamp for display in the chat bubble.
 * HH:MM for today, MMM DD HH:MM for older dates.
 */
export function formatTimestamp(ts) {
    if (!ts) return "";
    const d = new Date(ts);
    const now = new Date();
    const hours = String(d.getHours()).padStart(2, "0");
    const mins = String(d.getMinutes()).padStart(2, "0");
    const time = `${hours}:${mins}`;
    // Same day -> just time
    if (d.getDate() === now.getDate() &&
        d.getMonth() === now.getMonth() &&
        d.getFullYear() === now.getFullYear()) {
        return time;
    }
    // Different day -> include date
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return `${months[d.getMonth()]} ${d.getDate()} ${time}`;
}

// ────────────────────────────────────────────────────────────────────────
// Timing badge formatting
// ────────────────────────────────────────────────────────────────────────

/**
 * Format timing stats into a compact badge string.
 */
export function formatTimingBadge(timing) {
    if (!timing) return "";
    const duration = timing.duration_ms != null ? timing.duration_ms : 0;
    const tps = timing.tokens_per_second != null ? timing.tokens_per_second : 0;
    if (duration < 1000) {
        return `${duration}ms @ ${tps} tok/s`;
    }
    return `${(duration / 1000).toFixed(1)}s @ ${tps} tok/s`;
}

/**
 * Format timing tooltip with detailed stats.
 */
export function formatTimingTooltip(timing) {
    if (!timing) return "";
    const parts = [];
    if (timing.ttft_ms != null) parts.push(`TTFT: ${timing.ttft_ms}ms`);
    if (timing.token_count != null) parts.push(`Tokens: ${timing.token_count}`);
    if (timing.tokens_per_second != null) parts.push(`TPS: ${timing.tokens_per_second}`);
    if (timing.duration_ms != null) parts.push(`Duration: ${(timing.duration_ms / 1000).toFixed(1)}s`);
    return parts.join(" | ");
}

// ────────────────────────────────────────────────────────────────────────
// Think tag parsing (Qwen <think> + DeepSeek Thinking... + Gemma <|channel>)
// ────────────────────────────────────────────────────────────────────────

/**
 * Parse a message text for think/reasoning blocks.
 * Returns { thinking: string|null, response: string }
 */
export function parseThinkBlocks(text) {
    if (!text) return { thinking: null, response: text || "" };

    // Qwen-style: <think>...</think> (fully closed)
    const qwenMatch = text.match(/<think>([\s\S]*?)<\/think>\s*([\s\S]*)/);
    if (qwenMatch) {
        return { thinking: qwenMatch[1].trim(), response: qwenMatch[2].trim() };
    }

    // Qwen-style: <think> without closing tag (streaming — partial)
    const openThink = text.match(/<think>([\s\S]*)$/);
    if (openThink) {
        // Everything after <think> is in-progress thinking; no response yet
        return { thinking: openThink[1].trim(), response: "" };
    }

    // DeepSeek-style: starts with "Thinking..." then content until first double newline
    if (text.startsWith("Thinking...")) {
        const rest = text.slice("Thinking...".length);
        const endMatch = rest.match(/\n\n([\s\S]*)/);
        if (endMatch) {
            const thinking = "Thinking..." + rest.substring(0, endMatch.index).trim();
            return { thinking, response: endMatch[1].trim() };
        }
        // No double newline found — entire text is thinking
        return { thinking: text.trim(), response: "" };
    }

    // Gemma-style: <|channel>...<channel|> (fully closed)
    const gemmaMatch = text.match(/<\|channel>([\s\S]*?)<channel\|>([\s\S]*)/);
    if (gemmaMatch) {
        return { thinking: gemmaMatch[1].trim(), response: gemmaMatch[2].trim() };
    }

    // Gemma-style: <|channel> without closing tag (streaming — partial)
    const openGemma = text.match(/<\|channel>([\s\S]*)$/);
    if (openGemma) {
        // Everything after <|channel> is in-progress thinking; no response yet
        return { thinking: openGemma[1].trim(), response: "" };
    }

    return { thinking: null, response: text };
}

// ────────────────────────────────────────────────────────────────────────
// Context window estimation
// ────────────────────────────────────────────────────────────────────────

// Template-specific per-entry overhead (role prefixes + suffixes).
// Derived from CHAT_TEMPLATES in utils.py — each template has different
// formatting tokens wrapping user/assistant messages.
//
// Split into {user, assistant} for higher accuracy — some templates
// have asymmetric overhead (e.g., Mistral assistant has no suffix).
const TEMPLATE_OVERHEAD = {
    qwen:     { user: 4, assistant: 4 },  // <|im_start|>user\n + <|im_end|>\n
    llama:    { user: 6, assistant: 7 },  // <|start_header_id|>user<|end_header_id|>\n\n + <|eot_id|>
    mistral:  { user: 3, assistant: 0 },  // [INST] ... [/INST] + (no suffix)
    phi:      { user: 3, assistant: 4 },  // <|user|>\n + <|end|>\n
    deepseek: { user: 2, assistant: 3 },  // User: + \n\n
    gemma:    { user: 4, assistant: 5 },  // <bos><start_of_turn>user\n + <end_of_turn>\n
};
const DEFAULT_OVERHEAD = { user: 4, assistant: 4 };

/**
 * Roughly estimate token count from text, with content-type-aware heuristic.
 *
 * - Plain text:          ~4 chars/token (standard heuristic)
 * - Code/special chars:  ~2.5 chars/token (estimated when special char ratio > 30%)
 * - Base64 image data:   ~1.5 chars/token (6 base64 chars ≈ 4 bytes ≈ 3-5 tokens)
 */
export function estimateTokens(text) {
    if (!text) return 0;

    // Detect base64 data URI — very different char/token ratio
    if (text.startsWith("data:") && text.includes(";base64,")) {
        // Base64: ~6 chars = 4 bytes; each byte may encode ~0.25-0.5 tokens
        // Rough heuristic: 1.5 chars per token for base64 payloads
        return Math.round(text.length / 1.5);
    }

    // Check if text appears code-heavy (lots of special characters)
    const specialCharRatio = text.length > 0
        ? (text.match(/[^a-zA-Z0-9\s]/g) || []).length / text.length
        : 0;
    if (specialCharRatio > 0.3) {
        // Code-heavy or symbol-rich text: ~2-3 chars per token
        return Math.round(text.length / 2.5);
    }

    // Plain text: ~4 chars per token (standard heuristic)
    return Math.round(text.length / 4);
}

/**
 * Estimate the token cost of an attached image.
 *
 * For base64 data URIs: derives approximate byte size from the base64
 * payload length, then estimates ~1 token per 200 bytes of image data.
 * For raw filenames (uploaded images): returns a default per-image constant.
 *
 * @param {string|null} image - entry.image value (data URI, raw filename, or null)
 * @returns {number} Estimated token count for the image
 */
function estimateImageTokens(image) {
    if (!image) return 0;

    if (image.startsWith("data:") && image.includes(";base64,")) {
        // Extract just the base64 payload (after the comma)
        const commaIdx = image.indexOf(",");
        const b64Payload = commaIdx >= 0 ? image.substring(commaIdx + 1) : image;
        // Each base64 character represents ~0.75 bytes
        const approxBytes = Math.round(b64Payload.length * 0.75);
        // Vision encoders typically use ~1 token per ~200 bytes of image data
        // (varies by model: LLaVA, Qwen-VL, etc.)
        return Math.max(256, Math.round(approxBytes / 200));
    }

    // Raw filename — actual size unknown; use a reasonable default
    return 512;
}

/**
 * Estimate total context usage from chat history.
 *
 * Counts:
 *   1. Message text (content-type-aware)
 *   2. Role label / template formatting overhead (template-aware, split by role)
 *   3. Image token cost (per entry.image, if present)
 *   4. System prompt (if provided via options)
 *
 * Returns a breakdown object with total and component estimates,
 * allowing callers to display a detailed tooltip.
 *
 * @param {Array}  history              - Chat history entries
 * @param {Object} [options]            - Optional parameters
 * @param {string} [options.systemPrompt]  - System prompt text to include in the count
 * @param {string} [options.chatTemplate]  - Chat template name for overhead lookup
 * @param {boolean} [options.countImages]  - Whether to estimate image tokens (default: true)
 * @param {string} [options.pendingAssistantMessage] - In-progress assistant message to include
 *        (e.g., during streaming — the accumulated text not yet saved to _chatHistory)
 * @returns {{total: number, system: number, history: number, images: number}}
 */
// ────────────────────────────────────────────────────────────────────────
// Model Metadata Extraction (GGUF model browser)
// ────────────────────────────────────────────────────────────────────────

/**
 * Quantization pattern regex — matches common GGUF quantization tags.
 * Examples: Q4_K_M, Q6_K, Q8_0, F16, BF16, IQ3_XXS, IQ4_NL, etc.
 */
const _QUANT_PATTERN = /(?:^|[_\s-])(Q[0-9](?:_[KMLSZTXMLS]+)?|F16|BF16|IQ[0-9](?:_[A-Z]+)?)(?:\.gguf)?(?:\.[a-z]+)?$/i;

/**
 * Extract quantization type from a GGUF model filename.
 * @param {string} name - Filename (e.g., "qwen2.5-7b-instruct-q4_k_m.gguf")
 * @returns {string|null} Quantization string (e.g., "Q4_K_M") or null if not found
 */
export function extractQuantization(name) {
    if (!name) return null;
    const m = name.match(_QUANT_PATTERN);
    return m ? m[1].toUpperCase() : null;
}

/**
 * Known model family name patterns for human-readable display.
 * Maps regex patterns to display names, ordered by specificity (most specific first).
 */
const _MODEL_FAMILY_PATTERNS = [
    [/qwen2\.5/i,     "Qwen 2.5"],
    [/qwen2/i,        "Qwen 2"],
    [/qwen/i,         "Qwen"],
    [/llama-?3/i,     "Llama 3"],
    [/llama/i,        "Llama"],
    [/mistral/i,      "Mistral"],
    [/mixtral/i,      "Mixtral"],
    [/phi-?3/i,       "Phi-3"],
    [/phi/i,          "Phi"],
    [/deepseek/i,     "DeepSeek"],
    [/gemma-?2/i,     "Gemma 2"],
    [/gemma/i,        "Gemma"],
    [/command.?r/i,   "Command R"],
    [/dbrx/i,         "DBRX"],
    [/falcon/i,       "Falcon"],
    [/starcoder/i,    "StarCoder"],
    [/codellama/i,    "Code Llama"],
    [/yi/i,           "Yi"],
    [/nous/i,         "Nous"],
    [/hermes/i,       "Hermes"],
    [/solar/i,        "SOLAR"],
];

/**
 * Extract human-readable model family name from a filename.
 * @param {string} name - Filename (e.g., "qwen2.5-7b-instruct-q4_k_m.gguf")
 * @returns {string|null} Display name (e.g., "Qwen 2.5") or null if unknown
 */
export function extractModelFamily(name) {
    if (!name) return null;
    for (const [pattern, display] of _MODEL_FAMILY_PATTERNS) {
        if (pattern.test(name)) return display;
    }
    return null;
}

/**
 * Format a file size (bytes) into a human-readable string.
 * @param {number} bytes - File size in bytes
 * @param {number} [decimals=1] - Number of decimal places
 * @returns {string} Formatted size (e.g., "4.8 GB", "320.0 MB")
 */
export function formatFileSize(bytes, decimals = 1) {
    if (bytes == null || bytes < 0) return "Unknown";
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const factor = 1024;
    let unitIdx = 0;
    let size = bytes;
    while (size >= factor && unitIdx < units.length - 1) {
        size /= factor;
        unitIdx++;
    }
    return `${size.toFixed(decimals)} ${units[unitIdx]}`;
}

export function estimateContextTokens(history, options = {}) {
    if (!history || !history.length) {
        // Still count pending message + system prompt even with empty history
        if (!options.pendingAssistantMessage && !options.systemPrompt) {
            return { total: 0, system: 0, history: 0, images: 0 };
        }
    }

    const overhead = options.chatTemplate
        ? (TEMPLATE_OVERHEAD[options.chatTemplate] || DEFAULT_OVERHEAD)
        : DEFAULT_OVERHEAD;

    let historyTokens = 0;
    let imageTokens = 0;

    // 1-3. Process actual history entries
    if (history && history.length) {
        for (const entry of history) {
            // 1. Message text + template formatting overhead
            historyTokens += estimateTokens(entry.message || "");

            // 2. Role-specific template formatting overhead
            const role = entry.role === "assistant" ? "assistant" : "user";
            historyTokens += overhead[role] ?? overhead.user;

            // 3. Image token cost (if entry has an attached image)
            if (options.countImages !== false && entry.image) {
                imageTokens += estimateImageTokens(entry.image);
            }
        }
    }

    // 4. Pending assistant message (in-progress streaming text)
    if (options.pendingAssistantMessage) {
        historyTokens += estimateTokens(options.pendingAssistantMessage);
        historyTokens += overhead.assistant ?? overhead.user;
    }

    // 5. System prompt (if provided)
    let systemTokens = 0;
    if (options.systemPrompt) {
        systemTokens += estimateTokens(options.systemPrompt);
        // System block template overhead (system is wrapped similarly to user)
        systemTokens += overhead.user;
    }

    const total = historyTokens + systemTokens + imageTokens;

    return { total, system: systemTokens, history: historyTokens, images: imageTokens };
}
