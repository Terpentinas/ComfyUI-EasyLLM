← [Documentation Home](README.md)

---

## Chat UI Features

The **EasyLLM** and **EasyLLM GGUF** nodes include a built-in chat interface for a more natural interaction experience.

### Popup Chat

Double-click the node to open a popup chat window with:
- **Conversation history** — scrollable chat log with user/assistant messages
- **Multi-turn conversation** — the popup maintains full chat history across exchanges
- **Markdown rendering** — the LLM's responses are rendered with proper formatting (bold, lists, code blocks, etc.)
- **System prompt editor** — customize the system instruction
- **Model selector** — browse and select GGUF models directly from the popup
- **Image upload** — upload images for vision-language models (no wires needed)
- **Send button** — queue the workflow with your message
- **Enter-to-Send** — press Enter to send, Shift+Enter for newline
- **Token streaming** — responses appear token-by-token in real-time

![Chat Popup in Action](../media/chat-pop-up.png)

🎥 **[Watch the streaming chat demo](../media/simple-chat.mp4)**

### Canvas Widgets

The node displays key controls directly on the canvas:
- **Text input** — type your message
- **Send button** — click to queue generation
- **Response area** — read-only display of the last response
- **Mode selector** — switch between chat and enhancer modes
- **Prompt template** — configure enhancer output format

### Chat Modes

Each node has a mode badge displayed on the canvas:

| Badge | Mode | Description |
|-------|------|-------------|
| 💬 **CHAT** | Chat | Free-form conversation with the LLM |
| 🔧 **ENHANCER** | Enhancer | Transform simple prompts into detailed descriptions |

### Export

Chat and enhancer history can be exported as **Markdown** or **Plain Text** from the popup menu.

---

← [Back to Documentation](README.md)
