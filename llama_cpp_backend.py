"""
C++-accelerated GGUF inference backend via llama-cpp-python.

The entire autoregressive generation loop runs in native C++ with
hand-optimized CUDA kernels, eliminating Python loop overhead.

    Python Node
        │
        ▼
    llama.cpp C++ engine
        │
        ▼
    GPU (CUDA cores directly)

"""

import logging
import os
import re
import time
from typing import Optional

import jinja2.exceptions

try:
    import llama_cpp.llama_chat_format as _lcf
except ImportError:
    _lcf = None

# Attempt llama-cpp-python import; set availability flag for upstream nodes
LLAMA_CPP_AVAILABLE = False
LLAMA_CPP_VERSION = "unknown"
try:
    from llama_cpp import Llama as _Llama
    from llama_cpp import __version__ as _llama_cpp_version
    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_VERSION = _llama_cpp_version
    logging.info(
        f"[LLM Chat GGUF] llama-cpp-python version {LLAMA_CPP_VERSION} detected"
    )
except ImportError:
    _Llama = None  # type: ignore

# Version constants from _version_config.py
from ._version_config import MIN_VISION_VERSION, SAFE_BASELINE_VERSION


def _parse_version(version_str: str) -> tuple[int, int, int] | None:
    """Parse a semver string like '0.3.23' into a comparable tuple (0, 3, 23).

    Returns None if parsing fails.
    """
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _version_gte(version_str: str, min_version_str: str) -> bool:
    """Check if version_str >= min_version_str using semver comparison.

    Args:
        version_str: The installed version (e.g., '0.3.23')
        min_version_str: The minimum required version (e.g., '0.3.30')

    Returns:
        True if version_str >= min_version_str, False if below, None on parse error.
    """
    v_actual = _parse_version(version_str)
    v_min = _parse_version(min_version_str)
    if v_actual is None or v_min is None:
        return False  # Can't parse — assume too old to be safe
    return v_actual >= v_min


def _vision_version_ok() -> bool:
    """Check if the installed llama-cpp-python version supports vision/image_url."""
    return _version_gte(LLAMA_CPP_VERSION, MIN_VISION_VERSION)


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    """Strip image_url content from messages, keeping only text content.

    Used as a fallback when the installed llama-cpp-python version is too old
    to properly process image_url content types. Without this, the raw message
    dict gets rendered as Python repr in the prompt.

    Args:
        messages: OpenAI-compatible messages array potentially containing
                  image_url content types.

    Returns:
        Cleaned messages with only text content.
    """
    cleansed = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract only text items from the content array
            text_parts = [
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            cleansed.append({
                "role": msg["role"],
                "content": "\n".join(text_parts) if text_parts else ""
            })
        else:
            cleansed.append(msg)
    return cleansed


def _messages_have_images(messages: list[dict]) -> bool:
    """Check if any message in the array has image_url content type."""
    return any(
        isinstance(m.get("content"), list) and any(
            isinstance(c, dict) and c.get("type") == "image_url"
            for c in m["content"]
        )
        for m in messages
    )


class LlamaCppModel:
    """Wraps a GGUF model loaded via llama-cpp-python (C++ inference engine).

    Uses the same highly optimized C++ backend as Ollama and LM Studio.
    Typical speed for an 8B model with n_gpu_layers=-1: 30-50+ tokens/sec.

    Args:
        model_path: Path to the .gguf file on disk
        n_gpu_layers: Number of layers to offload to GPU.
                      -1 = all layers (maximum speed).
                      0 = CPU only (slow).
                      24 = partial offload (balanced VRAM).
        use_mlock: Lock model memory in RAM to prevent OS swapping.
                  Stabilizes shared RAM performance.
        n_ctx: Context window size (max tokens model can remember).
        verbose: Enable llama.cpp verbose logging (debugging only).

    Raises:
        ImportError: If llama-cpp-python is not installed.
        RuntimeError: If the model file cannot be loaded.
    """

    def __init__(
        self,
        model_path: str,
        mmproj: str = "",
        n_gpu_layers: int = -1,
        use_mlock: bool = True,
        n_ctx: int = 4096,
        verbose: bool = False,
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  llama-cpp-python is NOT installed or failed to import  ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
                "\n"
                "The LLM Chat GGUF node needs llama-cpp-python for 30-50+ tok/s\n"
                "C++ inference. Here's how to install it:\n"
                "\n"
                "  🚀 AUTOMATIC (recommended):\n"
                "     Run the install helper script:\n"
                "       python custom_nodes\\llm-chat\\install.py\n"
                "     This detects your ComfyUI Python, CUDA version, and\n"
                "     installs the correct pre-built wheel automatically.\n"
                "     Also works with ComfyUI Manager's auto-install hooks.\n"
                "\n"
                "  🔧 MANUAL (pick one):\n"
                "     CUDA 12.x:\n"
                "       pip install llama-cpp-python --extra-index-url\n"
                "       https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
                "     CUDA 11.x:\n"
                "       pip install llama-cpp-python --extra-index-url\n"
                "       https://abetlen.github.io/llama-cpp-python/whl/cu118\n"
                "     CPU only:\n"
                "       pip install llama-cpp-python --extra-index-url\n"
                "       https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
                "\n"
                "  📦 ComfyUI Windows Portable:\n"
                "     python_embeded\\python.exe -m pip install llama-cpp-python\n"
                "       --extra-index-url https://abetlen.github.io/...\n"
                "\n"
                "After installing, restart ComfyUI.\n"
            )

        self.model_path = model_path
        self.mmproj = mmproj
        self.n_gpu_layers = n_gpu_layers
        self.use_mlock = use_mlock
        self.n_ctx = n_ctx
        self.verbose = verbose
        self._model: Optional["_Llama"] = None
        self._metadata: dict = {}
        self._supports_vision: bool = False
        self._chat_handler = None
        self._template_suppressed: bool = False

    def load(self):
        """Load the GGUF model via llama.cpp C++ engine (GPU if n_gpu_layers > 0, else CPU).

        With n_gpu_layers=-1, all layers are loaded onto the GPU for maximum speed.

        Uses multi-tier path resolution via resolve_gguf_path() to find the model
        across various ComfyUI installation layouts (standard, Forge, Stability Matrix).

        Raises:
            RuntimeError: If the model file cannot be found or loaded.
        """
        if self._model is not None:
            logging.debug(
                f"[LLM Chat GGUF] Model already loaded: {self.model_path}"
            )
            return

        from .debug_profiler import profiler
        profiler.begin("model_load_total")

        import os
        from .utils import resolve_gguf_path, _get_searched_paths

        profiler.begin_sub("model_load_total", "path_resolve")
        resolved = resolve_gguf_path(self.model_path)
        if resolved is None:
            searched = _get_searched_paths(self.model_path)
            searched_str = "\n".join(f"  • {p}" for p in searched)
            raise RuntimeError(
                f"GGUF model file not found: {self.model_path}\n\n"
                "🔍 Searched in these locations:\n"
                f"{searched_str}\n\n"
                "💡 Tips:\n"
                "  • Open the LLM Chat popup and use the Browse button to\n"
                "    locate your .gguf file anywhere on the system.\n"
                "  • Use an absolute path, e.g.:\n"
                f"    {os.path.join('E:', os.sep, 'Models', 'GGUF', 'my_model.gguf')}\n"
                "  • Provide just the filename if the file is in a standard\n"
                "    ComfyUI models directory (text_encoders, checkpoints, etc.)\n"
                "  • The search now covers Ollama (~/.ollama/models) and\n"
                "    LM Studio (~/.cache/lm-studio/models) directories as fallback.\n"
                "  • Register custom directories in ComfyUI's extra_model_paths.yaml:\n"
                "      - Open [ComfyUI]/extra_model_paths.yaml\n"
                "      - Add an entry like:\n"
                "          my_gguf_models:\n"
                "            base_path: D:\\lmstudio\\models\n"
                "      - Restart ComfyUI — the directory will be searched automatically.\n"
                "  • Download a Q4_K_M model from HuggingFace, e.g.:\n"
                "    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF\n"
                "    https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
            )

        # Update to resolved absolute path for reliable loading
        self.model_path = resolved
        logging.info(
            f"[LLM Chat GGUF] Resolved model path: {self.model_path}"
        )

        # ── Resolve mmproj path (multimodal projection for vision models) ──
        if self.mmproj and self.mmproj.strip():
            from .utils import resolve_gguf_path, _get_searched_paths
            resolved_mmproj = resolve_gguf_path(self.mmproj)
            if resolved_mmproj is None:
                searched = _get_searched_paths(self.mmproj)
                searched_str = "\n".join(f"  • {p}" for p in searched)
                raise RuntimeError(
                    f"Multimodal projection file not found: {self.mmproj}\n\n"
                    "🔍 Searched in these locations:\n"
                    f"{searched_str}\n\n"
                    "The mmproj file is a small GGUF file that maps image embeddings\n"
                    "to the language model's text embedding space. It's usually named\n"
                    "like your main model with -mmproj in the filename."
                )
            self.mmproj = resolved_mmproj
            self._supports_vision = True
            logging.info(
                f"[LLM Chat GGUF] Resolved mmproj path: {self.mmproj}"
            )
        profiler.end_sub("model_load_total", "path_resolve")

        logging.info(
            f"[LLM Chat GGUF] Loading model via llama.cpp C++ engine:\n"
            f"  Model: {self.model_path}\n"
            f"  mmproj: {self.mmproj if self._supports_vision else '(none)'}\n"
            f"  n_gpu_layers: {self.n_gpu_layers}\n"
            f"  use_mlock: {self.use_mlock}\n"
            f"  n_ctx: {self.n_ctx}"
        )

        # Clear PyTorch CUDA cache before allocating C++ CUDA context.
        # Prevents conflicts between llama.cpp's cuBLAS context and PyTorch's
        # CUDA context when both try to use the GPU simultaneously.
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Ask ComfyUI to free VRAM before loading GGUF model ──
        # Image generation models (Anima, WanVAE, etc.) may be occupying VRAM.
        # free_memory() tells ComfyUI to unload them so llama.cpp can use the GPU.
        profiler.begin_sub("model_load_total", "vram_free")
        try:
            from .memory_manager import estimate_vram_needed
            import comfy.model_management
            memory_needed = estimate_vram_needed(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
            )
            comfy.model_management.free_memory(
                memory_required=memory_needed,
                device=torch.device("cuda"),
            )
            try:
                file_size_gb = os.path.getsize(self.model_path) / (1024**3)
                log_detail = (
                    f"(file: {file_size_gb:.1f} GiB, "
                    f"n_gpu_layers={self.n_gpu_layers})"
                )
            except Exception:
                log_detail = ""
            logging.info(
                f"[LLM Chat GGUF] Requested ComfyUI to free "
                f"{memory_needed / (1024**3):.1f} GiB VRAM "
                f"{log_detail}"
            )
        except Exception as e:
            logging.warning(
                f"[LLM Chat GGUF] Failed to request VRAM freeing: {e} — "
                "proceeding without explicit free_memory()"
            )
        profiler.end_sub("model_load_total", "vram_free")

        # ── CREATE CHAT HANDLER (multimodal vision support) ──
        # Llava15ChatHandler loads the mmproj via mtmd_cpp C++ backend,
        # processes image_url messages, and tokenizes images via mtmd_tokenize().
        # Without this, the mmproj file is never loaded (mmproj= is not a
        # recognized parameter of Llama.__init__() and is silently dropped).
        chat_handler = None
        profiler.begin_sub("model_load_total", "chat_handler")
        if self._supports_vision and self.mmproj:
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(
                    clip_model_path=self.mmproj,
                    verbose=self.verbose,
                )
                logging.info(
                    f"[LLM Chat GGUF] Created Llava15ChatHandler for mmproj: "
                    f"{self.mmproj}"
                )
            except Exception as e:
                logging.warning(
                    f"[LLM Chat GGUF] Failed to create vision chat handler: {e}. "
                    "Image input will fall back to text-only via try/except."
                )
                self._supports_vision = False
        profiler.end_sub("model_load_total", "chat_handler")

        # Store chat_handler reference so unload() can free its C-level mtmd_ctx
        self._chat_handler = chat_handler

        profiler.begin_sub("model_load_total", "llama_constructor")
        try:
            self._model = _Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_gpu_layers=self.n_gpu_layers,
                use_mlock=self.use_mlock,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                n_batch=1024,       # Larger batch speeds up prompt evaluation
                n_ubatch=512,       # Micro-batch for memory efficiency (half of n_batch)
            )
        except jinja2.exceptions.TemplateError as _tmpl_e:
            # ── Chat template contains unsupported Jinja2 tags (e.g., {% generation %}) ──
            # Some model variants (e.g., Dolphin) embed custom Jinja2 tags like
            # {% generation %} in tokenizer.chat_template metadata. Jinja2ChatFormatter
            # cannot parse these, crashing model load.
            #
            # Monkey-patch Jinja2ChatFormatter.__init__ to catch TemplateSyntaxError and
            # substitute a minimal safe template. kv_overrides does not affect metadata()
            # (it reads GGUF KV pairs via C API), so the template parsing loop always
            # sees the broken original. The patch is scoped to this retry and restored
            # in the finally block below.
            logging.warning(
                f"[LLM Chat GGUF] Model's chat template contains unsupported "
                f"Jinja2 tags ({_tmpl_e}). "
                f"Retrying with monkey-patched Jinja2ChatFormatter."
            )
            if _lcf is None:
                raise RuntimeError(
                    "Cannot import llama_cpp.llama_chat_format — cannot apply "
                    "template suppression monkey-patch."
                ) from _tmpl_e
            _original_init = _lcf.Jinja2ChatFormatter.__init__
            def _patched_jinja2_init(self_, template, *args, **kwargs):
                try:
                    _original_init(self_, template, *args, **kwargs)
                except jinja2.exceptions.TemplateSyntaxError:
                    # Fall back to a minimal safe template that always parses.
                    # The text-only path bypasses the model's built-in template
                    # entirely (uses format_prompt_by_template() + generate()).
                    _original_init(
                        self_, "{{ message['content'] }}", *args, **kwargs
                    )
            _lcf.Jinja2ChatFormatter.__init__ = _patched_jinja2_init
            try:
                self._model = _Llama(
                    model_path=self.model_path,
                    chat_handler=chat_handler,
                    n_gpu_layers=self.n_gpu_layers,
                    use_mlock=self.use_mlock,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose,
                    n_batch=1024,
                    n_ubatch=512,
                )
                self._template_suppressed = True
                logging.info(
                    f"[LLM Chat GGUF] Model loaded successfully with suppressed "
                    f"chat template. Text-only mode will use "
                    f"format_prompt_by_template()."
                )
            except Exception as _retry_e:
                raise RuntimeError(
                    f"Failed to load model even with template suppression: "
                    f"{_retry_e}"
                ) from _tmpl_e
            finally:
                # ⚠️ CRITICAL: Always restore the original __init__, even if
                # _Llama() raised a different exception. Otherwise ALL future
                # model loads in this process would use the patched version,
                # potentially masking real template issues on other models.
                _lcf.Jinja2ChatFormatter.__init__ = _original_init
        except OSError as _llama_e:
            _llama_err_str = str(_llama_e)
            if "0xc000001d" in _llama_err_str or "-1073741795" in _llama_err_str:
                # ── STATUS_ILLEGAL_INSTRUCTION: CPU doesn't support instruction set ──
                # This happens when the pre-built wheel was compiled with CPU instructions
                # (e.g., AVX-VNNI, AVX512-BF16) that the processor doesn't support.
                # The install.py handles this at install time by detecting old CPUs and
                # selecting the safe baseline. If we get here at runtime, it means either
                # --try-latest was used or the detection didn't catch this CPU.
                # Solution: re-run install.py with --backend cpu to get a CPU-only wheel.
                logging.error(
                    f"[LLM Chat GGUF] CPU instruction set incompatibility (0xc000001d) "
                    f"detected. The installed wheel requires CPU instructions not "
                    f"supported by this processor."
                )
                raise RuntimeError(
                    "CPU instruction set incompatibility detected.\n\n"
                    "The installed llama-cpp-python wheel was compiled with CPU "
                    "instructions not supported by your processor.\n\n"
                    "This CPU lacks AVX2 or other instructions required by the "
                    "pre-built wheel.\n\n"
                    "Solutions:\n"
                    "  1. Re-run install.py with CPU-only backend (uses a wheel\n"
                    "     compiled without AVX2):\n"
                    "       python install.py --backend cpu\n"
                    "  2. Or compile from source with conservative flags:\n"
                    "       set CMAKE_ARGS=-DGGML_AVX2=OFF -DGGML_FMA=OFF\n"
                    "       pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                    "  3. Or use a GPU backend (Vulkan) which offloads compute:\n"
                    "       python install.py --backend vulkan"
                ) from _llama_e
            else:
                raise  # Re-raise other OSError types as-is

        profiler.end_sub("model_load_total", "llama_constructor")

        # ── DIAGNOSTIC: Wrap eval() to log per-batch llama_decode() timing ──
        # Logs C-level llama_decode() duration for prompt eval batches and single-token generation.
        # Critical for identifying CUDA/cuBLAS re-initialization between calls.
        _original_eval = self._model.eval
        import functools
        @functools.wraps(_original_eval)
        def _timed_eval(tokens):
            t0 = time.perf_counter()
            result = _original_eval(tokens)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000
            if elapsed > 100:  # Only log if >100ms (avoids spam for fast single-token evals)
                logging.info(
                    f"[DIAG] llama.eval() batch={len(tokens)} "
                    f"took {elapsed:.1f}ms"
                )
            return result
        self._model.eval = _timed_eval
        profiler.end_sub("model_load_total", "eval_wrap")

        # Expose model metadata for chat template auto-detection
        self._metadata = getattr(self._model, "metadata", {}) or {}

        logging.info(
            f"[LLM Chat GGUF] Model loaded successfully — "
            f"ready for 30-50+ tok/s inference"
        )
        if self._metadata:
            logging.debug(
                f"[LLM Chat GGUF] Model metadata keys: {list(self._metadata.keys())}"
            )

        profiler.end("model_load_total")

    @property
    def metadata(self) -> dict:
        """Model metadata (architecture, tokenizer config, etc.) from the GGUF file."""
        return self._metadata

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 42,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate text using the C++ inference engine.

        Args:
            prompt: The formatted prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy/deterministic)
            top_k: Top-K sampling (higher = more diverse)
            top_p: Nucleus sampling threshold
            seed: Random seed for reproducibility.
                  0 = auto-randomize (different output each call).
            stop: List of stop tokens. If None, no stop tokens are passed
                  (model stops naturally at EOS). Caller should provide
                  model-appropriate stop tokens based on chat template.
            repetition_penalty: Penalty for repeating tokens.
                                1.0 = no penalty.
                                >1.0 discourages repetition (e.g., 1.2).
                                <1.0 encourages repetition.

        Returns:
            str: The generated text (without the input prompt)

        Raises:
            RuntimeError: If the model is not loaded (call load() first).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before generate().")

        # Auto-randomize seed when set to 0 (mirrors utils.auto_seed())
        if seed == 0:
            import random
            seed = random.randint(1, 0xFFFFFFFF)

        # ── DIAGNOSTIC: Measure total __call__ latency ──
        t1 = time.perf_counter()
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            echo=False,                     # Don't repeat the input prompt
            stop=stop or [],                # Configurable stop tokens from caller
            repeat_penalty=repetition_penalty,  # llama.cpp uses repeat_penalty
        )
        t2 = time.perf_counter()
        total_time = (t2 - t1) * 1000
        if total_time > 100:
            logging.info(
                f"[DIAG] generate: __call__ total={total_time:.1f}ms"
            )
        return output["choices"][0]["text"].strip()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 42,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
    ):
        """Generate text with token-by-token streaming via the C++ engine.

        Yields individual tokens one at a time as they are generated,
        enabling real-time display in the popup UI.

        Args:
            Same as generate(), except returns a generator yielding individual tokens.

        Yields:
            str: Individual tokens of generated text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before generate_stream().")

        # Auto-randomize seed when set to 0
        if seed == 0:
            import random
            seed = random.randint(1, 0xFFFFFFFF)

        # ── DIAGNOSTIC: Measure __call__ setup vs inference latency ──
        # T0 (start_time) set by caller (streaming.py). T1 = before __call__.
        # T2 = generator returned (tokenization+setup). T3 = first token = TTFT.
        t1 = time.perf_counter()
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            echo=False,
            stop=stop or [],
            repeat_penalty=repetition_penalty,
            stream=True,  # Enable streaming mode
        )
        t2 = time.perf_counter()
        setup_time = (t2 - t1) * 1000
        if setup_time > 10:  # Only log if >10ms (avoids spam for trivial setup)
            logging.info(
                f"[DIAG] generate_stream: __call__ setup took {setup_time:.1f}ms"
            )
        first_token = True
        for chunk in output:
            token_text = chunk["choices"][0]["text"]
            if token_text:
                if first_token:
                    t3 = time.perf_counter()
                    ttft = (t3 - t1) * 1000
                    if ttft > 100:  # Only log if >100ms
                        logging.info(
                            f"[DIAG] generate_stream: first-token TTFT={ttft:.1f}ms "
                            f"(setup={setup_time:.1f}ms, "
                            f"inference={ttft-setup_time:.1f}ms)"
                        )
                    first_token = False
                yield token_text

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 42,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate a chat response using create_chat_completion() (multimodal API).

        Uses the OpenAI-compatible chat API from llama-cpp-python, which handles
        chat templates internally from the model's GGUF metadata. Unlike generate(),
        this method:
          - Takes a messages array (not a formatted prompt string)
          - Returns choices[0].message.content (not choices[0].text)
          - Accepts optional stop tokens — forwarded to create_chat_completion()
            to prevent template token bleeding into generated output.

        Args:
            messages: OpenAI-compatible messages array, e.g.:
                      [{"role": "system", "content": "..."},
                       {"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                           {"type": "text", "text": "What's in this image?"}
                       ]}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy/deterministic)
            top_k: Top-K sampling (higher = more diverse)
            top_p: Nucleus sampling threshold
            seed: Random seed for reproducibility.
                  0 = auto-randomize (different output each call).
            repetition_penalty: Penalty for repeating tokens.
                                 1.0 = no penalty.
                                 >1.0 discourages repetition.
                                 <1.0 encourages repetition.
            stop: Optional list of stop strings. Forwarded to
                  create_chat_completion() to halt generation when these
                  sequences appear. Use CHAT_TEMPLATES["stop"] values
                  from utils.py to prevent template token bleeding.

        Returns:
            str: The generated chat response text

        Raises:
            RuntimeError: If the model is not loaded (call load() first).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before generate_chat().")

        # ── VERSION WARNING: Log when below recommended version ──
        _has_images = _messages_have_images(messages)
        if _has_images and not _vision_version_ok():
            logging.warning(
                f"[LLM Chat GGUF] llama-cpp-python {LLAMA_CPP_VERSION} is below "
                f"the recommended version {MIN_VISION_VERSION} for image_url support. "
                f"Attempting vision anyway — if create_chat_completion() fails, "
                f"images will be stripped and retried as text-only."
            )

        # ── DIAGNOSTIC: Log llama-cpp-python version and chat template ──
        try:
            import llama_cpp
            _lcpp_ver = getattr(llama_cpp, '__version__', 'unknown')
        except Exception:
            _lcpp_ver = 'unknown'
        _chat_template = getattr(self._model, 'chat_template', None) or \
                         self._metadata.get('tokenizer.chat_template', 'NOT FOUND')
        logging.info(
            f"[DIAG] generate_chat: llama-cpp-python={_lcpp_ver}, "
            f"supports_vision={self._supports_vision}, "
            f"mmproj={self.mmproj}, "
            f"messages_len={len(messages)}, "
            f"messages_roles={[m['role'] for m in messages]}, "
            f"chat_template_preview={repr(_chat_template[:150])}"
        )

        # Auto-randomize seed when set to 0 (mirrors generate() behavior)
        if seed == 0:
            import random
            seed = random.randint(1, 0xFFFFFFFF)

        # ── DIAGNOSTIC: Measure create_chat_completion total latency ──
        t1 = time.perf_counter()

        # ── TRY VISION; FALL BACK TO TEXT-ONLY ON FAILURE ──
        # Older llama-cpp-python versions may not handle image_url content type
        # properly. We try first, and if create_chat_completion raises an
        # exception, we strip images and retry as text-only.
        try:
            output = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                repeat_penalty=repetition_penalty,
                stop=stop or [],
            )
            t2 = time.perf_counter()
            total_time = (t2 - t1) * 1000
            if total_time > 100:
                logging.info(
                    f"[DIAG] generate_chat: create_chat_completion "
                    f"total={total_time:.1f}ms"
                )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if self._template_suppressed:
                # Template was suppressed due to incompatible Jinja2 tags.
                # create_chat_completion() requires a valid chat template to
                # format messages — without one, it cannot render the prompt.
                raise RuntimeError(
                    f"create_chat_completion failed because the model's chat "
                    f"template was suppressed due to incompatible Jinja2 tags "
                    f"({e}).\n\n"
                    f"This model (likely a Dolphin variant) has a custom "
                    f"'{{% generation %}}' tag in its tokenizer.chat_template "
                    f"metadata that Jinja2 cannot parse.\n\n"
                    f"For text-only usage, use format_prompt_by_template() "
                    f"with model.generate() — this path works perfectly.\n"
                    f"For multimodal/image input, please use a different model "
                    f"or upgrade llama-cpp-python."
                ) from e
            elif _has_images:
                logging.warning(
                    f"[LLM Chat GGUF] create_chat_completion failed with images "
                    f"({e}). Retrying with images stripped (text-only)."
                )
                text_messages = _strip_images_from_messages(messages)
                try:
                    t1b = time.perf_counter()
                    output = self._model.create_chat_completion(
                        messages=text_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repeat_penalty=repetition_penalty,
                        stop=stop or [],
                    )
                    t2b = time.perf_counter()
                    total_time_b = (t2b - t1b) * 1000
                    if total_time_b > 100:
                        logging.info(
                            f"[DIAG] generate_chat (fallback): "
                            f"create_chat_completion total={total_time_b:.1f}ms"
                        )
                    return output["choices"][0]["message"]["content"].strip()
                except Exception as e2:
                    raise RuntimeError(
                        f"create_chat_completion failed even after stripping images: {e2}"
                    ) from e2
            else:
                # No images involved — re-raise the original error
                raise

    def generate_chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = 42,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ):
        """Generate a chat response with token-by-token streaming (multimodal API).

        Uses create_chat_completion with stream=True, yielding individual tokens
        from the chunk deltas. Like generate_chat(), this method:
          - Takes a messages array (not a formatted prompt string)
          - Yields delta.get("content", "") from choices[0].delta (not choices[0].text)
          - Accepts optional stop tokens — forwarded to create_chat_completion()
            to prevent template token bleeding into generated output.

        Args:
            Same as generate_chat(), except returns a generator yielding individual tokens.
            stop: Optional list of stop strings. Forwarded to
                  create_chat_completion() to halt generation when these
                  sequences appear.

        Yields:
            str: Individual tokens of generated chat response text.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before generate_chat_stream().")

        # ── VERSION WARNING: Log when below recommended version ──
        _has_images = _messages_have_images(messages)
        if _has_images and not _vision_version_ok():
            logging.warning(
                f"[LLM Chat GGUF] llama-cpp-python {LLAMA_CPP_VERSION} is below "
                f"the recommended version {MIN_VISION_VERSION} for image_url support. "
                f"Attempting vision anyway — if create_chat_completion() fails, "
                f"images will be stripped and retried as text-only."
            )

        # ── DIAGNOSTIC: Log llama-cpp-python version and chat template ──
        try:
            import llama_cpp
            _lcpp_ver = getattr(llama_cpp, '__version__', 'unknown')
        except Exception:
            _lcpp_ver = 'unknown'
        _chat_template = getattr(self._model, 'chat_template', None) or \
                         self._metadata.get('tokenizer.chat_template', 'NOT FOUND')
        logging.info(
            f"[DIAG] generate_chat_stream: llama-cpp-python={_lcpp_ver}, "
            f"supports_vision={self._supports_vision}, "
            f"mmproj={self.mmproj}, "
            f"messages_len={len(messages)}, "
            f"messages_roles={[m['role'] for m in messages]}, "
            f"chat_template_preview={repr(_chat_template[:150])}"
        )

        # Auto-randomize seed when set to 0 (mirrors generate_stream() behavior)
        if seed == 0:
            import random
            seed = random.randint(1, 0xFFFFFFFF)

        # ── DIAGNOSTIC: Measure create_chat_completion setup vs inference ──
        t1 = time.perf_counter()

        # ── TRY VISION; FALL BACK TO TEXT-ONLY ON FAILURE ──
        # For streaming, the exception may occur during iteration (lazy eval),
        # so we wrap the entire generation loop in try/except.
        try:
            output = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                repeat_penalty=repetition_penalty,
                stop=stop or [],
                stream=True,
            )
            t2 = time.perf_counter()
            setup_time = (t2 - t1) * 1000
            if setup_time > 10:
                logging.info(
                    f"[DIAG] generate_chat_stream: create_chat_completion "
                    f"setup took {setup_time:.1f}ms"
                )
            first_token = True
            for chunk in output:
                delta = chunk["choices"][0].get("delta", {})
                token_text = delta.get("content", "")
                if token_text:
                    if first_token:
                        t3 = time.perf_counter()
                        ttft = (t3 - t1) * 1000
                        if ttft > 100:
                            logging.info(
                                f"[DIAG] generate_chat_stream: first-token "
                                f"TTFT={ttft:.1f}ms "
                                f"(setup={setup_time:.1f}ms, "
                                f"inference={ttft-setup_time:.1f}ms)"
                            )
                        first_token = False
                    yield token_text
        except Exception as e:
            if _has_images:
                logging.warning(
                    f"[LLM Chat GGUF] create_chat_completion failed with images "
                    f"({e}). Retrying with images stripped (text-only)."
                )
                text_messages = _strip_images_from_messages(messages)
                try:
                    t1b = time.perf_counter()
                    output = self._model.create_chat_completion(
                        messages=text_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        repeat_penalty=repetition_penalty,
                        stop=stop or [],
                        stream=True,
                    )
                    t2b = time.perf_counter()
                    setup_time_b = (t2b - t1b) * 1000
                    if setup_time_b > 10:
                        logging.info(
                            f"[DIAG] generate_chat_stream (fallback): "
                            f"setup took {setup_time_b:.1f}ms"
                        )
                    first_token_b = True
                    for chunk in output:
                        delta = chunk["choices"][0].get("delta", {})
                        token_text = delta.get("content", "")
                        if token_text:
                            if first_token_b:
                                t3b = time.perf_counter()
                                ttft_b = (t3b - t1b) * 1000
                                if ttft_b > 100:
                                    logging.info(
                                        f"[DIAG] generate_chat_stream (fallback): "
                                        f"first-token TTFT={ttft_b:.1f}ms "
                                        f"(setup={setup_time_b:.1f}ms, "
                                        f"inference={ttft_b-setup_time_b:.1f}ms)"
                                    )
                                first_token_b = False
                            yield token_text
                except Exception as e2:
                    raise RuntimeError(
                        f"create_chat_completion failed even after stripping images: {e2}"
                    ) from e2
            else:
                # No images involved — re-raise the original error
                raise

    @property
    def supports_vision(self) -> bool:
        """Whether this model instance has vision/multimodal support enabled.

        Returns True when an mmproj (multimodal projection) file was provided
        and resolved during load(). Vision models (LLaVA, BakLLaVA, Qwen-VL, etc.)
        can accept image inputs via generate_chat()/generate_chat_stream().
        """
        return self._supports_vision

    def unload(self):
        """Unload the model from GPU memory.

        Frees all VRAM allocated by this model — including both the main GGUF
        model weights/KV-cache (via llama.cpp C++ engine) AND the multimodal
        projection context (mtmd_ctx) if a vision chat handler was created.

        MUST be called in this order:
          1. Chat handler first (frees mtmd_ctx GPU memory)
          2. Main model second (frees model weights, KV cache)
          3. PyTorch CUDA cache emptied last

        Call this when you need to reclaim GPU memory for image generation
        (KSampler, etc.).
        """
        # ── 1. Close chat handler first (frees mtmd_ctx / multimodal GPU memory) ──
        # Llava15ChatHandler stores C-level multimodal context (mtmd_ctx) in an
        # ExitStack callback (mtmd_free). The handler never closes this ExitStack —
        # close it explicitly. Otherwise the multimodal projection model remains in
        # GPU memory after the main model unloads, leaking VRAM.
        if self._chat_handler is not None:
            try:
                if hasattr(self._chat_handler, '_exit_stack'):
                    self._chat_handler._exit_stack.close()
            except Exception as e:
                logging.warning(
                    f"[LLM Chat GGUF] Error closing chat handler ExitStack: {e}"
                )
            self._chat_handler = None

        # ── 2. Close main model (frees model weights, KV cache, context) ──
        # llama.cpp performs its own CUDA memory management — calling close()
        # on the _Llama instance triggers the ExitStack which frees the model,
        # context, and batch objects at the C level.
        if self._model is not None:
            self._model.close()
            self._model = None

            # Synchronize CUDA after closing the C++ engine to ensure all
            # llama.cpp CUDA operations complete before PyTorch takes over.
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Empty PyTorch's CUDA caching allocator so that freed C-level
                # CUDA memory is actually released back to the GPU driver.
                # Without this, torch.cuda.memory_allocated() may still show
                # VRAM in use even though llama.cpp freed everything.
                torch.cuda.empty_cache()

            logging.info(
                "[LLM Chat GGUF] Model unloaded from GPU — "
                "VRAM freed for image generation"
            )

    def __del__(self):
        """Ensure model is unloaded on garbage collection."""
        self.unload()
