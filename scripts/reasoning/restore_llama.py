# Pin `transformers.models.llama.{modeling,configuration}_llama` to the exact v4.46.1
# sources while LlamaEncoderModel is in use, with a fully symmetric undo so an embedding
# process (evaluation inside a training run) is left exactly as it was found.
#
# undo() restores everything apply() may have perturbed, including top-level lazy
# re-exports (`transformers.LlamaModel` etc.) that transformers caches on first attribute
# access: any such attribute cached *while the patch was active* aliases a vendored class
# and must be dropped so the next access re-resolves to the installed version. Snapshots
# read module `__dict__` directly (never getattr) so they cannot themselves trigger the
# lazy loader. State is kept per apply/undo cycle, so repeated cycles keep working.

# module attributes that may alias vendored objects on these holders
_HOLDER_ATTRS = {
    "transformers": (
        "LlamaModel", "LlamaPreTrainedModel", "LlamaForCausalLM",
        "LlamaForSequenceClassification", "LlamaForQuestionAnswering",
        "LlamaForTokenClassification", "LlamaConfig",
    ),
    "transformers.models": ("llama",),
    "transformers.models.llama": (
        "modeling_llama", "configuration_llama",
        "LlamaModel", "LlamaPreTrainedModel", "LlamaForCausalLM",
        "LlamaForSequenceClassification", "LlamaForQuestionAnswering",
        "LlamaForTokenClassification", "LlamaConfig", "LlamaDecoderLayer",
        "LlamaRMSNorm", "LlamaMLP", "LlamaAttention", "LlamaRotaryEmbedding",
    ),
}
_TARGET_MODULES = (
    "transformers.models.llama.modeling_llama",
    "transformers.models.llama.configuration_llama",
)

_patch_state = None  # None when inactive; else dict(modules=..., attrs=...)


def _snapshot():
    import sys
    state = {"modules": {}, "attrs": {}}
    for mod in _TARGET_MODULES:
        state["modules"][mod] = sys.modules.get(mod)
    for holder_name, attrs in _HOLDER_ATTRS.items():
        holder = sys.modules.get(holder_name)
        if holder is None:
            continue
        for attr in attrs:
            # __dict__ access only: getattr would trigger transformers' lazy loader
            if attr in holder.__dict__:
                state["attrs"][(holder_name, attr)] = (True, holder.__dict__[attr])
            else:
                state["attrs"][(holder_name, attr)] = (False, None)
    return state


def apply(
    force: bool = True,
    restore_config: bool = True,
    modeling_src: str = None,
    config_src: str = None,
):
    """
    Replace installed `transformers.models.llama.modeling_llama` (and optionally
    `configuration_llama`) with the exact v4.46.1 files. Undo with `undo()`.
    """
    global _patch_state
    import sys, types, importlib, os, linecache

    if _patch_state is not None:  # already active
        return True

    # prefer the vendored v4.46.1 sources (no network, no per-rank GitHub downloads)
    _vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendored_llama_v4461")
    if modeling_src is None:
        _local = os.path.join(_vendor, "modeling_llama.py")
        modeling_src = _local if os.path.exists(_local) else \
            "https://raw.githubusercontent.com/huggingface/transformers/refs/tags/v4.46.1/src/transformers/models/llama/modeling_llama.py"
    if config_src is None:
        _local = os.path.join(_vendor, "configuration_llama.py")
        config_src = _local if os.path.exists(_local) else \
            "https://raw.githubusercontent.com/huggingface/transformers/refs/tags/v4.46.1/src/transformers/models/llama/configuration_llama.py"

    PKG = "transformers.models.llama"
    MOD_MODEL = f"{PKG}.modeling_llama"
    MOD_CONF = f"{PKG}.configuration_llama"

    def _fetch_text(src: str) -> str:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                return f.read()
        try:
            import requests
            r = requests.get(src, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception:
            from urllib.request import urlopen
            with urlopen(src, timeout=30) as r:
                return r.read().decode("utf-8")

    # ensure parent packages are imported BEFORE the snapshot, so the snapshot records
    # their true pre-patch attribute state
    importlib.import_module("transformers")
    importlib.import_module("transformers.models")
    pkg = importlib.import_module(PKG)
    state = _snapshot()

    def _install_module(module_name: str, code_text: str, version_tag: str):
        pseudo_name = f"<restored {module_name} {version_tag}>"
        lines = code_text.splitlines(True)
        linecache.cache[pseudo_name] = (len(code_text), None, lines, pseudo_name)
        m = types.ModuleType(module_name)
        m.__dict__.update({
            "__name__": module_name,
            "__package__": PKG,
            "__file__": pseudo_name,
            "__hf_llama_restored__": version_tag,
        })
        code_obj = compile(code_text, pseudo_name, "exec")
        exec(code_obj, m.__dict__)
        sys.modules[module_name] = m
        setattr(pkg, module_name.rsplit(".", 1)[-1], m)
        return m

    # configuration first: the vendored modeling module resolves `.configuration_llama`
    # at exec time, and installing it second would leave an incoherent pair whose
    # LlamaConfig identities disagree
    if restore_config:
        _install_module(MOD_CONF, _fetch_text(config_src), "v4.46.1")
    _install_module(MOD_MODEL, _fetch_text(modeling_src), "v4.46.1")

    # point the top-level re-exports (`transformers.LlamaModel` etc.) at the vendored
    # classes explicitly: transformers' lazy loader keeps resolution caches that survive
    # a __dict__ pop, so re-resolution cannot be relied on, while a module __dict__ entry
    # always wins over __getattr__. undo() restores every entry from the snapshot.
    vendored_mod = sys.modules[MOD_MODEL]
    vendored_conf = sys.modules.get(MOD_CONF)
    for (holder_name, attr), _snap in state["attrs"].items():
        if attr in ("modeling_llama", "configuration_llama", "llama"):
            continue  # module attrs already set by _install_module / unchanged
        replacement = getattr(vendored_mod, attr, None)
        if replacement is None and vendored_conf is not None:
            replacement = getattr(vendored_conf, attr, None)
        if replacement is None:
            continue
        holder = sys.modules.get(holder_name)
        if holder is not None:
            holder.__dict__[attr] = replacement

    _patch_state = state
    return True


def undo():
    """Restore sys.modules and every snapshotted holder attribute to its pre-apply()
    state; attributes first cached while the patch was active are dropped so the next
    access lazily re-resolves to the installed transformers version."""
    global _patch_state
    import sys

    if _patch_state is None:
        return

    for mod, orig in _patch_state["modules"].items():
        if orig is not None:
            sys.modules[mod] = orig
        else:
            sys.modules.pop(mod, None)

    for (holder_name, attr), (present, value) in _patch_state["attrs"].items():
        holder = sys.modules.get(holder_name)
        if holder is None:
            continue
        if present:
            holder.__dict__[attr] = value
        else:
            holder.__dict__.pop(attr, None)

    _patch_state = None
