# restore_llama_4461_hotpatch.py
def apply(
    force: bool = True,
    restore_config: bool = True,
    modeling_src: str = "https://raw.githubusercontent.com/huggingface/transformers/refs/tags/v4.46.1/src/transformers/models/llama/modeling_llama.py",
    config_src:   str = "https://raw.githubusercontent.com/huggingface/transformers/refs/tags/v4.46.1/src/transformers/models/llama/configuration_llama.py",
):
    """
    Replace installed `transformers.models.llama.modeling_llama` (and optionally
    `configuration_llama`) with the exact v4.46.1 files. Safe to call multiple times.
    """
    import sys, types, importlib, os, linecache

    PKG = "transformers.models.llama"
    MOD_MODEL = f"{PKG}.modeling_llama"
    MOD_CONF  = f"{PKG}.configuration_llama"

    def _fetch_text(src: str) -> str:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                return f.read()
        # try requests, then urllib
        try:
            import requests
            r = requests.get(src, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception:
            from urllib.request import urlopen
            with urlopen(src, timeout=30) as r:
                return r.read().decode("utf-8")

    def _install_module(module_name: str, code_text: str, version_tag: str):
        # If already restored and not forcing, do nothing
        existing = sys.modules.get(module_name)
        if existing and getattr(existing, "__hf_llama_restored__", None) == version_tag:
            return existing
        # Backup once
        if existing and not getattr(existing, "__hf_llama_backup__", False):
            sys.modules[module_name + ".__backup__"] = existing
            setattr(existing, "__hf_llama_backup__", True)

        # Ensure parent packages imported
        importlib.import_module("transformers")
        importlib.import_module("transformers.models")
        pkg = importlib.import_module(PKG)

        # Prepare a deterministic pseudo-filename and preload linecache
        pseudo_name = f"<restored {module_name} {version_tag}>"
        # linecache expects a tuple: (size, mtime, lines_with_newlines, filename)
        lines = code_text.splitlines(True)
        linecache.cache[pseudo_name] = (len(code_text), None, lines, pseudo_name)

        # Create fresh module and exec into it
        m = types.ModuleType(module_name)
        m.__dict__.update({
            "__name__": module_name,
            "__package__": PKG,
            "__file__": pseudo_name,
            "__hf_llama_restored__": version_tag,
        })
        code_obj = compile(code_text, pseudo_name, "exec")
        exec(code_obj, m.__dict__)

        # Register
        sys.modules[module_name] = m
        setattr(pkg, module_name.rsplit(".", 1)[-1], m)
        return m

    # Idempotence short-circuit
    if (not force) and (MOD_MODEL in sys.modules) and getattr(sys.modules[MOD_MODEL], "__hf_llama_restored__", None) == "v4.46.1":
        return True

    # Fetch and install modeling (and optionally config) from v4.46.1
    modeling_code = _fetch_text(modeling_src)
    _install_module(MOD_MODEL, modeling_code, "v4.46.1")

    if restore_config:
        conf_code = _fetch_text(config_src)
        _install_module(MOD_CONF, conf_code, "v4.46.1")

    return True


def undo():
    """Revert to the original modules if backups exist."""
    import sys, importlib
    PKG = "transformers.models.llama"
    for name in ("modeling_llama", "configuration_llama"):
        mod = f"{PKG}.{name}"
        bkp = mod + ".__backup__"
        if bkp in sys.modules:
            sys.modules[mod] = sys.modules[bkp]
            del sys.modules[bkp]
            pkg = importlib.import_module(PKG)
            setattr(pkg, name, sys.modules[mod])
