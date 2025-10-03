import sys, platform, importlib, json

PKGS = ["torch","transformers","datasets","evaluate","accelerate",
        "sentence_transformers","faiss","fastapi","gradio","sklearn"]

def print_env():
    info = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for p in PKGS:
        try:
            modname = {"sentence_transformers":"sentence_transformers"}.get(p,p)
            m = importlib.import_module(modname)
            ver = getattr(m, "__version__", "unknown")
        except Exception as e:
            ver = f"not_found({e})"
        info[p] = ver
    print("[ENV]", json.dumps(info, ensure_ascii=False))
