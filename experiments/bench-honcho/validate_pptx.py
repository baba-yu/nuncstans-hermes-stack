#!/usr/bin/env python3
"""Validate the .pptx output of an experiment.
Emits a JSON line to stdout: {ok, path, slides, size, error}"""
import sys, os, json, pathlib

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: validate_pptx.py <path>"})); sys.exit(2)
    p = pathlib.Path(sys.argv[1])
    if not p.exists():
        print(json.dumps({"ok": False, "path": str(p), "error": "file not found"})); return
    try:
        from pptx import Presentation
    except ImportError:
        # Fallback: treat as zip and count ppt/slides/slide*.xml
        import zipfile
        try:
            with zipfile.ZipFile(p) as z:
                slides = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            print(json.dumps({"ok": len(slides) == 5, "path": str(p), "slides": len(slides),
                              "size": p.stat().st_size, "method": "zip-fallback"}))
        except Exception as e:
            print(json.dumps({"ok": False, "path": str(p), "error": f"zip-fallback: {e}",
                              "size": p.stat().st_size}))
        return
    try:
        prs = Presentation(str(p))
        n = len(prs.slides)
        print(json.dumps({"ok": n == 5, "path": str(p), "slides": n,
                          "size": p.stat().st_size, "method": "python-pptx"}))
    except Exception as e:
        # Fall back to zip-based slide counting (python-pptx can be overly
        # strict about XML control chars that still open fine in PowerPoint).
        import zipfile
        try:
            with zipfile.ZipFile(p) as z:
                slides = [n for n in z.namelist()
                          if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            print(json.dumps({"ok": len(slides) == 5, "path": str(p),
                              "slides": len(slides), "size": p.stat().st_size,
                              "method": "zip-fallback-after-pptx-error",
                              "pptx_error": str(e)}))
        except Exception as e2:
            print(json.dumps({"ok": False, "path": str(p), "error": str(e),
                              "fallback_error": str(e2), "size": p.stat().st_size}))

if __name__ == "__main__":
    main()
