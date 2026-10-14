/**
 * ComfyUI-Chatterbox frontend extension: node-error toasts.
 *
 * Served from the package's WEB_DIRECTORY at /extensions/ComfyUI-Chatterbox/
 * and dynamic-imported at boot by the core extension loader. Plain ESM: no
 * build step, no dependencies.
 */
import { app } from "../../scripts/app.js";

/** Node ids whose execution failures deserve a toast. Kept in sync with the
 *  Python schema by tests/test_frontend_node_set.py. */
const CHATTERBOX_NODE_TYPES = new Set([
    "ChatterboxTTS",
    "ChatterboxMTLTTS",
    "ChatterboxSRT",
    "ChatterboxDialog",
    "ChatterboxVC",
]);

const EXTENSION_NAME = "Chatterbox.ErrorToasts";

/** Toast lifetime and dedupe window, both in milliseconds. */
const TOAST_DURATION = 10000;
const DEDUPE_WINDOW = 3000;

/**
 * Opt-out via the core `Comfy.Extension.Disabled` setting.
 *
 * That setting gates extension *callbacks* (the store invokes only enabled
 * extensions); it does not gate side effects performed at module load. So the
 * list is consulted here, at load. A missing store, a missing key or a throwing
 * read fails open: toasts stay on. Silencing every node error with the same
 * broken read that skipped the registration would be the worse failure.
 */
function isDisabledByName() {
    try {
        const setting = app.extensionManager && app.extensionManager.setting;
        if (!setting || typeof setting.get !== "function") return false;
        const disabled = setting.get("Comfy.Extension.Disabled");
        return Array.isArray(disabled) && disabled.includes(EXTENSION_NAME);
    } catch (error) {
        return false;
    }
}

let shownKey = null;
let shownAt = 0;

/**
 * Show one toast for one execution_error event, honouring the dedupe window.
 * Never throws: a broken toast must not take core error rendering with it.
 */
function showErrorToast(detail) {
    const summary = detail.exception_message || "Chatterbox node failed.";
    const nodeType = detail.node_type || "unknown node";
    const description =
        "Node: " + nodeType +
        (detail.exception_type ? " (" + detail.exception_type + ")" : "");

    const now = Date.now();
    const key = nodeType + "\n" + summary;
    if (key === shownKey && now - shownAt < DEDUPE_WINDOW) {
        return; // the same fault is already on screen
    }
    shownKey = key;
    shownAt = now;

    const toast = app.extensionManager?.toast;
    if (toast && typeof toast.error === "function") {
        // Current API: error(title, options).
        toast.error(summary, {
            description: description,
            duration: TOAST_DURATION,
            closable: true,
        });
    } else if (toast && typeof toast.add === "function") {
        // Deprecated API, still present in older frontends.
        toast.add({
            severity: "error",
            summary: summary,
            detail: description,
            life: TOAST_DURATION,
            closable: true,
        });
    } else {
        console.error("[Chatterbox TTS] Node error:", summary, description);
    }
}

if (!isDisabledByName()) {
    // Registered at module load, never in setup(): a lazily loaded extension
    // may never get setup() called, and a listener registered late silently
    // misses every event that fired before it.
    app.api.addEventListener("execution_error", (event) => {
        try {
            const detail = event && event.detail;
            if (!detail) return;
            if (!CHATTERBOX_NODE_TYPES.has(detail.node_type)) return; // not ours
            showErrorToast(detail);
        } catch (error) {
            console.error("[Chatterbox TTS] Node-error toast failed:", error);
        }
    });
}

app.registerExtension({ name: EXTENSION_NAME });
