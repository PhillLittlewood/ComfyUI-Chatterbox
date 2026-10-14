/**
 * node:test suite for the shipped js/extension.js. Zero dependencies: the
 * built-in runner, the standard library, nothing else.
 *
 * The extension imports ComfyUI's frontend `app` module, which does not exist
 * outside a browser. So every case stages a throwaway tree in the OS temp dir
 * shaped like a real deployment:
 *
 *     <stage>/scripts/app.js            <- the stub below, configured by stub.json
 *     <stage>/ComfyUI-Chatterbox/js/extension.js   <- a copy of the shipped file
 *
 * The extension is imported *from the staged package path*, so its
 * `import { app } from "../../scripts/app.js"` resolves to the stub two
 * directories up -- the same relative shape the core extension loader serves it
 * under. Each case gets a fresh stage, hence a fresh module instance, hence a
 * fresh listener.
 */
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..");
const SHIPPED_EXTENSION = path.join(REPO_ROOT, "js", "extension.js");

const EXTENSION_NAME = "Chatterbox.ErrorToasts";
const DISABLED_SETTING = "Comfy.Extension.Disabled";
const TOAST_DURATION = 10000;
const DEDUPE_WINDOW = 3000;

// The two messages the ask names, verbatim. Anything that reformats them --
// a rewrap, a dropped article -- breaks the toast the user reads.
const FLASH_MESSAGE =
    "Chatterbox Flash ships no built-in voice -- connect a reference " +
    "audio clip to the node's audio prompt input.";
const VC_MESSAGE =
    "The Chatterbox voice pack 'chatterbox-flash' loaded without a " +
    "conversion model. Use the Chatterbox TTS node for it.";

/**
 * A minimal stand-in for the frontend app module. Its shape comes from
 * stub.json, written next to this file's parent directory by stage(), so one
 * stub plays every delivery branch the extension feature-detects.
 */
const STUB_SOURCE = `import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(readFileSync(join(here, "..", "stub.json"), "utf8"));

const state = {
    listeners: [],
    registered: [],
    modern: [],
    legacy: [],
    settingReads: [],
};

const manager = {};

if (config.toast === "modern") {
    manager.toast = {
        error(title, options) {
            state.modern.push({ title, options });
        },
    };
} else if (config.toast === "legacy") {
    manager.toast = {
        add(options) {
            state.legacy.push(options);
        },
    };
}

if (config.disabled === true) {
    manager.setting = {
        get(key) {
            state.settingReads.push(key);
            return config.disabledEntries;
        },
    };
}

export const app = {
    api: {
        addEventListener(type, handler) {
            state.listeners.push({ type, handler });
        },
    },
    registerExtension(extension) {
        state.registered.push(extension);
    },
    extensionManager: manager,
};

export { state };
`;

/** Stage a throwaway deployment tree and return its paths. */
function stage(config) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "chatterbox-toast-"));
    const packageJs = path.join(root, "ComfyUI-Chatterbox", "js");
    fs.mkdirSync(packageJs, { recursive: true });
    const extensionPath = path.join(packageJs, "extension.js");
    fs.copyFileSync(SHIPPED_EXTENSION, extensionPath);

    const scriptsDir = path.join(root, "scripts");
    fs.mkdirSync(scriptsDir, { recursive: true });
    const stubPath = path.join(scriptsDir, "app.js");
    fs.writeFileSync(stubPath, STUB_SOURCE, "utf8");

    fs.writeFileSync(
        path.join(root, "stub.json"),
        JSON.stringify(config),
        "utf8"
    );
    return { root, stubPath, extensionPath };
}

function discard(staged) {
    fs.rmSync(staged.root, { recursive: true, force: true });
}

/**
 * Import both modules from the staged tree. The extension's own import of the
 * app stub resolves to the same file URL, so the two calls share one module
 * instance and the test sees exactly what the extension saw.
 */
async function load(staged) {
    await import(pathToFileURL(staged.extensionPath).href);
    const stub = await import(pathToFileURL(staged.stubPath).href);
    return { stub };
}

/** Fire a real execution_error event at the recorded listener. */
function dispatch(state, detail) {
    assert.equal(state.listeners.length, 1, "exactly one listener registered");
    const [listener] = state.listeners;
    assert.equal(listener.type, "execution_error");
    listener.handler(new CustomEvent("execution_error", { detail }));
}

/** The payload ComfyUI's execution layer broadcasts for a failed node. */
function chatterboxError(overrides = {}) {
    return {
        prompt_id: "prompt-1",
        node_id: "7",
        node_type: "ChatterboxTTS",
        executed: ["7"],
        exception_message: FLASH_MESSAGE,
        exception_type: "ValueError",
        traceback: ["Traceback (most recent call last):", "ValueError: ..."],
        current_inputs: { text: "Hello.", audio_prompt: null },
        current_outputs: {},
        ...overrides,
    };
}

function frozenClock(start) {
    const real = Date.now;
    let now = start;
    Date.now = () => now;
    return {
        advance(ms) {
            now += ms;
        },
        restore() {
            Date.now = real;
        },
    };
}

test("1. the execution_error listener is registered at module load", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    // Nothing has called setup() -- nothing in this harness can -- yet the
    // listener is already there, and a dispatch right now already toasts. A
    // listener registered lazily would miss every event that fired first.
    assert.equal(stub.state.listeners.length, 1);
    assert.equal(stub.state.listeners[0].type, "execution_error");
    assert.equal(stub.state.modern.length, 0, "importing must not toast");

    dispatch(stub.state, chatterboxError());
    assert.equal(stub.state.modern.length, 1);
});

test("2. a Chatterbox failure toasts the message verbatim with node and type", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    dispatch(stub.state, chatterboxError());

    assert.equal(stub.state.legacy.length, 0, "the deprecated branch must not fire");
    assert.equal(stub.state.modern.length, 1);
    const { title, options } = stub.state.modern[0];
    assert.equal(title, FLASH_MESSAGE);
    assert.ok(options.description.includes("ChatterboxTTS"), options.description);
    assert.ok(options.description.includes("ValueError"), options.description);
    assert.equal(options.closable, true);
    assert.equal(options.duration, TOAST_DURATION);
});

test("3. a non-Chatterbox node_type produces no toast", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    dispatch(
        stub.state,
        chatterboxError({ node_type: "SomeOtherPackNode", node_id: "99" })
    );

    assert.equal(stub.state.modern.length, 0, "another pack's failure is not ours");
    assert.equal(stub.state.legacy.length, 0);
});

test("4. an identical message is deduped inside the window and re-toasts outside it", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    const clock = frozenClock(0);
    try {
        dispatch(stub.state, chatterboxError());
        assert.equal(stub.state.modern.length, 1);

        dispatch(stub.state, chatterboxError());
        assert.equal(stub.state.modern.length, 1, "the same fault twice in a row");

        clock.advance(DEDUPE_WINDOW - 1);
        dispatch(stub.state, chatterboxError());
        assert.equal(stub.state.modern.length, 1, "still inside the window");

        clock.advance(1); // exactly at the boundary: not inside it
        dispatch(stub.state, chatterboxError());
        assert.equal(stub.state.modern.length, 2, "past the window it toasts again");

        // The key names the node, so the same message on a different node is a
        // different fault and is never suppressed.
        dispatch(stub.state, chatterboxError({ node_type: "ChatterboxVC" }));
        assert.equal(stub.state.modern.length, 3, "a different node is not a duplicate");
    } finally {
        clock.restore();
    }
});

test("5. without toast.error the deprecated toast.add is used", async (t) => {
    const staged = stage({ toast: "legacy" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    dispatch(stub.state, chatterboxError());

    assert.equal(stub.state.modern.length, 0, "the modern branch must not fire");
    assert.equal(stub.state.legacy.length, 1);
    const call = stub.state.legacy[0];
    assert.equal(call.severity, "error");
    assert.equal(call.summary, FLASH_MESSAGE);
    assert.ok(call.detail.includes("ChatterboxTTS"), call.detail);
    assert.ok(call.detail.includes("ValueError"), call.detail);
    assert.equal(call.life, TOAST_DURATION);
    assert.equal(call.closable, true);
});

test("6. without a toast manager it logs to the console and never throws", async (t) => {
    const staged = stage({ toast: "none" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    const logged = [];
    const realError = console.error;
    console.error = (...args) => logged.push(args);
    try {
        dispatch(stub.state, chatterboxError());
        // Not our node: filtered out before any delivery, so no log either.
        dispatch(stub.state, chatterboxError({ node_type: "SomeOtherPackNode" }));
        // A malformed event must not throw either.
        dispatch(stub.state, null);
    } finally {
        console.error = realError;
    }

    assert.equal(logged.length, 1, "only the Chatterbox failure logs");
    assert.match(String(logged[0][0]), /\[Chatterbox TTS\] Node error:/);
    assert.ok(String(logged[0][1]).includes(FLASH_MESSAGE));
    assert.ok(String(logged[0][2]).includes("ChatterboxTTS"));
});

test("7. the extension registers under its own name", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    assert.deepEqual(stub.state.registered, [{ name: EXTENSION_NAME }]);
});

test("8. both named messages pass through verbatim as the toast title", async (t) => {
    const staged = stage({ toast: "modern" });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    dispatch(stub.state, chatterboxError());
    dispatch(
        stub.state,
        chatterboxError({
            node_type: "ChatterboxVC",
            exception_message: VC_MESSAGE,
        })
    );

    assert.deepEqual(
        stub.state.modern.map((entry) => entry.title),
        [FLASH_MESSAGE, VC_MESSAGE]
    );
});

test("9. naming the extension in Comfy.Extension.Disabled skips the listener", async (t) => {
    const staged = stage({
        toast: "modern",
        disabled: true,
        disabledEntries: [EXTENSION_NAME],
    });
    t.after(() => discard(staged));
    const { stub } = await load(staged);

    // The setting was really read -- the skip is the opt-out, not a shortcut.
    assert.deepEqual(stub.state.settingReads, [DISABLED_SETTING]);
    assert.equal(stub.state.listeners.length, 0, "a disabled extension registers nothing");
    assert.deepEqual(
        stub.state.registered,
        [{ name: EXTENSION_NAME }],
        "the name still registers, so it appears in the core disabled list"
    );
});
