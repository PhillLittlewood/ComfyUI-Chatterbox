"""V3 node implementations for ComfyUI-Chatterbox.

Each node lives in its own module so the schemas can be read side by side
against ``tests/frozen_widget_snapshot.json``:

* :mod:`nodes.tts_node` -- ``ChatterboxTTS``, speech synthesis.
* :mod:`nodes.mtl_node` -- ``ChatterboxMTLTTS``, multilingual synthesis.
* :mod:`nodes.srt_node` -- ``ChatterboxSRT``, timed subtitles.
* :mod:`nodes.dialog_node` -- ``ChatterboxDialog``, multi-speaker.
* :mod:`nodes.vc_node` -- ``ChatterboxVC``, voice conversion.

:mod:`nodes.common` holds the pieces the schemas share -- the appended
``device`` / ``dtype`` / ``attention_mode`` widgets and the queue-time checks --
so the nodes cannot drift apart. ``dialog_node`` also borrows the multilingual
node's language and checkpoint-version tables rather than restating them.

The import order is the order ComfyUI is handed the nodes in
(:meth:`ChatterboxExtension.get_node_list`), and the order inside each
``define_schema`` is a frozen compatibility contract for the two original nodes
-- see the module docstrings. Nothing here may reorder a widget on either of
them; the four later nodes are new ids and cannot affect a stored value on any
other node.
"""

from .tts_node import ChatterboxTTSNode
from .mtl_node import ChatterboxMTLTTSNode
from .srt_node import ChatterboxSRTNode
from .dialog_node import ChatterboxDialogNode
from .vc_node import ChatterboxVCNode

__all__ = [
    "ChatterboxTTSNode",
    "ChatterboxMTLTTSNode",
    "ChatterboxSRTNode",
    "ChatterboxDialogNode",
    "ChatterboxVCNode",
]