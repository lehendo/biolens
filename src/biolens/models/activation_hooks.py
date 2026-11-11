"""
Hook-based activation extraction for models that don't support output_hidden_states.

Used by adapters (e.g. Evo 2) where the
architecture requires hooking specific submodules rather than using a built-in
hidden-states flag.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle


class ResidualStreamCapture:
    """Captures the output of a named submodule during a forward pass.

    Example::

        with ResidualStreamCapture(model.encoder.layer[3]) as cap:
            outputs = model(**inputs)
        hidden = cap.output   # (N, L, D)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self.output: Tensor | None = None
        self._handle: RemovableHandle | None = None

    def __enter__(self) -> ResidualStreamCapture:
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        # Most transformer layers return (hidden_states, ...) or just hidden_states.
        if isinstance(output, tuple):
            self.output = output[0].detach()
        else:
            self.output = output.detach()


@contextmanager
def capture_layer_outputs(
    model: torch.nn.Module,
    layer_accessor: Callable[[torch.nn.Module, int], torch.nn.Module],
    layer_indices: list[int],
) -> Generator[dict[int, list[Tensor]], None, None]:
    """
    Context manager that hooks multiple layers simultaneously.

    Args:
        model:           The full model.
        layer_accessor:  Callable(model, layer_idx) → submodule to hook.
                         E.g., for Geneformer: lambda m, i: m.bert.encoder.layer[i]
        layer_indices:   Which layers to capture.

    Yields:
        A dict mapping layer index → list of activation tensors.
        The list is populated during forward passes inside the context.

    Example::

        def accessor(m, i):
            return m.bert.encoder.layer[i]

        with capture_layer_outputs(model, accessor, [3, 7, 11]) as caps:
            for batch in dataloader:
                model(**batch)   # activations accumulate in caps

        # caps[3] is a list of Tensors, one per forward pass
    """
    captures: dict[int, list[Tensor]] = {i: [] for i in layer_indices}
    handles = []

    for idx in layer_indices:
        submodule = layer_accessor(model, idx)

        def make_hook(layer_idx: int) -> Callable:
            def hook(_module: Any, _input: Any, output: Any) -> None:
                acts = output[0] if isinstance(output, tuple) else output
                captures[layer_idx].append(acts.detach())
            return hook

        handles.append(submodule.register_forward_hook(make_hook(idx)))

    try:
        yield captures
    finally:
        for h in handles:
            h.remove()
