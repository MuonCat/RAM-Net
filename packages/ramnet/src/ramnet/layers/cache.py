from typing import Any, Callable, NamedTuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class RAMState(NamedTuple):
    """Recurrent memory contents and mass."""

    slot: torch.Tensor  # (..., H, A, C)
    mass: torch.Tensor  # (..., H, A)


class RAMCacheLayer(CacheLayerMixin):
    """One recurrent state, its physical length, and per-sample CAPE positions."""

    is_compileable = False
    is_sliding = False
    is_croppable = False
    supports_early_init = False
    max_cache_len = -1

    def __init__(self):
        super().__init__()
        self.state: RAMState | None = None
        self.next_position: torch.Tensor | None = None
        self.cache_length = 0
        self.is_initialized = False
        self.device = None

    def lazy_initialization(self, recurrent_state: RAMState) -> None:
        self.device = recurrent_state.slot.device
        self.is_initialized = True

    def update(
            self,
            recurrent_state: RAMState | tuple[torch.Tensor, torch.Tensor],
            offset: int = 1,
            cache_kwargs: dict[str, Any] | None = None,
    ) -> RAMState:
        state = RAMState(*recurrent_state)
        if not self.is_initialized:
            self.lazy_initialization(state)

        # Assign functional outputs directly to preserve autograd across chunks.
        self.state = state
        self.cache_length += offset
        if cache_kwargs is not None:
            self.next_position = cache_kwargs.get('next_position', self.next_position)
        return state

    def get_seq_length(self) -> int:
        return self.cache_length

    def get_mask_sizes(self, query_length: int | torch.Tensor) -> tuple[int, int]:
        if isinstance(query_length, torch.Tensor):
            query_length = query_length.numel()
        return self.cache_length + query_length, 0

    def get_max_cache_shape(self) -> int:
        return -1

    def get_max_length(self) -> int:
        return -1

    @property
    def batch_size(self) -> int:
        return self.state.slot.size(0) if self.state is not None else -1

    @property
    def max_batch_size(self) -> int:
        return self.batch_size

    def _map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if self.state is not None:
            self.state = RAMState(*(fn(value) for value in self.state))
        if self.next_position is not None:
            self.next_position = fn(self.next_position)

    def offload(self) -> None:
        self._map_tensors(lambda value: value.to('cpu', non_blocking=True))

    def prefetch(self) -> None:
        if self.is_initialized:
            self._map_tensors(lambda value: value.to(self.device, non_blocking=True))

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def batch_select_indices(self, indices: torch.LongTensor) -> None:
        self._map_tensors(lambda value: value.index_select(0, indices.to(value.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._map_tensors(lambda value: value.repeat_interleave(repeats, dim=0))

    def crop(self, max_length: int) -> None:
        raise ValueError('RAMCache cannot roll back recurrent states; replay the retained prefix instead.')

    def reset(self) -> None:
        self.state = None
        self.next_position = None
        self.cache_length = 0
        self.is_initialized = False


class RAMCache(Cache):
    """A Hugging Face cache with FLA-style ``recurrent_state`` layer entries.

    Storage is allocated by attention on first use, in the operator's state dtype.
    Cache length counts all processed tokens; CAPE positions count valid tokens.
    """

    def __init__(self) -> None:
        super().__init__(layer_class_to_replicate=RAMCacheLayer)

    def get_layer(self, layer_idx: int) -> RAMCacheLayer:
        if layer_idx < 0:
            raise ValueError('layer_idx must be non-negative.')
        while len(self.layers) <= layer_idx:
            self.layers.append(RAMCacheLayer())
        return self.layers[layer_idx]

    def update(
            self,
            recurrent_state: RAMState | tuple[torch.Tensor, torch.Tensor],
            layer_idx: int,
            offset: int = 1,
            cache_kwargs: dict[str, Any] | None = None,
    ) -> RAMState:
        return self.get_layer(layer_idx).update(recurrent_state, offset, cache_kwargs)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return -1

    def __getitem__(self, layer_idx: int) -> dict[str, RAMState | None]:
        return {'recurrent_state': self.layers[layer_idx].state}

    def __iter__(self):
        for layer_idx in range(len(self.layers)):
            yield self[layer_idx]
