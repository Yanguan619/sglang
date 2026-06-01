from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadataBuilder,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    allocate_and_compress_keys,
    compressed_attention,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


logger = logging.getLogger(__name__)


@dataclass
class ForwardMetadata:
    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None

    # MiniCPM sparse attention fields
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    cache_seqlens_int32: Optional[torch.Tensor] = None
    max_seq_len_q: int = 1

    # MiniCPM sparse attention — compression metadata (k1/k2)
    k1: Optional[object] = None
    k2: Optional[object] = None

    # MiniCPM sparse attention — prefill
    sparse_bs_list: Optional[List[int]] = None
    seqlen_k_sparse_bs_tensor: Optional[torch.Tensor] = None
    token_to_bs: Optional[torch.Tensor] = None
    token_pos_in_bs: Optional[torch.Tensor] = None
    sparse_page_table: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q_cpu: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q: Optional[torch.Tensor] = None
    old_bs_to_new_bs_range: Optional[List[int]] = None
    sparse_max_seq_len_q: int = 1

    # MiniCPM sparse attention — decode
    sparse_cache_seqlens_int32: Optional[torch.Tensor] = None
    sparse_cu_seqlens_k: Optional[torch.Tensor] = None

    # MiniCPM sparse attention — stage1 optimisation / adjusted metadata
    cache_seqlens_int32_stage1: Optional[torch.Tensor] = None
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: Optional[int] = None


class AscendAttnMaskBuilder:
    def __init__(self, model_runner: ModelRunner, device, use_fia, use_mla):
        """
        Initialize the AscendAttnMaskBuilder class.

        :param model_runner: ModelRunner instance for model execution.
        :param device: Device to run the model on (e.g., 'cuda', 'npu').
        :param use_fia: Boolean flag to indicate if environment variable ASCEND_USE_FIA is set to 1.
        """
        self.use_fia = use_fia
        self.model_runner = model_runner
        self.device = device

        # Initialize mask
        mask_len = 128
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(self.device)

        # Initialize FIA mask
        fia_mask_len = 2048
        self.fia_mask = self.generate_mask_flag(fia_mask_len).to(self.device)

        # Initialize MTP mask
        mtp_mask_len = 2048
        self.mtp_mask = self.generate_mask_flag(mtp_mask_len).to(self.device)

        # Initialize mixed chunk mask cache
        mixed_mask_len = 2048
        self.mixed_chunk_attn_mask = self.get_splitfuse_attn_mask(mixed_mask_len)

        if use_mla:
            # Initialize RingMla mask
            ringmla_mask_len = 512
            self.ringmla_mask = self.generate_attn_mask(
                ringmla_mask_len, "norm", torch.bfloat16
            ).to(self.device)

    @staticmethod
    def generate_mask_flag(max_seq_len):
        """
        Generate a mask flag for attention masks.

        :param max_seq_len: Maximum sequence length for the mask.
        :return: A boolean tensor representing the mask flag.
        """
        # Construct lower triangle matrix.
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Create upper triangle matrix used to mark mask positions.
        mask_flag = ~mask_flag
        return mask_flag

    @staticmethod
    def generate_attn_mask(max_seq_len, mode, dtype=torch.float16):
        """
        Generate an attention mask.

        :param max_seq_len: Maximum sequence length for the mask.
        :param mode: Mode of the mask ('mix' or 'norm').
        :param dtype: Data type of the mask tensor.
        :return: A tensor representing the attention mask.
        """
        mask_flag = AscendAttnMaskBuilder.generate_mask_flag(max_seq_len)
        if mode == "mix":
            mask_value = float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
        else:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        attn_mask = (
            torch.zeros(size=(max_seq_len, max_seq_len))
            .masked_fill_(mask_flag, mask_value)
            .to(dtype)
        )
        return attn_mask

    @staticmethod
    def get_attention_mask_id(seq_lens, extend_lens):
        """
        Generate attention mask IDs based on sequence lengths and extended lengths.

        :param seq_lens: Sequence lengths.
        :param extend_lens: Extended lengths.
        :return: A tensor containing the attention mask IDs.
        """
        starts = seq_lens - extend_lens
        ends = seq_lens

        # Use torch.stack to stack the start and end indices together
        ranges = torch.stack((starts, ends), dim=-1)

        # Use list comprehension to generate tensors for each range and concatenate them
        attn_mask_id = torch.cat([torch.arange(start, end) for start, end in ranges])
        return attn_mask_id

    def update_attn_cache(
        self,
        seqlen: int,
        mask_cache: torch.Tensor,
        seq_len_cached: int,
        dtype: torch.dtype,
        mode,
    ):
        """
        Update the attention mask cache.

        :param seqlen: Maximum sequence length.
        :param mask_cache: Current attention mask cache.
        :param seq_len_cached: Cached sequence length.
        :param dtype: Data type of the mask tensor.
        :param mode: Mode of the mask ('mix' or 'norm').
        :return: Updated mask cache and sequence length cache.
        """
        if seqlen > seq_len_cached:
            seq_len_cached = seqlen
            mask_cache = self.generate_attn_mask(seqlen, mode, dtype)
        if mask_cache.dtype != dtype:
            mask_cache = mask_cache.to(dtype)
        return mask_cache, seq_len_cached

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :return: A tensor representing the splitfuse attention mask.
        """
        attn_mask = (
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1).to(torch.int8).to(self.device)
        )
        return attn_mask


class AscendAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.q_head_dim = self.qk_rope_head_dim + model_runner.model_config.qk_nope_head_dim
        self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.speculative_num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.ascend_attn_mask_builder = AscendAttnMaskBuilder(
            model_runner, self.device, self.use_fia, self.use_mla
        )
        self.mask, self.fia_mask, self.mtp_mask, self.mix_mask = (
            self.ascend_attn_mask_builder.mask,
            self.ascend_attn_mask_builder.fia_mask,
            self.ascend_attn_mask_builder.mtp_mask,
            self.ascend_attn_mask_builder.mixed_chunk_attn_mask,
        )
        if self.use_mla:
            self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask

        # MiniCPM sparse attention
        hf_config = getattr(model_runner.model_config, "hf_config", None)
        self.has_minicpm_sparse = hf_config is not None and getattr(
            hf_config, "has_sparse_attention", False
        )
        if self.has_minicpm_sparse:
            logger.info("======== Detected minicpm sparse.")
            self.kernel_size = hf_config.sparse_kernel_size
            self.kernel_stride = hf_config.sparse_kernel_stride
            self.block_size = hf_config.sparse_block_size
            self.window_size = hf_config.sparse_window_size
            self.sparse_topk = hf_config.sparse_topk + (self.window_size // self.block_size)
            self.num_sparse_topk_tokens = self.block_size * self.sparse_topk
            self.dense_len = hf_config.sparse_dense_len
            self.init_blocks = hf_config.sparse_init_blocks
            self.local_blocks = self.window_size // self.block_size
            self.k1_kernel_size = self.kernel_size
            self.k1_kernel_stride = self.kernel_stride
            self.k2_kernel_size = self.kernel_size * 4
            self.k2_kernel_stride = self.kernel_stride * 4

            sparse_config = SparseConfig.from_model_config(hf_config, model_runner.model_config)
            self.sparse_batch_analyzer = SparseBatchAnalyzer(sparse_config)
            self.sparse_metadata_builder = SparseMetadataBuilder(
                sparse_config,
                num_kv_heads=model_runner.model_config.num_key_value_heads
                // get_tensor_model_parallel_world_size(),
                max_context_len=self.max_context_len,
            )
            self.max_sparse_pages = (
                self.num_sparse_topk_tokens + self.page_size - 1
            ) // self.page_size
            self.pages_per_block = self.block_size // self.page_size  # 0 if block < page

            # Additional sparse config for batch analysis
            self.dense_as_sparse = True
            self.split_stage1 = model_runner.server_args.split_stage1
            self.head_dim = model_runner.model_config.head_dim
            self.head_group_num = model_runner.model_config.num_key_value_heads
            tp_q = model_runner.model_config.num_attention_heads
            self.heads_per_group = tp_q // self.head_group_num

            self.req_to_sparse_k1_token = getattr(
                model_runner.req_to_token_pool, "req_to_sparse_k1_token", None
            )
            self.req_to_sparse_k2_token = getattr(
                model_runner.req_to_token_pool, "req_to_sparse_k2_token", None
            )

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        seq_lens_max = forward_batch.seq_lens.max()
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        self.forward_metadata.block_tables = (
            forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :seq_lens_max
            ][:, :: self.page_size]
            // self.page_size
        )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens_cpu_int = (
                forward_batch.extend_seq_lens.cpu().int()
            )
        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens

        if (
            self.use_mla
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            and not forward_batch.forward_mode.is_target_verify()
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        ):
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to("cpu")
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(forward_batch.req_pool_indices.tolist(), seq_prefix_lens):
                req_indices = forward_batch.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = req_indices[:seq_len][:: self.page_size] // self.page_size
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        if self.has_minicpm_sparse:
            self._init_minicpm_sparse_metadata(forward_batch)
            self.update_batch_for_sparse(forward_batch, self.forward_metadata)

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        metadata = ForwardMetadata()

        metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        metadata.seq_lens = seq_lens
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i * 1 for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ):
        metadata = self.graph_metadata[bs]
        max_len = seq_lens_cpu[:bs].max().item()
        if forward_mode.is_target_verify():
            max_len += self.speculative_num_draft_tokens
        max_seq_pages = (max_len + self.page_size - 1) // self.page_size

        metadata.block_tables[:bs, :max_seq_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
            // self.page_size
        )
        metadata.block_tables[:bs, max_seq_pages:].fill_(0)
        metadata.block_tables[bs:, :].fill_(0)
        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])

        self.forward_metadata = metadata

        self.graph_mode = True

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def do_cp_balance_attn(
        self,
        q_nope,
        k_nope,
        q_pe,
        k_pe,
        topk_indices,
        layer,
        actual_seq_qlen,
        actual_seq_lengths_kv,
    ):
        seq_len = q_nope.shape[0]
        split_len = (seq_len + 1) // 2
        q_nope_prev, q_nope_next = torch.split(q_nope, split_len, dim=0)
        q_rope_prev, q_rope_next = torch.split(q_pe, split_len, dim=0)
        q_nope_prev = q_nope_prev.contiguous()
        q_nope_next = q_nope_next.contiguous()
        q_rope_prev = q_rope_prev.contiguous()
        q_rope_next = q_rope_next.contiguous()
        topk_indices_prev, topk_indices_next = topk_indices

        actual_seq_qlen_prev, actual_seq_qlen_next = actual_seq_qlen
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        attn_out_prev = torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope_prev,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_prev,
            key_rope=k_pe,
            sparse_indices=topk_indices_prev,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        attn_out_next = torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope_next,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_next,
            key_rope=k_pe,
            sparse_indices=topk_indices_next,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        return torch.cat([attn_out_prev, attn_out_next], dim=0)

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        if is_prefill:
            if self.forward_metadata.actual_seq_lengths_q is not None:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q
            else:
                actual_seq_qlen = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            if self.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    actual_seq_qlen = (
                        torch.arange(
                            self.speculative_num_draft_tokens,
                            self.speculative_num_draft_tokens + q.shape[0],
                            self.speculative_num_draft_tokens,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                elif forward_batch.forward_mode.is_draft_extend():
                    actual_seq_qlen = (
                        forward_batch.extend_seq_lens.cumsum().to(q.device).to(torch.int32)
                    )
                else:
                    actual_seq_qlen = torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = self.forward_metadata.actual_seq_lengths_kv
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens

        if is_prefill and is_nsa_enable_prefill_cp() and forward_batch.nsa_cp_metadata is not None:
            attn_out = self.do_cp_balance_attn(
                q_nope,
                k_nope,
                q_pe,
                k_pe,
                topk_indices,
                layer,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            )
        else:
            attn_out = torch.ops.custom.npu_sparse_flash_attention(
                query=q_nope,
                key=k_nope,
                value=k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                sparse_indices=topk_indices,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
            )

        return attn_out

    def _init_minicpm_sparse_metadata(self, forward_batch: ForwardBatch):
        """Build base MiniCPM sparse metadata in forward_metadata.

        This stores the basic metadata (cu_seqlens, cache_seqlens, page_table).
        K1/K2 compression and sparse batch analysis are done in update_batch_for_sparse.
        """
        m = self.forward_metadata
        bs = forward_batch.batch_size
        device = self.device
        seqlens = forward_batch.seq_lens

        m.cache_seqlens_int32 = seqlens.to(torch.int32)
        m.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
        m.cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=device)

        if forward_batch.forward_mode.is_decode_or_idle():
            m.max_seq_len_q = 1
        elif forward_batch.forward_mode.is_extend():
            m.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
            m.cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(forward_batch.extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )

        m.page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seq_len_k
        ]

    def update_batch_for_sparse(self, forward_batch: ForwardBatch, metadata: ForwardMetadata):
        """Build sparse batch metadata after base metadata is initialised.

        This mirrors MiniCPMSparseBackend.update_batch_for_sparse and
        covers both prefill and decode paths: K1/K2 compression, sparse batch
        sequence lengths, page tables, token mappings, and stage1 adjustments.
        """
        cu_seqlens_q = metadata.cu_seqlens_q

        compression_metadata = self.sparse_metadata_builder.build_k1_k2_compression_metadata(
            forward_batch=forward_batch,
            base_metadata=metadata,
            req_to_sparse_k1_token=self.req_to_sparse_k1_token,
            req_to_sparse_k2_token=self.req_to_sparse_k2_token,
            k1_kernel_size=self.k1_kernel_size,
            k1_kernel_stride=self.k1_kernel_stride,
            k2_kernel_size=self.k2_kernel_size,
            k2_kernel_stride=self.k2_kernel_stride,
            cu_seqlens_q=cu_seqlens_q,
        )

        metadata.k1 = compression_metadata["k1"]
        metadata.k2 = compression_metadata["k2"]

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.sparse_bs_list = self.sparse_batch_analyzer.identify_sparse_batches(
                forward_batch, self.dense_as_sparse
            )

            seqlen_q_sparse_bs, metadata.seqlen_k_sparse_bs_tensor = (
                self.sparse_metadata_builder.build_sequence_lengths(
                    cu_seqlens_q,
                    forward_batch.extend_prefix_lens,
                    metadata.sparse_bs_list,
                )
            )

            cu_seqlens_q_sparse_bs = torch.tensor(
                [0] + seqlen_q_sparse_bs, dtype=torch.int32, device=cu_seqlens_q.device
            ).cumsum(dtype=torch.int32, dim=0)

            extend_prefix_lens_sparse = torch.tensor(
                [forward_batch.extend_prefix_lens_cpu[bs] for bs in metadata.sparse_bs_list],
                dtype=torch.long,
                device="cpu",
            )

            metadata.token_to_bs, metadata.token_pos_in_bs = (
                self.sparse_metadata_builder.build_token_mappings(
                    cu_seqlens_q_sparse_bs,
                    extend_prefix_lens_sparse,
                    seqlen_q_sparse_bs,
                )
            )
            metadata.token_to_bs = metadata.token_to_bs.to(device=metadata.cu_seqlens_q.device)
            metadata.token_pos_in_bs = metadata.token_pos_in_bs.to(
                device=metadata.cu_seqlens_q.device
            )

            prefill_metadata = self.sparse_metadata_builder.build_sparse_prefill_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                sparse_bs_list=metadata.sparse_bs_list,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
                cu_seqlens_q=cu_seqlens_q,
                sparse_page_table_dtype=metadata.page_table.dtype,
                sparse_page_table_device=metadata.page_table.device,
            )

            metadata.sparse_page_table = prefill_metadata["sparse_page_table"]
            metadata.sparse_cu_seqlens_q_cpu = prefill_metadata["sparse_cu_seqlens_q_cpu"]
            metadata.sparse_cu_seqlens_q = prefill_metadata["sparse_cu_seqlens_q"]
            metadata.old_bs_to_new_bs_range = prefill_metadata["old_bs_to_new_bs_range"]
            metadata.sparse_max_seq_len_q = prefill_metadata["sparse_max_seq_len_q"]

            forward_batch.sparse_batch_size = len(metadata.sparse_bs_list)
            forward_batch.sparse_idx = prefill_metadata["sparse_idx"]

            # Stage1 optimisation metadata for prefill mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            seqlens_q_sparse_list = []
            for i in range(forward_batch.batch_size):
                if forward_batch.seq_lens_cpu[i] >= self.dense_len:
                    seqlens_q_sparse_list.append(forward_batch.extend_seq_lens_cpu[i])

            seqlen_q_sparse_tensor = torch.tensor(
                seqlens_q_sparse_list, dtype=torch.int32, device=metadata.cu_seqlens_q.device
            )
            cu_seqlen_q_sparse_tensor = torch.nn.functional.pad(
                torch.cumsum(seqlen_q_sparse_tensor, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.cu_seqlens_q_adjusted = cu_seqlen_q_sparse_tensor * self.heads_per_group
            metadata.max_seqlen_q_adjusted = (
                seqlen_q_sparse_tensor.max().item() * self.heads_per_group
            )
        else:
            decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )

            metadata.sparse_cache_seqlens_int32 = decode_metadata["sparse_cache_seqlens_int32"]
            metadata.sparse_cu_seqlens_k = decode_metadata["sparse_cu_seqlens_k"]
            metadata.sparse_cu_seqlens_q = decode_metadata["sparse_cu_seqlens_q"]
            metadata.sparse_page_table = decode_metadata["sparse_page_table"]
            metadata.token_to_bs = decode_metadata["token_to_bs"]

            # Stage1 optimisation metadata for decode mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            metadata.cu_seqlens_q_adjusted = metadata.cu_seqlens_q * self.heads_per_group
            metadata.max_seqlen_q_adjusted = metadata.max_seq_len_q * self.heads_per_group

    def _minicpm_sparse_to_npu_block_table(self, topk_idx, page_table, batch_size, max_seq_len_k):
        """Convert MiniCPM top-k block indices to per-head-group NPU block_table.

        topk_idx: [num_heads, num_tokens, topk]  block-level indices (all heads)
        page_table: [batch, max_seq_len]  token-level positions from req_to_token

        Splits topk_idx into 2 head groups internally so each group gets its own
        sparse page selection.  Returns 2 × batch rows.

        Returns:
            block_table: [batch * 2, max_sparse_pages] per-head-group page indices
            ctx_lens: [batch * 2] per-head-group number of sparse tokens
        """
        num_heads, total_q, topk = topk_idx.shape
        half_heads = num_heads // 2
        max_sparse_pages = self.max_sparse_pages
        tokens_per_block = self.block_size

        block_table = torch.zeros(
            batch_size * 2, max_sparse_pages, dtype=torch.int32, device=self.device
        )
        ctx_lens = torch.zeros(batch_size * 2, dtype=torch.int32, device=self.device)

        for b in range(batch_size):
            for gi, h_start in enumerate([0, half_heads]):
                h_end = h_start + half_heads
                group_blocks = topk_idx[h_start:h_end, b, :]  # [half_heads, topk]
                blocks_flat = group_blocks.unique()
                blocks_flat = blocks_flat[blocks_flat >= 0]

                if blocks_flat.numel() == 0:
                    continue

                page_positions = blocks_flat * tokens_per_block // self.page_size

                unique_pages = torch.unique(page_positions)
                n_pages = unique_pages.numel()
                if n_pages == 0:
                    continue
                n_pages = min(n_pages, max_sparse_pages)

                logical_page_offsets = unique_pages[:n_pages].to(torch.int64) * self.page_size
                physical_pages = page_table[b, logical_page_offsets]
                physical_pages = torch.where(
                    logical_page_offsets < max_seq_len_k,
                    physical_pages // self.page_size,
                    torch.zeros_like(physical_pages),
                )

                out_idx = b * 2 + gi
                block_table[out_idx, :n_pages] = physical_pages.to(torch.int32)
                ctx_lens[out_idx] = n_pages * self.page_size

        return block_table, ctx_lens

    def _compute_sparse_cache_lens(self, m, max_seq_len_k):
        """Compute cache_lens for stage1 sparse top-k optimisation.

        Mirrors MiniCPMSparseBackend.sparse_get_topk_impl logic.
        """
        if m.cache_seqlens_int32_stage1 is not None:
            return m.cache_seqlens_int32_stage1
        if max_seq_len_k > m.max_seq_len_q:
            if m.max_seq_len_q == 1:
                return m.cache_seqlens_int32 - 1
            seq_lens_k = m.cu_seqlens_k[1:] - m.cu_seqlens_k[:-1]
            seq_lens_q = m.cu_seqlens_q[1:] - m.cu_seqlens_q[:-1]
            return seq_lens_k - seq_lens_q
        batch_size = m.cu_seqlens_q.shape[0] - 1
        return torch.zeros(batch_size, dtype=torch.int32, device=m.cu_seqlens_q.device)

    def forward_minicpm_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Forward for MiniCPM sparse attention on NPU.

        Pipeline:
          1. Save KV cache
          2. Compute compressed keys (k1/k2) → topk block indices (stage1 opt)
          3. Build per-head-group NPU block_table from topk indices
          4. Gather sparse KV pages per group; handle dense/sparse mixed batch
          5. Call npu_fused_infer_attention_score with TND layout (2× batch)
        """
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        m = self.forward_metadata
        bs = forward_batch.batch_size

        is_prefill = forward_batch.forward_mode.is_extend()
        max_seq_len_k = forward_batch.seq_lens_cpu.max().item()

        token_pt = self.req_to_token[forward_batch.req_pool_indices, :max_seq_len_k]

        q_full = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        # --- Step 1: Compute topk block indices ---
        k1_token_nums = (
            sum(m.k1.total_compress_token_nums[i].item() for i in range(bs))
            if hasattr(m, "k1") and m.k1 is not None
            else 0
        )
        k2_token_nums = (
            sum(m.k2.total_compress_token_nums[i].item() for i in range(bs))
            if hasattr(m, "k2") and m.k2 is not None
            else 0
        )
        logger.info(
            f"MiniCPM sparse attention: k1_token_nums={k1_token_nums}, k2_token_nums={k2_token_nums}"
        )
        if k1_token_nums == 0:
            if self.graph_mode:
                return self.forward_decode_graph(q, k, v, layer, forward_batch, False)
            attn_output = torch.empty(
                q_full.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                device=self.device,
                dtype=q.dtype,
            )
            key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if self.forward_metadata.seq_lens_cpu_int is not None:
                context_lens = self.forward_metadata.seq_lens_cpu_int
            else:
                context_lens = forward_batch.seq_lens.to(torch.int32)
            torch_npu._npu_paged_attention(
                query=q_full,
                key_cache=key_cache,
                value_cache=value_cache,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                scale_value=layer.scaling,
                block_table=self.forward_metadata.block_tables,
                context_lens=context_lens,
                out=attn_output,
            )
            return attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        # Compress keys
        full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
            layer=layer,
            forward_batch=forward_batch,
            metadata=m,
            k1_token_nums=k1_token_nums,
            k2_token_nums=k2_token_nums,
            dtype=torch.bfloat16,
            device=self.device,
            max_context_length=self.max_context_len,
            split_stage1=True,
        )

        # --- Stage1 optimisation: compute cache_lens ---
        cache_lens = self._compute_sparse_cache_lens(m, max_seq_len_k)

        # Compute compressed attention → topk block indices
        topk_idx = compressed_attention(
            q_full,
            full_compressed_k1,
            full_compressed_k2,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.sparse_topk,
            m.cu_seqlens_q,
            m.k1.cu_seqlens,
            m.k2.cu_seqlens,
            m.max_seq_len_q,
            self.max_context_len,
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            cu_seqlens_q_adjusted=m.cu_seqlens_q_adjusted,
            max_seqlen_q_adjusted=m.max_seqlen_q_adjusted,
            split_stage1=self.split_stage1,
        )

        # --- Step 2: Build per-head-group sparse block table ---
        # topk_idx: [num_heads, total_q, topk]; _minicpm_sparse_to_npu_block_table
        # internally splits into 2 head groups → [bs * 2, max_sparse_pages]
        block_table, ctx_lens = self._minicpm_sparse_to_npu_block_table(
            topk_idx, token_pt, bs, max_seq_len_k
        )

        # --- Step 3: Reshape q for head groups (interleaved → blocked layout) ---
        # q_full: [total_q, tp_q_head_num, head_dim]
        # After split: [total_q, 2, tp_q_head_num // 2, head_dim]
        total_q_len = q_full.shape[0]
        half_heads = layer.tp_q_head_num // 2
        q_grouped = q_full.reshape(total_q_len, 2, half_heads, layer.head_dim)
        # Rearrange from interleaved [g0_t0, g1_t0, g0_t1, g1_t1, ...]
        # to blocked [g0_t0, g0_t1, ..., g1_t0, g1_t1, ...]
        # matching block_table layout (group0 rows, then group1)
        q_g0 = q_grouped[:, 0, :, :].contiguous()  # [total_q, half_heads, head_dim]
        q_g1 = q_grouped[:, 1, :, :].contiguous()  # [total_q, half_heads, head_dim]
        q_tnd = torch.cat([q_g0, q_g1], dim=0)  # [total_q * 2, half_heads, head_dim]

        # --- Step 4: Dense / sparse mixed batch detection ---
        has_sparse_bs_list = hasattr(m, "sparse_bs_list") and m.sparse_bs_list is not None
        if has_sparse_bs_list and len(m.sparse_bs_list) < bs:
            is_sparse_mask = torch.zeros(bs, dtype=torch.bool, device=self.device)
            is_sparse_mask[m.sparse_bs_list] = True
        else:
            is_sparse_mask = torch.ones(bs, dtype=torch.bool, device=self.device)

        # --- Step 5: Handle dense/sparse mixed batch — dense rows in block_table ---
        if has_sparse_bs_list and len(m.sparse_bs_list) < bs:
            dense_bs_list = [i for i in range(bs) if i not in m.sparse_bs_list]
            for dense_bs in dense_bs_list:
                kv_len = forward_batch.seq_lens_cpu[dense_bs]
                for gi in range(2):
                    out_idx = dense_bs * 2 + gi
                    n_pages = kv_len // self.page_size
                    n_pages = min(n_pages, self.max_sparse_pages)
                    if n_pages == 0:
                        continue
                    dense_pages = (
                        self.req_to_token[
                            forward_batch.req_pool_indices[dense_bs : dense_bs + 1], :kv_len
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    block_table[out_idx, :n_pages] = dense_pages[0, :n_pages].to(torch.int32)
                    ctx_lens[out_idx] = n_pages * self.page_size

            # Note: q_tnd is already in blocked layout ([g0_all_tokens, g1_all_tokens])
            # matching the TND cu_seqlens layout, so no rearrangement needed.

        # --- Step 6: Gather sparse (or dense) KV pages per-group ---
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        )

        k_chunks = []
        v_chunks = []
        all_lens = []
        for b in range(bs):
            for gi in range(2):
                out_idx = b * 2 + gi
                if is_sparse_mask[b]:
                    n_valid = min(ctx_lens[out_idx].item() // self.page_size, self.max_sparse_pages)
                else:
                    n_valid = forward_batch.seq_lens_cpu[b] // self.page_size
                if n_valid == 0:
                    all_lens.append(0)
                    continue
                if is_sparse_mask[b]:
                    pages = block_table[out_idx, :n_valid].long()
                else:
                    pages_block = (
                        self.req_to_token[
                            forward_batch.req_pool_indices[b : b + 1],
                            : forward_batch.seq_lens_cpu[b],
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    pages = pages_block[0, :n_valid].long()

                k_gather = k_cache[pages]
                v_gather = v_cache[pages]
                actual_tokens = n_valid * self.page_size
                k_chunks.append(
                    k_gather.reshape(-1, layer.tp_k_head_num, layer.head_dim)[:actual_tokens]
                )
                v_chunks.append(
                    v_gather.reshape(-1, layer.tp_v_head_num, layer.head_dim)[:actual_tokens]
                )
                all_lens.append(actual_tokens)

        k_sparse = (
            torch.cat(k_chunks, dim=0)
            if k_chunks
            else k_cache.new_empty(0, layer.tp_k_head_num, layer.head_dim)
        )
        v_sparse = (
            torch.cat(v_chunks, dim=0)
            if v_chunks
            else v_cache.new_empty(0, layer.tp_v_head_num, layer.head_dim)
        )
        all_lens_t = torch.tensor(all_lens, dtype=torch.int32, device=self.device)

        # --- Step 7: Fused attention with TND layout, 2× batch ---
        if is_prefill:
            extend_lens = forward_batch.extend_seq_lens
            actual_seq_lengths_g0 = torch.cumsum(extend_lens, dim=0).to(torch.int32)
            actual_seq_lengths_g1 = (
                actual_seq_lengths_g0 + actual_seq_lengths_g0[-1]
                if bs > 0
                else actual_seq_lengths_g0
            )
            actual_seq_lengths = torch.cat([actual_seq_lengths_g0, actual_seq_lengths_g1], dim=0)
        else:
            actual_seq_lengths = torch.arange(1, bs * 2 + 1, dtype=torch.int32, device=self.device)

        actual_seq_lengths_kv = torch.cumsum(all_lens_t, dim=0)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_tnd,
            k_sparse,
            v_sparse,
            num_heads=half_heads,
            num_key_value_heads=layer.tp_k_head_num // 2,
            input_layout="TND",
            scale=layer.scaling,
            next_tokens=0,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )

        # --- Step 8: Reshape output back from head-group layout ---
        # attn_output: [total_q * 2, half_heads, v_head_dim] (blocked layout)
        # Split into group0 / group1 and interleave back to original head layout
        attn_g0, attn_g1 = attn_output[:total_q_len, :, :], attn_output[total_q_len:, :, :]
        result = torch.stack([attn_g0, attn_g1], dim=1).reshape(
            total_q_len, layer.tp_q_head_num, layer.v_head_dim
        )

        return result.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if self.has_minicpm_sparse and not self.use_mla:
            # logger.info("======== forward_extend => forward_minicpm_sparse")
            return self.forward_minicpm_sparse(q, k, v, layer, forward_batch, save_kv_cache)
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            if is_mla_preprocess_enabled():
                save_kv_cache = False
            return self.forward_mtp(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            else:
                if layer.qk_head_dim <= 128:
                    query = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                    attn_output = torch.empty(
                        (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                        dtype=query.dtype,
                        device=query.device,
                    )

                    torch_npu._npu_flash_attention_qlens(
                        query=query,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        mask=self.mask,
                        block_table=self.forward_metadata.block_tables,
                        seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                        context_lens=self.forward_metadata.seq_lens_cpu_int,
                        scale_value=layer.scaling,
                        num_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        out=attn_output,
                    )
                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(q)

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    causal = True
                    if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
                        causal = False

                    self.native_attn._run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                    )
        elif sum(forward_batch.extend_prefix_lens_cpu) > 0:
            num_token_padding = q.shape[0]
            q, k, v = [data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]]
            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

            # 1st, compute extend tokens to get attn_output and attn_lse
            num_tokens = q_nope.size(0)
            attn_output = torch.zeros(
                num_tokens,
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            attn_lse = torch.zeros(
                layer.tp_q_head_num,
                num_tokens,
                dtype=torch.float32,
                device=q_nope.device,
            )
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_rope,
                k_nope=k_nope,
                k_rope=k_rope,
                value=v,
                mask=self.ringmla_mask,
                seqlen=self.forward_metadata.extend_seq_lens_cpu_int,
                head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
                pre_out=None,
                prev_lse=None,
                qk_scale=layer.scaling,
                kernel_type="kernel_type_high_precision",
                mask_type="mask_type_triu",
                calc_type="calc_type_first_ring",
                output=attn_output,
                softmax_lse=attn_lse,
            )

            # 2nd, load history kvcache(kv_a and k_pe) and calculate k_nope
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_cached = torch.index_select(
                k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
            )
            k_rope_cached = torch.index_select(
                v_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
            ).flatten(0, 1)

            assert layer.kv_b_proj is not None
            kv = layer.kv_b_proj(kv_cached)[0].view(
                -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
            )
            k_nope, v = kv.split([self.qk_nope_head_dim, layer.v_head_dim], dim=-1)

            # 3rd, compute history kv to attn_out
            k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
            seq_len = torch.stack(
                [
                    self.forward_metadata.extend_seq_lens_cpu_int,
                    self.forward_metadata.prefix_lens,
                ]
            )
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_rope,
                k_nope=k_nope,
                k_rope=k_rope,
                value=v,
                mask=self.ringmla_mask,
                seqlen=seq_len,
                head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
                pre_out=attn_output,
                prev_lse=attn_lse,
                qk_scale=layer.scaling,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                calc_type="calc_type_default",
                output=attn_output,
                softmax_lse=attn_lse,
            )
            attn_output = attn_output.reshape([-1, layer.tp_q_head_num, layer.v_head_dim])
            if num_token_padding != forward_batch.num_token_non_padded_cpu:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            assert layer.qk_head_dim != layer.v_head_dim, (
                "FIA only supports qk_head_dim != v_head_dim"
            )

            num_token_padding = q.shape[0]
            q, k, v = [data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]]

            q_nope, q_rope = q.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([layer.v_head_dim, self.qk_rope_head_dim], dim=-1)

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=layer.tp_q_head_num,
                input_layout="TND",
                atten_mask=self.fia_mask,
                sparse_mode=3,
                actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                scale=layer.scaling,
                next_tokens=0,
            )

            attn_output = attn_output.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)
            if num_token_padding != forward_batch.num_token_non_padded_cpu:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        return attn_output

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
            )
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
            )
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + query.shape[0],
                    self.speculative_num_draft_tokens,
                )

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                atten_mask=self.mtp_mask,
                scale=layer.scaling,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                sparse_mode=3,
            )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if not self.graph_mode and forward_batch.num_token_non_padded_cpu != num_token_padding:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - forward_batch.num_token_non_padded_cpu,
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(-1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank)

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + q_nope.shape[0],
                    self.speculative_num_draft_tokens,
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
            )
            attn_output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="TND",
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                sparse_mode=3,
                atten_mask=self.mtp_mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                workspace=workspace,
                out=[attn_output, softmax_lse],
            )
            attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            if not self.graph_mode and forward_batch.num_token_non_padded_cpu != num_token_padding:
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if not self.use_mla:
            num_tokens = q.shape[0]
            """PA will support bs<tp in the later version of CANN"""
            if num_tokens < get_attention_tp_size():
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
                    -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                )
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).view(
                    -1, self.page_size, layer.tp_v_head_num * layer.v_head_dim
                )
                query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                num_tokens = query.shape[0]
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSH",
                    scale=layer.scaling,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                )
                output = torch.empty(
                    (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
                torch_npu.npu_fused_infer_attention_score.out(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSH",
                    scale=layer.scaling,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    workspace=workspace,
                    out=[output, softmax_lse],
                )
                return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
            else:
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = torch.from_numpy(
                        np.array(self.forward_metadata.seq_lens_cpu_list).astype(np.int32)
                    )
                else:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=actual_seq_len_kv,
                    out=attn_output,
                )
                return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(-1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank)

            q_nope = q.view(-1, 1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, 1, layer.tp_q_head_num, self.qk_rope_head_dim)

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if self.has_minicpm_sparse and not self.use_mla:
            # logger.info("======== forward_decode => forward_minicpm_sparse")
            return self.forward_minicpm_sparse(q, k, v, layer, forward_batch, save_kv_cache)
        if is_mla_preprocess_enabled():
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )

        if self.graph_mode and (not self.enable_torch_compile):
            return self.forward_decode_graph(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        if not self.use_mla:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            num_tokens = q.shape[0]
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            if self.use_fia:
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        forward_batch.batch_size,
                        -1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
                    v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    scale=layer.scaling,
                )
            else:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )

                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    num_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                if is_fia_nz():
                    kv_c = _reshape_kv_for_fia_nz(
                        kv_c, layer.tp_k_head_num, self.kv_lora_rank, self.page_size
                    )
                    k_pe = _reshape_kv_for_fia_nz(
                        k_pe, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                    )
                else:
                    kv_c = kv_c.view(-1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank)
                    k_pe = k_pe.view(
                        -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                    )
                q = q.view(forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank)
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
            else:
                assert self.graph_mode == False  # _npu_paged_attention_mla not support graph mode
                q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if (
            topk_indices is not None
            or self.use_mla
            or (not self.use_fia and layer.qk_head_dim > 128)
        ):
            raise NotImplementedError(
                "The 'enable-mixed-chunk' feature is currently unsupported in the following scenarios: "
                "1. When using the MLA backend on Ascend NPU devices, "
                "2. When using the deepseekv3.2 model on Ascend NPU devices, "
                "3. When the environment variable ASCEND_USE_FIA is set to 0 and qk_head_dim exceeds 128 on Ascend NPU devices."
            )
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        num_block, block_size, _, _ = k_cache.shape
        key = k_cache.view(num_block, block_size, -1)
        value = v_cache.view(num_block, block_size, -1)

        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            block_size=block_size,
            block_table=self.forward_metadata.block_tables,
            atten_mask=self.mix_mask,
            sparse_mode=3,
            actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
            actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
            scale=layer.scaling,
        )

        return attn_output.view(attn_output.shape[0], layer.tp_q_head_num * layer.v_head_dim)


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(AscendAttnBackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
