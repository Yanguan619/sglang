from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.hardware_backend.npu.attention.minicpm_sparse_npu import (
    get_block_table_v2_ref,
    get_block_table_v3_ref,
    infllmv2_attn_stage1_npu,
    max_pooling_1d_varlen_npu,
)
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minicpm_sparse_utils import (
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadataBuilder,
    allocate_and_compress_keys,
    compressed_attention,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_bool_env_var

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

    # sparse attention fields (MiniCPM)
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seq_len_q: Optional[int] = None
    max_seq_len_k: Optional[int] = None
    cache_seqlens_int32: Optional[torch.Tensor] = None
    k1: Optional[object] = None
    k2: Optional[object] = None
    sparse_bs_list: Optional[list] = None
    sparse_page_table: Optional[torch.Tensor] = None
    sparse_cache_seqlens_int32: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q: Optional[torch.Tensor] = None
    sparse_cu_seqlens_k: Optional[torch.Tensor] = None
    sparse_max_seq_len_q: Optional[int] = None
    token_to_bs: Optional[torch.Tensor] = None
    token_pos_in_bs: Optional[torch.Tensor] = None
    cache_seqlens_int32_stage1: Optional[torch.Tensor] = None
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: Optional[int] = None
    seqlen_k_sparse_bs_tensor: Optional[torch.Tensor] = None
    sparse_idx: Optional[torch.Tensor] = None


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
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(
            self.device
        )

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
            mask_value = (
                float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
            )
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
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1)
            .to(torch.int8)
            .to(self.device)
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
            self.q_head_dim = (
                self.qk_rope_head_dim + model_runner.model_config.qk_nope_head_dim
            )
        self.native_attn = TorchNativeAttnBackend(model_runner)
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
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

        # Sparse attention configuration (MiniCPM)
        self.use_sparse_attention = False
        self.sparse_config = None
        self.sparse_batch_analyzer = None
        self.sparse_metadata_builder = None
        hf_config = getattr(model_runner.model_config, "hf_config", None)
        if hf_config is not None and getattr(hf_config, "has_sparse_attention", False):
            self.use_sparse_attention = True
            self.sparse_kernel_size = hf_config.sparse_kernel_size
            self.sparse_kernel_stride = hf_config.sparse_kernel_stride
            self.init_blocks = hf_config.sparse_init_blocks
            self.block_size = hf_config.sparse_block_size
            self.sparse_window_size = hf_config.sparse_window_size
            self.dense_len = hf_config.sparse_dense_len
            topk = hf_config.sparse_topk
            self.local_blocks = self.sparse_window_size // self.block_size
            self.sparse_topk = topk + self.local_blocks
            self.num_sparse_topk_tokens = self.block_size * self.sparse_topk
            self.head_dim = model_runner.model_config.head_dim
            self.head_group_num = model_runner.model_config.num_key_value_heads
            self.num_kv_heads = self.head_group_num
            self.heads_per_group = (
                model_runner.model_config.num_attention_heads // self.head_group_num
            )
            self.k1_kernel_size = self.sparse_kernel_size
            self.k1_kernel_stride = self.sparse_kernel_stride
            self.k2_kernel_size = self.sparse_kernel_size * 4
            self.k2_kernel_stride = self.sparse_kernel_stride * 4
            self.split_stage1 = getattr(
                model_runner.server_args, "split_stage1", False
            )

            import sglang.srt.layers.attention.minicpm_sparse_utils as _sparse_utils

            _sparse_utils.infllmv2_attn_stage1 = infllmv2_attn_stage1_npu
            _sparse_utils.max_pooling_1d_varlen = max_pooling_1d_varlen_npu

            sparse_config = SparseConfig.from_model_config(
                hf_config, model_runner.model_config
            )
            self.sparse_batch_analyzer = SparseBatchAnalyzer(sparse_config)
            self.sparse_metadata_builder = SparseMetadataBuilder(
                sparse_config,
                num_kv_heads=self.head_group_num,
                max_context_len=self.max_context_len,
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
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to(
                "cpu"
            )
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(
                forward_batch.req_pool_indices.tolist(), seq_prefix_lens
            ):
                req_indices = forward_batch.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        self.graph_mode = False

        if self.use_sparse_attention:
            self._init_minicpm_sparse_metadata(forward_batch)

    def _init_minicpm_sparse_metadata(self, forward_batch: ForwardBatch):
        fm = self.forward_metadata
        bs = forward_batch.batch_size
        device = self.device

        fm.page_table = fm.block_tables  # alias for get_compress_k_v2

        seqlens_in_batch = forward_batch.seq_lens
        fm.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
        fm.cache_seqlens_int32_stage1 = fm.cache_seqlens_int32 - 1

        # Build the cu_seqlens and page_table (same as MiniCPMBackend's pattern)
        if forward_batch.forward_mode.is_decode_or_idle():
            fm.max_seq_len_q = 1
            fm.max_seq_len_k = seqlens_in_batch.max().item()
            cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=device
            )
            fm.cu_seqlens_q = cu_seqlens_q
            fm.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            fm.sparse_max_seq_len_q = 1
            fm.cu_seqlens_q_adjusted = cu_seqlens_q * self.heads_per_group
            fm.max_seqlen_q_adjusted = 1 * self.heads_per_group
            # K1/K2 compression metadata
            compression = (
                self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                    forward_batch=forward_batch,
                    base_metadata=fm,
                    req_to_sparse_k1_token=(
                        forward_batch.req_to_token_pool.req_to_sparse_k1_token
                    ),
                    req_to_sparse_k2_token=(
                        forward_batch.req_to_token_pool.req_to_sparse_k2_token
                    ),
                    k1_kernel_size=self.k1_kernel_size,
                    k1_kernel_stride=self.k1_kernel_stride,
                    k2_kernel_size=self.k2_kernel_size,
                    k2_kernel_stride=self.k2_kernel_stride,
                    cu_seqlens_q=cu_seqlens_q,
                )
            )
            fm.k1 = compression["k1"]
            fm.k2 = compression["k2"]

            decode_md = self.sparse_metadata_builder.build_sparse_decode_metadata(
                forward_batch=forward_batch,
                base_metadata=fm,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )
            fm.sparse_cache_seqlens_int32 = decode_md["sparse_cache_seqlens_int32"]
            fm.sparse_cu_seqlens_k = decode_md["sparse_cu_seqlens_k"]
            fm.sparse_cu_seqlens_q = decode_md["sparse_cu_seqlens_q"]
            fm.sparse_page_table = decode_md["sparse_page_table"]
            fm.token_to_bs = decode_md["token_to_bs"]

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            # Use prefill metadata
            cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                cu_seqlens_q = cu_seqlens_k

            fm.cu_seqlens_q = cu_seqlens_q
            fm.cu_seqlens_k = cu_seqlens_k
            fm.max_seq_len_q = (
                forward_batch.extend_seq_lens.max().item()
                if any(forward_batch.extend_prefix_lens_cpu)
                else seqlens_in_batch.max().item()
            )
            fm.max_seq_len_k = seqlens_in_batch.max().item()

            compression = (
                self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                    forward_batch=forward_batch,
                    base_metadata=fm,
                    req_to_sparse_k1_token=(
                        forward_batch.req_to_token_pool.req_to_sparse_k1_token
                    ),
                    req_to_sparse_k2_token=(
                        forward_batch.req_to_token_pool.req_to_sparse_k2_token
                    ),
                    k1_kernel_size=self.k1_kernel_size,
                    k1_kernel_stride=self.k1_kernel_stride,
                    k2_kernel_size=self.k2_kernel_size,
                    k2_kernel_stride=self.k2_kernel_stride,
                    cu_seqlens_q=cu_seqlens_q,
                )
            )
            fm.k1 = compression["k1"]
            fm.k2 = compression["k2"]

            fm.sparse_bs_list = (
                self.sparse_batch_analyzer.identify_sparse_batches(
                    forward_batch, dense_as_sparse=False
                )
            )

            seqlen_q_sparse_bs, fm.seqlen_k_sparse_bs_tensor = (
                self.sparse_metadata_builder.build_sequence_lengths(
                    cu_seqlens_q,
                    forward_batch.extend_prefix_lens,
                    fm.sparse_bs_list,
                )
            )

            cu_seqlens_q_sparse_bs = torch.tensor(
                [0] + seqlen_q_sparse_bs, dtype=torch.int32, device=device
            ).cumsum(dtype=torch.int32, dim=0)

            extend_prefix_lens_sparse = torch.tensor(
                [
                    forward_batch.extend_prefix_lens_cpu[bs]
                    for bs in fm.sparse_bs_list
                ],
                dtype=torch.long,
                device="cpu",
            )

            fm.token_to_bs, fm.token_pos_in_bs = (
                self.sparse_metadata_builder.build_token_mappings(
                    cu_seqlens_q_sparse_bs,
                    extend_prefix_lens_sparse,
                    seqlen_q_sparse_bs,
                )
            )
            fm.token_to_bs = fm.token_to_bs.to(device=device)
            fm.token_pos_in_bs = fm.token_pos_in_bs.to(device=device)

            prefill_md = (
                self.sparse_metadata_builder.build_sparse_prefill_metadata(
                    forward_batch=forward_batch,
                    base_metadata=fm,
                    sparse_bs_list=fm.sparse_bs_list,
                    head_group_num=self.head_group_num,
                    dense_len=self.dense_len,
                    sparse_topk=self.sparse_topk,
                    block_size=self.block_size,
                    cu_seqlens_q=cu_seqlens_q,
                    sparse_page_table_dtype=fm.block_tables.dtype,
                    sparse_page_table_device=device,
                )
            )

            fm.sparse_page_table = prefill_md["sparse_page_table"]
            fm.sparse_cu_seqlens_q = prefill_md["sparse_cu_seqlens_q"]
            fm.sparse_max_seq_len_q = prefill_md["sparse_max_seq_len_q"]
            fm.sparse_idx = prefill_md["sparse_idx"]
            fm.old_bs_to_new_bs_range = prefill_md["old_bs_to_new_bs_range"]
            fm.sparse_cu_seqlens_q_cpu = prefill_md["sparse_cu_seqlens_q_cpu"]

            fm.cu_seqlens_q_adjusted = fm.sparse_cu_seqlens_q * self.heads_per_group
            fm.max_seqlen_q_adjusted = (
                fm.sparse_max_seq_len_q * self.heads_per_group
            )

        else:
            if forward_batch.forward_mode.is_target_verify():
                raise NotImplementedError(
                    "MiniCPM sparse backend does not support target verify"
                )

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
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
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
                        forward_batch.extend_seq_lens.cumsum()
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = self.forward_metadata.actual_seq_lengths_kv
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens

        if (
            is_prefill
            and is_nsa_enable_prefill_cp()
            and forward_batch.nsa_cp_metadata is not None
        ):
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

    def _get_topk_for_sparse(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        fm = self.forward_metadata

        if is_prefill:
            cu_seqlens_q = fm.sparse_cu_seqlens_q
            cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            max_seqlen_q = fm.sparse_max_seq_len_q
            max_seqlen_k = forward_batch.seq_lens_cpu.max().item()

            total_k1 = fm.k1.cu_total_compress_token_nums[-1].item()
            total_k2 = fm.k2.cu_total_compress_token_nums[-1].item()

            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=fm,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                dtype=torch.bfloat16,
                device=self.device,
                max_context_length=self.max_context_len,
            )

            topk_idx = compressed_attention(
                q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim),
                full_compressed_k1,
                full_compressed_k2,
                self.k1_kernel_size,
                self.k1_kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                fm.k1.cu_seqlens,
                fm.k2.cu_seqlens,
                max_seqlen_q,
                self.max_context_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cu_seqlens_q_adjusted=fm.cu_seqlens_q_adjusted,
                max_seqlen_q_adjusted=fm.max_seqlen_q_adjusted,
            )
        else:
            total_k1 = forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride
            total_k2 = forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride

            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=fm,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                dtype=torch.bfloat16,
                device=self.device,
                max_context_length=self.max_context_len,
                split_stage1=self.split_stage1,
            )

            q_1d = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            cu_seqlens_q = fm.cu_seqlens_q
            cu_seqlens_k = fm.cu_seqlens_k
            max_seqlen_q = fm.max_seq_len_q
            max_seqlen_k = fm.max_seq_len_k

            topk_idx = compressed_attention(
                q_1d.unsqueeze(0),
                full_compressed_k1,
                full_compressed_k2,
                self.k1_kernel_size,
                self.k1_kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                fm.k1.cu_seqlens,
                fm.k2.cu_seqlens,
                max_seqlen_q,
                self.max_context_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cu_seqlens_q_adjusted=fm.cu_seqlens_q_adjusted,
                max_seqlen_q_adjusted=fm.max_seqlen_q_adjusted,
                split_stage1=self.split_stage1,
            )

        return topk_idx

    def _forward_extend_minicpm_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError("MiniCPM sparse backend does not support cross attention")

        bs = forward_batch.batch_size
        fm = self.forward_metadata

        if k is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        topk_idx = self._get_topk_for_sparse(q_reshaped, layer, forward_batch, is_prefill=True)

        page_table = fm.block_tables
        if max(forward_batch.seq_lens_cpu) >= self.dense_len:
            sparse_pt_sparse_bs = get_block_table_v2_ref(
                topk_idx,
                page_table,
                fm.token_to_bs,
                fm.token_pos_in_bs,
                fm.seqlen_k_sparse_bs_tensor,
                self.sparse_topk,
                self.block_size,
            )
            fm.sparse_page_table[fm.sparse_idx, :self.num_sparse_topk_tokens] = (
                sparse_pt_sparse_bs[:, :self.num_sparse_topk_tokens]
            )

        q_by_hg = q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim)

        if fm.sparse_bs_list is not None and len(fm.sparse_bs_list) < bs:
            dense_bs_list = [i for i in range(bs) if i not in fm.sparse_bs_list]
            for dense_bs in dense_bs_list:
                sparse_page_table_idx_start = fm.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = fm.old_bs_to_new_bs_range[dense_bs + 1]
                if sparse_page_table_idx_end - sparse_page_table_idx_start == 2:
                    ps = fm.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start].item()
                    length = fm.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1].item() - ps
                    if length > 0:
                        t = q_by_hg[ps:ps + 2 * length, :, :].clone()
                        q_by_hg[ps:ps + length, :, :] = t[0::2, :, :]
                        q_by_hg[ps + length:ps + 2 * length, :, :] = t[1::2, :, :]
                        kv_len = forward_batch.seq_lens_cpu[dense_bs]
                        fm.sparse_page_table[sparse_page_table_idx_start, :kv_len] = page_table[dense_bs, :kv_len]
                        fm.sparse_page_table[sparse_page_table_idx_start + 1, :kv_len] = page_table[dense_bs, :kv_len]

        sparse_cache_seqlens = (fm.sparse_page_table != 0).sum(dim=1).to(
            dtype=torch.int32, device=self.device
        )

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        sparse_pt = fm.sparse_page_table.to(torch.int32).long()
        sparse_pt = sparse_pt.clamp(0, key_cache.shape[0] - 1)
        k_gathered = key_cache[sparse_pt].squeeze(2)
        v_gathered = value_cache[sparse_pt].squeeze(2)

        scale = layer.scaling
        attn_output = torch.zeros(
            q_by_hg.shape[0], layer.tp_q_head_num // 2, layer.v_head_dim,
            dtype=q.dtype, device=self.device,
        )

        pos = 0
        for i in range(sparse_pt.shape[0]):
            ctx_len = sparse_cache_seqlens[i].item()
            if ctx_len <= 0:
                pos += 1
                continue
            q_i = q_by_hg[pos:pos + 1]
            k_i = k_gathered[i, :ctx_len]
            v_i = v_gathered[i, :ctx_len]
            scores = torch.einsum('qd,kmd->qkm', q_i.float(), k_i.float()) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output[pos] = torch.einsum('qkm,kmd->qd', attn_weights, v_i.float()).to(q.dtype)
            pos += 1

        if fm.sparse_bs_list is not None and len(fm.sparse_bs_list) < bs:
            for dense_bs in dense_bs_list:
                sparse_page_table_idx_start = fm.old_bs_to_new_bs_range[dense_bs]
                if sparse_page_table_idx_start >= attn_output.shape[0]:
                    continue
                ps = fm.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start].item()
                length = fm.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1].item() - ps
                if length >= 2:
                    t = attn_output[ps:ps + 2 * length, :, :].clone()
                    attn_output[ps:ps + 2 * length:2, :, :] = t[:length, :, :]
                    attn_output[ps + 1:ps + 2 * length:2, :, :] = t[length:2 * length, :, :]

        return attn_output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode_minicpm_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError("MiniCPM sparse backend does not support cross attention")

        bs = forward_batch.batch_size
        fm = self.forward_metadata

        if k is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        page_table = fm.block_tables
        cache_seqlens = fm.cache_seqlens_int32

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        topk_idx = self._get_topk_for_sparse(q_reshaped, layer, forward_batch, is_prefill=False)

        sparse_page_table = get_block_table_v3_ref(
            topk_idx,
            page_table,
            fm.token_to_bs,
            cache_seqlens,
            cache_seqlens,
            self.sparse_topk,
            self.block_size,
        )
        fm.sparse_page_table[:2 * bs, :self.num_sparse_topk_tokens] = (
            sparse_page_table[:, :self.num_sparse_topk_tokens]
        )

        q_by_hg = q.reshape(-1, layer.tp_q_head_num // 2, layer.head_dim)

        sparse_cache_seqlens = (fm.sparse_page_table != 0).sum(dim=1).to(
            dtype=torch.int32, device=self.device
        )

        _, _, full_kv_heads, head_dim = key_cache.shape

        attn_output = torch.empty(
            q_by_hg.shape[0], layer.tp_q_head_num // 2, layer.v_head_dim,
            dtype=q.dtype, device=self.device,
        )

        torch_npu._npu_paged_attention(
            query=q_by_hg,
            key_cache=key_cache,
            value_cache=value_cache,
            num_heads=layer.tp_q_head_num // 2,
            num_kv_heads=layer.tp_k_head_num // 2,
            scale_value=layer.scaling,
            block_table=sparse_pt,
            context_lens=sparse_cache_seqlens,
            out=attn_output,
        )

        return attn_output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

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
        if self.use_sparse_attention:
            return self._forward_extend_minicpm_sparse(
                q, k, v, layer, forward_batch, save_kv_cache
            )
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
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )

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
                    if (
                        layer.is_cross_attention
                        or layer.attn_type == AttentionType.ENCODER_ONLY
                    ):
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
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]
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
            attn_output = attn_output.reshape(
                [-1, layer.tp_q_head_num, layer.v_head_dim]
            )
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
            assert (
                layer.qk_head_dim != layer.v_head_dim
            ), "FIA only supports qk_head_dim != v_head_dim"

            num_token_padding = q.shape[0]
            q, k, v = [
                data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
            ]

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
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
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
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
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
                c_kv_cache = c_kv.view(
                    -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
                )

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
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
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
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
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
                query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                num_tokens = query.shape[0]
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
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
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                attn_output = torch.empty(
                    (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                    dtype=query.dtype,
                    device=query.device,
                )
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = torch.from_numpy(
                        np.array(self.forward_metadata.seq_lens_cpu_list).astype(
                            np.int32
                        )
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
                return attn_output.view(
                    num_tokens, layer.tp_q_head_num * layer.v_head_dim
                )
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
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )

            q_nope = q.view(-1, 1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, 1, layer.tp_q_head_num, self.qk_rope_head_dim)

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

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
        if self.use_sparse_attention:
            return self._forward_decode_minicpm_sparse(
                q, k, v, layer, forward_batch, save_kv_cache
            )
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
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q.view(
                        forward_batch.batch_size,
                        -1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    ),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim
                    ),
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
                    kv_c = kv_c.view(
                        -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                    )
                    k_pe = k_pe.view(
                        -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                    )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
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
                assert (
                    self.graph_mode == False
                )  # _npu_paged_attention_mla not support graph mode
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
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
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

        return attn_output.view(
            attn_output.shape[0], layer.tp_q_head_num * layer.v_head_dim
        )


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

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
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
