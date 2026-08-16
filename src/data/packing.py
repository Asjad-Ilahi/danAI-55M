"""
Document-aware packing and mask construction per §6 & §18.

When packing multiple EOS-terminated documents into a fixed-length training sequence,
naive causal attention allows tokens in document B to attend to tokens in document A.
This module constructs block-diagonal causal masks and tracks document segment IDs
so tokens can only attend to earlier tokens WITHIN THEIR OWN DOCUMENT.
"""

import torch
from typing import List, Tuple, Dict, Any, Iterator


def pack_documents(
    documents_tokens: List[List[int]],
    max_seq_len: int,
    eos_token_id: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Pack EOS-terminated documents into fixed-size sequences of length max_seq_len.
    Returns (packed_token_ids, segment_ids).
    
    segment_ids: List[int] of length max_seq_len indicating which document index
    within the packed sequence each token belongs to (e.g. [0, 0, 0, 1, 1, 2, ...]).
    """
    current_tokens = []
    current_segments = []
    doc_index = 0

    for doc in documents_tokens:
        # Ensure EOS termination
        if not doc or doc[-1] != eos_token_id:
            doc = list(doc) + [eos_token_id]

        doc_pos = 0
        while doc_pos < len(doc):
            space = max_seq_len - len(current_tokens)
            chunk = doc[doc_pos : doc_pos + space]
            current_tokens.extend(chunk)
            current_segments.extend([doc_index] * len(chunk))
            doc_pos += len(chunk)

            if len(current_tokens) == max_seq_len:
                yield current_tokens, current_segments
                current_tokens = []
                current_segments = []
                doc_index = 0

        doc_index += 1


def create_block_diagonal_causal_mask(
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a 2D or 4D block-diagonal causal mask from segment IDs.
    
    Args:
        segment_ids: Tensor of shape (batch_size, seq_len) or (seq_len,) containing
                     integer document segment indices.
                     
    Returns:
        mask: Boolean Tensor of shape (batch_size, 1, seq_len, seq_len) where:
              True/1  = token i CAN attend to token j (same document AND j <= i)
              False/0 = token i CANNOT attend to token j (different document OR j > i)
    """
    if segment_ids.ndim == 1:
        segment_ids = segment_ids.unsqueeze(0)  # (1, seq_len)

    batch_size, seq_len = segment_ids.shape

    # Same document mask: (B, S, 1) == (B, 1, S) -> (B, S, S)
    same_doc_mask = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)

    # Standard causal mask: j <= i -> (S, S) lower triangular
    causal_tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=segment_ids.device))

    # Combine: same document AND lower triangular
    # Shape: (batch_size, 1, seq_len, seq_len)
    block_causal_mask = (same_doc_mask & causal_tril).unsqueeze(1)

    return block_causal_mask
