#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.


# Adapted from:
# https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/models/rnn_state_encoder.py
# This is a filtered down version that only supports LSTM

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class RNNStateEncoder(nn.Module):
    r"""RNN encoder for use with RL and possibly IL.

    The main functionality this provides over just using PyTorch's RNN interface directly
    is that it takes an addition masks input that resets the hidden state between two adjacent
    timesteps to handle episodes ending in the middle of a rollout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.num_recurrent_layers = num_layers * 2

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.contiguous()

    def single_forward(
        self, x: torch.Tensor, hidden_states: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a non-sequence input"""

        hidden_states = torch.where(masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(()))

        x, hidden_states = self.rnn(x.unsqueeze(0), self.unpack_hidden(hidden_states))
        hidden_states = self.pack_hidden(hidden_states)

        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """

        (
            x_seq,
            hidden_states,
        ) = build_rnn_inputs(x, hidden_states, masks, rnn_build_seq_info)

        rnn_ret = self.rnn(x_seq, self.unpack_hidden(hidden_states))
        x_seq: PackedSequence = rnn_ret[0]  # type: ignore
        hidden_states: torch.Tensor = rnn_ret[1]  # type: ignore
        hidden_states = self.pack_hidden(hidden_states)

        x, hidden_states = build_rnn_out_from_seq(
            x_seq,
            hidden_states,
            rnn_build_seq_info,
        )

        return x, hidden_states

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            assert rnn_build_seq_info is None
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            assert rnn_build_seq_info is not None
            x, hidden_states = self.seq_forward(x, hidden_states, masks, rnn_build_seq_info)

        hidden_states = hidden_states.permute(1, 0, 2)

        return x, hidden_states


class LSTMStateEncoder(RNNStateEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__(input_size, hidden_size, num_layers)

    # Note: Type handling mypy errors in pytorch libraries prevent
    # directly setting hidden_states type
    def pack_hidden(self, hidden_states: Any) -> torch.Tensor:  # type is Tuple[torch.Tensor, torch.Tensor]
        return torch.cat(hidden_states, 0)

    def unpack_hidden(self, hidden_states: torch.Tensor) -> Any:  # type is Tuple[torch.Tensor, torch.Tensor]
        lstm_states = torch.chunk(hidden_states.contiguous(), 2, 0)
        return (lstm_states[0], lstm_states[1])


def build_rnn_inputs(
    x: torch.Tensor,
    rnn_states: torch.Tensor,
    not_dones: torch.Tensor,
    rnn_build_seq_info: Dict[str, torch.Tensor],
) -> Tuple[
    PackedSequence,
    torch.Tensor,
]:
    r"""Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.

    Use the returned select_inds and build_rnn_out_from_seq to invert this.

    :param x: A (T * N, -1) tensor of the data to build the PackedSequence out of
    :param rnn_states: A (-1, N, -1) tensor of the rnn_hidden_states

    :return: tuple(x_seq, rnn_states, select_inds, rnn_state_batch_inds, last_sequence_in_batch_mask)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN

        rnn_states are the corresponding rnn state

        select_inds can be passed to build_rnn_out_from_seq to retrieve the
            RNN output

        rnn_state_batch_inds indicates which of the rollouts in the batch a hidden
            state came from/is for

        last_sequence_in_batch_mask indicates if an episode is the last in that batch.
            There will be exactly N places where this is True

    """

    select_inds = rnn_build_seq_info["select_inds"]
    num_seqs_at_step = rnn_build_seq_info["cpu_num_seqs_at_step"]

    x_seq = PackedSequence(x.index_select(0, select_inds), num_seqs_at_step, None, None)

    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    sequence_starts = rnn_build_seq_info["sequence_starts"]

    # Just select the rnn_states by batch index, the masking bellow will set things
    # to zero in the correct locations
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    # Now zero things out in the correct locations
    rnn_states.masked_fill_(
        torch.logical_not(not_dones.view(1, -1, 1).index_select(1, sequence_starts)),
        0,
    )

    return (
        x_seq,
        rnn_states,
    )


def build_rnn_out_from_seq(
    x_seq: PackedSequence,
    hidden_states: torch.Tensor,
    rnn_build_seq_info: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Construct the output of the rnn from a packed sequence returned by
        forward propping an RNN on the packed sequence returned by :ref:`build_rnn_inputs`.

    :param x_seq: The packed sequence output from the rnn
    :param hidden_statess: The hidden states output from the rnn
    :param select_inds: Returned from :ref:`build_rnn_inputs`
    :param rnn_state_batch_inds: Returned from :ref:`build_rnn_inputs`
    :param last_sequence_in_batch_mask: Returned from :ref:`build_rnn_inputs`
    :param N: The number of simulator instances in the batch of experience.
    """
    select_inds = rnn_build_seq_info["select_inds"]
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))

    last_sequence_in_batch_inds = rnn_build_seq_info["last_sequence_in_batch_inds"]
    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    output_hidden_states = hidden_states.index_select(
        1,
        last_sequence_in_batch_inds[_invert_permutation(rnn_state_batch_inds[last_sequence_in_batch_inds])],
    )

    return x, output_hidden_states


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    orig_size = permutation.size()
    permutation = permutation.view(-1)
    output = torch.empty_like(permutation)
    output.scatter_(
        0,
        permutation,
        torch.arange(0, permutation.numel(), device=permutation.device),
    )
    return output.view(orig_size)
