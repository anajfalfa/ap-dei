import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Ws = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )
        
        ###raise NotImplementedError("Add your implementation.")

    def forward(self, query, encoder_outputs, src_lengths):
        query = query.unsqueeze(2)  # (batch_size, max_tgt_len, hidden_size)
        encoder_outputs = encoder_outputs.unsqueeze(1)  # (batch_size, 1, max_src_len, hidden_size)

        # Compute attention scores
        scores = self.v(torch.tanh(self.Ws(query) + self.Wh(encoder_outputs))).squeeze(-1)  # (batch_size, max_tgt_len, max_src_len)

        # Mask padding positions
        attn_mask = self.sequence_mask(src_lengths).unsqueeze(1)  # (batch_size, 1, max_src_len)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        alignment = torch.softmax(scores, dim=2)  # (batch_size, max_tgt_len, max_src_len)

        # Context computation
        context = torch.bmm(alignment, encoder_outputs.squeeze(1))  # (batch_size, max_tgt_len, hidden_size)

        attn_out = self.mlp(torch.cat([context, query.squeeze(2)], dim=-1))  # (batch_size, max_tgt_len, hidden_size)

        return attn_out, alignment
        ###raise NotImplementedError("Add your implementation.")

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################

        encoded = self.embedding(src)
        encoded = self.dropout(encoded)
        packed_encoded = torch.nn.utils.rnn.pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden_states, cell_states) = self.lstm(packed_encoded)
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        enc_output = self.dropout(enc_output)

        return enc_output, (hidden_states, cell_states)

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
            # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        ###raise NotImplementedError("Add your implementation.")


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
            # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
        
        lstm_outs = []

        embedded_tgt = self.embedding(tgt)
        embedded_tgt = self.dropout(embedded_tgt)

        for i in range(embedded_tgt.size(1)):
            embedded_tgt_token = embedded_tgt[:, i, :]              # (batch_size, hidden_size)
            embedded_tgt_token = embedded_tgt_token.unsqueeze(1)    # (batch_size, 1, hidden_size)

            output_tgt_token, dec_state = self.lstm(embedded_tgt_token, dec_state)
            output_tgt_token = self.dropout(output_tgt_token)

            if self.attn is not None:
                output_tgt_token, _ = self.attn(output_tgt_token, encoder_outputs, src_lengths)

            lstm_outs.append(output_tgt_token)

        outputs = torch.cat(lstm_outs, dim=1)

        return outputs, dec_state
        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        ###raise NotImplementedError("Add your implementation.")


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
