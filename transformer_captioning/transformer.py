# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
#done
class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):

        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # TODO : Compute attention 

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        #compute dot-product attention. Don't forget the scaling value!
        #Expected shape of dot_product is (N, S, T)
        dot_product = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.embed_dim) # Indent this line to match the level of 'query = self.query_proj(query)'

        if attn_mask is not None:
            additive_mask = attn_mask.masked_fill(attn_mask == 0, float('-1e20'))
            dot_product += additive_mask

        attention = torch.softmax(dot_product, dim=-1)
        attention = self.dropout(attention)
        y = torch.bmm(attention, value)
        return y

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        #done
        # TODO: Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        qkv = self.head_proj(query).view(N, S, H, 3 * self.head_dim)
        query, key, value = qkv.chunk(3, dim=-1)

        # TODO : Compute multi-head attention

        #done
        #project query, key and value
        #after projection, split the embedding across num_heads
        #eg - expected shape for value is (N, H, T, D/H)
        query = query.permute(0, 2, 1, 3)  
        key = key.permute(0, 2, 1, 3)      
        value = value.permute(0, 2, 1, 3)


        #done
        #compute dot-product attention separately for each head. Don't forget the scaling value!
        #Expected shape of dot_product is (N, H, S, T)
        dot_product = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            #done
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            additive_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            dot_product += additive_mask

        #done
        # apply softmax, dropout, and use value
        attn_weights = F.softmax(dot_product, dim=-1)
        attn_weights = self.dropout(attn_weights)
        y = y = torch.matmul(attn_weights, value)

        #done
        # concat embeddings from different heads, and project
        y = y.transpose(1, 2).contiguous().view(N, S, D)
        output = self.out_proj(y)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        attn_output = self.self_attn(seq, seq, seq, mask)
        attn_output = self.dropout(attn_output)
        out = self.norm(attn_output + seq)
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, cond, mask=None):
        attn_output = self.cross_attn(seq, cond, cond, mask)
        attn_output = self.dropout(attn_output)
        out = self.norm(attn_output + seq)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        # TODO: Initialize the following. 
        # MLP has the following layers : linear, relu, dropout, linear ; hidden dim of linear is given by dim_feedforward
        self.mlp = ...
        self.dropout = ...
        self.norm = ...
       

    def forward(self, seq):
         ############# TODO - MLP on the sequence. Add dropout to mlp layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        # TODO - get caption and feature embeddings 
        # Don't forget position embeddings for captions!
        # expected caption embedding output shape : (N, T, D)

        # Unsqueeze feature embedding along dimension 1
        # expected feature embedding output shape : (N, 1, D) 
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        #TODO - get causal mask. This should be a matrix of shape (_len, _len). 
        # This mask is multiplicative
        # setting mask[i,j] = 0 means jth element of the sequence is not used 
        # to predict the ith element of the sequence.
        return mask
                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


