import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import clip model
import transformers
from transformers import AutoTokenizer, CLIPTextModel




from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(f'{os.getcwd()}/AudioCLIP/audioclip')))


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()


        self.model = transformers.CLIPTextModel.from_pretrained(model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)


        for param in self.model.parameters():
            param.requires_grad = trainable


        self.target_token_idx = 0


    def forward(self, input_text):
        input_ids = self.text_tokenizer(input_text, padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # outputs = self.text_tokenizermodel(**inputs)
        # last_hidden_state = outputs.last_hidden_state # output.last_hidden_state.shape is [2,7,512]
        # pooled_output = outputs.pooler_output  # pooled (EOS token) states
        output = self.model(**input_ids)
        last_hidden_state = output.last_hidden_state


        return last_hidden_state
    
class ProjectionHead(nn.Module):
    # Projection head for CLIP
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)


        x += projected


        return self.layer_norm(x)



class BottleNeckAudioVideo(nn.Module):
    def __init__(self, num_layers, d_size = 256, in_size_video=512, in_size_audio=128, out_size=512) -> None:
        super().__init__()
        self.num_layers = num_layers
        if type(d_size) == int:
            self.d_size = [d_size]*num_layers
        else:
            self.d_size = d_size
        
        self.linear_down_v = nn.Linear(in_size_video, self.d_size[0])
        self.linear_down_a = nn.Linear(in_size_audio, self.d_size[0])
        self.layers_video = nn.ModuleList()
        for i in range(num_layers):
            self.layers_video.append(TransformerBlock(
                    self.d_size[i], 8,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.5,
                    proj_pdrop=0.5,
                    mha_win_size=19,
                    # use_rel_pe=self.use_rel_pe
                ))
        
        self.layers_audio = nn.ModuleList()
        for a in range(num_layers):
            self.layers_audio.append(TransformerBlock(
                    self.d_size[a], 8,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.5,
                    proj_pdrop=0.5,
                    mha_win_size=19
                ))
        self.layers_audio.append(nn.Linear(self.d_size[-1], out_size))
        
        self.layers_out = nn.ModuleList()
        for c in range(num_layers):
            self.layers_out.append(nn.Conv1d(self.d_size[-1]*2, out_size, 1))
        
        
        self.out = nn.Conv1d(out_size*num_layers, out_size, 1)
        
            
            

        
    def forward(self,embeddings_video, embeddings_audio, mask) -> torch.Tensor:
        assert len(embeddings_video) == len(embeddings_audio) == self.num_layers, "embeddings_video and embeddings_audio must have the same length"
        # assert embeddings_video[0].shape == embeddings_audio[0].shape, "embeddings_video and embeddings_audio must have the same shape"
        
        
        bfl = []
        for i in range(self.num_layers):
            em_vid = self.linear_down_v(embeddings_video[i].transpose(1,2)).transpose(1,2)
            em_vid = F.tanh(em_vid)
            em_aud = self.linear_down_a(embeddings_audio[i].transpose(1,2)).transpose(1,2)
            em_aud = F.tanh(em_aud)
            embedding_video, mask = self.layers_video[i](em_vid,mask,text=em_aud,cross_attn=True)
            embedding_audio, mask = self.layers_audio[i](em_aud,mask,text=em_vid,cross_attn=True)
            cat = torch.cat((embedding_video, embedding_audio), dim=1)
            bfl.append(self.layers_out[i](cat))
            
            # i += 2 # increment the layer number
        
        return self.out(torch.cat(bfl, dim=1)).transpose(1,2)
    

class BottleNeckAudioVideoRMAttn(nn.Module):
    def __init__(self, out_size = 256, in_size_video=512, in_size_audio=128, stem_size=512) -> None:
        super().__init__()
        
        self.out_size = out_size
            
        
        self.linear_down_v1 = nn.Linear(in_size_video, stem_size)
        self.linear_down_a1 = nn.Linear(in_size_audio, stem_size)
        self.linear_down_v2 = nn.Linear(stem_size, stem_size)
        self.linear_down_a2 = nn.Linear(stem_size, stem_size)
        
        self.linear_out = nn.Linear(stem_size, self.out_size)
        
        
    def forward(self,embeddings_video, embeddings_audio, mask) -> torch.Tensor:
        assert len(embeddings_video) == len(embeddings_audio), "embeddings_video and embeddings_audio must have the same length"
        assert embeddings_video[-1].shape[-1] == embeddings_audio[-1].shape[-1], "embeddings_video and embeddings_audio must have the same shape"
        
        # take the last embedding
        embeddings_audio = embeddings_audio[-1]
        embeddings_video = embeddings_video[-1]
        
        # audio
        em_aud = self.linear_down_a1(embeddings_audio.transpose(1,2))
        em_aud = F.tanh(em_aud)
        em_aud = self.linear_down_a2(em_aud)
        
        # video
        
        em_vid = self.linear_down_v1(embeddings_video.transpose(1,2))
        em_vid = F.tanh(em_vid)
        em_vid = self.linear_down_v2(em_vid)
        
        # cross addition
        
        em = torch.add(em_vid, em_aud)
        
        # resudulal connection audio and video
        
        em_aud = torch.add(em_aud, em)
        em_vid = torch.add(em_vid, em)
        
        # tanh on both
        
        em_aud = F.tanh(em_aud)
        em_vid = F.tanh(em_vid)
        
        # addition
        
        em = torch.add(em_vid, em_aud)
        
        em = self.linear_out(em)
        em = F.gelu(em)
        
        return em.transpose(1,2)
        




class ConvTransformerBackbone_StemOnly(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        path_pretrained=None,   # path to pretrained model
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        
        

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

       
        if( path_pretrained is not None):
            try:
                print("Loaded pretrained af model")
                print(self)
                checkpoint = torch.load(path_pretrained)
                self.load_state_dict(checkpoint['state_dict'], strict=False)
            except:
                print("Could not load pretrained af model")
                pass
        else:
             # init weights
            self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        outs_video = []
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
            outs_video.append(x)

        return x, mask, outs_video

@register_backbone("convTransformer")
class ConvTransformerBackbone(ConvTransformerBackbone_StemOnly):
    
    def forward(self, x, mask):
        x, mask, _ = super().forward(x,mask)        # main branch with downsampling
        
        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("AVFusionConvTransformer")
class AVFusionConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5, 2, 5),      # (#convs, #video stem transformers, #branch transformers, #bottle neck transformers #audio stem)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        use_text = False,      # use text embedding
        use_audio = False,      # use audio embedding
        aformer_path = None,
        freeze_aformer = False,
        **kwargs
    ):
        super().__init__()
        assert len(arch) == 5
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_text = use_text
        self.use_audio = use_audio
        self.audio_embed = 128
        
        if use_text:
            self.text_encoder = TextEncoder("openai/clip-vit-base-patch32", trainable=True)
            self.text_embedder = ProjectionHead(512, 512, 0.1)
        elif use_audio:
            self.video_stem = ConvTransformerBackbone_StemOnly(n_in,                  # input feature dimension
                n_embd,                # embedding dimension (after convolution)
                n_head,                # number of head for self-attention in transformers
                n_embd_ks,             # conv kernel size of the embedding network
                max_len,               # max sequence length
                arch=arch[:3],      # (#convs, #video stem transformers, #branch transformers, #bottle neck transformers #audio stem)
                mha_win_size = [-1]*6, # size of local window for mha
                scale_factor = 2,      # dowsampling rate for the branch
                with_ln = False,       # if to attach layernorm after conv
                attn_pdrop = 0.0,      # dropout rate for the attention map
                proj_pdrop = 0.0,      # dropout rate for the projection / MLP
                path_pdrop = 0.0,      # droput rate for drop path
                use_abs_pe = False,    # use absolute position embedding
                use_rel_pe = False,    # use relative position embedding
                path_pretrained=aformer_path
            )
            
        else:
            raise ValueError("Must use either text or audio")
            
        
            
        # stem network using (vanilla) transformer
        self.stem_audio = nn.ModuleList()
        for idx in range(arch[4]):
            self.stem_audio.append(
                TransformerBlock(
                    self.audio_embed, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
        self.n_fusion_layers = arch[3]
        # stem network using (vanilla) transformer
        # self.bottle_neck = BottleNeckAudioVideo(self.n_fusion_layers, d_size=n_embd//2, out_size=512)
        self.bottle_neck = BottleNeckAudioVideoRMAttn(out_size=512, in_size_video=n_embd, in_size_audio=self.audio_embed, stem_size=self.audio_embed)

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_video, mask_video, kv, cross_attn=False):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        if (self.use_text or self.use_audio) and kv is None:
            # Throw an error if we're using text but don't have any
            raise ValueError("text is None but use_text is True")
        
        if cross_attn:
            #print("text", len(text))
            # randomly turn on/off the text embedding
            if(self.use_text):
                # get the text embedding for this layer
                text_enc = self.text_encoder(kv) # get token embeddings
                x_audio = self.text_embedder(text_enc) # get projection head embeddings from the CLIP model  
            elif(self.use_audio):
               x_audio = kv.transpose(1,2)
            else:
                x_audio = x_video
            
        B, C, T = x_video.size()
        outs_audio = []
        
        # stem transformer
        if self.use_text:
            raise NotImplementedError("text embedding not implemented")
        else:
            # Use audio
            x_video, mask_video, outs_video = self.video_stem(x_video, mask_video)
        
        # Filter out the layers we want to fuse
        tmp = []
        for idx in range(len(outs_video)):
            if(len(outs_video) - idx <= self.n_fusion_layers):
                tmp.append(x_video)
        outs_video = tmp
        
        for idx in range(len(self.stem_audio)):
            x_audio, mask_audio = self.stem_audio[idx](x_audio, mask_video)
            if(len(self.stem_audio) - idx <= self.n_fusion_layers):
                outs_audio.append(x_audio)
                
        
                
        x = self.bottle_neck(outs_video, outs_audio, mask_video)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask_video, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask_video = self.branch[idx](x, mask_video)
            out_feats += (x, )
            out_masks += (mask_video, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks