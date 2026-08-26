'''
负责融合排序。

根据意图权重：

Score = w_text * S_text
      + w_simclr * S_simclr
      + w_clip * S_clip
      + w_spatial * S_spatial
      + w_semantic * S_semantic
      + w_meta * S_meta

'''