# Attention-Guided Neural Networks for Low-Resource Machine Translation

## Abstract

Neural machine translation has advanced rapidly over the past decade, yet
low-resource language pairs remain challenging. In this work we introduce an
attention-guided training scheme that reuses monolingual data to regularize a
sequence-to-sequence model. We evaluate on three language pairs and report
consistent BLEU improvements. Our approach requires no additional parallel
corpora and integrates cleanly with existing transformer architectures [1].

## Introduction

Machine translation systems built on the transformer architecture have become
the dominant paradigm [1]. These models rely on self-attention to capture
long-range dependencies between tokens. However, their performance degrades
sharply when parallel training data is scarce, a common situation for the vast
majority of the world's languages. Prior work has explored back-translation as
a means of exploiting monolingual corpora [2], and semi-supervised objectives
have also been proposed. Despite this progress, a principled way to guide
attention using unlabeled data has not been fully explored.

We argue that the attention distribution itself carries a useful training
signal. When a model is uncertain, its attention tends to spread diffusely
across the source sequence. By encouraging sharper, more consistent attention
on monolingual data, we can regularize the model without any new parallel
sentences. This paper makes three contributions. First, we formalize an
attention-consistency loss. Second, we show it is complementary to
back-translation. Third, we release our training code for reproducibility.

## Methodology

Our model follows the standard encoder-decoder transformer of Vaswani and
colleagues [1]. Given a source sentence, the encoder produces contextual
representations that the decoder attends to at each step. We add an auxiliary
objective computed on monolingual target-language text. Specifically, we run
the decoder in a language-modeling mode and penalize the divergence between the
attention distributions of two stochastically augmented views of the same
sentence. The augmentation consists of token dropout and span masking, similar
in spirit to contrastive representation learning [3].

The total loss is a weighted sum of the standard cross-entropy translation loss
and the attention-consistency term. We tune the weighting coefficient on a
development set for each language pair. Optimization uses Adam with a warmup
schedule. All experiments were run on a single GPU, and each configuration was
repeated three times with different random seeds to estimate variance.

## Results

Across all three language pairs we observe BLEU gains ranging from 1.2 to 2.8
points over a strong back-translation baseline. The improvements are largest in
the lowest-resource setting, which supports our hypothesis that the
attention-consistency signal is most valuable when parallel data is scarce.
Ablation studies confirm that both token dropout and span masking contribute to
the final result. Removing the consistency term entirely recovers the baseline,
indicating that the gains are attributable to our proposed objective and not to
incidental hyperparameter changes [4].

## Discussion

The results suggest that attention distributions are an underused source of
self-supervision. Unlike back-translation, our method does not require an
auxiliary reverse model, which simplifies the training pipeline and reduces
compute. A limitation is that the augmentation hyperparameters must be tuned per
language pair. Future work could explore learned augmentation policies and
extend the approach to multilingual settings where several low-resource
languages are trained jointly.

## Conclusion

We presented an attention-guided training scheme that improves low-resource
machine translation using only monolingual data. The method is simple, adds
little computational overhead, and is complementary to existing techniques. We
hope this encourages further study of attention as a training signal rather than
merely an interpretability tool.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in
Neural Information Processing Systems, 30.

[2] Sennrich, R., Haddow, B., & Birch, A. (2016). Improving neural machine
translation models with monolingual data. Proceedings of the 54th Annual
Meeting of the Association for Computational Linguistics.
https://doi.org/10.18653/v1/P16-1009

[3] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple
framework for contrastive learning of visual representations. Proceedings of
the 37th International Conference on Machine Learning.

[4] Fictional, A. B., & Nonexistent, C. D. (2023). A completely fabricated paper
about attention consistency that does not exist anywhere.
https://doi.org/10.9999/this.doi.is.fake.2023
