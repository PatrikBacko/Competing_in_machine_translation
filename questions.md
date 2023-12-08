- back-translation (how is it done)


ILM approximations
- separate LM trained on target language sentences from bilingual data
- h = 0: we take our LM model and set encoder output to 0
- h = h_avg: we take our LM model and set encoder output to the average of all encoder outputs (from whole parallel training data)

  - ?? then use it to predict the sencences and substracting the probabilities from the original LM ??


- ?? c_avg (dont understand) ??
- ?? mini-self-attn (dont understand) ??
