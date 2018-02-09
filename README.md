## nlp-gan

This is a project I started for fun in the summer of 2017 after reading about GANs and wondering if they could be applied to natural language processing. 

The primary challenge of applying GANs to NLP is that language is generally a discrete space (each word is a distinct point), and GANs require a continuous output space in order to propagate gradients back and forth between the discriminator and the generator. My solution to this problem is essentially just to use word vectors as a continuous input/output space. The outputs of the generator do not necessarily fall directly on existing words, so they are really more of a sequence of “meanings” than discrete English words. In order to get actual text back from the generator for humans to read, I preform a nearest-neighbor search among my dictionary of word vectors.

For the purposes of this project, I used pre-trained word vectors from GloVe, which you can find [here]( https://nlp.stanford.edu/projects/glove/) (not included in repo).

For both the generator and the discriminator, I used a basic LSTM architecture with no peephole connections.

Training the network proved difficult, and I often ran into mode collapse with the generator. I tried to cripple the discriminator using dropout and instance noise, but I found most of my improvements in avoiding collapse came from adjusting my training schedule.

Towards the end, I began to produce sentences that had variety and were sometimes somewhat grammatically correct. However, they were still nowhere near convincing nor did they compare to other, simpler, NLG techniques. I abandoned the project in the fall as I got busy with school (and college apps). The endeavor was primarily a learning experience, but who knows, I may come back to it someday.
