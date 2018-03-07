## nlg-gan

This is a project I started for fun after reading about GANs and wondering if they could be applied to natural language processing. The project began with the intent of collaborating with other people, which is why it is structured and documented as such.

The primary challenge of applying GANs to NLP is that language is generally a discrete space (each word is a distinct point), and GANs require a continuous output space in order to propagate gradients back and forth between the discriminator and the generator. My solution to this problem is essentially just to use word vectors as a continuous input/output space. The outputs of the generator do not necessarily fall directly on existing words, but can be interpreted more as "meanings" in the word vector space. In order to get actual text back from the generator for humans to read, I preform a nearest-neighbor search among the dictionary of word vectors.

For the purposes of this project, I used pre-trained word vectors from GloVe, which you can find [here]( https://nlp.stanford.edu/projects/glove/) (not included in repo).

For both the generator and the discriminator, I used a basic LSTM architecture with no peephole connections.

Training the network proved difficult, and I often ran into mode collapse with the generator. Adding dropout and instance noise helped some, but I found most of my improvements in avoiding collapse came from adjusting my training schedule.

Towards the end, I began to produce sentences that had variety and were sometimes somewhat grammatically correct. However, they were still nowhere near convincing nor did they compare to other, simpler, NLG techniques. The endeavor was primarily a learning experience, helping familiarize myself with Tensorflow and deep learning in general.
