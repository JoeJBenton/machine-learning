This is a set of programs that together attempt to replicate the MCTS self-play and learning algorithms implemented by AlphaZero as explained in the following:
https://arxiv.org/pdf/1712.01815.pdf
https://www.nature.com/articles/nature24270.epdf
https://web.stanford.edu/~surag/posts/alphazero.html

At the minute, there is a problem with the learning rate of the position evaluation part of the neural network.
This is causing the learning of the valuation to stagnate, so the current version doesn't accurately evaluate positions.
