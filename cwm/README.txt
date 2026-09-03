Ever since hearing about world models, I've started to believe that
world models are the future to all of AI. Basically a world model is
just a model's understanding of the world. You may have heard of world
models like JEPA, which use its understanding of the world to do stuff
like image and video generation and robotics, but really a world model
is just any model of the world. For instance, LLMs like Othello-GPT
have shown that LLMs can form internal world models in order to handle
tasks in those worlds. That being said, I think we'll start seeing a
rise in models being trained with world models in mind, like Facebook
Research's CWM, because it enables models to actually reason about its
world, rather than following reasoning techniques from SFT or be
dependent on an actual world like RL.

Another prediction I have for world models is their application to
test-time training. Humans basically live their entire lives doing
test-time training, and yet, they don't require tens of thousands of
examples to learn something like an LLM would. I think this boils down
to humans having well-developed world models and so learning something
typically builds around the world model, leaving it mostly unchanged.
There are obvious edge cases like, for example, when I learned how to
code, it was really difficult and I had to reframe a lot of how I
thought about things. But I think this just further proves the world
model hypothesis: that I had to train a new world model for coding,
which required a lot of training, but now that I know how to code,
learning new CS topics is easy since I'm training around the world
model. I believe that this is the key to test-time training. If a model
has a mature world model, test-time training will be lightweight and
require much fewer examples.

Altogether, I think this will help make desktop models a much realer
reality. If we can train models to have thorough, accurate world models
and be able to test-time train on them, models for personal use would
be able to reach or even beat the abilities of much larger pretrained
models.