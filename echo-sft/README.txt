I tried to do SFT with ECHO, but I don't think it worked. Firstly, I did
end up completing training (~27 hours/$43), and when it was finished, I
told Cursor to test out the model on some SWE tasks and download it, but
I was in the middle of making dinner and so, around 30 minutes later, I
figured the downloading must've been done and I shut down the cluster
the model was trained on--which was a huge mistake. It was only around
two thirds downloaded and so what I had locally was garbage. Before
that, though, Cursor decided to do two runs (seriously? only two?) in
which one was successful (`django__django-11099`) and one was not
(`astropy__astropy-12907`). The successful one was apparently a pretty
easy one where the agent just had to find and modify a regex, but the
unsuccessful one was interesting because the model just scrolled through
a single file looking for something that wasn't even in that file. It, I
assume, didn't learn `grep`.

I think the issue was that SFT on trajectories is inherently a way to
train action prediction. It tries to learn what the teacher model does,
not what it sees. And I think even with hybrid optimization, learning
observations is tricky because, while everything needed to understand
why a particular action was chosen fits inside the trajectory, knowing
why calling a given tool has a given result requires knowing either the
environment to understand the tool, or the tool to understand the
environment. Trajectories tend to provide neither and so the model
fails to generalize what it's seeing. I think I need to train a model
to first understand some basic tools, which it can then use to
understand more tools.

On the upside, however, I think this means that it is not enough to
train a model on actions. I think there's a certain granularity of
knowledge that's just difficult to train through just rewarding a model
for doing well. I think, learning from this experiment, I want to just
focus on training observation prediction first, since I think it's
different enough and not as explored.