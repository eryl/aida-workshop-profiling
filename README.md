# AIDA Technical Workshop - Profiling and improving pytorch experiments

This repository contain material for the AIDA technical workshop on profiling and improving pytorch experiments. The workshop is intended to be run in a terminal. To have a similar experiment environment, we can use google colab and create terminals in a notebook by following this [guide](https://colab.research.google.com/github/eryl/aida-workshop-profiling/blob/main/profiling_experiment.ipynb).

You can find the supporting presentation [here](https://docs.google.com/presentation/d/1DU_cAQYHfvAD4vPH91oH5huNqX0SoAnz/edit?usp=sharing&ouid=116285713938211495704&rtpof=true&sd=true).

If you want to use your local computer, you can create a conda environment with the supplied `environment.yml`:
```shell
$ conda env create -f environment.yml
$ conda activate dataloading_workshop
```

# Part 1: the inefficient script

We'll be working with the training script in `train.py`. This is intended to serve as a minimal experiment example, with the kind of common issues you find in running pytorch experiments. These mainly have to do with data loading and we'll look at how we can diagnose the issues and some common techniques to solve them.

The training script takes a configuration file as input which is located in `configs/example_config.py`, and we start the training by running:

```shell
$ python train.py configs/example_config.py
```

This should first download the dataset we will use: the Oxford III Pets dataset and after that training should start. While training is progressing, you might as yourself: "is this really the fastest it can go?" and our first step will be to look at resource utilization.

There are many good utilities for looking at this, but here we'll use `nvitop` since it's easy to install on colab. If you've followed the instructions in the colab notebook or created the conda envrionment, it should already be installed. Otherwise you can install it running:

```shell
$ pip install nvitop
```

You can now run `nvitop` in a terminal and it should display the CPU and GPU usage.

**Can you say anything about CPU and GPU utilization?**

# Part 2: Identify the inefficiencies

While this script could be easy to analyze manually, more complex training pipelines are more challanging and it's good to have tools which allows us to inspect the runtime. There are many so called _profilers_ which allow you to do that, but a light weight easy to use one is `line_profiler` which we will use here. Installing the package is easily done with pip:

```shell
$ pip install line_profiler
```

We tell line_profiler what parts of our code to profile by wrapping functions with the `@profile` decorator, for example to profile the function below:

```python
def function_to_profile(*args, **kwargs):
    # Here's the function we want to look at which does lots of stuff
```

We add `@profile` before the function definition:

```python
@profile
def function_to_profile(*args, **kwargs):
    # Here's the function we want to look at which does lots of stuff
```

We can then profile the program by substituting `python` in our usual call with `kernprof -l`:

```shell
$ kernprof -lv train.py configs/example_config.py
```

This then starts the training and samples what functions are being called as the program is running. Once the process exits (by us pressing `CTRL-C` for example) a runtime profile is generated with the name `train.py.prof`. This file is in a binary format, but we can inspect it calling:

```shell
$ python -m line_profiler train.py.prof
```

Which will display the profiling information which looks something like this:
```
Pystone(1.1) time for 50000 passes = 2.48
This machine benchmarks at 20161.3 pystones/second
Wrote profile results to pystone.py.lprof
Timer unit: 1e-06 s

File: pystone.py
Function: Proc2 at line 149
Total time: 0.606656 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   149                                           @profile
   150                                           def Proc2(IntParIO):
   151     50000        82003      1.6     13.5      IntLoc = IntParIO + 10
   152     50000        63162      1.3     10.4      while 1:
   153     50000        69065      1.4     11.4          if Char1Glob == 'A':
   154     50000        66354      1.3     10.9              IntLoc = IntLoc - 1
   155     50000        67263      1.3     11.1              IntParIO = IntLoc - IntGlob
   156     50000        65494      1.3     10.8              EnumLoc = Ident1
   157     50000        68001      1.4     11.2          if EnumLoc == Ident1:
   158     50000        63739      1.3     10.5              break
   159     50000        61575      1.2     10.1      return IntParIO
```

We get one table for each function we have decorated with `@profile`. The table contains the listing of the function code. This is not recorded in the profile, but read from the source file when you run the display, which means that if you've changed your file since profiling you can't trust the values.

Often the interesting part is looking at `% Time`, this tells us how much of total runtime this line accounts for.


Decorate the function `main()` in the script `train.py` and run it like above.

**Which lines is our training spending most of its time in?**

## NameError: name 'profile' not defined
As you are experimenting with the code, you might get the error:

```
Exception has occurred: NameError
name 'profile' is not defined
  File "/home/erik/src/dataloading_workshop/train.py", line 88, in <module>
    @profile
     ^^^^^^^
NameError: name 'profile' is not defined
```

This is because the `profile` decorator is defined in a preamble executed when we run the script through `kernprof`, but is not defined if we run the script directly with the regular python. To solve this we can add the following snippet in the beginning of our script:

```python
try:
    @profile
    def foo():
        return
except NameError:
    def profile(f):
        return f
```
This checks whether `profile` exists in the global namespace, and if not it creates the "no-op" decorator `profile`

# Part 3: First fix

The finding in the profiling above should be that a lot of time is spent on the line which begins the main training-loop, at the point where we iterate over the batches. The reason for why this is slow is that we are loading the data on the _critical path_ of training. All the transformations (and simulated slowness) all the slow parts of data loading blocks training progress because we're doing it sequentially.

We can easily fix this by using the built-in _multiprocessing_ capabilites of the `DataLoader`:
Change:
```python
training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size)
```
To:
```python
training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_processes=2)
```

Any value of `num_processes > 0` will make the loading of data take place in a seperate process from the one doing the training. This allows the seperate processes to fetch data from the dataset at the same time as we're training. With enough CPU cores, we can set this number to something large like 12, and then at least that many batches will be created in parallell so when we ask for a new batch from the dataloader, there will likely be one waiting for us.

**Try setting num processes to some positive number, what happens?**

# Part 4: Inspect the data loading
When we increase the number of processes in the dataloader, we can offload expensive CPU calculations to other processes, but that might not be the actual bottleneck. By looking at the performance timelines in `nvitop` you might see that the CPU utlilization is not maxed out, even though we use more processes than cores on the machine. We can inspect the loading in our dataset.

**add a `@profile` decoration before the call to `__getitem__()` in the `CustomSlowDataset`. Change `num_processes=0` in the dataloader. Run a profiling session and report on the result.**

It might be that our dataloader is not bottlenecked by needing too much CPU, but instead by IO bottlenecks. If that is the case there are some things we can try if the IO bottleneck is hardware-dependent (e.g. data is on network attached storage) we need to solve it at another level than our training script (changing where data is stored, how its stored, etc.).

# Part 5: Fixing compute bottlenecks with caching
Now we might end up in a situation where we're saturating all of our CPU cores, so we can't speed up dataloading further by parallalizing the process. What we can do instead is to trade compute with storage by pre-calculating batches and store them to disk. We might be able to do this on another computer with more CPU cores or make rerunning experiments faster. 

**Try implementing a wrapper for your slow dataset which first check whether there's a cached version of the item on disk before taking it from the wrapped dataset**


# Part 6: Pre-fetching batches

Another issue might be that the bottleneck is not in preparing the data from disk, but actually transferring the data to the GPU. This will show up as taking long time when you make a call like this (having set `CUDA_LAUNCH_BLOCKING=1`):

```python
x = x.to(torch.device('cuda'))
```

If that line is taking a lot of time from training, it's likely because `x` is large and takes a lot of time to transfer to the GPU. We can get around this by actually calling this transfer on the previous batch (remember to set `CUDA_LAUNCH_BLOCKING=0`, otherwise this won't work):

 ```python
    for epoch in tqdm.trange(config.max_epochs, desc='Epoch'):
        training_pbar = tqdm.tqdm(training_dataloader, desc='Training batch')
        n_batches = len(training_dataloader)
        
        next_batch = next(training_pbar)
        next_input, next_target = next_input.to(device), next_target.to(device)
        for i in range(n_batches - 1):
            optimizer.zero_grad()
            input, target = next_input, next_target
            next_batch = next(training_pbar)
            next_input, next_target = next_input.to(device), next_target.to(device)

            input = input.to(device)
            target = target.to(device)
            logits = model(input)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            historic_losses.append(loss.item())
            if i % config.update_iterations == 0:
                mean_loss = mean(historic_losses)
                training_pbar.set_description(f'Training batch ({mean_loss})')
                
        if hasattr(training_dataset, 'next_epoch'):
            training_dataset.next_epoch()
 ```

 **Why would this not work if we have `CUDA_LAUNCH_BLOCKING=1`?**