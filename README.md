# R222 Project

## Requirements

- Python
- NumPy
- SciPy

## Warning

A serious experiment uses a lot of resources. For reference, a text file containing 47 million 300-dimensional vectors takes up 221 GB of space. It takes hours to write and read such a file.

## Input

At the top level, create the directories `aux`, `data`, and `output`:

```
aux
> animals.txt       # one word per line
> countries.txt     # one word per line
> degree_input.txt  # line format: [adjective] [noun];[target word]
> nearest_input.txt # line format: [adjective] [noun] OR [word]
> occupations.txt   # one word per line
> sports.txt        # one word per line
data
> vectors-sk-an.lemmas.50-min-count.15-iters
```

`output` is empty.

## Running

All logging is saved in `output`.

To create `adjectives.txt`, `an.txt`, and `nouns.txt` in `data`, run:

```
python -m bin.create_an_lexicon
```

Run the following to create the true and false vectors for each variant of the conjunction operator:

```
python -m bin.conj1
python -m bin.conj2
python -m bin.conj3
python -m bin.conj4 # unsupported, takes ages, don't do this!
```

The first experiment is run using:

```
python -m bin.sim_expt
```

It will ignore a missing `conj4.txt` in `data`.

The second experiment is run using:

```
python -m bin.gen_space
```

which takes literally ages (and tons of space). Parallelise it by modifying the `NUM_PROCESSES` variable in `bin/gen_space.py`, according to the number of processors you have.

This is followed by running

```
python -m bin.nearest_expt
```

for the actual experiment. This may take ages too, because reading the vectors from disk can be slow.

The final experiment is run using:

```
python -m bin.degree_expt
```

## Notes

- The `maxiter` parameter for SciPy's `optimize.minimize()` method can be changed in `r222/utils.py`.
- `best_sn()` (of `conj4`) is currently unsupported. I have never seen it complete _one_ iteration; it takes that long.
- There are multiple versions of `dotkron()`, `big_dotkron()`, and `big_dotkron_single()` in `r222/utils.py`, using different methods (`einsum`, `dot`, `tensordot`) to achieve the same result with different time and space requirements.


