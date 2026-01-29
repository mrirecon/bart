# Streaming

In a typical BART reconstruction, the user invokes several BART
commands to achieve a desired result.
Every intermediate result in the chain of BART commands is
stored on disk, resulting in two files (hdr & cfl) per reconstruction step.

Instead of storing intermediate results on disk,
the user can also *stream* BART output.

By specifying a dash `-` as the input/output filename,
one can advise BART to stream data to standard output (stdout), or
to read data from standard input (stdin).
Using the pipe `|` operator available in common Unix shells,
multiple (BART) processes can be started with
their stdout linked to the stdin of the subsequent command.

For example, these commands produce the same result:
```
 $ bart phantom -k ksp; bart fft -i 3 ksp img
 $ bart phantom -k - | bart fft -i 3 - img
```

In case b), no persistent temporary file (ksp) is stored on disk,
and both processes (`phantom` and `fft`) actually
run in parallel.
The `fft`-tool still waits for the `phantom`-tool to produce data.

One can also stream multiple input/output files using named pipes.
These are detected in BART based on file-naming convention, ie
a named pipe must always end in `.fifo`. For example:
```
bart traj -r -o2 -              |\
bart tee trj.fifo               |\
bart phantom -k -t - -          |\
bart nufft -i -t trj.fifo - img
```
Here,
- `traj` creates a trajectory and writes it to stdout
- `tee` receives the trajectory on stdin and duplicates it,
  writing it to
    - a newly created fifo trj.fifo and
    - stdout
- `phantom` receives the trajectory on stdin
  and outputs a phantom k-space on stdout.
- `nufft` receives the trajectory from trj.fifo
  and the k-space on stdin.
- The reconstruction is written to a file img.{cfl,hdr}

The trj.fifo-file is automatically created and deleted in this job.

# Multidimensional Array Slicing

BART tools operate on multidimensional cfl-arrays.
Instead of operating on a complete array, BART can also
operate on *slices*.
A slice is a subset of the original array in which several indices are fixed.

In conjunction with streaming/looping,
this enables sequential and parallel processing
of data in BART reconstruction pipelines.

To illustrate sequential data processing, consider the following example.
Note that this requires a fairly recent version of view[^3].

```
bart phantom -B -                                           |\
    bart resize -c 0 200 1 200 - -                          |\
    bart conway -P -n1000 - -                               |\
    bart copy --stream $(bart bitmask 2) --delay 0.01 - -   |\
    view --real-time 2 -
```

- First, the `phantom`-tool produces a dataset showing the BART-Logo.
- This is `resize`d to 200x200 pixels using zero-padding.
- Then, `conway` simulates Conways game of life on the input dataset.
  The output contains iterations along the third dimension, ie dimension 2, counting from 0.
- The output is streamed - the subsequent processes receive every iteration
  as soon as it has been computed.
- The following `copy` delays every iteration by 10 milliseconds, for better visualization.
- The mrirecon viewer[^3] is then used to visualize the process in real-time.
    - Here, `--real-time 2` specifies that dimension 2 is streamed,
      and the viewer should update to new frames along this dimension.

See our preprint[^1] for precise description, more code examples[^2] and applications.

# Caveats

- Overwriting a file using several chained commands must be avoided:
  Expression like `bart copy x - | bart scale 2 - x` result in undefined output!
- Input/Output files are opened and processed in BART using a specific order,
  which can cause deadlocks in combination with streaming, when multiple inputs/outputs
  are streamed.
  For example, `bart phantom ksp; bart nlinv ksp img.fifo coils.fifo & bart fmac img.fifo coils.fifo cimg`,
  won't work.
- For efficiency, streams rely on local shared memory via mmap. To send output to a remote target,
  the `bart --stream-bin-out`-option can be used.

# References

[^1]: https://arxiv.org/abs/2512.02748 BART Streams: Real-time Reconstruction Using a Modular Framework for Pipeline Processing.

[^2]: https://gitlab.tugraz.at/ibi/mrirecon/bart-streams Code examples for BART Streams paper.

[^3]: https://codeberg.org/IBI-TUGraz/view mrirecon viewer
