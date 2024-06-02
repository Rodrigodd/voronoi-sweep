# Voronoi-Sweep

This is a almost direct implementation of Fortune's original description of his
sweepline algorithm for generating voronoi diagrams, in the Rust programming
language.

There are many small extensions to the original algorithm to make it handle
non-unique bottommost points (based on Kenny Wong's paper), and many small
corner cases (like coincident sites, sites coincident with intersection events,
sites lying on hyperbolas, etc.) that are not explicitly handled by the original
algorithm.

Currently it only implements the algorithm for point sites, thought the core
algorithm can be hopefully extended to handle line segments and other types of
sites.

## Animation

You can run a really cool animation of the algorithm by running the following
command:

```sh
cargo run --example fortune-anim
```

## References

- S.J. Fortune, A sweepline algorithm for Voronoi diagrams, Algorithmica 2 (1987) 153–174.
- Kenny Wong, Hausi A. Müller, An Efficient Implementation of Fortune's Plane-Sweep Algorithm for Voronoi Diagrams
- https://jacquesheunis.com/post/fortunes-algorithm/
