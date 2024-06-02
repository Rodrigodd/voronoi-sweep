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

At time of writing, the algorithm is fully functional, but I still need to work
on performance improvements (it still not $`O(n \log n)`$), refactor the code,
and do more extensive testing.

## Animation and Explanation

You can see a really cool animation of the inner workings of the algorithm by
running the following command:

```sh
cargo run --example fortune-anim
```

[Link to animation](https://github.com/Rodrigodd/voronoi-sweep/assets/51273772/18e6957b-d9c7-4866-9b43-e67f03e7240a)

In the animation, black dots are site events or intersection events. The
horizontal line is the sweepline, representing the beachline in $`*`$-mapping
space, and the green hyperbolas are the $`*`$-mapped bisectors. Below the
sweepline is the partially built Voronoi diagram, where the parabolas represent
the non-$`*`$-mapped beachline. Each parabola is a section of the benchline,
delimeted by the green dots.

The algorithm processes events in order from bottom to top:
- On site events, a new bisector is added to the beachline, adding events for
  any $`*`$-mapped intersections with its neighbors, and invalidating
  intersections of no-longer neighboring bisectors.
- On intersection events, a new vertex is added to the Voronoi diagram (or a new
  triangle to the Delaunay diagram), and the two bisectors are merged into a
  single bisector of the two highest points, invalidating intersections with
  no-longer existing bisectors.

The mapping $`*`$ is defined as $`*(z) = (z_x, z_y + d(z))`$, where $`d`$ is the
distance to the closest site. I did not read all the lemmas and theorems in
Fortune's paper until understanding it, but my understanding is that this map
allows the algorithm to commit to a vertex only after finding all 3 closest
points.

Later papers describe this algorithm without mentioning the mapping $`*`$,
working directly with the parabolas shown in the animation, but that was not
considered in my implementation, only included in the animation for visual
flair.

## References

- S.J. Fortune, A sweepline algorithm for Voronoi diagrams, Algorithmica 2 (1987) 153–174.
- Kenny Wong, Hausi A. Müller, An Efficient Implementation of Fortune's Plane-Sweep Algorithm for Voronoi Diagrams
