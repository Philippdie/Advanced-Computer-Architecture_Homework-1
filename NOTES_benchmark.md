## Benchmarks

Every result is in seconds.

### Data

**Follow line**
* `total_time` (for 100 iterations): 0.980788
* `single_iteration_time`: 0.009808

**Recognize object**
* `total_time` (for 30 iterations): 7.368362
* `single_iteration_time`: 0.245612

**Infrared obstacle check**
* `total_time` (for 100 iterations): 0.042494
* `single_iteration_time`: 0.000425

### Results

The *object recognition* is the only real heavy task, taking ≈0.25 seconds.

The *line following* and the *infrared obstacle check* take ≈1 millisecond each.
