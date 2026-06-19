from src.parameter_space import (
    DiffPairParameterSpace
)

from src.spice_generator import (
    DiffPairSpiceGenerator
)

from src.simulation_runner import (
    NgspiceRunner
)

from src.dataset_builder import (
    DatasetBuilder
)

space = DiffPairParameterSpace()

samples = space.generate_samples(5)

space.save_samples(samples)

generator = DiffPairSpiceGenerator()

netlists = generator.generate_batch(samples)

runner = NgspiceRunner()

results = runner.run_batch(netlists)

print("\nSimulation Results:\n")

for r in results:
    print(r)
    
runner.save_results(results)

builder = DatasetBuilder()

dataset = builder.build_dataset(
    samples,
    results
)

builder.save_dataset(dataset)