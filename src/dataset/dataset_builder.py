import json
import os


class DatasetBuilder:

    def build_dataset(
        self,
        samples,
        simulation_results
    ):

        dataset = []

        for sample, result in zip(
            samples,
            simulation_results
        ):

            entry = {
                **sample,
                **result
            }

            dataset.append(entry)

        return dataset

    def save_dataset(self, dataset):

        os.makedirs(
            "data/datasets",
            exist_ok=True
        )

        filepath = (
            "data/datasets/final_dataset.json"
        )

        with open(filepath, "w") as f:

            json.dump(dataset, f, indent=2)

        print(
            f"\nDataset saved to {filepath}"
        )