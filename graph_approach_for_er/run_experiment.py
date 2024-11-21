import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from graph_approach_for_er.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="same_source_0de51797d9a6c73a84ef8f72a7e35914")],
                workers=1,
                local_scheduler=True)


if __name__ == "__main__":
    main()
