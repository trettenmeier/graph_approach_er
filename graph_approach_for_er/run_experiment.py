import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from graph_approach_for_er.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="pan_a21262b92b0c211d325da68a1c33c672")],
                workers=1,
                local_scheduler=True)

if __name__ == "__main__":
    main()
