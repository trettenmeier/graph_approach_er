import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from graph_approach_for_er.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="mrsp_0c52f3736fe2fdbe788695bfc49c7cc5")],
                workers=1,
                local_scheduler=True)


if __name__ == "__main__":
    main()
