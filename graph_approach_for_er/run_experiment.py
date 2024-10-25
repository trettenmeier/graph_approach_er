import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from graph_approach_for_er.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="mrsp_636144bca17ed3f0aa20963fc7191602")],
                workers=1,
                local_scheduler=True)  # the baseline for quora questions


if __name__ == "__main__":
    main()
