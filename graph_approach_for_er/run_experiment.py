import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from graph_approach_for_er.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="lfw_1ac3213e42d6c0d9b264c3230adc905a")],
                workers=1,
                local_scheduler=True)

    # baseline: lfw_31df52e921ad7d09a6f2640c6f524804

if __name__ == "__main__":
    main()
