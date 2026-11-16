from argparse import ArgumentParser
from .run import STAGES, run_pipeline

parser = ArgumentParser(description="Run the mental health pipeline stages.")
parser.add_argument("stage", nargs="?", default="all", choices=[*STAGES, "all"])
args = parser.parse_args()

run_pipeline(args.stage)
