# Illustrative curriculum, not an exact reproduction of the published checkpoint.
# Run from the repository root; install extras with: python -m pip install -e ".[test,logging]"
$ErrorActionPreference = 'Stop'
python run.py train scratch hard 1 100000 models/curriculum_01 --seed 42 --lr 0.0003 --eval-every 10000 --tensorboard-log runs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python run.py train models/curriculum_01.zip mixed 2 25000 models/curriculum_02 --seed 42 --lr 0.0003 --eval-every 10000 --tensorboard-log runs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python run.py train models/curriculum_02.zip mixed 2 25000 models/curriculum_03 --seed 42 --lr 0.0003 --eval-every 10000 --tensorboard-log runs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python run.py train models/curriculum_03.zip mixed 3 25000 models/curriculum_04 --seed 42 --lr 0.0003 --eval-every 10000 --tensorboard-log runs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python run.py train models/curriculum_04.zip mixed 2-5 10000 models/curriculum_05 --seed 42 --lr 0.0001 --eval-every 10000 --tensorboard-log runs
exit $LASTEXITCODE
