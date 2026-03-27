# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Unit tests for _validateCheckpoint in RavenSampled and GeneticAlgorithm.
  Uses lightweight stub objects to avoid instantiating abstract classes.
"""
import copy
import os
import sys

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()

from ravenframework.Optimizers.RavenSampled import RavenSampled
from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
results = {'pass': 0, 'fail': 0}


def checkTrue(comment, condition):
  if condition:
    results['pass'] += 1
  else:
    print(f'FAIL: {comment}')
    results['fail'] += 1


def checkRaises(comment, func, errType):
  """Assert that func() raises errType."""
  try:
    func()
    print(f'FAIL: {comment} | expected {errType.__name__} but no exception raised')
    results['fail'] += 1
  except errType:
    results['pass'] += 1
  except Exception as e:
    print(f'FAIL: {comment} | expected {errType.__name__}, got {type(e).__name__}: {e}')
    results['fail'] += 1


def checkNoRaise(comment, func):
  """Assert that func() does not raise."""
  try:
    func()
    results['pass'] += 1
  except Exception as e:
    print(f'FAIL: {comment} | unexpected {type(e).__name__}: {e}')
    results['fail'] += 1


# ---------------------------------------------------------------------------
# Stub classes
# ---------------------------------------------------------------------------
class _MockBase:
  """Minimal stub for testing RavenSampled._validateCheckpoint."""
  name = 'testOpt'
  toBeSampled = {'x': None, 'y': None}
  _objectiveVar = ['ans']
  _minMax = ['min']
  _warnings = None

  def __init__(self):
    self._warnings = []

  def raiseAnError(self, errType, *args):
    raise errType(*args)

  def raiseAWarning(self, *args):
    self._warnings.append(args[0] if args else '')


class _MockGA(_MockBase):
  """Minimal stub for testing GeneticAlgorithm._validateCheckpoint."""
  _populationSize = 10
  _isMultiObjective = False


def _baseValidate(mock, checkpoint):
  """Call RavenSampled._validateCheckpoint bound to mock."""
  RavenSampled._validateCheckpoint(mock, checkpoint)


def _gaValidate(mock, checkpoint):
  """Call GeneticAlgorithm._validateCheckpoint bound to mock."""
  GeneticAlgorithm._validateCheckpoint(mock, checkpoint)


# ---------------------------------------------------------------------------
# Base valid checkpoint (must match _MockBase attributes)
# ---------------------------------------------------------------------------
_VALID_BASE = {
  'version': '1.0',
  'optimizerType': '_MockBase',
  'optimizerName': 'testOpt',
  'settings': {
    'variables': ['x', 'y'],   # sorted, must match sorted(toBeSampled.keys())
    'objectiveVars': ['ans'],
    'minMax': ['min'],
  }
}

_VALID_GA = {
  'version': '1.0',
  'optimizerType': '_MockGA',
  'optimizerName': 'testOpt',
  'settings': {
    'variables': ['x', 'y'],
    'objectiveVars': ['ans'],
    'minMax': ['min'],
    'populationSize': 10,
    'isMultiObjective': False,
  }
}

# ---------------------------------------------------------------------------
# TC-18  Valid base checkpoint passes without error or warning
# ---------------------------------------------------------------------------
mock = _MockBase()
checkNoRaise('TC-18 valid base checkpoint', lambda: _baseValidate(mock, copy.deepcopy(_VALID_BASE)))
checkTrue('TC-18 no warnings issued', len(mock._warnings) == 0)

# ---------------------------------------------------------------------------
# TC-19  Version mismatch → warning only, no IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['version'] = '0.9'
checkNoRaise('TC-19 version mismatch no error', lambda: _baseValidate(mock, ckpt))
checkTrue('TC-19 version mismatch raised warning', len(mock._warnings) > 0)

# ---------------------------------------------------------------------------
# TC-20  Name mismatch → warning only, no IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['optimizerName'] = 'differentName'
checkNoRaise('TC-20 name mismatch no error', lambda: _baseValidate(mock, ckpt))
checkTrue('TC-20 name mismatch raised warning', len(mock._warnings) > 0)

# ---------------------------------------------------------------------------
# TC-21  Optimizer type mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['optimizerType'] = 'GeneticAlgorithm'
checkRaises('TC-21 type mismatch raises IOError', lambda: _baseValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-22  Variable set mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['settings']['variables'] = ['x', 'z']
checkRaises('TC-22 variable mismatch raises IOError', lambda: _baseValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-23  Objective variable mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['settings']['objectiveVars'] = ['cost']
checkRaises('TC-23 objective mismatch raises IOError', lambda: _baseValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-24  minMax mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockBase()
ckpt = copy.deepcopy(_VALID_BASE)
ckpt['settings']['minMax'] = ['max']
checkRaises('TC-24 minMax mismatch raises IOError', lambda: _baseValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-25  GA population size mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockGA()
ckpt = copy.deepcopy(_VALID_GA)
ckpt['settings']['populationSize'] = 20   # mock has 10
checkRaises('TC-25 GA popSize mismatch raises IOError', lambda: _gaValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-26  GA population size absent in checkpoint → no error (backward compat)
# ---------------------------------------------------------------------------
mock = _MockGA()
ckpt = copy.deepcopy(_VALID_GA)
del ckpt['settings']['populationSize']
checkNoRaise('TC-26 GA popSize absent no error', lambda: _gaValidate(mock, ckpt))

# ---------------------------------------------------------------------------
# TC-27  GA isMultiObjective mismatch → IOError
# ---------------------------------------------------------------------------
mock = _MockGA()
ckpt = copy.deepcopy(_VALID_GA)
ckpt['settings']['isMultiObjective'] = True  # mock has False
checkRaises('TC-27 GA multiobj mismatch raises IOError', lambda: _gaValidate(mock, ckpt), IOError)

# ---------------------------------------------------------------------------
# TC-28  GA isMultiObjective absent → no error (backward compat)
# ---------------------------------------------------------------------------
mock = _MockGA()
ckpt = copy.deepcopy(_VALID_GA)
del ckpt['settings']['isMultiObjective']
checkNoRaise('TC-28 GA multiobj absent no error', lambda: _gaValidate(mock, ckpt))

# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
print(f'Pass: {results["pass"]}, Fail: {results["fail"]}')
sys.exit(results['fail'])
