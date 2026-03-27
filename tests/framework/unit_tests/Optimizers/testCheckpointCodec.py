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
  Unit tests for the checkpoint encoder/decoder (_encodeCheckpointState and
  _decodeCheckpointState) in RavenSampled.  Each test performs a full
  encode → JSON-serialise → JSON-deserialise → decode round-trip and checks
  that the reconstructed value matches the original.
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np
import xarray as xr

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()

from ravenframework.Optimizers.RavenSampled import _encodeCheckpointState, _decodeCheckpointState

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


def checkEqual(comment, value, expected):
  res = (value == expected)
  if res:
    results['pass'] += 1
  else:
    print(f'FAIL: {comment} | got {value!r}, expected {expected!r}')
    results['fail'] += 1


def checkFloat(comment, value, expected, tol=1e-12):
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if res:
    results['pass'] += 1
  else:
    print(f'FAIL: {comment} | got {value!r}, expected {expected!r}')
    results['fail'] += 1


def round_trip(obj):
  """Encode → JSON serialise/deserialise → decode."""
  encoded = _encodeCheckpointState(obj)
  serialised = json.loads(json.dumps(encoded))
  return _decodeCheckpointState(serialised)


# ---------------------------------------------------------------------------
# TC-1  plain float passes through unchanged
# ---------------------------------------------------------------------------
checkFloat('TC-1 plain float round-trip', round_trip(3.14), 3.14)

# ---------------------------------------------------------------------------
# TC-2  NaN → sentinel → NaN
# ---------------------------------------------------------------------------
result = round_trip(float('nan'))
checkTrue('TC-2 NaN round-trip is nan', math.isnan(result))

# ---------------------------------------------------------------------------
# TC-3  +Inf → sentinel → +Inf
# ---------------------------------------------------------------------------
result = round_trip(float('inf'))
checkTrue('TC-3 +inf round-trip is +inf', math.isinf(result) and result > 0)

# ---------------------------------------------------------------------------
# TC-4  -Inf → sentinel → -Inf
# ---------------------------------------------------------------------------
result = round_trip(float('-inf'))
checkTrue('TC-4 -inf round-trip is -inf', math.isinf(result) and result < 0)

# ---------------------------------------------------------------------------
# TC-5  numpy float64 scalar
# ---------------------------------------------------------------------------
checkFloat('TC-5 np.float64 value preserved', round_trip(np.float64(2.718)), 2.718)

# ---------------------------------------------------------------------------
# TC-6  numpy int32 scalar
# ---------------------------------------------------------------------------
checkEqual('TC-6 np.int32 value preserved', round_trip(np.int32(42)), 42)

# ---------------------------------------------------------------------------
# TC-7  numpy bool_ scalar
# ---------------------------------------------------------------------------
checkTrue('TC-7 np.bool_ True preserved', round_trip(np.bool_(True)) is True)
checkTrue('TC-7 np.bool_ False preserved', round_trip(np.bool_(False)) is False)

# ---------------------------------------------------------------------------
# TC-8  1-D float64 ndarray
# ---------------------------------------------------------------------------
orig = np.array([1.1, 2.2, 3.3], dtype=np.float64)
decoded = round_trip(orig)
checkTrue('TC-8 ndarray type', isinstance(decoded, np.ndarray))
checkEqual('TC-8 ndarray dtype', str(decoded.dtype), 'float64')
checkTrue('TC-8 ndarray values', np.allclose(decoded, orig))

# ---------------------------------------------------------------------------
# TC-9  2-D int32 ndarray
# ---------------------------------------------------------------------------
orig = np.array([[1, 2], [3, 4]], dtype=np.int32)
decoded = round_trip(orig)
checkTrue('TC-9 2D ndarray type', isinstance(decoded, np.ndarray))
checkEqual('TC-9 2D ndarray shape', decoded.shape, (2, 2))
checkEqual('TC-9 2D ndarray dtype', str(decoded.dtype), 'int32')
checkTrue('TC-9 2D ndarray values', np.array_equal(decoded, orig))

# ---------------------------------------------------------------------------
# TC-10  xarray DataArray
# ---------------------------------------------------------------------------
orig = xr.DataArray([10.0, 20.0, 30.0], coords=[('x', [0, 1, 2])])
decoded = round_trip(orig)
checkTrue('TC-10 DataArray type', isinstance(decoded, xr.DataArray))
try:
  xr.testing.assert_equal(orig, decoded)
  results['pass'] += 1
except AssertionError as e:
  print(f'FAIL: TC-10 DataArray values: {e}')
  results['fail'] += 1

# ---------------------------------------------------------------------------
# TC-11  deque with maxlen
# ---------------------------------------------------------------------------
orig = deque([1, 2, 3], maxlen=5)
decoded = round_trip(orig)
checkTrue('TC-11 deque type', isinstance(decoded, deque))
checkEqual('TC-11 deque maxlen', decoded.maxlen, 5)
checkEqual('TC-11 deque contents', list(decoded), [1, 2, 3])

# ---------------------------------------------------------------------------
# TC-12  deque without maxlen (maxlen=None)
# ---------------------------------------------------------------------------
orig = deque([7, 8])
decoded = round_trip(orig)
checkTrue('TC-12 deque no-maxlen type', isinstance(decoded, deque))
checkTrue('TC-12 deque no-maxlen maxlen is None', decoded.maxlen is None)
checkEqual('TC-12 deque no-maxlen contents', list(decoded), [7, 8])

# ---------------------------------------------------------------------------
# TC-13  set of hashable values
# ---------------------------------------------------------------------------
orig = {10, 20, 30}
decoded = round_trip(orig)
checkTrue('TC-13 set type', isinstance(decoded, set))
checkEqual('TC-13 set elements', decoded, {10, 20, 30})

# ---------------------------------------------------------------------------
# TC-14  tuple
# ---------------------------------------------------------------------------
orig = (1, 'hello', 3.0)
decoded = round_trip(orig)
checkTrue('TC-14 tuple type', isinstance(decoded, tuple))
checkEqual('TC-14 tuple contents', decoded, (1, 'hello', 3.0))

# ---------------------------------------------------------------------------
# TC-15  dict with integer keys: _decode leaves keys as strings
#         (int conversion is done later in _restoreCheckpointState)
# ---------------------------------------------------------------------------
orig = {0: 'traj0', 1: 'traj1'}
decoded = round_trip(orig)
checkTrue('TC-15 int-keyed dict: keys remain strings after decode',
          all(isinstance(k, str) for k in decoded.keys()))
checkEqual('TC-15 int-keyed dict value under str key', decoded['0'], 'traj0')
# Manual int conversion (mirrors _restoreCheckpointState behaviour):
restored = {int(k): v for k, v in decoded.items()}
checkEqual('TC-15 int-key restore', restored[0], 'traj0')

# ---------------------------------------------------------------------------
# TC-16  list of mixed types
# ---------------------------------------------------------------------------
orig = [1, np.float64(2.5), 'text', None, True]
decoded = round_trip(orig)
checkTrue('TC-16 list type', isinstance(decoded, list))
checkEqual('TC-16 list length', len(decoded), 5)
checkEqual('TC-16 list int element', decoded[0], 1)
checkFloat('TC-16 list float element', decoded[1], 2.5)
checkEqual('TC-16 list str element', decoded[2], 'text')
checkTrue('TC-16 list None element', decoded[3] is None)
checkTrue('TC-16 list bool element', decoded[4] is True)

# ---------------------------------------------------------------------------
# TC-17  nested structure with NaN inside an ndarray inside a dict
# ---------------------------------------------------------------------------
orig = {
  'traj': {
    0: {'val': np.array([float('nan'), 1.0])},
    'flag': np.bool_(False),
  }
}
decoded = round_trip(orig)
inner_val = decoded['traj']['0']['val']   # key '0' because JSON stringifies int keys
checkTrue('TC-17 nested ndarray type', isinstance(inner_val, np.ndarray))
checkTrue('TC-17 nested NaN preserved', math.isnan(inner_val[0]))
checkFloat('TC-17 nested finite value', inner_val[1], 1.0)
checkTrue('TC-17 nested bool preserved', decoded['traj']['flag'] is False)

# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
print(f'Pass: {results["pass"]}, Fail: {results["fail"]}')
sys.exit(results['fail'])
