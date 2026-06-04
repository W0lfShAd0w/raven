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
  Implementation of crossovers for crossover process of Genetic Algorithm
  currently the implemented crossover algorithms are:
  1.  onePointCrossover
  2.  uniformCrossover
  3.  twoPointsCrossover

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""

import numpy as np
from copy import deepcopy
from scipy.special import comb
from itertools import combinations
import xarray as xr
from ...utils import randomUtils
from ...utils.SSChecker import EQChecker, SingleCycleChecker


# @profile
def onePointCrossover(parents,**kwargs):
  """
    Method designed to perform crossover by swapping chromosome portions before/after specified or sampled location
    @ In, parents, xr.DataArray, parents involved in the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
          variables, list, variables names.
    @ Out, children, np.array, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  # Number of children = 2* (nParents choose 2)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),nGenes)),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                  'Gene':kwargs['variables']})


  # defaults
  if (kwargs['crossoverProb'] == None) or ('crossoverProb' not in kwargs.keys()):
    crossoverProb = randomUtils.random(dim=1, samples=1)
  else:
    crossoverProb = kwargs['crossoverProb']

  # create children
  parentsPairs = list(combinations(parents,2))

  for ind,parent in enumerate(parentsPairs):
    parent = np.array(parent).reshape(2,-1) # two parents at a time

    if randomUtils.random(dim=1,samples=1) <= crossoverProb:
      if (kwargs['points'] == None) or ('points' not in kwargs.keys()):
        point = list([randomUtils.randomIntegers(1,nGenes-1,None)])
      elif (any(i>=nGenes-1 for i in kwargs['points'])):
        raise ValueError('Crossover point cannot be larger than number of Genes (variables)')
      else:
        point = kwargs['points']
      for i in range(nGenes):
        if len(point)>1:
          raise ValueError('In one Point Crossover a single crossover location should be provided!')
        children[2*ind:2*ind+2,i] = parent[np.arange(0,2)*(i<point[0])+np.arange(-1,-3,-1)*(i>=point[0]),i]
    else:
      # Each child is just a copy of the parents
      children[2*ind:2*ind+2,:] = parent

  return children

def uniformCrossover(parents,**kwargs):
  """
    Method designed to perform crossover by swapping genes one by one
    @ In, parents, xr.DataArray, parents involved in the mating process.
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process.
          Shape is nParents x len(chromosome) i.e, number of Genes/Vars
    @ Out, children, xr.DataArray, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),np.shape(parents)[1])),
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                      'Gene':parents.coords['Gene'].values})

  if (kwargs['crossoverProb'] == None) or ('crossoverProb' not in kwargs.keys()):
    crossoverProb = randomUtils.random(dim=1, samples=1)
  else:
    crossoverProb = kwargs['crossoverProb']

  # check for EQ or single-cycle (Nth cycle) input
  EQFlag = False
  SCFlag = False
  if any("prlodata" in sublist for sublist in kwargs["files"]):
    inpfile = [sublist[-1] for sublist in kwargs["files"] if sublist[1]=='prlodata'][0]
    EQObject = EQChecker(inpfile.getPath()+inpfile.getFilename())
    effectiveType = EQObject.prloData.phase1CalcType if EQObject.prloData.calculationType == "coupled_transient" else EQObject.prloData.calculationType
    EQFlag = effectiveType in ["eq_cycle","eq_uprate"]
    SCFlag = effectiveType in ["single_cycle","single_uprate"] and EQObject.prloData.numBatches > 1
  if SCFlag:
    SCObject = SingleCycleChecker(inpfile.getPath()+inpfile.getFilename())

  index = 0
  parentsPairs = list(combinations(parents,2))
  for parentPair in parentsPairs:
    parent1 = parentPair[0].values
    parent2 = parentPair[1].values
    if EQFlag:
      children1,children2 = uniformEQCrossoverMethod(parent1,parent2,crossoverProb,EQObject)
    elif SCFlag:
      children1,children2 = uniformSCCrossoverMethod(parent1,parent2,crossoverProb,SCObject)
    else:
      children1,children2 = uniformCrossoverMethod(parent1,parent2,crossoverProb)
    children[index]   = children1
    children[index+1] = children2
    index +=  2
  return children


def twoPointsCrossover(parents, **kwargs):
  """
    Method designed to perform a two point crossover on 2 parents:
    Partition each parents in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1 B2 C1
    children2 = A2 B1 C2
    @ In, parents, xr.DataArray, parents involved in the mating process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          parents, 2D array, parents in the current mating process.
          Shape is nParents x len(chromosome) i.e, number of Genes/Vars
          crossoverProb, float, crossoverProb determines when child takes genes from a specific parent, default is random
          points, integer, point at which the cross over happens, default is random
    @ Out, children, xr.DataArray, children resulting from the crossover. Shape is nParents x len(chromosome) i.e, number of Genes/Vars
  """
  nParents,nGenes = np.shape(parents)
  children = xr.DataArray(np.zeros((int(2*comb(nParents,2)),np.shape(parents)[1])),
                              dims=['chromosome','Gene'],
                              coords={'chromosome': np.arange(int(2*comb(nParents,2))),
                                      'Gene':parents.coords['Gene'].values})
  parentPairs = list(combinations(parents,2))
  index = 0
  if nGenes<=2:
    ValueError('In Two point Crossover the number of genes should be >=3!')
  for couples in parentPairs:
    [loc1,loc2] = randomUtils.randomChoice(list(range(1,nGenes)), size=2, replace=False, engine=None)
    if loc1 > loc2:
      locL = loc2
      locU = loc1
    else:
      locL=loc1
      locU=loc2
    parent1 = couples[0]
    parent2 = couples[1]
    children1,children2 = twoPointsCrossoverMethod(parent1,parent2,locL,locU)

    children[index]   = children1
    children[index+1] = children2
    index = index + 2

  return children

__crossovers = {}
__crossovers['onePointCrossover']  = onePointCrossover
__crossovers['twoPointsCrossover'] = twoPointsCrossover
__crossovers['uniformCrossover']   = uniformCrossover
#!__crossovers['EQCrossover']         = EQCrossover #!TODO(rollnk):deprecated; remove.


def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __crossovers:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __crossovers[name]

def twoPointsCrossoverMethod(parent1,parent2,locL,locU):
  """
    Method designed to perform a twopoint crossover on 2 arrays:
    Partition each array in three sequences (A,B,C):
    parent1 = A1 B1 C1
    parent2 = A2 B2 C2
    Then:
    children1 = A1 B2 C1
    children2 = A2 B1 C2
    @ In, parent1: first array
    @ In, parent2: second array
    @ In, LocL: first location
    @ In, LocU: second location
    @ Out, children1: first generated array
    @ Out, children2: second generated array
  """
  children1 = parent1.copy(deep=True)
  children2 = parent2.copy(deep=True)

  seqB1 = parent1.values[locL:locU]
  seqB2 = parent2.values[locL:locU]

  children1[locL:locU] = seqB2
  children2[locL:locU] = seqB1
  return children1,children2

def uniformCrossoverMethod(parent1,parent2,crossoverProb):
  """
    Method designed to perform a uniform crossover on 2 arrays
    @ In, parent1: first array
    @ In, parent2: second array
    @ In, crossoverProb: crossover probability for each gene
    @ Out, children1: first generated array
    @ Out, children2: second generated array
  """
  children1 = np.zeros(parent1.size)
  children2 = np.zeros(parent2.size)

  for pos in range(parent1.size):
    if randomUtils.random(dim=1,samples=1)<crossoverProb:
      children1[pos] = parent2[pos]
      children2[pos] = parent1[pos]
    else:
      children1[pos] = parent1[pos]
      children2[pos] = parent2[pos]

  return children1,children2

def uniformEQCrossoverMethod(parent1,parent2,crossoverProb,eqchecker):
  """
    Uniform crossover for equilibrium-cycle (EQ) PRLO shuffling schemes.
    Two crossover passes are performed within each retry iteration:

    Pass 1 — Fresh-batch (batch-1) crossover: positions where both parents carry
    a batch-1 gene are eligible for fuel-type exchange.  When types differ,
    updateFATypes propagates the change through the downstream reload chain.
    This preserves the ILB behaviour for the fresh batch.

    Pass 2 — Reload-batch crossover: positions where both children (after Pass 1)
    carry the same batch number N > 1 AND the same fuel type are eligible for
    source-location exchange.  This recombines the shuffling scheme — which
    batch-(N-1) source feeds each batch-N destination — between the two parents.
    A per-batch-level demand check prevents over-demanding any source location
    beyond its symmetry multiplicity.  Positions where the proposed source is
    type-incompatible in the receiving child are skipped; these are conservatively
    deferred rather than producing a genome that fails checkGenome.

    Both passes use the same crossoverProb gate.  The existing checkGenome + retry
    loop provides final validation; the demand and type checks in Pass 2 reduce
    the rejection rate significantly.

    @ In, parent1, numpy.array, first parent chromosome
    @ In, parent2, numpy.array, second parent chromosome
    @ In, crossoverProb, float, per-gene crossover probability
    @ In, eqchecker, EQChecker, logical constraint handler for EQ-cycle cases
    @ Out, child1, numpy.array, first generated child chromosome
    @ Out, child2, numpy.array, second generated child chromosome
  """
  solnLen   = eqchecker.prloData.solnLen
  numBatches = eqchecker.prloData.numBatches
  symMult   = eqchecker.prloData.symmetricMultiplicity

  maxiter = 1000; iter = 0
  flag = False
  while not flag:
    if iter >= maxiter:
      raise ValueError("UniformEQCrossoverMethod has failed to generate a valid genome.")

    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    # Pass 1 — Fresh-batch crossover (ILB behaviour preserved).
    for pos in range(parent1.size):
      #!NOTE(rollnk):this behavior of passing an eqchecker attribute into an eqchecker function is temporary until the EQ functions can be merged into the PRLO plugin.
      p1decoded = eqchecker.decodeFAID(parent1[pos], solnLen, numBatches)
      p2decoded = eqchecker.decodeFAID(parent2[pos], solnLen, numBatches)
      if p1decoded[1] == p2decoded[1] == 1: # only crossover if batch numbers match at batch-1.
        if randomUtils.random(dim=1,samples=1)<crossoverProb:
          child1[pos] = parent2[pos]
          child2[pos] = parent1[pos]
          if p1decoded[2] != p2decoded[2]: # FA types don't match; propagate type change to reload chain.
            #!NOTE(rollnk):this behavior of passing an eqchecker attribute into an eqchecker function is temporary until the EQ functions can be merged into the PRLO plugin.
            child1 = updateFATypes(pos+1,p1decoded,p2decoded[2],child1,parent1,eqchecker)
            child2 = updateFATypes(pos+1,p2decoded,p1decoded[2],child2,parent2,eqchecker)

    # Pass 2 — Reload-batch crossover: recombine shuffling schemes for reload batches.
    # Decode children AFTER Pass 1 to reflect any type changes made by updateFATypes.
    decoded_c1 = [eqchecker.decodeFAID(int(child1[pos]), solnLen, numBatches) for pos in range(parent1.size)]
    decoded_c2 = [eqchecker.decodeFAID(int(child2[pos]), solnLen, numBatches) for pos in range(parent1.size)]

    # Build per-batch-level source demand counters from the current child state.
    d1 = {}
    d2 = {}
    for pos in range(parent1.size):
      if decoded_c1[pos][1] > 1:
        key = (decoded_c1[pos][1], decoded_c1[pos][0])
        d1[key] = d1.get(key, 0) + symMult[pos+1]
      if decoded_c2[pos][1] > 1:
        key = (decoded_c2[pos][1], decoded_c2[pos][0])
        d2[key] = d2.get(key, 0) + symMult[pos+1]

    for pos in range(parent1.size):
      c1dec = decoded_c1[pos]
      c2dec = decoded_c2[pos]
      N = c1dec[1]
      if N < 2 or N != c2dec[1]:
        continue  # not a matching-batch reload position
      if c1dec[2] != c2dec[2]:
        continue  # mismatched fuel type; type-propagation path deferred to future work
      if randomUtils.random(dim=1,samples=1) >= crossoverProb:
        continue
      src1, src2 = c1dec[0], c2dec[0]
      if src1 == src2:
        continue  # same source; swap is a no-op
      myMult = symMult[pos+1]
      T = c1dec[2]
      # Verify that the proposed new source has the correct batch number and fuel type
      # in each child (may differ from parent state after Pass 1 type propagation).
      if decoded_c1[src2-1][1] != N-1 or decoded_c1[src2-1][2] != T:
        continue  # src2 in child1 is batch-incompatible or type-incompatible
      if decoded_c2[src1-1][1] != N-1 or decoded_c2[src1-1][2] != T:
        continue  # src1 in child2 is batch-incompatible or type-incompatible
      # Accept swap only if neither child over-demands the source's symmetry multiplicity.
      if (d1.get((N, src2), 0) + myMult <= symMult[src2] and
          d2.get((N, src1), 0) + myMult <= symMult[src1]):
        child1[pos], child2[pos] = int(child2[pos]), int(child1[pos])
        d1[(N, src1)] = d1.get((N, src1), 0) - myMult
        d1[(N, src2)] = d1.get((N, src2), 0) + myMult
        d2[(N, src2)] = d2.get((N, src2), 0) - myMult
        d2[(N, src1)] = d2.get((N, src1), 0) + myMult

    #!NOTE(rollnk):this behavior of passing an eqchecker attribute into an eqchecker function is temporary until the EQ functions can be merged into the PRLO plugin.
    flag = all((eqchecker.checkGenome(child1,symMult)[0],
                eqchecker.checkGenome(child2,symMult)[0]))
    iter += 1

  return child1,child2

def updateFATypes(sourceLoc,sourceDecoded,faType,child,parent,eqobj):
  """
    Utility for the uniformEQCrossoverMethod. If a crossover operation results in a different FA type
    at a given location, all associated reloaded FA's must also have their FA types updated.
  """
  sourceLocsList = [(sourceLoc,sourceDecoded[1]+1)] # position, batch number
  antihang = 0
  while len(sourceLocsList) != 0:
    antihang += 1
    if antihang >= 10000:
      raise ValueError("uniformEQCrossoverMethod failed to update FA types in chromosome; possible recursive reloading detected in generated shuffling scheme.")
    pos, batchNum = sourceLocsList.pop()
    #!NOTE(rollnk):this behavior of passing an eqchecker attribute into an eqchecker function is temporary until the EQ functions can be merged into the PRLO plugin.
    reloadFAID = eqobj.encodeFAID((pos,batchNum,sourceDecoded[2]),eqobj.prloData.solnLen,eqobj.prloData.numBatches) # calculate ID for fuel of type sourceDecoded[2] and batch batchNum at location pos
    updatedFAID = eqobj.encodeFAID((pos,batchNum,faType),eqobj.prloData.solnLen,eqobj.prloData.numBatches) # calculate ID for fuel of type faType and batch batchNum at location pos
    reloadLocs = [i for i in range(len(parent)) if parent[i] == reloadFAID]
    for i in reloadLocs:
      sourceLocsList.append((i+1,batchNum+1))
      child[i] = updatedFAID
  return child

def uniformSCCrossoverMethod(parent1,parent2,crossoverProb,scchecker):
  """
    Method designed to perform a uniform crossover on 2 arrays for the single-cycle
    (Nth cycle) shuffling scheme.

    Genes are swapped in a single pass with demand-tracking feasibility checks so
    that the children are always valid without rejection sampling.  Three classes of
    position are handled:
      - fresh-fresh (batchNum==1 in both parents): swap is unconditionally valid
        because each fresh gene encodes sourceLoc==i+1 by construction.
      - reload-reload (batchNum==2 in both parents): swap is accepted only when the
        updated source-location demand in each child does not exceed symMult.
      - mixed (batchNums differ): skipped so that each child inherits its base
        parent's fresh-fuel count, preserving the feedBatchSize constraint.

    @ In, parent1, numpy.array, first parent chromosome
    @ In, parent2, numpy.array, second parent chromosome
    @ In, crossoverProb, float, per-gene crossover probability
    @ In, scchecker, SingleCycleChecker, logical constraint handler for single-cycle cases.
    @ Out, child1, numpy.array, first generated child chromosome
    @ Out, child2, numpy.array, second generated child chromosome
  """
  solnLen   = scchecker.prloData.solnLen
  numBatches= scchecker.prloData.numBatches
  symMult   = scchecker.prloData.symmetricMultiplicity

  child1 = deepcopy(parent1)
  child2 = deepcopy(parent2)

  # Decode all genes once to avoid repeated calls inside the loop.
  decoded1 = [scchecker.decodeFAID(int(g), solnLen, numBatches) for g in parent1]
  decoded2 = [scchecker.decodeFAID(int(g), solnLen, numBatches) for g in parent2]

  # Initialise per-source demand counters for each child (reload genes only).
  d1 = {s:0 for s in range(1,len(parent1)+1)}  # demand on each source location in child1
  d2 = {s:0 for s in range(1,len(parent2)+1)}  # demand on each source location in child2
  for i in range(len(parent1)):
    if decoded1[i][1] == 2:
      s = decoded1[i][0]; d1[s] += symMult[i+1]
    if decoded2[i][1] == 2:
      s = decoded2[i][0]; d2[s] += symMult[i+1]

  for i in range(len(parent1)):
    b1 = decoded1[i][1]
    b2 = decoded2[i][1]

    if b1 != b2:
      continue  # mixed position: skip to preserve fresh-fuel counts in each child

    if randomUtils.random(dim=1,samples=1) >= crossoverProb:
      continue

    if b1 == 1:
      # Fresh-fresh swap: always valid.
      child1[i] = parent2[i]
      child2[i] = parent1[i]

    else:  # b1 == b2 == 2
      sl1 = decoded1[i][0]  # current source of child1[i]
      sl2 = decoded2[i][0]  # current source of child2[i]
      if sl1 == sl2:
        continue  # same source; swap is a no-op
      myMult = symMult[i+1]
      # Accept swap only if neither child's source-location demand is exceeded.
      if (d1.get(sl2,0) + myMult <= symMult[sl2] and
          d2.get(sl1,0) + myMult <= symMult[sl1]):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
        d1[sl1] -= myMult;  d1[sl2] += myMult
        d2[sl2] -= myMult;  d2[sl1] += myMult

  if not all((scchecker.checkGenome(child1,symMult)[0],
              scchecker.checkGenome(child2,symMult)[0])):
    raise ValueError("uniformSCCrossoverMethod produced an invalid genome; this is a bug.")

  return child1,child2