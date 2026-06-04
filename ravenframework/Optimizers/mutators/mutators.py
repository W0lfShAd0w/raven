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
  Implementation of mutators for Mutation process of Genetic Algorithm
  currently the implemented mutation algorithms are:
  1.  swapMutator
  2.  scrambleMutator
  3.  bitFlipMutator
  4.  inversionMutator
  5.  randomMutator

  Created June,16,2020
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi, Junyung Kim
"""
import numpy as np
import xarray as xr
from operator import itemgetter
from ...utils import utils, randomUtils
from ...utils.SSChecker import EQChecker, SingleCycleChecker

def swapMutator(offSprings, distDict, **kwargs):
  """
    This method performs the swap mutator. For each child, two genes are sampled and switched
    E.g.:
    child=[a,b,c,d,e] --> b and d are selected --> child = [a,d,c,b,e]
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  loc1, loc2 = locationsGenerator(offSprings, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})
  for i in range(np.shape(offSprings)[0]):
    children[i] = offSprings[i]
    ## TODO What happens if loc1 or 2 is out of range?! should we raise an error?
    if randomUtils.random(dim=1,samples=1)<=kwargs['mutationProb']:
      # convert loc1 and loc2 in terms of cdf values
      cdf1 = distDict[offSprings.coords['Gene'].values[loc1]].cdf(float(offSprings[i,loc1].values))
      cdf2 = distDict[offSprings.coords['Gene'].values[loc2]].cdf(float(offSprings[i,loc2].values))
      children[i,loc1] = distDict[offSprings.coords['Gene'].values[loc1]].ppf(cdf2)
      children[i,loc2] = distDict[offSprings.coords['Gene'].values[loc2]].ppf(cdf1)
  return children

def swapMutatorSS(offSprings, distDict, **kwargs):
  """
    User-facing swap mutator dispatcher for PRLO shuffling scheme optimization.
    Routes to swapMutatorEQ for equilibrium-cycle problems or swapMutatorSingleCycle
    for single-cycle (Nth cycle) problems based on the calculationType in the prlodata file.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, see swapMutatorEQ / swapMutatorSingleCycle for full parameter list.
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  if not any("prlodata" in sublist for sublist in kwargs["files"]):
    raise ValueError("'swapMutatorSS' requires a File of type 'prlodata'.")
  inpfile = [sublist[-1] for sublist in kwargs["files"] if sublist[1]=='prlodata'][0]
  prloData = EQChecker.PRLODataParser(inpfile.getPath()+inpfile.getFilename(), verbosity='reduced')
  effectiveType = prloData.phase1CalcType if prloData.calculationType == "coupled_transient" else prloData.calculationType
  if effectiveType in ["eq_cycle","eq_uprate"]:
    return swapMutatorEQ(offSprings, distDict, **kwargs)
  elif effectiveType in ["single_cycle","single_uprate"] and prloData.numBatches > 1:
    return swapMutatorSingleCycle(offSprings, distDict, **kwargs)
  raise ValueError(f"'swapMutatorSS' does not support calculationType '{prloData.calculationType}' with the given parameters.")

def swapMutatorEQ(offSprings, distDict, **kwargs):
  """
    Swap mutator for equilibrium-cycle (EQ) PRLO shuffling schemes.
    Two symmetry-equivalent gene locations are selected and swapped.  Two
    distinct swap modes are applied depending on whether the selected genes
    belong to the same batch or different batches:

    Same-batch swap: exchanges the CDF-space values at loc1 and loc2, then
    updates the immediate downstream reload reference for each position.  This
    shuffles assemblies within the current zoning map without changing which
    locations belong to which batch.

    Cross-batch swap: exchanges the batch assignments of loc1 and loc2 by
    directly re-encoding new FAID values.  loc1 takes on loc2's batch number,
    reload source, and fuel type; loc2 takes on loc1's batch number and type
    (re-encoded as fresh if loc1 was batch-1, otherwise inheriting loc1's
    source).  Immediate downstream reload references for both positions are
    updated so that the reload chain remains consistent.  This allows the GA to
    explore different zoning map configurations during EQ optimisation runs.

    In both modes the result is validated by checkGenome and retried if invalid.

    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary of distributions associated with each gene
          (used for CDF/PPF transforms in the same-batch case)
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that a mutation is attempted
          variables, list, variable names
          files, list, input file list (must contain a prlodata entry)
    @ Out, children, xr.DataArray, the mutated chromosome.
  """
  # check for EQ input
  EQFlag = False
  if any("prlodata" in sublist for sublist in kwargs["files"]):
    inpfile = [sublist[-1] for sublist in kwargs["files"] if sublist[1]=='prlodata'][0]
    EQObject = EQChecker(inpfile.getPath()+inpfile.getFilename())
    symMult = EQObject.prloData.symmetricMultiplicity
    effectiveType = EQObject.prloData.phase1CalcType if EQObject.prloData.calculationType == "coupled_transient" else EQObject.prloData.calculationType
    EQFlag = effectiveType in ["eq_cycle","eq_uprate"]
  if not EQFlag:
    raise ValueError("'swapMutatorEQ' is only appropriate of the 'eq_cycle' calculationType.")

  solnLen   = EQObject.prloData.solnLen
  numBatches = EQObject.prloData.numBatches

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})

  for i in range(np.shape(offSprings)[0]):
    antihang = 0
    flag = False
    while not flag: # ensure a valid selection.
      children[i] = offSprings[i]
      antihang += 1
      if antihang >= 1000:
        raise ValueError("swapMutatorEQ has failed to generate a valid genome.")
      loc1, loc2 = locationsGenerator(offSprings, kwargs['locs'])
      if symMult[loc1+1] != symMult[loc2+1]:
        flag = False
        continue

      if randomUtils.random(dim=1,samples=1)<=kwargs['mutationProb']:
        # Decode original gene values to determine swap mode.
        decoded1 = EQObject.decodeFAID(int(offSprings[i,loc1].values), solnLen, numBatches)
        decoded2 = EQObject.decodeFAID(int(offSprings[i,loc2].values), solnLen, numBatches)
        source1, batchNum1, type1 = decoded1
        source2, batchNum2, type2 = decoded2

        if batchNum1 == batchNum2:
          # Same-batch swap: spatial shuffle within the current zoning map.
          cdf1 = distDict[offSprings.coords['Gene'].values[loc1]].cdf(float(offSprings[i,loc1].values))
          cdf2 = distDict[offSprings.coords['Gene'].values[loc2]].cdf(float(offSprings[i,loc2].values))
          children[i,loc1] = distDict[offSprings.coords['Gene'].values[loc1]].ppf(cdf2)
          children[i,loc2] = distDict[offSprings.coords['Gene'].values[loc2]].ppf(cdf1)
          # update any reloaded FA's pointing to the swapped positions for loc1
          reloadedFA = EQObject.encodeFAID((loc1+1, batchNum1+1, type1), solnLen, numBatches)
          updatedFA  = EQObject.encodeFAID((loc2+1, batchNum1+1, type1), solnLen, numBatches)
          for pos in range(np.shape(children[i])[0]):
            if children[i,pos] == reloadedFA:
              children[i,pos] = updatedFA
          # update any reloaded FA's pointing to the swapped positions for loc2
          reloadedFA = EQObject.encodeFAID((loc2+1, batchNum2+1, type2), solnLen, numBatches)
          updatedFA  = EQObject.encodeFAID((loc1+1, batchNum2+1, type2), solnLen, numBatches)
          for pos in range(np.shape(children[i])[0]):
            if children[i,pos] == reloadedFA:
              children[i,pos] = updatedFA

        else:
          # Cross-batch swap: exchange batch assignments to explore different zoning map configurations.
          # Fresh assemblies (batch-1) must encode their own position as the source; reload assemblies
          # inherit the source from the gene being moved to that location.
          children[i,loc1] = (EQObject.encodeFAID((loc1+1, 1, type2), solnLen, numBatches)
                              if batchNum2 == 1
                              else EQObject.encodeFAID((source2, batchNum2, type2), solnLen, numBatches))
          children[i,loc2] = (EQObject.encodeFAID((loc2+1, 1, type1), solnLen, numBatches)
                              if batchNum1 == 1
                              else EQObject.encodeFAID((source1, batchNum1, type1), solnLen, numBatches))
          # Update downstream reload reference for loc1
          if batchNum1 < numBatches:
            oldReload = EQObject.encodeFAID((loc1+1, batchNum1+1, type1), solnLen, numBatches)
            newReload = EQObject.encodeFAID((loc2+1, batchNum1+1, type1), solnLen, numBatches)
            for pos in range(np.shape(children[i])[0]):
              if children[i,pos] == oldReload:
                children[i,pos] = newReload
          # Update downstream reload reference for loc2
          if batchNum2 < numBatches:
            oldReload = EQObject.encodeFAID((loc2+1, batchNum2+1, type2), solnLen, numBatches)
            newReload = EQObject.encodeFAID((loc1+1, batchNum2+1, type2), solnLen, numBatches)
            for pos in range(np.shape(children[i])[0]):
              if children[i,pos] == oldReload:
                children[i,pos] = newReload

      flag = EQObject.checkGenome(children[i],symMult)[0]

  return children

def swapMutatorSingleCycle(offSprings, distDict, **kwargs):
  """
    Swap mutator for single-cycle (Nth cycle) shuffling schemes.
    For each child, two genes at symmetry-equivalent locations are swapped and
    the result is validated against the SingleCycleChecker. No reload chain
    propagation is needed since reload sources are fixed by the reload geometry.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          locs, list, the 2 locations of the genes to be swapped
          mutationProb, float, probability that governs the mutation process
          variables, list, variables names.
          files, list, list of input files (must include a prlodata file).
    @ Out, children, xr.DataArray, the mutated chromosome, i.e., the child.
  """
  if not any("prlodata" in sublist for sublist in kwargs["files"]):
    raise ValueError("'swapMutatorSingleCycle' requires a File of type 'prlodata'.")
  inpfile = [sublist[-1] for sublist in kwargs["files"] if sublist[1]=='prlodata'][0]
  SCObject = SingleCycleChecker(inpfile.getPath()+inpfile.getFilename())
  symMult = SCObject.prloData.symmetricMultiplicity

  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})

  for i in range(np.shape(offSprings)[0]):
    antihang = 0
    flag = False
    while not flag:
      children[i] = offSprings[i]
      antihang += 1
      if antihang >= 1000:
        raise ValueError("swapMutatorSingleCycle has failed to generate a valid genome.")
      loc1, loc2 = locationsGenerator(offSprings, kwargs['locs'])
      if symMult[loc1+1] != symMult[loc2+1]:
        flag = False
        continue

      if randomUtils.random(dim=1,samples=1)<=kwargs['mutationProb']:
        gene1 = int(children[i,loc1].values)
        gene2 = int(children[i,loc2].values)
        _,b1,t1 = SCObject.decodeFAID(gene1,SCObject.prloData.solnLen,SCObject.prloData.numBatches)
        _,b2,t2 = SCObject.decodeFAID(gene2,SCObject.prloData.solnLen,SCObject.prloData.numBatches)
        # Re-encode with the new destination location
        children[i,loc1] = SCObject.encodeFAID((loc1+1,1,t2),SCObject.prloData.solnLen,SCObject.prloData.numBatches) if b2==1 else gene2
        children[i,loc2] = SCObject.encodeFAID((loc2+1,1,t1),SCObject.prloData.solnLen,SCObject.prloData.numBatches) if b1==1 else gene1

      flag = SCObject.checkGenome(children[i],symMult)[0]

  return children

# @profile
def scrambleMutator(offSprings, distDict, **kwargs):
  """
    This method performs the scramble mutator. For each child, a subset of genes is chosen
    and their values are shuffled randomly.
    @ In, offSprings, xr.DataArray, offsprings after crossover
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          chromosome, numpy.array, the chromosome that will mutate to the new child
          locs, list, the locations of the genes to be randomly scrambled
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
          variables, list, variables names.
    @ Out, child, np.array, the mutated chromosome, i.e., the child.
  """
  locs = locationsGenerator(offSprings, kwargs['locs'])

  # initializing children
  children = xr.DataArray(np.zeros((np.shape(offSprings))),
                          dims=['chromosome','Gene'],
                          coords={'chromosome': np.arange(np.shape(offSprings)[0]),
                                  'Gene':kwargs['variables']})

  for i in range(np.shape(offSprings)[0]):
    for j in range(np.shape(offSprings)[1]):
      children[i,j] = distDict[offSprings[i].coords['Gene'].values[j]].cdf(float(offSprings[i,j].values))

  for i in range(np.shape(offSprings)[0]):
    for ind,element in enumerate(locs):
      if randomUtils.random(dim=1,samples=1)< kwargs['mutationProb']:
        children[i,locs[0]:locs[-1]+1] = randomUtils.randomPermutation(list(children.data[i,locs[0]:locs[-1]+1]),None)

  for i in range(np.shape(offSprings)[0]):
    for j in range(np.shape(offSprings)[1]):
      children[i,j] = distDict[offSprings.coords['Gene'].values[j]].ppf(float(children[i,j].values))

  return children

def bitFlipMutator(offSprings, distDict, **kwargs):
  """
    This method is designed to flip a single gene in each chromosome with probability = mutationProb.
    E.g. gene at location loc is flipped from current value to newValue
    The gene to be flipped is completely random.
    The new value of the flipped gene is is completely random.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by bitFlipMutator')

  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc-1]
      oldCDFvalue = distDict[geneIDToBeChanged].cdf(child.values[loc-1])
      newCDFValue = 1.0 - oldCDFvalue
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc-1] = newValue
  return offSprings

def randomMutator(offSprings, distDict, **kwargs):
  """
    This method is designed to randomly mutate a single gene in each chromosome with probability = mutationProb.
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  if kwargs['locs'] is not None and 'locs' in kwargs.keys():
    raise ValueError('Locs arguments are not being used by randomMutator')
  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # sample gene location to be flipped: i.e., determine loc
      chromosomeSize = child.values.shape[0]
      loc = randomUtils.randomIntegers(0, chromosomeSize, caller=None, engine=None)
      # gene at location loc is flipped from current value to newValue
      geneIDToBeChanged = child.coords['Gene'].values[loc-1]
      newCDFValue = randomUtils.random()
      newValue = distDict[geneIDToBeChanged].ppf(newCDFValue)
      child.values[loc-1] = newValue
  return offSprings

def inversionMutator(offSprings, distDict, **kwargs):
  """
    This method is designed mirror a sequence of genes in each chromosome with probability = mutationProb.
    The sequence of genes to be mirrored is completely random.
    E.g. given chromosome C = [0,1,2,3,4,5,6,7,8,9] and sampled locL=2 locU=6;
         New chromosome  C' = [0,1,6,5,4,3,2,7,8,9]
    @ In, offSprings, xr.DataArray, children resulting from the crossover process
    @ In, distDict, dict, dictionary containing distribution associated with each gene
    @ In, kwargs, dict, dictionary of parameters for this mutation method:
          mutationProb, float, probability that governs the mutation process, i.e., if prob < random number, then the mutation will occur
    @ Out, offSprings, xr.DataArray, children resulting from the crossover process
  """
  # sample gene locations: i.e., determine locL and locU
  locL, locU = locationsGenerator(offSprings, kwargs['locs'])

  for child in offSprings:
    # the mutation is performed for each child independently
    if randomUtils.random(dim=1,samples=1)<kwargs['mutationProb']:
      # select sequence to be mirrored and mirror it
      seq = np.arange(locL,locU+1)
      allElems = []
      for i,elem in enumerate(seq):
        allElems.append(distDict[child.coords['Gene'].values[i]].cdf(float(child[elem].values)))

      mirrSeq = allElems[::-1]
      mirrElems = []
      for elem in mirrSeq:
        mirrElems.append(distDict[child.coords['Gene'].values[i]].ppf(elem))
      # insert mirrored sequence into child
      child.values[locL:locU+1]=mirrElems

  return offSprings

def locationsGenerator(offSprings,locs):
  """
  Methods designed to process the locations for the mutators. These locations can be either user specified or
  randomly generated.
  @ In, offSprings, xr.DataArray, children resulting from the crossover process
  @ In, locs, list, the two locations of the genes to be swapped
  @ Out, loc1, loc2, int, the two ordered processed locations required by the mutators
  """
  if locs is None:
    locs = list(set(randomUtils.randomChoice(list(np.arange(offSprings.data.shape[1])),size=2,replace=False)))
  loc1 = np.minimum(locs[0], locs[1])
  loc2 = np.maximum(locs[0], locs[1])
  return loc1, loc2

__mutators = {}
__mutators['swapMutator']         = swapMutator
__mutators['swapMutatorSS']       = swapMutatorSS
__mutators['scrambleMutator']     = scrambleMutator
__mutators['bitFlipMutator']      = bitFlipMutator
__mutators['inversionMutator']    = inversionMutator
__mutators['randomMutator']       = randomMutator


def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __mutators:
    cls.raiseAnError (IOError, "{} MECHANISM NOT IMPLEMENTED!!!!!".format(name))
  return __mutators[name]
