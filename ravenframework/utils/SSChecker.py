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

import re
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET

##!TODO(rollnk): This entire file is a reproduction of the equivalent in the PRLO plugin.
##!              This file is now deprecated and will soon be deleted.


class _PRLOCheckerBase():
  """
  Base class for PRLO shuffling scheme checkers.
  Provides shared utilities: PRLODataParser, decodeFAID, encodeFAID.
  Not intended for direct instantiation.
  """

  class PRLODataParser():
    def __init__(self, inputFile, verbosity="full"):
      """
      Constructor. This XML file parser supports a large number of input tags. The
      'verbosity' argument can be provided to reduce the time and memory spent
      parsing the input.

      @ In, inputFile, string, xml PARCSv345 parameters file
      @ Out, None
      """
      self.dataFile = Path(inputFile)
      dorm = ET.parse(self.dataFile)
      root = dorm.getroot()

      if verbosity in ['calcType','full','reduced']:
        self.calculationType     = '_'.join(self.parseXMLInput(root,'calculationType',default="single_cycle").lower().split())
        _p53dNode                = root.find("PARCSR53DSettings")
        _p53dRoot                = _p53dNode if _p53dNode is not None else root
        _phase1Raw               = self.parseXMLInput(_p53dRoot,'phase1CalcType',default=None)
        self.phase1CalcType      = '_'.join(_phase1Raw.lower().split()) if _phase1Raw else None
      if verbosity in ['full','reduced']:
        # unstructured_opt calcs may have no LWR assembly geometry;
        # skip the blocks below to avoid AttributeError on tags with no default.
        if self.calculationType == 'unstructured_opt':
          self.numBatches = 1
          self.feedBatchSizeLimits = None
          self.colLabels = self.rowLabels = self.geometry = None
          self.faDict = self.fuelFADict = []
          self.numTypes = self.solnLen = self.numAssemblies = 0
          self.wabaTypes = set()
          self.crBankLocSet = set()
          return

        self.numBatches          = self.parseXMLInput(root,'numBatches',datatype=int,default=1)
        self.feedBatchSizeLimits = self.parseXMLInput(root,'feedBatchSizeLimits',default=None)
        self.colLabels           = self.parseXMLInput(root,'colLabels')
        self.rowLabels           = self.parseXMLInput(root,'rowLabels')
        self.geometry            = self.parseXMLInput(root,'geometry')
        countFuelLocs = len([int(x) for x in self.parseXMLInput(root,'geometry').split() if "00" not in x and "r" not in x])
        self.numAssemblies       = self.parseXMLInput(root,'numAssemblies',datatype=int,default=countFuelLocs)
        self.reloadCycle     = self.parseXMLInput(root,'reloadCycle',datatype=Path,default=None)
        self.reloadGeometry  = self.parseXMLInput(root,'reloadGeometry',default=None)

        self.coreShape = re.sub(r"r\d",'0',re.sub(r"\d+",'1',self.geometry.replace('00',' ')))
        self.solnLen = max([int(s) for s in self.geometry.split() if s.isdigit()])
        self.faDict = []
        for fa in root.iter('FA'):
          self.faDict.append(fa.attrib)
        self.fuelFADict = [fa for fa in self.faDict if fa['type'] == 'fuel']
        self.numTypes = len(self.fuelFADict)
        self.wabaTypes = {i + 1 for i, fa in enumerate(self.fuelFADict)
                          if fa.get('waba', 'false').lower() in ['true', '1', 'yes', 't', 'y']}
        self.crBank = self.parseXMLInput(root, 'crBank', default=None)
        self.xsDict =[]
        for xs in root.iter('XS'):
          self.xsDict.append(xs.attrib)

        # Resolve calculated values from user-provided data
        if not self.feedBatchSizeLimits:
          self.feedBatchSizeLimits = (np.ceil(self.numAssemblies/self.numBatches),self.numAssemblies) # logical extremes for feed batch size
        else:
          self.feedBatchSizeLimits = tuple([int(val.strip()) for val in self.feedBatchSizeLimits.replace(',',' ').split()])
          if len(self.feedBatchSizeLimits) == 1:
            # if only one value is given, assume that value is the maximum limit.
            self.feedBatchSizeLimits = (np.ceil(self.numAssemblies/self.numBatches),self.feedBatchSizeLimits[0])
          else:
            self.feedBatchSizeLimits = (self.feedBatchSizeLimits[0],self.feedBatchSizeLimits[-1]) #ensure only two values are provided.

        # Resolve CR bank location set from crBank token string (parallel to geometry)
        crBankLocSet = set()
        if getattr(self, 'crBank', None) is not None:
          crBankTokens = self.crBank.strip().split()
          geom = self.geometry.strip().split()
          for geomToken, crToken in zip(geom, crBankTokens):
            if str(geomToken).isdigit() and int(geomToken) != 0 and int(crToken) == 1:
              crBankLocSet.add(int(geomToken))
        self.crBankLocSet = crBankLocSet

        if getattr(self, 'wabaTypes', set()) and not crBankLocSet:
          print("WARNING: WABA FA types are defined but no <crBank> was provided. WABA assemblies will not be excluded from any location.")

      if verbosity in ['full']:
        self.useTemplate     = self.strToBool(self.parseXMLInput(root,'useTemplate',default="False"))
        dflt = "no default" if not self.useTemplate else None
        self.THFlag          = self.strToBool(self.parseXMLInput(root,'THFlag',default="False"))
        self.pinPowerRecFlag = self.strToBool(self.parseXMLInput(root,'pinPowerRecFlag',default="False"))
        self.power           = self.parseXMLInput(root,'power',datatype=float,default=100.0)
        self.coreType        = self.parseXMLInput(root,'coreType',default="PWR")
        self.initialExposure = self.parseXMLInput(root,'initialExposure',datatype=float,default=0.00)
        self.initialBoron    = self.parseXMLInput(root,'initialBoron',datatype=int,default=1000)
        self.numAxial        = self.parseXMLInput(root,'numAxial',datatype=int,default=dflt)
        self.gridZ           = self.parseXMLInput(root,'gridZ',default=dflt)
        self.BC              = self.parseXMLInput(root,'BC',default=dflt)
        self.faPower         = self.parseXMLInput(root,'faPower',datatype=float,default=dflt)
        self.faPitch         = self.parseXMLInput(root,'faPitch',datatype=float,default=dflt)
        self.inletTemp       = self.parseXMLInput(root,'inletTemp',datatype=float,default=dflt)
        self.flow            = self.parseXMLInput(root,'flow',datatype=float,default=dflt) # coolant mass flow in kg/s/FA
        self.depHistory      = self.parseXMLInput(root,'depHistory',default=dflt)
        self.inpHistFile     = self.parseXMLInput(root,'inpHistFile',default=None)
        self.xsLib           = self.parseXMLInput(root,'xsLib',datatype=Path,default='.')
        self.xsExtension     = self.parseXMLInput(root,'xsExtension',default='')

        fuelMap = [int(s) for s in self.geometry.split() if s.isdigit()]
        self.symmetricMultiplicity = {i:fuelMap.count(i) for i in fuelMap}
        del self.symmetricMultiplicity[0] #locations are 1-indexed; '0' is the void space.

    def parseXMLInput(self,root,tag,datatype=str,default="no default"):
      """
      Method to parse input values from the PRLO data XLM file.
      @ In, root, ElementTree.Element, the contents of the PRLO data XLM input file.
      @ In, tag, str, name of the XML node or tag to be parsed.
      @ In, datatype, type, the desired datatype of the parsed value.
      @ In, default, the default value to be returned if tag is not found in root.
      @ Out, str or datatype, parsed value from the PRLO data XML file.
      """
      parsedValue = root.find(tag)
      if default == "no default":
        try:
          return datatype(parsedValue.text.strip())
        except AttributeError as e:
          raise AttributeError(f"Tag {tag} in provided PRLOdata.xml file is missing and has no default value.")
      else:
        return datatype(parsedValue.text.strip()) if parsedValue is not None else default

    def strToBool(self,string):
      if string.lower() in ['t','true','1','yes','y']:
        return True
      elif string.lower() in ['f','false','0','no','n']:
        return False
      else:
        raise ValueError(f"Failed to convert string '{string}' to boolean.")

  def decodeFAID(self,faID,solnLen,numBatches):
    """
      PRLO uses a numbering scheme for the solution encoding that assumes
      all possible FA's are by batch, then location, then type, for a total
      count of possible FA designations = numBatches*numAssemblies*numTypes
      @ In, faID, str, the FA ID according to the PRLO encoding scheme (0-indexed)
      @ In, solnLen, int, the number of fuel locations in the solution
      @ In, numBatches, int, the maximum number of fuel batches.
      @ Out, locNum, int, source location in the solution for faID (1-indexed)
      @ Out, batchNum, int, batch number for faID (1-indexed)
      @ Out, typeNum, int, FA type number for faID (1-indexed)
    """
    # decode location
    locNum = int(np.ceil(((faID + 1) % (solnLen*numBatches)) / numBatches))
    if locNum == 0:
      locNum = solnLen
    # decode batch number
    batchNum = 1 + int(faID % numBatches)
    # decode FA type
    typeNum = 1 + int(faID / (solnLen * numBatches))
    return (locNum, batchNum, typeNum)

  def encodeFAID(self,FA,solnLen,numBatches):
    """
      PRLO uses a numbering scheme for the solution encoding that assumes
      all possible FA's are by batch, then location, then type, for a total
      count of possible FA designations = numBatches*numAssemblies*numTypes
      @ In, FA, tuple, the FA source location, batch number, and type (all 1-indexed)
      @ In, solnLen, int, the number of fuel locations in the solution
      @ In, numBatches, int, the maximum number of fuel batches.
      @ Out, faID, str, the FA ID according to the PRLO encoding scheme (0-indexed)
    """
    if FA[0] > solnLen or FA[1] > numBatches:
      return None # prevent eroneous FA's from being given ID's.
    else:
      return (FA[0]-1)*numBatches + solnLen*numBatches*(FA[2]-1) + (FA[1]-1)


class EQChecker(_PRLOCheckerBase):
  """
  EQ checker and genome modifier for existing EQ cycle genomes.
  """
  def __init__(self,prloDataInputFile):
    """
      Initializing variables.
      @ In, None
      @ Out, None
    """
    self.prloData = self.PRLODataParser(prloDataInputFile)

  ## function for shuffling scheme logic
  def checkGenome(self,genome,symMult):
    """
      Validate if the shuffling scheme represented by "genome" and "symMult" correspond to
      a valid equilibrium cycle shuffling scheme.
      @ In, genome, list, the faID selections that define the current solution.
      @ In, symMult, dict, multiplicity of each location in the zoning map (1-indexed).
      @ Out, bool, False if any check is violated; True otherwise.
      @ Out, int, Error code to indicate what caused the failure.
    """
  ## Assert: genome length matches desired solution length
    if len(genome) != self.prloData.solnLen:
      return False, 7

    decodedGenomeWithMult = [(self.decodeFAID(genome[l],self.prloData.solnLen,self.prloData.numBatches),symMult[l+1]) for l in range(len(genome))]
    zoneMap = [l[0][1] for l in decodedGenomeWithMult]

  ## Assert: no batch is larger than the previous one.
    batchCount = {}
    for batNum in set(zoneMap):
      batchCount[batNum] = sum([symMult[i+1] for i in range(len(zoneMap)) if zoneMap[i] == batNum])
    for batNum in batchCount.keys():
      if batNum == 1:
        continue
      else:
        if batchCount[batNum] > batchCount[batNum-1]:
          return False, 1

  ## Assert: every FA has a valid source
    for gene in genome:
      currentFA = self.decodeFAID(gene, self.prloData.solnLen, self.prloData.numBatches)
      if currentFA[1] == 1: #fresh fuel FA
        continue
      else:
        sourceFA = self.decodeFAID(genome[currentFA[0]-1], self.prloData.solnLen, self.prloData.numBatches)
        # check that FA types match and current batchNum = source batchNum + 1
        if currentFA[2] != sourceFA[2] or currentFA[1] - sourceFA[1] != 1:
          return False, 2

  ## Assert: no shuffling motion violates multiplicity
    for i in range(len(decodedGenomeWithMult)):
      gene, multVal = decodedGenomeWithMult[i]
      reloadedCount = sum([loc[1] for loc in decodedGenomeWithMult if loc[0] == (i+1, gene[1]+1, gene[2])])
      if reloadedCount > multVal:
        return False, 3

  ## Assert: FA type doesn't change during reshuffle and batch number increments correctly.
  ##         Fresh fuel (batch 1) is skipped: its source location is itself by convention.
    for i in range(len(decodedGenomeWithMult)):
      gene, multVal = decodedGenomeWithMult[i]
      if gene[1] == 1: # skip fresh fuel
        continue
      sourceLoc = gene[0]-1 # 1-indexed to 0-indexed
      if gene[2] != decodedGenomeWithMult[sourceLoc][0][2]: # do FA types match?
        return False, 4
      elif decodedGenomeWithMult[sourceLoc][0][1] - gene[1] != -1: # does batch number increment by 1 after reload?
        return False, 5

  ## Assert: no fresh WABA assembly is placed at a CR bank location
    wabaTypes = getattr(self.prloData, 'wabaTypes', set())
    crBankLocSet = getattr(self.prloData, 'crBankLocSet', set())
    if wabaTypes and crBankLocSet:
      for i in range(len(genome)):
        _, batchNum, typeNum = self.decodeFAID(genome[i], self.prloData.solnLen, self.prloData.numBatches)
        if batchNum == 1 and typeNum in wabaTypes and (i + 1) in crBankLocSet:
          return False, 6

  ## Assert: all FAID values are valid
    maxFAID = self.prloData.solnLen * self.prloData.numBatches * self.prloData.numTypes - 1
    for gene in genome:
      if gene > maxFAID:
        return False, 8

    return True, 0


class SingleCycleChecker(_PRLOCheckerBase):
  """
  Checker for the direct single-cycle (Nth cycle) FAID interpretation.
  Fresh fuel genes must resolve to their own destination location; reload genes
  must resolve to valid entries in the user-provided reloadGeometry and respect
  symmetry-based source multiplicity limits.
  Inherits decodeFAID, encodeFAID, and PRLODataParser from _PRLOCheckerBase.
  """
  def __init__(self,prloDataInputFile):
    """
      Initializing variables.
      @ In, prloDataInputFile, str, path to the PRLO data XML input file.
      @ Out, None
    """
    self.prloData = self.PRLODataParser(prloDataInputFile, verbosity='reduced')
    # Compute symmetricMultiplicity (not provided by 'reduced' verbosity).
    fuelMap = [int(s) for s in self.prloData.geometry.split() if s.isdigit()]
    self.prloData.symmetricMultiplicity = {i: fuelMap.count(i) for i in fuelMap}
    del self.prloData.symmetricMultiplicity[0]
    self.reloadMap = None if self.prloData.numBatches == 1 else self._parseReloadGeometry()

  def _parseReloadGeometry(self):
    """
      Convert the user-provided reload geometry into a symmetry-reduced reload map.
      The reload geometry is interpreted as a fixed FA-type map using the labels
      defined in faDict.
      @ In, None
      @ Out, reloadMap, dict, parsed reload-map metadata for single-cycle sampling.
    """
    if self.prloData.reloadGeometry is None:
      raise ValueError("SingleCycleChecker requires 'reloadGeometry' be provided in the PRLO data input.")

    reloadTokens = self.prloData.reloadGeometry.strip().split()
    geometryTokens = self.prloData.geometry if isinstance(self.prloData.geometry, list) else self.prloData.geometry.strip().split()
    if len(reloadTokens) != len(geometryTokens):
      raise ValueError("reloadGeometry must contain the same number of entries as geometry.")

    fuelLabelMap = {fa['label']: i + 1 for i, fa in enumerate(self.prloData.fuelFADict)}
    reloadMapDict = {}
    for geomToken, reloadToken in zip(geometryTokens, reloadTokens):
      if not str(geomToken).isdigit() or int(geomToken) == 0:
        continue
      locNum = int(geomToken)
      if reloadToken not in fuelLabelMap:
        raise ValueError(f"reloadGeometry token '{reloadToken}' for destination location {locNum} does not match a fuel FA label in faDict.")
      reloadType = fuelLabelMap[reloadToken]
      if locNum in reloadMapDict and reloadMapDict[locNum] != reloadType:
        raise ValueError(f"reloadGeometry is inconsistent across symmetric copies of location {locNum}.")
      reloadMapDict[locNum] = reloadType

    missingLocs = [loc for loc in range(1, self.prloData.solnLen + 1) if loc not in reloadMapDict]
    if missingLocs:
      raise ValueError(f"reloadGeometry is missing reload FA types for solution locations {missingLocs}.")

    return {'mode': 'fixedType', 'values': [reloadMapDict[loc] for loc in range(1, self.prloData.solnLen + 1)]}

  def checkGenome(self, genome, symMult):
    """
      Validate the direct single-cycle FAID interpretation.
      Fresh fuel genes must resolve to their own destination location, while reload genes
      must resolve to valid entries in reloadGeometry and respect the same symmetry-based
      source multiplicity limits enforced in eq-cycle shuffling.
      @ In, genome, list, FAID-encoded genome for the sampled single-cycle loading pattern.
      @ In, symMult, dict, multiplicity of each location in the zoning map (1-indexed).
      @ Out, valid, bool, True if the genome is compatible with the single-cycle semantics.
      @ Out, int, Error code to indicate what caused the failure.
    """
    if len(genome) != self.prloData.solnLen:
      return False, 1

    reloadValues = None if self.reloadMap is None else self.reloadMap['values']
    freshFuelCount = 0
    sourceDemand = {}

    for i, gene in enumerate(genome):
      sourceLoc, batchNum, faType = self.decodeFAID(int(gene), self.prloData.solnLen, self.prloData.numBatches)
      if batchNum == 1:
        if sourceLoc != i + 1:
          return False, 2
        freshFuelCount += symMult[i + 1]
      elif batchNum == 2:
        if self.reloadMap is None:
          return False, 3
        if faType != reloadValues[sourceLoc - 1]:
          return False, 4
        sourceDemand[sourceLoc] = sourceDemand.get(sourceLoc, 0) + symMult[i + 1]
      else:
        return False, 5

    for sourceLoc, demand in sourceDemand.items():
      if demand > symMult[sourceLoc]:
        return False, 6

    if self.prloData.numBatches == 1:
      if freshFuelCount != self.prloData.numAssemblies:
        return False, 7
    elif not self.prloData.feedBatchSizeLimits[0] <= freshFuelCount <= self.prloData.feedBatchSizeLimits[1]:
      return False, 8

  ## Assert: no fresh WABA assembly is placed at a CR bank location
    wabaTypes = getattr(self.prloData, 'wabaTypes', set())
    crBankLocSet = getattr(self.prloData, 'crBankLocSet', set())
    if wabaTypes and crBankLocSet:
      for i, gene in enumerate(genome):
        _, batchNum, typeNum = self.decodeFAID(int(gene), self.prloData.solnLen, self.prloData.numBatches)
        if batchNum == 1 and typeNum in wabaTypes and (i + 1) in crBankLocSet:
          return False, 9

  ## Assert: all FAID values are valid
    maxFAID = self.prloData.solnLen * self.prloData.numBatches * self.prloData.numTypes - 1
    for gene in genome:
      if gene > maxFAID:
        return False, 10

    return True, 0
