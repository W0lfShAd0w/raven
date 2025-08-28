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
  Implementation of survivorSelection step for new generation
  selection process in Genetic Algorithm.
  NOTE: this file only exists to call methods in survivorSelectors.py, making for a confusing and convoluted call stack. - rollnk

  Created Apr,3,2024
  @authors: Mohammad Abdo, Junyung Kim
"""

def singleObjSurvivorSelect(self, info, rlz, traj, individuals, individualFitness, objectiveVal, g):
  """
    process of selecting survivors for single objective problems
    @ In, self, Instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations
    @ In, traj, dict, dictionary of trajectories
    @ In, individuals, list, list of individuals
    @ In, individualFitness, list, list of individual fitness
    @ In, objectiveVal, list, floats of objective values
    @ In, g, xr.DataArray, constraint data
  """
  if self.counter > 1:
    self.matingPop_inputs, self.matingPop_fitness,\
    self.matingPop_ages,self.matingPop_objvals = self._survivorSelectionInstance(age=self.matingPop_ages,
                                                                    variables=list(self.toBeSampled),
                                                                    population=self.matingPop_inputs,
                                                                    fitness=self.matingPop_fitness,
                                                                    objVar = self._objectiveVar[0],
                                                                    newRlz=rlz,
                                                                    individualsFitness=individualFitness,
                                                                    popObjectiveVal=self.matingPop_objvals)
  else:
    self.matingPop_inputs = individuals
    self.matingPop_fitness = individualFitness
    self.matingPop_objvals = rlz[self._objectiveVar[0]].data

def multiObjSurvivorSelect(self, info, rlz, traj, individuals, individualFitness, objectiveVal, g):
  """
    process of selecting survivors for multi-objective problems
    @ In, self, instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations (including values of all objectives)
    @ In, traj, dict, dictionary of trajectories
    @ In, individuals, list, list of individual individuals
    @ In, individualFitness, list, list of fitness values for individual individuals
    @ In, objectiveVal, list, values of the objectives (for ranking and crowding distance calculation)
    @ In, g, xr.DataArray, constraint data
  """
  if self.counter > 1:
    self.matingPop_inputs,self.matingPop_ranks, \
    self.matingPop_ages,self.matingPop_CD, \
    self.matingPop_objvals,self.matingPop_fitness, \
    self.matingPop_g                  = self._survivorSelectionInstance(age=self.matingPop_ages,
                                                                         variables=list(self.toBeSampled),
                                                                         population=self.matingPop_inputs,
                                                                         individuals=rlz,
                                                                         popObjectiveVal=self.matingPop_objvals,
                                                                         offObjectiveVal=objectiveVal,
                                                                         popFit = self.matingPop_fitness,
                                                                         offFit = individualFitness,
                                                                         popConstV = self.matingPop_g,
                                                                         direction=self._minMax,
                                                                         offConstV = g)
  else:
    self.matingPop_inputs = individuals
    self.matingPop_fitness = individualFitness
    self.matingPop_objvals = objectiveVal
    self.matingPop_g = g
