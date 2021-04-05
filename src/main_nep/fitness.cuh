/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

struct Cost {
  float weight_force;
  float weight_energy;
  float weight_stress;
  float force_std;     // std of force
  float potential_std; // std of potential
  float virial_std;    // std of virial
};

class Fitness
{
public:
  Fitness(char*);
  void compute(const int, const float*, float*);
  void predict(char*, const float*);
  void report_error(
    char* input_dir,
    const int generation,
    const float loss_total,
    const float loss_L1,
    const float loss_L2,
    const float* elite);
  int number_of_variables; // number of variables in the potential
  int maximum_generation;  // maximum number of generations for SNES;

protected:
  // output files:
  FILE* fid_train_out;

  // functions related to initialization
  void read_Nc(FILE*);
  void read_Na(FILE*);
  void read_potential(char*);
  void read_train_in(char*);

  // functions related to fitness evaluation
  void predict_energy_or_stress(FILE*, float*, float*);
  float get_fitness_force(void);
  float get_fitness_energy(void);
  float get_fitness_stress(void);

  // input potential parameters:
  int num_neurons_2b = 0;
  float rc_2b;
  int num_neurons_3b = 0;
  float rc_3b;
  int num_neurons_mb = 0;
  int n_max, L_max;

  int potential_type;              // 0=NN2B
  int Nc;                          // number of configurations
  int N;                           // total number of atoms (sum of Na[])
  int max_Na;                      // number of atoms in the largest configuration
  int num_virial_configurations;   // number of configurations having virial
  GPU_Vector<int> Na;              // number of atoms in each configuration
  GPU_Vector<int> Na_sum;          // prefix sum of Na
  std::vector<int> has_virial;     // 1 if has virial for a configuration, 0 otherwise
  GPU_Vector<float> atomic_number; // atomic number (number of protons)

  GPU_Vector<float> r;          // position
  GPU_Vector<float> force;      // force
  GPU_Vector<float> pe;         // potential energy
  GPU_Vector<float> virial;     // per-atom virial tensor
  GPU_Vector<float> h;          // box and inverse box
  GPU_Vector<float> pe_ref;     // reference energy for the whole box
  GPU_Vector<float> virial_ref; // reference virial for the whole box
  GPU_Vector<float> force_ref;  // reference force
  std::vector<float> error_cpu; // error in energy, virial, or force
  GPU_Vector<float> error_gpu;  // error in energy, virial, or force

  // other classes
  Neighbor neighbor;
  std::unique_ptr<Potential> potential;
  Cost cost;
};
