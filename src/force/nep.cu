/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "nep.cuh"
#include "nep_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP::initialize_dftd3()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  has_dftd3 = false;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "dftd3") {
        has_dftd3 = true;
        if (tokens.size() != 4) {
          std::cout << "dftd3 must have 3 parameters\n";
          exit(1);
        }
        std::string xc_functional = tokens[1];
        float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
        float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
        dftd3.initialize(xc_functional, rc_potential, rc_coordination_number);
        break;
      }
    }
  }

  input_run.close();
}

NEP::NEP(const char* file_potential, const int num_atoms)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4") {
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5") {
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.version = 5;
    zbl.enabled = true;
  }  else if (tokens[0] == "nep4_temperature") {
    paramb.version = 4;
    paramb.model_type = 3;
  } else if (tokens[0] == "nep4_zbl_temperature") {
    paramb.version = 4;
    paramb.model_type = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_dipole") {
    paramb.version = 4;
    paramb.model_type = 1;
  } else if (tokens[0] == "nep4_polarizability") {
    paramb.version = 4;
    paramb.model_type = 2;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("Use the NEP%d potential with %d atom type.\n", paramb.version, paramb.num_types);
  } else {
    printf("Use the NEP%d potential with %d atom types.\n", paramb.version, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
    printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  } else {
    printf("    cutoff = \n");
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
      printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
  }
  paramb.rc_radial_max_inv = 1.0f / paramb.rc_radial_max;

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);
  if (paramb.has_q_222) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_1111) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_112) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_123) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_233) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_134) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  nep_model_type = paramb.model_type;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }
  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial_max; // largest cutoff
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else if (paramb.version == 5) {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  if (paramb.model_type == 2) {
    // Polarizability models have twice as many parameters
    annmb.num_para_ann *= 2;
  }
  printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para + annmb.dim);
  for (int n = 0; n < annmb.num_para + annmb.dim; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // Store descriptor coefficients for each type pair contiguously. All the
  // descriptor and force kernels consume one type pair at a time.
  std::vector<float> descriptor_parameters(num_para_descriptor);
  const int radial_basis_count =
    (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  const int angular_basis_count =
    (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1);
  for (int type_pair = 0; type_pair < paramb.num_types_sq; ++type_pair) {
    for (int basis = 0; basis < radial_basis_count; ++basis) {
      descriptor_parameters[type_pair * radial_basis_count + basis] =
        parameters[annmb.num_para_ann + basis * paramb.num_types_sq + type_pair];
    }
    for (int basis = 0; basis < angular_basis_count; ++basis) {
      descriptor_parameters[paramb.num_c_radial + type_pair * angular_basis_count + basis] =
        parameters[
          annmb.num_para_ann + paramb.num_c_radial + basis * paramb.num_types_sq + type_pair];
    }
  }
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(parameters.data());
  nep_data.descriptor_parameters_type_pair.resize(num_para_descriptor);
  nep_data.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.c_type_pair = nep_data.descriptor_parameters_type_pair.data();
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  neighbor.initialize(rc, num_atoms, paramb.MN_radial);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(static_cast<size_t>(num_atoms) * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(static_cast<size_t>(num_atoms) * annmb.dim);
  nep_data.sum_fxyz.resize(
    static_cast<size_t>(num_atoms) * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

  initialize_dftd3();
  need_peratom_virial = check_need_peratom_virial();
  B_projection_size = annmb.num_neurons1 * (annmb.dim + 2);
}

NEP::~NEP(void)
{
  // nothing
}

void NEP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  pointer += 1;

  // Possibly read polarizability parameters, which are placed after the regular nep parameters.
  if (paramb.model_type == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
    }
    ann.b1_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
}

static __global__ void find_neighbor_list_large_box(
  NEP::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[static_cast<size_t>(N) * i1 + n1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    int t2 = g_type[n2];
    float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[static_cast<size_t>(N) * count_radial++ + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

template <bool use_radial_basis_sums>
static __global__ void find_descriptor(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const bool is_polarizability,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz,
  bool need_B_projection,
  double* B_projection,
  int B_projection_size)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    if (use_radial_basis_sums) {
      extern __shared__ float radial_basis_sums[];
      const int basis_count = paramb.basis_size_radial + 1;
      for (int k = 0; k < basis_count; ++k) {
        radial_basis_sums[k * blockDim.x + threadIdx.x] = 0.0f;
      }

      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
        int t2 = g_type[n2];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
        for (int k = 0; k < basis_count; ++k) {
          radial_basis_sums[k * blockDim.x + threadIdx.x] += fn12[k];
        }
      }

      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float value = 0.0f;
        for (int k = 0; k < basis_count; ++k) {
          value += radial_basis_sums[k * blockDim.x + threadIdx.x] *
            annmb.c_type_pair[n * basis_count + k];
        }
        q[n] = value;
      }
    } else {
      // get radial descriptors
      for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
        int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index =
              (t1 * paramb.num_types + t2) *
                ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1)) +
              n * (paramb.basis_size_radial + 1) + k;
            gn12 += fn12[k] * annmb.c_type_pair[c_index];
          }
          q[n] += gn12;
        }
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index =
            paramb.num_c_radial +
            (t1 * paramb.num_types + t2) *
              ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1)) +
            n * (paramb.basis_size_angular + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (is_polarizability) {
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[t1],
        annmb.b0_pol[t1],
        annmb.w1_pol[t1],
        annmb.b1_pol,
        q,
        F,
        Fp);
      // Add the potential F for this atom to the diagonal of the virial
      g_virial[n1] = F;
      g_virial[n1 + N * 1] = F;
      g_virial[n1 + N * 2] = F;

      // Reset the potential and forces such that they
      // are zero for the next call to the model. The next call
      // is not used in the case of is_pol = True, but it doesn't
      // hurt to clean up.
      F = 0.0f;
      for (int d = 0; d < annmb.dim; ++d) {
        Fp[d] = 0.0f;
      }
    }

    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    } else {
      if (!need_B_projection)
        apply_ann_one_layer(
          annmb.dim,
          annmb.num_neurons1,
          annmb.w0[t1],
          annmb.b0[t1],
          annmb.w1[t1],
          annmb.b1,
          q,
          F,
          Fp);
      else
        apply_ann_one_layer(
          annmb.dim,
          annmb.num_neurons1,
          annmb.w0[t1],
          annmb.b0[t1],
          annmb.w1[t1],
          annmb.b1,
          q,
          F,
          Fp,
          B_projection + n1 * B_projection_size);
    }
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const bool is_dipole,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int basis = n * (paramb.basis_size_radial + 1) + k;
          int basis_count = (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
          gnp12 +=
            fnp12[k] * annmb.c_type_pair[(t1 * paramb.num_types + t2) * basis_count + basis];
          gnp21 +=
            fnp12[k] * annmb.c_type_pair[(t2 * paramb.num_types + t1) * basis_count + basis];
        }
        float tmp12 = g_Fp[static_cast<size_t>(N) * n + n1] * gnp12 * d12inv;
        float tmp21 = g_Fp[static_cast<size_t>(N) * n + n2] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      if (is_dipole) {
        // The dipole is proportional to minus the sum of the virials times r12
        float r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_sxx -= r12_square * f21[0];
        s_syy -= r12_square * f21[1];
        s_szz -= r12_square * f21[2];
      } else {
        s_sxx += r12[0] * f21[0];
        s_syy += r12[1] * f21[1];
        s_szz += r12[2] * f21[2];
      }
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

static __global__ void find_partial_force_angular(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[static_cast<size_t>(N) * (paramb.n_max_radial + 1 + d) + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index =
            paramb.num_c_radial +
            (t1 * paramb.num_types + t2) *
              ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1)) +
            n * (paramb.basis_size_angular + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __device__ __forceinline__ void add_angular_derivative(
  const float gn_scale,
  const float gnp_scale,
  const float value,
  const float gradient_x,
  const float gradient_y,
  const float gradient_z,
  const float inverse_distance,
  const float* unit,
  float* force)
{
  const float projection = gradient_x * unit[0] + gradient_y * unit[1] + gradient_z * unit[2];
  const float radial = gnp_scale * value;
  const float angular = gn_scale * inverse_distance;
  force[0] += radial * unit[0] + angular * (gradient_x - unit[0] * projection);
  force[1] += radial * unit[1] + angular * (gradient_y - unit[1] * projection);
  force[2] += radial * unit[2] + angular * (gradient_z - unit[2] * projection);
}

static __device__ __forceinline__ int get_angular_channel(const int component)
{
  if (component < 3) {
    return 0;
  }
  if (component < 8) {
    return 1;
  }
  if (component < 15) {
    return 2;
  }
  return 3;
}

static __device__ __forceinline__ float get_angular_weight(const int component)
{
  switch (component) {
    case 0:
      return 2.0f * 0.238732414637843f;
    case 1:
    case 2:
      return 4.0f * 0.119366207318922f;
    case 3:
      return 2.0f * 0.099471839432435f;
    case 4:
    case 5:
      return 4.0f * 0.596831036594608f;
    case 6:
    case 7:
      return 4.0f * 0.149207759148652f;
    case 8:
      return 2.0f * 0.139260575205408f;
    case 9:
    case 10:
      return 4.0f * 0.104445431404056f;
    case 11:
    case 12:
      return 4.0f * 1.044454314040563f;
    case 13:
    case 14:
      return 4.0f * 0.174075719006761f;
    case 15:
      return 2.0f * 0.011190581936149f;
    case 16:
    case 17:
      return 4.0f * 0.223811638722978f;
    case 18:
    case 19:
      return 4.0f * 0.111905819361489f;
    case 20:
    case 21:
      return 4.0f * 1.566681471060845f;
    default:
      return 4.0f * 0.195835183882606f;
  }
}

struct AngularCubicTerm {
  unsigned char component0;
  unsigned char component1;
  unsigned char component2;
  unsigned char coefficient;
  signed char sign;
};

__constant__ AngularCubicTerm Q112_TERMS[8] = {
  {0, 0, 3, 0, 1}, {0, 1, 4, 1, 1}, {0, 2, 5, 1, 1}, {3, 1, 1, 2, 1},
  {3, 2, 2, 2, 1}, {6, 1, 1, 3, 1}, {6, 2, 2, 3, -1}, {1, 2, 7, 4, 1}};

__constant__ AngularCubicTerm Q123_TERMS[21] = {
  {12, 2, 4, 6, 1}, {11, 2, 5, 6, -1}, {1, 11, 4, 6, 1}, {1, 12, 5, 6, 1},
  {0, 11, 6, 5, 1}, {0, 12, 7, 5, 1}, {14, 2, 6, 3, 1}, {13, 2, 7, 3, -1},
  {1, 13, 6, 3, 1}, {1, 14, 7, 3, 1}, {10, 0, 5, 4, 1}, {0, 4, 9, 4, 1},
  {10, 2, 3, 1, 1}, {0, 3, 8, 1, 1}, {1, 3, 9, 1, 1}, {10, 2, 6, 0, 1},
  {10, 1, 7, 0, -1}, {2, 7, 9, 0, -1}, {1, 6, 9, 0, -1}, {2, 5, 8, 2, -1},
  {1, 4, 8, 2, -1}};

__constant__ AngularCubicTerm Q233_TERMS[24] = {
  {3, 8, 8, 0, 1}, {10, 10, 3, 1, 1}, {3, 9, 9, 1, 1}, {10, 10, 6, 2, -1},
  {6, 9, 9, 2, 1}, {4, 8, 9, 3, 1}, {10, 5, 8, 3, 1}, {13, 13, 3, 4, -1},
  {14, 14, 3, 4, -1}, {14, 7, 9, 5, -1}, {13, 6, 9, 5, -1}, {10, 14, 6, 5, -1},
  {10, 13, 7, 5, 1}, {10, 7, 9, 6, 1}, {11, 6, 8, 7, -1}, {12, 7, 8, 7, -1},
  {11, 4, 9, 8, 1}, {12, 5, 9, 8, 1}, {10, 12, 4, 8, 1}, {10, 11, 5, 8, -1},
  {12, 14, 4, 9, 1}, {11, 14, 5, 9, 1}, {13, 11, 4, 9, 1}, {13, 12, 5, 9, -1}};

__constant__ AngularCubicTerm Q134_TERMS[31] = {
  {10, 15, 2, 0, -1}, {1, 15, 9, 0, -1}, {0, 15, 8, 1, 1}, {1, 13, 18, 2, -1},
  {1, 14, 19, 2, -1}, {2, 14, 18, 2, -1}, {2, 13, 19, 2, 1}, {10, 18, 2, 3, -1},
  {1, 10, 19, 3, 1}, {1, 18, 9, 3, 1}, {2, 19, 9, 3, 1}, {1, 16, 8, 4, 1},
  {2, 17, 8, 4, 1}, {0, 10, 17, 5, 1}, {0, 16, 9, 5, 1}, {1, 11, 16, 5, -1},
  {1, 12, 17, 5, -1}, {2, 12, 16, 5, -1}, {2, 11, 17, 5, 1}, {1, 13, 22, 6, 1},
  {1, 14, 23, 6, 1}, {2, 14, 22, 6, -1}, {2, 13, 23, 6, 1}, {0, 11, 18, 7, 1},
  {0, 12, 19, 7, 1}, {0, 13, 20, 8, 1}, {0, 14, 21, 8, 1}, {1, 11, 20, 9, 1},
  {1, 12, 21, 9, 1}, {2, 12, 20, 9, -1}, {2, 11, 21, 9, 1}};

static __device__ __noinline__ void add_cubic_pull(
  const AngularCubicTerm* terms,
  const int num_terms,
  const float* coefficients,
  const float Fp,
  const float* sum,
  float* pull)
{
  for (int index = 0; index < num_terms; ++index) {
    const AngularCubicTerm term = terms[index];
    const float scale = Fp * coefficients[term.coefficient] * static_cast<float>(term.sign);
    pull[term.component0] += scale * sum[term.component1] * sum[term.component2];
    pull[term.component1] += scale * sum[term.component0] * sum[term.component2];
    pull[term.component2] += scale * sum[term.component0] * sum[term.component1];
  }
}

static __device__ __forceinline__ void build_l4_angular_pull(
  const NEP::ParaMB paramb,
  const float* sum,
  const float* Fp,
  float* pull)
{
#pragma unroll
  for (int component = 0; component < 24; ++component) {
    pull[component] =
      get_angular_weight(component) * sum[component] * Fp[get_angular_channel(component)];
  }

  int channel = 4;
  if (paramb.has_q_222) {
    const float s3 = sum[3];
    const float s4 = sum[4];
    const float s5 = sum[5];
    const float s6 = sum[6];
    const float s7 = sum[7];
    const float Fp_q222 = Fp[channel++];
    pull[3] += Fp_q222 *
      (-0.022498442479992f * s3 * s3 - 0.134990654879954f * (s4 * s4 + s5 * s5) +
       0.067495327439977f * (s6 * s6 + s7 * s7));
    pull[4] += Fp_q222 *
      (-0.269981309759908f * s3 * s4 - 0.809943929279722f * s6 * s4 -
       0.809943929279723f * s5 * s7);
    pull[5] += Fp_q222 *
      (-0.269981309759908f * s3 * s5 + 0.809943929279722f * s6 * s5 -
       0.809943929279723f * s4 * s7);
    pull[6] += Fp_q222 *
      (0.134990654879954f * s3 * s6 + 0.404971964639861f * (s5 * s5 - s4 * s4));
    pull[7] += Fp_q222 *
      (0.134990654879954f * s3 * s7 - 0.809943929279723f * s4 * s5);
  }

  if (paramb.has_q_1111) {
    const float s0_squared = sum[0] * sum[0];
    const float s12_squared = sum[1] * sum[1] + sum[2] * sum[2];
    const float Fp_q1111 = Fp[channel++];
    pull[0] += Fp_q1111 *
      (0.106387242824456f * sum[0] * s0_squared + 0.106387242824454f * sum[0] * s12_squared);
    pull[1] += Fp_q1111 *
      (0.106387242824454f * s0_squared * sum[1] + 0.106387242824456f * s12_squared * sum[1]);
    pull[2] += Fp_q1111 *
      (0.106387242824454f * s0_squared * sum[2] + 0.106387242824456f * s12_squared * sum[2]);
  }
  if (paramb.has_q_112) {
    add_cubic_pull(Q112_TERMS, 8, C4B2, Fp[channel++], sum, pull);
  }
  if (paramb.has_q_123) {
    add_cubic_pull(Q123_TERMS, 21, C4B_123, Fp[channel++], sum, pull);
  }
  if (paramb.has_q_233) {
    add_cubic_pull(Q233_TERMS, 24, C4B_233, Fp[channel++], sum, pull);
  }
  if (paramb.has_q_134) {
    add_cubic_pull(Q134_TERMS, 31, C4B_134, Fp[channel], sum, pull);
  }
}

static __device__ __forceinline__ void accumulate_angular_component(
  const int component,
  const float gn_scale,
  const float gnp_scale,
  const float inverse_distance,
  const float* unit,
  float* force)
{
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x2_minus_y2 = x2 - y2;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;

  switch (component) {
    case 0:
      add_angular_derivative(gn_scale, gnp_scale, z, 0.0f, 0.0f, 1.0f, inverse_distance, unit, force);
      break;
    case 1:
      add_angular_derivative(gn_scale, gnp_scale, x, 1.0f, 0.0f, 0.0f, inverse_distance, unit, force);
      break;
    case 2:
      add_angular_derivative(gn_scale, gnp_scale, y, 0.0f, 1.0f, 0.0f, inverse_distance, unit, force);
      break;
    case 3:
      add_angular_derivative(
        gn_scale, gnp_scale, -1.0f + 3.0f * z2, 0.0f, 0.0f, 6.0f * z, inverse_distance, unit, force);
      break;
    case 4:
      add_angular_derivative(gn_scale, gnp_scale, z * x, z, 0.0f, x, inverse_distance, unit, force);
      break;
    case 5:
      add_angular_derivative(gn_scale, gnp_scale, z * y, 0.0f, z, y, inverse_distance, unit, force);
      break;
    case 6:
      add_angular_derivative(
        gn_scale, gnp_scale, x2_minus_y2, 2.0f * x, -2.0f * y, 0.0f, inverse_distance, unit, force);
      break;
    case 7:
      add_angular_derivative(
        gn_scale, gnp_scale, two_xy, 2.0f * y, 2.0f * x, 0.0f, inverse_distance, unit, force);
      break;
    case 8:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        -3.0f * z + 5.0f * z * z2,
        0.0f,
        0.0f,
        -3.0f + 15.0f * z2,
        inverse_distance,
        unit,
        force);
      break;
    case 9: {
      const float a = -1.0f + 5.0f * z2;
      add_angular_derivative(gn_scale, gnp_scale, a * x, a, 0.0f, 10.0f * x * z, inverse_distance, unit, force);
      break;
    }
    case 10: {
      const float a = -1.0f + 5.0f * z2;
      add_angular_derivative(gn_scale, gnp_scale, a * y, 0.0f, a, 10.0f * y * z, inverse_distance, unit, force);
      break;
    }
    case 11:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        z * x2_minus_y2,
        2.0f * x * z,
        -2.0f * y * z,
        x2_minus_y2,
        inverse_distance,
        unit,
        force);
      break;
    case 12:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        z * two_xy,
        2.0f * y * z,
        2.0f * x * z,
        two_xy,
        inverse_distance,
        unit,
        force);
      break;
    case 13:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        x3_minus_3xy2,
        3.0f * x2 - 3.0f * y2,
        -6.0f * x * y,
        0.0f,
        inverse_distance,
        unit,
        force);
      break;
    case 14:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        three_x2y_minus_y3,
        6.0f * x * y,
        3.0f * x2 - 3.0f * y2,
        0.0f,
        inverse_distance,
        unit,
        force);
      break;
    case 15:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        3.0f - 30.0f * z2 + 35.0f * z2 * z2,
        0.0f,
        0.0f,
        -60.0f * z + 140.0f * z * z2,
        inverse_distance,
        unit,
        force);
      break;
    case 16: {
      const float z_factor = -3.0f * z + 7.0f * z * z2;
      add_angular_derivative(
        gn_scale, gnp_scale, z_factor * x, z_factor, 0.0f, (-3.0f + 21.0f * z2) * x, inverse_distance, unit, force);
      break;
    }
    case 17: {
      const float z_factor = -3.0f * z + 7.0f * z * z2;
      add_angular_derivative(
        gn_scale, gnp_scale, z_factor * y, 0.0f, z_factor, (-3.0f + 21.0f * z2) * y, inverse_distance, unit, force);
      break;
    }
    case 18: {
      const float a = -1.0f + 7.0f * z2;
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        a * x2_minus_y2,
        2.0f * x * a,
        -2.0f * y * a,
        14.0f * z * x2_minus_y2,
        inverse_distance,
        unit,
        force);
      break;
    }
    case 19: {
      const float a = -1.0f + 7.0f * z2;
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        a * two_xy,
        2.0f * y * a,
        2.0f * x * a,
        14.0f * z * two_xy,
        inverse_distance,
        unit,
        force);
      break;
    }
    case 20:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        z * x3_minus_3xy2,
        z * (3.0f * x2 - 3.0f * y2),
        z * (-6.0f * x * y),
        x3_minus_3xy2,
        inverse_distance,
        unit,
        force);
      break;
    case 21:
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        z * three_x2y_minus_y3,
        z * 6.0f * x * y,
        z * (3.0f * x2 - 3.0f * y2),
        three_x2y_minus_y3,
        inverse_distance,
        unit,
        force);
      break;
    case 22: {
      const float value = x * x3_minus_3xy2 - y * three_x2y_minus_y3;
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        value,
        4.0f * x * x2 - 12.0f * x * y2,
        -12.0f * x2 * y + 4.0f * y * y2,
        0.0f,
        inverse_distance,
        unit,
        force);
      break;
    }
    default: {
      const float value = x * three_x2y_minus_y3 + y * x3_minus_3xy2;
      add_angular_derivative(
        gn_scale,
        gnp_scale,
        value,
        12.0f * x2 * y - 4.0f * y * y2,
        4.0f * x * x2 - 12.0f * x * y2,
        0.0f,
        inverse_distance,
        unit,
        force);
      break;
    }
  }
}

template <int num_angular_orders>
static __device__ __forceinline__ void find_angular_radial_response(
  const int basis_size,
  const float cutoff,
  const float inverse_cutoff,
  const float distance,
  const float* coefficients,
  float (&gn)[num_angular_orders],
  float (&gnp)[num_angular_orders])
{
  float fc = 0.0f;
  float fcp = 0.0f;
  find_fc_and_fcp(cutoff, inverse_cutoff, distance, fc, fcp);
  float fn[MAX_NUM_N];
  float fnp[MAX_NUM_N];
  find_fn_and_fnp(basis_size, inverse_cutoff, distance, fc, fcp, fn, fnp);
  for (int k = 0; k <= basis_size; ++k) {
#pragma unroll
    for (int n = 0; n < num_angular_orders; ++n) {
      const float coefficient = coefficients[n * (basis_size + 1) + k];
      gn[n] += fn[k] * coefficient;
      gnp[n] += fnp[k] * coefficient;
    }
  }
}

template <int num_angular_orders, int subwarp_size>
static __device__ __forceinline__ void contract_angular_pull(
  const float* pull,
  const int source_lane,
  const float (&gn)[num_angular_orders],
  const float (&gnp)[num_angular_orders],
  float& gn_scale,
  float& gnp_scale)
{
  gn_scale = 0.0f;
  gnp_scale = 0.0f;
#pragma unroll
  for (int n = 0; n < num_angular_orders; ++n) {
    const float pull_value = __shfl_sync(0xffffffffu, pull[n], source_lane, subwarp_size);
    gn_scale += pull_value * gn[n];
    gnp_scale += pull_value * gnp[n];
  }
}

template <int num_angular_orders>
static __global__ void find_force_angular_l4(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  constexpr int num_angular_components = 24;
  constexpr int max_angular_channels = 10;
  constexpr int atoms_per_warp = 4;
  constexpr int edges_per_atom = 8;
  constexpr int rounds = num_angular_components / edges_per_atom;
  constexpr int sum_per_atom = num_angular_orders * num_angular_components;
  constexpr int Fp_per_atom = num_angular_orders * max_angular_channels;
  const int lane = threadIdx.x;
  const int atom_in_warp = lane / edges_per_atom;
  const int edge_lane = lane % edges_per_atom;
  const int atom_base = N1 + blockIdx.x * atoms_per_warp;
  const int n1 = atom_base + atom_in_warp;
  const bool valid_atom = n1 < N2;
  const int source_atom = valid_atom ? n1 : N1;

  __shared__ float shared_sum[atoms_per_warp * sum_per_atom];
  __shared__ float shared_Fp[atoms_per_warp * Fp_per_atom];
  __shared__ float shared_pull[atoms_per_warp * sum_per_atom];

  for (int index = lane; index < atoms_per_warp * sum_per_atom; index += 32) {
    const int local_atom = index % atoms_per_warp;
    const int component = index / atoms_per_warp;
    const int atom = atom_base + local_atom < N2 ? atom_base + local_atom : N1;
    shared_sum[local_atom * sum_per_atom + component] =
      g_sum_fxyz[static_cast<size_t>(N) * component + atom];
  }
  for (int index = lane; index < atoms_per_warp * Fp_per_atom; index += 32) {
    const int local_atom = index % atoms_per_warp;
    const int component = index / atoms_per_warp;
    const int order = component / max_angular_channels;
    const int channel = component % max_angular_channels;
    const int atom = atom_base + local_atom < N2 ? atom_base + local_atom : N1;
    float value = 0.0f;
    if (channel < paramb.num_L) {
      const int descriptor = paramb.n_max_radial + 1 + channel * num_angular_orders + order;
      value = g_Fp[static_cast<size_t>(N) * descriptor + atom];
    }
    shared_Fp[local_atom * Fp_per_atom + component] = value;
  }
  __syncthreads();

  if (edge_lane < num_angular_orders) {
    const float* sum =
      shared_sum + atom_in_warp * sum_per_atom + edge_lane * num_angular_components;
    const float* Fp =
      shared_Fp + atom_in_warp * Fp_per_atom + edge_lane * max_angular_channels;
    float* pull =
      shared_pull + atom_in_warp * sum_per_atom + edge_lane * num_angular_components;
    build_l4_angular_pull(paramb, sum, Fp, pull);
  }
  __syncthreads();

  float owned_pull[rounds][num_angular_orders];
#pragma unroll
  for (int round = 0; round < rounds; ++round) {
#pragma unroll
    for (int n = 0; n < num_angular_orders; ++n) {
      owned_pull[round][n] = shared_pull[
        atom_in_warp * sum_per_atom + n * num_angular_components + round * edges_per_atom + edge_lane];
    }
  }

  int neighbor_count = 0;
  int type1 = 0;
  double x1 = 0.0;
  double y1 = 0.0;
  double z1 = 0.0;
  if (edge_lane == 0 && valid_atom) {
    neighbor_count = g_NN_angular[n1];
    type1 = g_type[n1];
    x1 = g_x[n1];
    y1 = g_y[n1];
    z1 = g_z[n1];
  }
  neighbor_count = __shfl_sync(0xffffffffu, neighbor_count, 0, edges_per_atom);
  type1 = __shfl_sync(0xffffffffu, type1, 0, edges_per_atom);
  x1 = __shfl_sync(0xffffffffu, x1, 0, edges_per_atom);
  y1 = __shfl_sync(0xffffffffu, y1, 0, edges_per_atom);
  z1 = __shfl_sync(0xffffffffu, z1, 0, edges_per_atom);

  int maximum_neighbor_count = neighbor_count;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    maximum_neighbor_count = max(
      maximum_neighbor_count, __shfl_xor_sync(0xffffffffu, maximum_neighbor_count, offset));
  }

  float center_force[3] = {0.0f};
  float center_virial[9] = {0.0f};
  const int basis_count = paramb.basis_size_angular + 1;
  const int coefficient_count_per_pair = num_angular_orders * basis_count;
  const int batch_count = (maximum_neighbor_count + edges_per_atom - 1) / edges_per_atom;
  for (int batch = 0; batch < batch_count; ++batch) {
    const int slot = batch * edges_per_atom + edge_lane;
    const bool active_edge = valid_atom && slot < neighbor_count;
    int n2 = source_atom;
    float r12[3] = {1.0f, 0.0f, 0.0f};
    float distance = 1.0f;
    float inverse_distance = 1.0f;
    float unit[3] = {1.0f, 0.0f, 0.0f};
    float gn[num_angular_orders] = {0.0f};
    float gnp[num_angular_orders] = {0.0f};
    if (active_edge) {
      n2 = g_NL_angular[n1 + N * slot];
      r12[0] = g_x[n2] - x1;
      r12[1] = g_y[n2] - y1;
      r12[2] = g_z[n2] - z1;
      apply_mic(box, r12[0], r12[1], r12[2]);
      distance = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      inverse_distance = 1.0f / distance;
      unit[0] = r12[0] * inverse_distance;
      unit[1] = r12[1] * inverse_distance;
      unit[2] = r12[2] * inverse_distance;
      const int type2 = g_type[n2];
      const float cutoff = (paramb.rc_angular[type1] + paramb.rc_angular[type2]) * 0.5f;
      const float* coefficients = annmb.c_type_pair + paramb.num_c_radial +
        (type1 * paramb.num_types + type2) * coefficient_count_per_pair;
      find_angular_radial_response<num_angular_orders>(
        paramb.basis_size_angular, cutoff, 1.0f / cutoff, distance, coefficients, gn, gnp);
    }

    float force[3] = {0.0f};
#pragma unroll
    for (int round = 0; round < rounds; ++round) {
#pragma unroll
      for (int source_lane = 0; source_lane < edges_per_atom; ++source_lane) {
        float gn_scale = 0.0f;
        float gnp_scale = 0.0f;
        contract_angular_pull<num_angular_orders, edges_per_atom>(
          owned_pull[round], source_lane, gn, gnp, gn_scale, gnp_scale);
        accumulate_angular_component(
          round * edges_per_atom + source_lane,
          gn_scale,
          gnp_scale,
          inverse_distance,
          unit,
          force);
      }
    }

    if (active_edge) {
      center_force[0] += force[0];
      center_force[1] += force[1];
      center_force[2] += force[2];
      atomicAdd(g_fx + n2, -static_cast<double>(force[0]));
      atomicAdd(g_fy + n2, -static_cast<double>(force[1]));
      atomicAdd(g_fz + n2, -static_cast<double>(force[2]));
      center_virial[0] -= r12[0] * force[0];
      center_virial[1] -= r12[1] * force[1];
      center_virial[2] -= r12[2] * force[2];
      center_virial[3] -= r12[0] * force[1];
      center_virial[4] -= r12[0] * force[2];
      center_virial[5] -= r12[1] * force[2];
      center_virial[6] -= r12[1] * force[0];
      center_virial[7] -= r12[2] * force[0];
      center_virial[8] -= r12[2] * force[1];
    }
  }

#pragma unroll
  for (int offset = edges_per_atom / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int component = 0; component < 3; ++component) {
      center_force[component] +=
        __shfl_down_sync(0xffffffffu, center_force[component], offset, edges_per_atom);
    }
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      center_virial[component] +=
        __shfl_down_sync(0xffffffffu, center_virial[component], offset, edges_per_atom);
    }
  }
  if (edge_lane == 0 && valid_atom) {
    atomicAdd(g_fx + n1, static_cast<double>(center_force[0]));
    atomicAdd(g_fy + n1, static_cast<double>(center_force[1]));
    atomicAdd(g_fz + n1, static_cast<double>(center_force[2]));
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      g_virial[n1 + static_cast<size_t>(component) * N] += center_virial[component];
    }
  }
}

static bool launch_find_force_angular(
  const int grid_size,
  const int block_size,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_Fp,
  const float* g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z,
  const bool is_dipole,
  const bool need_peratom_virial,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const int num_angular_orders = paramb.n_max_angular + 1;
  const int num_angular_channels = paramb.L_max +
    static_cast<int>(paramb.has_q_222 != 0) +
    static_cast<int>(paramb.has_q_1111 != 0) +
    static_cast<int>(paramb.has_q_112 != 0) +
    static_cast<int>(paramb.has_q_123 != 0) +
    static_cast<int>(paramb.has_q_233 != 0) +
    static_cast<int>(paramb.has_q_134 != 0);
  const bool use_direct_force =
    !is_dipole && !need_peratom_virial && paramb.model_type == 0 && paramb.L_max == 4 &&
    paramb.num_L == num_angular_channels &&
    paramb.dim_angular == num_angular_orders * num_angular_channels &&
    (num_angular_orders == 3 || num_angular_orders == 5);
  if (use_direct_force) {
    const int tile_grid_size = (N2 - N1 + 3) / 4;
    if (num_angular_orders == 3) {
      find_force_angular_l4<3><<<tile_grid_size, 32>>>(
        paramb, annmb, N, N1, N2, box, g_NN_angular, g_NL_angular, g_type, g_x, g_y, g_z,
        g_Fp, g_sum_fxyz, g_fx, g_fy, g_fz, g_virial);
    } else {
      find_force_angular_l4<5><<<tile_grid_size, 32>>>(
        paramb, annmb, N, N1, N2, box, g_NN_angular, g_NL_angular, g_type, g_x, g_y, g_z,
        g_Fp, g_sum_fxyz, g_fx, g_fy, g_fz, g_virial);
    }
  } else {
    find_partial_force_angular<<<grid_size, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      g_NN_angular,
      g_NL_angular,
      g_type,
      g_x,
      g_y,
      g_z,
      g_Fp,
      g_sum_fxyz,
      g_f12x,
      g_f12y,
      g_f12z);
  }
  return use_direct_force;
}

static __global__ void find_force_ZBL(
  NEP::ParaMB paramb,
  const int N,
  const NEP::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

// large box fo MD applications
void NEP::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor.find_neighbor_global(
    rc,
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor.NN.data(),
    neighbor.NL.data(),
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  bool is_polarizability = paramb.model_type == 2;
  const size_t radial_basis_sum_bytes = static_cast<size_t>(BLOCK_SIZE) *
    (paramb.basis_size_radial + 1) * sizeof(float);
  if (paramb.num_types == 1) {
    find_descriptor<true><<<grid_size, BLOCK_SIZE, radial_basis_sum_bytes>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      is_polarizability,
      potential_per_atom.data(),
      nep_data.Fp.data(),
      virial_per_atom.data(),
      nep_data.sum_fxyz.data(),
      need_B_projection,
      B_projection,
      B_projection_size);
  } else {
    find_descriptor<false><<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      is_polarizability,
      potential_per_atom.data(),
      nep_data.Fp.data(),
      virial_per_atom.data(),
      nep_data.sum_fxyz.data(),
      need_B_projection,
      B_projection,
      B_projection_size);
  }
  GPU_CHECK_KERNEL

  bool is_dipole = paramb.model_type == 1;
  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  bool angular_force_is_direct = launch_find_force_angular(
    grid_size,
    BLOCK_SIZE,
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    is_dipole,
    need_peratom_virial,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (!angular_force_is_direct) {
    find_properties_many_body(
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      nep_data.f12x.data(),
      nep_data.f12y.data(),
      nep_data.f12z.data(),
      is_dipole,
      position_per_atom,
      force_per_atom,
      virial_per_atom);
    GPU_CHECK_KERNEL
  }

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  const bool is_polarizability = paramb.model_type == 2;
  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    is_polarizability,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data(),
    need_B_projection,
    B_projection,
    B_projection_size);
  GPU_CHECK_KERNEL

  bool is_dipole = paramb.model_type == 1;
  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}

void NEP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);
  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

static __global__ void find_descriptor(
  const float temperature,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index =
            (t1 * paramb.num_types + t2) *
              ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1)) +
            n * (paramb.basis_size_radial + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index =
            paramb.num_c_radial +
            (t1 * paramb.num_types + t2) *
              ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1)) +
            n * (paramb.basis_size_angular + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // nomalize descriptor
    q[annmb.dim - 1] = temperature;
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

void NEP::compute_large_box(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor.find_neighbor_global(
    rc,
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor.NN.data(),
    neighbor.NL.data(),
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    temperature,
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  bool is_dipole = paramb.model_type == 1;
  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  bool angular_force_is_direct = launch_find_force_angular(
    grid_size,
    BLOCK_SIZE,
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    is_dipole,
    need_peratom_virial,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (!angular_force_is_direct) {
    find_properties_many_body(
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      nep_data.f12x.data(),
      nep_data.f12y.data(),
      nep_data.f12z.data(),
      is_dipole,
      position_per_atom,
      force_per_atom,
      virial_per_atom);
    GPU_CHECK_KERNEL
  }

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP::compute_small_box(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    temperature,
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  bool is_dipole = paramb.model_type == 1;
  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    is_dipole,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

void NEP::compute(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);

  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

    compute_small_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else {
    compute_large_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  }

  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

const GPU_Vector<int>& NEP::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP::get_NL_radial_ptr() { return nep_data.NL_radial; }
