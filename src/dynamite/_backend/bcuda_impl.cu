
#include "bcuda_impl.h"

#ifdef PETSC_USE_64BIT_INDICES
  #define CUDA_POPCOUNT(x) (__popcll(x))
  #define CUDA_CTZ(x) (__ffsll(x)-1)
#else
  #define CUDA_POPCOUNT(x) (__popc(x))
  #define CUDA_CTZ(x) (__ffs(x)-1)
#endif

#define CUDA_PARITY(x) (CUDA_POPCOUNT(x)&1)

#define TERM_REAL_CUDA(mask, sign) (!CUDA_PARITY((mask) & (sign)))

/* subspace functions for the GPU */

PetscErrorCode CopySubspaceData_CUDA_Full(data_Full** out_p, const data_Full* in) {
  PetscCallCUDA(cudaMalloc((void **) out_p, sizeof(data_Full)));
  PetscCallCUDA(cudaMemcpy(*out_p, in, sizeof(data_Full), cudaMemcpyHostToDevice));
  return 0;
}

PetscErrorCode DestroySubspaceData_CUDA_Full(data_Full* data) {
  PetscCallCUDA(cudaFree(data));
  return 0;
}

__device__ PetscInt S2I_CUDA_Full(PetscInt state, const data_Full* data) {
  return state;
}

__device__ PetscInt I2S_CUDA_Full(PetscInt idx, const data_Full* data) {
  return idx;
}

PetscErrorCode CopySubspaceData_CUDA_Parity(data_Parity** out_p, const data_Parity* in) {
  PetscCallCUDA(cudaMalloc((void **) out_p, sizeof(data_Parity)));
  PetscCallCUDA(cudaMemcpy(*out_p, in, sizeof(data_Parity), cudaMemcpyHostToDevice));
  return 0;
}

PetscErrorCode DestroySubspaceData_CUDA_Parity(data_Parity* data) {
  PetscCallCUDA(cudaFree(data));
  return 0;
}

__device__ PetscInt S2I_CUDA_Parity(PetscInt state, const data_Parity* data) {
  return (CUDA_PARITY(state) == data->space) ? state>>1 : (PetscInt)(-1);
}

__device__ PetscInt I2S_CUDA_Parity(PetscInt idx, const data_Parity* data) {
  return (idx<<1) | (CUDA_PARITY(idx) ^ data->space);
}

PetscErrorCode CopySubspaceData_CUDA_SpinConserve(data_SpinConserve** out_p, const data_SpinConserve* in) {
  PetscInt len_nchoosek = (in->k+1)*in->ld_nchoosek;

  data_SpinConserve cpu_data;

  PetscCall(PetscMemcpy(&cpu_data, in, sizeof(data_SpinConserve)));

  PetscCallCUDA(cudaMalloc(&(cpu_data.nchoosek), sizeof(PetscInt)*len_nchoosek));
  PetscCallCUDA(cudaMemcpy(cpu_data.nchoosek, in->nchoosek,
                           sizeof(PetscInt)*len_nchoosek, cudaMemcpyHostToDevice));

  PetscCallCUDA(cudaMalloc((void **) out_p, sizeof(data_SpinConserve)));
  PetscCallCUDA(cudaMemcpy(*out_p, &cpu_data, sizeof(data_SpinConserve), cudaMemcpyHostToDevice));

  return 0;
}

PetscErrorCode DestroySubspaceData_CUDA_SpinConserve(data_SpinConserve* data) {
  data_SpinConserve cpu_data;

  PetscCallCUDA(cudaMemcpy(&cpu_data, data, sizeof(data_SpinConserve), cudaMemcpyDeviceToHost));

  PetscCallCUDA(cudaFree(cpu_data.nchoosek));
  PetscCallCUDA(cudaFree(data));
  return 0;
}

__device__ PetscInt S2I_CUDA_SpinConserve(PetscInt state, const data_SpinConserve* data) {
  PetscInt n, k=0, idx=0;

  if (state >> data->L) return (PetscInt)(-1);
  if (CUDA_POPCOUNT(state) != data->k) return (PetscInt)(-1);

  while (state) {
    n = CUDA_CTZ(state);
    k++;
    if (k <= n) idx += data->nchoosek[k*data->ld_nchoosek + n];
    state &= state-1;  // pop least significant bit off of state
  }

  return idx;
}

__device__ PetscInt I2S_CUDA_SpinConserve(PetscInt idx, const data_SpinConserve* data) {
  PetscInt state = 0;
  PetscInt k = data->k;
  PetscInt current;
  for (PetscInt n=data->L; n>0; --n) {
    state <<= 1;
    current = (k > n-1) ? 0 : data->nchoosek[k*data->ld_nchoosek + n-1];
    if (idx >= current) {
        idx -= current;
        k--;
        state |= 1 ;
    }
  }
  return state;
}

PetscErrorCode CopySubspaceData_CUDA_Explicit(data_Explicit** out_p, const data_Explicit* in) {
  data_Explicit cpu_data;

  PetscCall(PetscMemcpy(&cpu_data, in, sizeof(data_Explicit)));

  PetscCallCUDA(cudaMalloc(&(cpu_data.state_map), sizeof(PetscInt)*in->dim));
  PetscCallCUDA(cudaMemcpy(cpu_data.state_map, in->state_map,
    sizeof(PetscInt)*in->dim, cudaMemcpyHostToDevice));

  if (in->rmap_indices != PETSC_NULLPTR) {
    PetscCallCUDA(cudaMalloc(&(cpu_data.rmap_indices), sizeof(PetscInt)*in->dim));
    PetscCallCUDA(cudaMemcpy(cpu_data.rmap_indices, in->rmap_indices,
                             sizeof(PetscInt)*in->dim, cudaMemcpyHostToDevice));
  }

  PetscCallCUDA(cudaMalloc(&(cpu_data.rmap_states), sizeof(PetscInt)*in->dim));
  PetscCallCUDA(cudaMemcpy(cpu_data.rmap_states, in->rmap_states,
    sizeof(PetscInt)*in->dim, cudaMemcpyHostToDevice));

  PetscCallCUDA(cudaMalloc((void **) out_p, sizeof(data_Explicit)));
  PetscCallCUDA(cudaMemcpy(*out_p, &cpu_data, sizeof(data_Explicit), cudaMemcpyHostToDevice));

  return 0;
}

PetscErrorCode DestroySubspaceData_CUDA_Explicit(data_Explicit* data) {
  data_Explicit cpu_data;

  PetscCallCUDA(cudaMemcpy(&cpu_data, data, sizeof(data_Explicit), cudaMemcpyDeviceToHost));

  PetscCallCUDA(cudaFree(cpu_data.state_map));
  if (cpu_data.rmap_indices != PETSC_NULLPTR) {
    PetscCallCUDA(cudaFree(cpu_data.rmap_indices));
  }
  PetscCallCUDA(cudaFree(cpu_data.rmap_states));
  PetscCallCUDA(cudaFree(data));
  return 0;
}

/* TODO: this is really not well suited for GPUs */
/* but I bet we can do something clever! */
__device__ PetscInt S2I_CUDA_Explicit(PetscInt state, const data_Explicit* data) {
  PetscInt left, right, mid;
  left = 0;
  right = data->dim;
  while (left <= right) {
    mid = left + (right-left)/2;
    if (data->rmap_states[mid] == state) {
      if (data->rmap_indices != PETSC_NULLPTR) {
        return data->rmap_indices[mid];
      } else {
        return mid;
      }
    }

    if (data->rmap_states[mid] < state) {
      left = mid + 1;
    }
    else {
      right = mid - 1;
    }
  }
  /* element was not in the array */
  return -1;
}

__device__ PetscInt I2S_CUDA_Explicit(PetscInt idx, const data_Explicit* data) {
  return data->state_map[idx];
}

PetscErrorCode MatCreateVecs_GPU(Mat mat, Vec *right, Vec *left)
{
  PetscInt M, N;

  PetscCall(MatGetSize(mat, &M, &N));

  if (right) {
    PetscCall(VecCreate(PetscObjectComm((PetscObject)mat),right));
    PetscCall(VecSetSizes(*right, PETSC_DECIDE, N));
    PetscCall(VecSetFromOptions(*right));
  }
  if (left) {
    PetscCall(VecCreate(PetscObjectComm((PetscObject)mat),left));
    PetscCall(VecSetSizes(*left, PETSC_DECIDE, M));
    PetscCall(VecSetFromOptions(*left));
  }

  return 0;
}

__device__ static __inline__ void add_real(PetscScalar *x, PetscReal r) {
  PetscReal *real_part;
  real_part = (PetscReal*) x;
  (*real_part) += r;
}

__device__ static __inline__ void add_imag(PetscScalar *x, PetscReal c) {
  PetscReal *imag_part;
  imag_part = ((PetscReal*)x) + 1;
  (*imag_part) += c;
}

// defines used in the various templates
#define Full_SP 0
#define Parity_SP 1
#define SpinConserve_SP 2
#define Explicit_SP 3

#define SUBSPACE Full
  #include "bcuda_template_1.cu"
#undef SUBSPACE

#define SUBSPACE Parity
  #include "bcuda_template_1.cu"
#undef SUBSPACE

#define SUBSPACE SpinConserve
  #include "bcuda_template_1.cu"
#undef SUBSPACE

#define SUBSPACE Explicit
  #include "bcuda_template_1.cu"
#undef SUBSPACE

#define LEFT_SUBSPACE Full
  #define RIGHT_SUBSPACE Full
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE SpinConserve
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Explicit
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE Parity
  #define RIGHT_SUBSPACE Full
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE SpinConserve
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Explicit
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE SpinConserve
  #define RIGHT_SUBSPACE Full
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE SpinConserve
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Explicit
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE

#define LEFT_SUBSPACE Explicit
  #define RIGHT_SUBSPACE Full
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Parity
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE SpinConserve
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE

  #define RIGHT_SUBSPACE Explicit
    #include "bcuda_template_2.cu"
  #undef RIGHT_SUBSPACE
#undef LEFT_SUBSPACE
