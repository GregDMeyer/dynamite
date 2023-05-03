
#include "bcuda_template_2_private.h"

PetscErrorCode C(BuildGPUShell,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  int xparity,
  Mat *A)
{
  PetscInt M, N, mpi_size;
  shell_context *ctx;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  if (mpi_size > 1) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
      "Shell GPU matrices currently only implemented for 1 MPI process.");
  }

  /* N is dimension of right subspace, M of left */
  M = C(Dim,LEFT_SUBSPACE)(left_subspace_data);
  N = C(Dim,RIGHT_SUBSPACE)(right_subspace_data);
  if (xparity) {
    M /= 2;
    N /= 2;
  }

  PetscCall(C(BuildContext_CUDA,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
    msc, left_subspace_data, right_subspace_data, &ctx));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, M, N, M, N, ctx, A));

  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
    (void(*)(void))C(MatMult_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));
  PetscCall(MatShellSetOperation(*A, MATOP_NORM,
    (void(*)(void))C(MatNorm_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));
  PetscCall(MatShellSetOperation(*A, MATOP_CREATE_VECS,
    (void(*)(void))MatCreateVecs_GPU));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
    (void(*)(void))C(MatDestroyCtx_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))));

  return 0;
}

PetscErrorCode C(BuildContext_CUDA,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  const msc_t *msc,
  const C(data,LEFT_SUBSPACE)* left_subspace_data,
  const C(data,RIGHT_SUBSPACE)* right_subspace_data,
  shell_context **ctx_p)
{
  /* NOTE: some data shared by GPU and CPU implementations is set in BuildMat */

  PetscReal *cpu_real_coeffs, real_part;
  PetscInt nterms, i;
  shell_context *ctx;

  PetscCall(PetscMalloc(sizeof(shell_context), ctx_p));
  ctx = (*ctx_p);

  ctx->gpu = PETSC_TRUE;
  nterms = msc->mask_offsets[msc->nmasks];

  PetscCallCUDA(cudaMalloc((void **) &(ctx->masks),
    sizeof(PetscInt)*msc->nmasks));
  PetscCallCUDA(cudaMemcpy(ctx->masks, msc->masks, sizeof(PetscInt)*msc->nmasks,
    cudaMemcpyHostToDevice));

  PetscCallCUDA(cudaMalloc((void **) &(ctx->mask_offsets),
    sizeof(PetscInt)*(msc->nmasks+1)));
  PetscCallCUDA(cudaMemcpy(ctx->mask_offsets, msc->mask_offsets, sizeof(PetscInt)*(msc->nmasks+1),
    cudaMemcpyHostToDevice));

  PetscCallCUDA(cudaMalloc((void **) &(ctx->signs), sizeof(PetscInt)*nterms));
  PetscCallCUDA(cudaMemcpy(ctx->signs, msc->signs, sizeof(PetscInt)*nterms,
    cudaMemcpyHostToDevice));

  PetscCallCUDA(cudaMalloc((void **) &(ctx->real_coeffs), sizeof(PetscReal)*nterms));
  /*
   * we need a CPU vector in which we will store the real coefficients, then we'll copy
   * from that over to the CPU.
   */
  PetscCall(PetscMalloc1(nterms, &cpu_real_coeffs));
  for (i=0; i < nterms; ++i) {
    real_part = PetscRealPart(msc->coeffs[i]);
    cpu_real_coeffs[i] = (real_part != 0) ? real_part : PetscImaginaryPart(msc->coeffs[i]);
  }
  PetscCallCUDA(cudaMemcpy(ctx->real_coeffs, cpu_real_coeffs, sizeof(PetscReal)*nterms,
    cudaMemcpyHostToDevice));
  PetscCall(PetscFree(cpu_real_coeffs));

  PetscCall(C(CopySubspaceData_CUDA,LEFT_SUBSPACE)(
    (C(data,LEFT_SUBSPACE)**)&(ctx->left_subspace_data),
    (C(data,LEFT_SUBSPACE)*)left_subspace_data));
  PetscCall(C(CopySubspaceData_CUDA,RIGHT_SUBSPACE)(
    (C(data,RIGHT_SUBSPACE)**)&(ctx->right_subspace_data),
    (C(data,RIGHT_SUBSPACE)*)right_subspace_data));

  return 0;
}

PetscErrorCode C(MatDestroyCtx_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A)
{
  shell_context *ctx;

  PetscCall(MatShellGetContext(A, &ctx));

  PetscCallCUDA(cudaFree(ctx->masks));
  PetscCallCUDA(cudaFree(ctx->mask_offsets));
  PetscCallCUDA(cudaFree(ctx->signs));
  PetscCallCUDA(cudaFree(ctx->real_coeffs));

  if (ctx->diag) {
    PetscCallCUDA(cudaFree(ctx->diag));
  }

  PetscCall(C(DestroySubspaceData_CUDA,LEFT_SUBSPACE)(
    (C(data,LEFT_SUBSPACE)*) ctx->left_subspace_data));
  PetscCall(C(DestroySubspaceData_CUDA,RIGHT_SUBSPACE)(
    (C(data,RIGHT_SUBSPACE)*) ctx->right_subspace_data));

  PetscCall(PetscFree(ctx));

  return 0;
}

PetscErrorCode C(MatMult_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, Vec x, Vec b)
{
  shell_context *ctx;

  const PetscScalar* xarray;
  PetscScalar* barray;
  PetscInt size;

  PetscCall(VecSet(b, 0));

  PetscCall(MatShellGetContext(A, &ctx));

  PetscCall(VecCUDAGetArrayRead(x, &xarray));
  PetscCall(VecCUDAGetArray(b, &barray));

  PetscCall(VecGetSize(b, &size));

  PetscCallCUDA(cudaDeviceSynchronize());

  C(device_MatMult,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))<<<GPU_BLOCK_NUM,GPU_BLOCK_SIZE>>>(
    size,
    ctx->masks,
    ctx->mask_offsets,
    ctx->signs,
    ctx->real_coeffs,
    ctx->nmasks,
    (C(data,LEFT_SUBSPACE)*) ctx->left_subspace_data,
    (C(data,RIGHT_SUBSPACE)*) ctx->right_subspace_data,
    ctx->diag,
    xarray,
    barray);

  PetscCallCUDA(cudaDeviceSynchronize());

  PetscCall(VecCUDARestoreArrayRead(x, &xarray));
  PetscCall(VecCUDARestoreArray(b, &barray));

  return 0;
}

__global__ void C(device_MatMult,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt size,
  PetscInt* masks,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  PetscInt nmasks,
  C(data,LEFT_SUBSPACE) *left_subspace_data,
  C(data,RIGHT_SUBSPACE) *right_subspace_data,
  PetscReal* diag,
  const PetscScalar* xarray,
  PetscScalar* barray)
{

  /* the following four lines come from the PETSc cuda source */
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar tmp, val;
  PetscReal sign;
  PetscInt bra, ket, row_idx, col_idx, mask_idx, term_idx, this_start;

  this_start = vec_start_index + threadIdx.x;

  for (row_idx = this_start; row_idx < vec_stop_index; row_idx += blockDim.x) {
    ket = C(I2S_CUDA,LEFT_SUBSPACE)(row_idx,left_subspace_data);

    if (diag) {
      val = diag[row_idx] * xarray[row_idx];
      mask_idx = 1;
    } else {
      val = 0;
      mask_idx = 0;
    }

    for (; mask_idx<nmasks; ++mask_idx) {
      tmp = 0;
      bra = ket ^ masks[mask_idx];

      col_idx = C(S2I_CUDA,RIGHT_SUBSPACE)(bra, right_subspace_data);
      if (col_idx == -1) {  // state is outside of the subspace; skip it
        continue;
      }

      /* sum all terms for this matrix element */
      for (term_idx = mask_offsets[mask_idx]; term_idx < mask_offsets[mask_idx+1]; ++term_idx) {
#if defined(PETSC_USE_64BIT_INDICES)
        sign = __popcll(bra & signs[term_idx])&1;
#else
        sign = __popc(bra & signs[term_idx])&1;
#endif
        sign = 1 - 2*sign;
        if TERM_REAL_CUDA(masks[mask_idx], signs[term_idx]) {
	  add_real(&tmp, sign * real_coeffs[term_idx]);
        }
        else {
          add_imag(&tmp, sign * real_coeffs[term_idx]);
        }
      }
      val += tmp * xarray[col_idx];
    }

    barray[row_idx] = val;

  }
}

PetscErrorCode C(MatNorm_GPU,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(Mat A, NormType type, PetscReal *nrm)
{
  shell_context *ctx;

  PetscReal *d_maxs,*h_maxs;
  PetscInt i, M;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Only NORM_INFINITY is implemented for shell matrices.");
  }

  PetscCall(MatShellGetContext(A, &ctx));

  /*
    keep the norm cached so we don't have to compute it all the time.
    if we already have it, just return it
  */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return 0;
  }

  PetscCallCUDA(cudaMalloc((void **) &d_maxs, sizeof(PetscReal)*GPU_BLOCK_NUM));
  PetscCall(PetscMalloc1(GPU_BLOCK_NUM, &h_maxs));

  PetscCall(MatGetSize(A, &M, NULL));

  C(device_MatNorm,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))<<<GPU_BLOCK_NUM, GPU_BLOCK_SIZE, sizeof(PetscReal)*GPU_BLOCK_SIZE>>>(
    M,
    ctx->masks,
    ctx->mask_offsets,
    ctx->signs,
    ctx->real_coeffs,
    ctx->nmasks,
    (C(data,LEFT_SUBSPACE)*) ctx->left_subspace_data,
    (C(data,RIGHT_SUBSPACE)*) ctx->right_subspace_data,
    d_maxs);

  PetscCallCUDA(cudaDeviceSynchronize());

  PetscCallCUDA(cudaMemcpy(h_maxs, d_maxs, sizeof(PetscReal)*GPU_BLOCK_NUM, cudaMemcpyDeviceToHost));

  /* now do max of h_maxs */
  (*nrm) = 0;
  for (i = 0; i < GPU_BLOCK_NUM; ++i) {
    if (h_maxs[i] > (*nrm)) (*nrm) = h_maxs[i];
  }

  ctx->nrm = (*nrm);

  PetscCallCUDA(cudaFree(d_maxs));
  PetscCall(PetscFree(h_maxs));

  return 0;
}

__global__ void C(device_MatNorm,C(LEFT_SUBSPACE,RIGHT_SUBSPACE))(
  PetscInt size,
  PetscInt* masks,
  PetscInt* mask_offsets,
  PetscInt* signs,
  PetscReal* real_coeffs,
  PetscInt nmasks,
  C(data,LEFT_SUBSPACE) *left_subspace_data,
  C(data,RIGHT_SUBSPACE) *right_subspace_data,
  PetscReal *d_maxs)
{
  extern __shared__ PetscReal threadmax[];

  /* the following four lines come from the PETSc cuda source */
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscReal sum,v1,v2,sign;
  PetscScalar csum;
  PetscInt ket, bra, row_idx, mask_idx, term_idx, i;

  /* first find this thread's max and put it in threadmax */

  threadmax[threadIdx.x] = 0;
  for (row_idx = vec_start_index+threadIdx.x; row_idx < vec_stop_index; row_idx += blockDim.x) {
    ket = C(I2S_CUDA,LEFT_SUBSPACE)(row_idx,left_subspace_data);
    sum = 0;
    for (mask_idx = 0; mask_idx < nmasks; ++mask_idx) {
      csum = 0;
      bra = ket ^ masks[mask_idx];

      if (C(S2I_CUDA,RIGHT_SUBSPACE)(bra, right_subspace_data) == -1) {
	continue;
      }

      /* sum all terms for this matrix element */
      for (term_idx = mask_offsets[mask_idx]; term_idx < mask_offsets[mask_idx+1]; ++term_idx) {
#if defined(PETSC_USE_64BIT_INDICES)
        sign = __popcll(bra & signs[term_idx])&1;
#else
        sign = __popc(bra & signs[term_idx])&1;
#endif
        sign = 1 - 2*sign;
        if TERM_REAL_CUDA(masks[mask_idx], signs[term_idx]) {
	  add_real(&csum, sign * real_coeffs[term_idx]);
        }
        else {
          add_imag(&csum, sign * real_coeffs[term_idx]);
        }
      }
      sum += abs(csum);
    }
    if (sum > threadmax[threadIdx.x]) {
      threadmax[threadIdx.x] = sum;
    }
  }
  __syncthreads();

  /* now do the coolest reduce ever on the shared memory and hand it off to CPU */

  for (i=1; i<blockDim.x; i*=2) {
    if (threadIdx.x % (2*i) == 0) {
      v1 = threadmax[threadIdx.x];
      v2 = threadmax[threadIdx.x + i];
      threadmax[threadIdx.x] = v1>v2 ? v1 : v2;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) d_maxs[blockIdx.x] = threadmax[0];
}
