
#include "backend_impl.h"

#undef  __FUNCT__
#define __FUNCT__ "BuildMat_Full"
PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,
                             const PetscInt* masks,
                             const PetscInt* signs,
                             const PetscScalar* coeffs,
                             Subspaces s,
                             Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,M,i,Istart,Iend,nrows,local_bits,nonlocal_mask;
  int mpi_rank,mpi_size;
  PetscInt d_nz,o_nz;

  PetscInt ket,ketidx,bra,braidx,sign;
  PetscScalar tmp_val;

  /* N is dimension of right subspace, M of left */
  N = get_dimension(L,s.right_type,s.right_space);
  M = get_dimension(L,s.left_type,s.left_space);

  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);

  ierr = MatCreate(PETSC_COMM_WORLD,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*A);CHKERRQ(ierr);

  nrows = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&nrows,&N);

  /* find how many spins are local, for allocation purposes */
  local_bits = 0;
  while(nrows >>= 1) ++local_bits;

  /* now count how many terms are not on our processor */

  /* this creates something that looks like 111110000000 */
  nonlocal_mask = (-1) << local_bits;

  /* if we are going to parity subspace, the leftmost bit doesn't matter */
  if (s.left_type == PARITY) {
    nonlocal_mask &= PARITY_MASK(L);
  }

  d_nz = o_nz = 0;
  for (i=0;i<nterms;++i) {
    /* only count each element once, even though
       there might be a few terms that contribute to it */
    if (i>0 && masks[i-1] == masks[i]) continue;

    if (masks[i] & nonlocal_mask) ++o_nz;
    else ++d_nz;
  }

  /* it seems from the documentation you can just call both of these and it will use the right one */
  ierr = MatSeqAIJSetPreallocation(*A,d_nz,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*A,d_nz,NULL,o_nz,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(*A,&Istart,&Iend);CHKERRQ(ierr);

  /* this is where the magic happens */

  /*
   * the subspace operations in these loops aren't very well optimized, but matrix
   * building isn't usually a big bottleneck, so it should be OK. at some point they
   * could be optimized further for some weird cases where matrix building is a
   * bottleneck.
   */

  for (ketidx=Istart;ketidx<Iend;++ketidx) {
    ket = ketidx;
    if (s.left_type == PARITY) {
      ket = ketidx | (((__builtin_popcount(ketidx)&1)^(s.left_space)) << L);
    }
    for (i=0;i<nterms;) {
      bra = ket ^ masks[i];
      braidx = bra;

      if (s.right_type == PARITY) {
        /*
         * if this value doesn't fit into our subspace, skip it!
         * we assume in this backend code that the operation has
         * already been checked to obey the subspaces
         */
        if ((__builtin_popcount(bra)&1) != s.right_space) continue;
        else braidx = bra & PARITY_MASK(L);
      }

      tmp_val = 0;
      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(bra & signs[i]) % 2);
        tmp_val += sign * coeffs[i];
        ++i;
      } while (i<nterms && masks[i-1] == masks[i]);

      ierr = MatSetValues(*A,1,&ketidx,1,&braidx,&tmp_val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "BuildMat_Shell"
PetscErrorCode BuildMat_Shell(PetscInt L,PetscInt nterms,
                              const PetscInt* masks,
                              const PetscInt* signs,
                              const PetscScalar* coeffs,
                              Subspaces s,
                              Mat *A)
{
  PetscErrorCode ierr;
  PetscInt N,M,n,m;
  shell_context *ctx;

  N = get_dimension(L,s.right_type,s.right_space);
  M = get_dimension(L,s.left_type,s.left_space);

  n = PETSC_DECIDE;
  m = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);
  PetscSplitOwnership(PETSC_COMM_WORLD,&m,&M);

  ierr = BuildContext(L,nterms,masks,signs,coeffs,s,&ctx);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,m,n,M,N,ctx,A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))MatMult_Shell);
  ierr = MatShellSetOperation(*A,MATOP_NORM,(void(*)(void))MatNorm_Shell);

  return ierr;

}

static inline void _radd(PetscScalar *x,PetscReal c)
{
  (*x) += c;
}

static inline void _cadd(PetscScalar *x,PetscReal c)
{
  (*x) += I*c;
}

#undef  __FUNCT__
#define __FUNCT__ "MatMult_Shell"
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b)
{
  PetscErrorCode ierr;
  PetscInt state,i,Istart,Iend,start,block_start,sign,m,mt,s,*l;
  PetscInt proc,proc_idx,proc_mask,proc_size,log2size,local_size;
  PetscBool assembling,r,do_xmul;
  const PetscScalar *x_array, *x_begin, *xp;
  PetscReal c,tmp_c;
  shell_context *ctx;

  /* cache */
  PetscInt *lidx;
  PetscInt cache_idx,lkp_idx,cache_idx_max,stop_count;
  PetscInt iterate_max, ms_parity;
  PetscScalar *summed_c,*cp,*values,*vp;

  PetscInt mpi_rank,mpi_size;
  MPI_Comm_rank(PETSC_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  ierr = PetscMalloc1(VECSET_CACHE_SIZE,&lidx);CHKERRQ(ierr);
  ierr = PetscMalloc1(VECSET_CACHE_SIZE,&values);CHKERRQ(ierr);
  ierr = PetscMalloc1(VECSET_CACHE_SIZE,&summed_c);CHKERRQ(ierr);

  /* clear out the b vector */
  ierr = VecSet(b,0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);

  /* first need to compute some constants etc */
  log2size = __builtin_ctz(mpi_size);

  /* this computes a mask over the indices that define the processor */
  if (ctx->s.left_type == FULL) {
    local_size = ctx->L - log2size;
  }
  else if (ctx->s.left_type == PARITY) {
    local_size = ctx->L - log2size - 1;
  }

  proc_mask = ((1 << log2size)-1) << local_size;
  proc_idx = mpi_rank << local_size;
  proc_size = Iend-Istart;
  x_begin = x_array - Istart;

  /* we are not already sending values to another processor */
  assembling = PETSC_FALSE;

  for (proc=0;proc < mpi_size;++proc) {

    /*
     * NOTE: these breaks and continues are dangerous if
     * the data doesn't divide perfectly across processors.
     * When switching to a subspace that's not a power of 2
     * need to be careful!
     */

    /* if there are none for this process, skip it */
    if (ctx->mask_starts[proc] == ctx->mask_starts[proc+1]) continue;

    /* if we've hit the end of the masks, stop */
    if (ctx->mask_starts[proc] == ctx->nterms) break;

    start = proc_mask & (proc_idx ^ ctx->masks[ctx->mask_starts[proc]]);
    for (block_start=start;
         block_start<start+proc_size;
         block_start+=VECSET_CACHE_SIZE) {

      cache_idx_max = intmin((start+proc_size)-block_start,VECSET_CACHE_SIZE);

      ierr = PetscMemzero(values,sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);
      ierr = PetscMemzero(summed_c,sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);

      for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
        lidx[cache_idx] = block_start + cache_idx;
      }

      for (i=ctx->mask_starts[proc];i<ctx->mask_starts[proc+1];++i) {

        m = ctx->masks[i];
        s = ctx->signs[i];
        ms_parity = __builtin_popcount(m&s)&1;
        c = -(ms_parity^(ms_parity-1))*ctx->coeffs[i];
        r = ctx->creal[i];

        l = ctx->lookup[s&LKP_MASK];

        /*
         * the lookup table and process distribution run into each other
         * for very small matrices, so just compute those in a boring way
         * NOTE that this means bugs in the optimized large-matrix code will
         * NOT show up when testing on small matrices!
         */
        if (cache_idx_max != VECSET_CACHE_SIZE || (block_start&LKP_MASK) != 0) {
          for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
            sign = __builtin_popcount(lidx[cache_idx]&s)&1;
            tmp_c = -(sign^(sign-1))*c;
            if (r) summed_c[cache_idx] += tmp_c;
            else summed_c[cache_idx] += I*tmp_c;
          }
          ierr = PetscInfo(A,"Using small-matrix shell code...");CHKERRQ(ierr);
        }
        /* otherwise we can really blaze. the loop limits are all #defined! */
        else {

/* this is the interior of the for loop. The compiler wasn't
 * doing a good enough job unswitching it so I write a macro
 * to unswitch it manually.
 */
/**********/
#define INNER_LOOP(sign_flip,add_func)                                          \
          for (cache_idx=0;cache_idx<VECSET_CACHE_SIZE;) {                      \
            sign = __builtin_popcount((cache_idx+block_start)&(~LKP_MASK)&s)&1; \
            tmp_c = -(sign^(sign-1))*c;                                         \
            for (lkp_idx=0;lkp_idx<LKP_SIZE;++lkp_idx,++cache_idx) {            \
              add_func(summed_c+cache_idx,(sign_flip)*tmp_c);                   \
            }                                                                   \
          }
/**********/

          /* if the sign mask cares about the leftmost bit and we're in a parity
           * conserving subspace, this will get the sign wrong sometimes. it takes
           * too long to fix it here, we just check later and flip the ones that
           * need flipping!
           */
          if (s&LKP_MASK) {
            if (r) {INNER_LOOP(l[lkp_idx],_radd)}
            else {INNER_LOOP(l[lkp_idx],_cadd)}
          }
          else {
            if (r) {INNER_LOOP(1,_radd)}
            else {INNER_LOOP(1,_cadd)}
          }
        }

        do_xmul = (i+1) == ctx->mask_starts[proc+1] || m != ctx->masks[i+1];

        /*
         * if we are finalizing this one, or if the next sign has a different value
         * for the parity bit, we should flip all the signs that are wrong.
         */
        if (ctx->s.left_type == PARITY) {
          /* make sure the signs are right */

          /*
           * this is honestly extremely confusing so it warrants a paragraph comment.
           *
           * in the case that the sign mask cares about the bit that we're ignoring when
           * we conserve parity, all of the ones in which that bit should be set to 1 will
           * be wrong. fortunately, it's the same ones all the time. So the plan is to just
           * let them be wrong, until we encounter a sign flip mask that doesn't care about that
           * bit. Before we change whether we care about it, or before we multiply by the x
           * vector, we should flip the signs of those terms. that's what the for loop in here does.
           */


          /* about to multiply and it's wrong  ||  not about to multiply, but next sign changes whether we care */
          if ((do_xmul && (s&PARITY_BIT(ctx->L))) || (!do_xmul && ((s^(ctx->signs[i+1]))&PARITY_BIT(ctx->L)) )) {
            for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
              sign = (__builtin_popcount(cache_idx+block_start)&1) ^ ctx->s.left_space;
              summed_c[cache_idx] *= -(sign^(sign-1));
            }
          }
        }

        /* need to finally multiply by x */
        if (do_xmul) {

          /* TODO: need to be careful here if going from full space to a subspace */
          if (ctx->s.left_type == PARITY) {
            mt = m & PARITY_MASK(ctx->L);
          }
          else mt = m;

          iterate_max = 1<<__builtin_ctz(mt);
          if (iterate_max < ITER_CUTOFF) {
            for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
              state = (block_start+cache_idx) ^ mt;
              values[cache_idx] += summed_c[cache_idx]*x_begin[state];
            }
          }
          else {
            for (cache_idx=0;cache_idx<cache_idx_max;++cache_idx) {
              state = (block_start+cache_idx) ^ mt;

              vp = values + cache_idx;
              xp = x_begin + state;
              cp = summed_c + cache_idx;
              (*vp) += (*cp)*(*xp);

              stop_count = intmin(iterate_max-(state%iterate_max),cache_idx_max-cache_idx);
              cache_idx += stop_count - 1;

              while (--stop_count) {
                ++xp;
                ++vp;
                ++cp;
                (*vp) += (*cp)*(*xp);
              }
            }
          }

          ierr = PetscMemzero(summed_c,sizeof(PetscScalar)*VECSET_CACHE_SIZE);CHKERRQ(ierr);
        }
      }

      /* set the last values that weren't set at the end there */
      if (assembling) {
        ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
        assembling = PETSC_FALSE;
      }
      ierr = VecSetValues(b,cache_idx_max,lidx,values,ADD_VALUES);CHKERRQ(ierr);

      ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
      assembling = PETSC_TRUE;
    }
  }

  if (assembling) {
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  ierr = PetscFree(lidx);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(summed_c);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "MatNorm_Shell"
PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm)
{
  PetscErrorCode ierr;
  PetscInt m,idx,state,i,Istart,Iend;
  PetscScalar csum,sign;
  PetscReal sum,local_max,global_max;
  shell_context *ctx;

  if (type != NORM_INFINITY) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Only NORM_INFINITY is implemented for shell matrices.");
  }

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  /*
    keep the norm cached so we don't have to compute it all the time.
    if we already have it, just return it
  */
  if (ctx->nrm != -1) {
    (*nrm) = ctx->nrm;
    return ierr;
  }

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  local_max = 0;
  for (idx=Istart;idx<Iend;++idx) {
    sum = 0;
    for (i=0;i<ctx->nterms;) {
      csum = 0;
      m = ctx->masks[i];
      if (ctx->s.right_type == PARITY) {
        state = PARITY_ItoS(idx,ctx->s.right_space,ctx->L);
      }
      else state = idx;

      /* sum all terms for this matrix element */
      do {
        /* this requires gcc builtins */
        sign = 1 - 2*(__builtin_popcount(state & ctx->signs[i]) % 2);
        if (!ctx->creal[i]) sign *= I;
        csum += sign * ctx->coeffs[i];
        ++i;
      } while (i<ctx->nterms && m == ctx->masks[i]);

      sum += PetscAbsComplex(csum);
    }
    if (sum > local_max) {
      local_max = sum;
    }
  }

  ierr = MPIU_Allreduce(&local_max,&global_max,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

  ctx->nrm = global_max;
  (*nrm) = global_max;

  return ierr;

}

#undef  __FUNCT__
#define __FUNCT__ "BuildContext"
PetscErrorCode BuildContext(PetscInt L,
                            PetscInt nterms,
                            const PetscInt* masks,
                            const PetscInt* signs,
                            const PetscScalar* coeffs,
                            Subspaces s,
                            shell_context **ctx_p)
{
  PetscErrorCode ierr;
  shell_context *ctx;
  PetscInt i,j,mpi_size,proc_size,N,m,tmp;

  ierr = PetscMalloc1(1,ctx_p);CHKERRQ(ierr);
  ctx = (*ctx_p);

  ctx->L = L;
  ctx->nterms = nterms;
  ctx->nrm = -1;
  ctx->s = s;
  ctx->gpu = PETSC_FALSE;

  /* we need to keep track of this stuff on our own. the numpy array might get garbage collected */
  ierr = PetscMalloc1(nterms,&(ctx->masks));CHKERRQ(ierr);
  ierr = PetscMalloc1(nterms,&(ctx->signs));CHKERRQ(ierr);
  ierr = PetscMalloc1(nterms,&(ctx->coeffs));CHKERRQ(ierr);
  ierr = PetscMalloc1(nterms,&(ctx->creal));CHKERRQ(ierr);

  for (i=0;i<nterms;++i) {
    ctx->masks[i] = masks[i];
    ctx->signs[i] = signs[i];

    ctx->coeffs[i] = PetscRealPart(coeffs[i]);
    ctx->creal[i] = (ctx->coeffs[i] != 0);
    if (!ctx->creal[i]) ctx->coeffs[i] = PetscImaginaryPart(coeffs[i]);
  }

  /* compute the sign lookup tables */
  ierr = PetscMalloc1(LKP_SIZE,&(ctx->lookup));CHKERRQ(ierr);
  for (i=0;i<LKP_SIZE;++i) {
    ierr = PetscMalloc1(LKP_SIZE,ctx->lookup+i);CHKERRQ(ierr);
    for (j=0;j<LKP_SIZE;++j) {
      tmp = __builtin_popcount(i&j)&1;
      ctx->lookup[i][j] = -(tmp^(tmp-1));
    }
  }

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpi_size);

  /* count the number of masks going to each processor */
  ierr = PetscMalloc1(mpi_size+1,&(ctx->mask_starts));CHKERRQ(ierr);

  proc_size = PETSC_DECIDE;
  N = get_dimension(L,s.left_type,s.left_space);
  ierr = PetscSplitOwnership(PETSC_COMM_WORLD,&proc_size,&N);CHKERRQ(ierr);

  ctx->mask_starts[0] = 0;
  m = proc_size;
  if (s.left_type == PARITY) {
    for (i=1;i<ctx->nterms;++i) {
      for (;m <= PARITY_ItoS(ctx->masks[i],s.left_space,L);m += proc_size) {
        ctx->mask_starts[m / proc_size] = i;
      }
    }
  }
  else {
    for (i=1;i<ctx->nterms;++i) {
      for (;m<=ctx->masks[i];m += proc_size) {
        ctx->mask_starts[m / proc_size] = i;
      }
    }
  }
  /* maybe a few of the last ones had none, have
   * to iterate through those
   */
  for (i=m/proc_size;i<mpi_size+1;++i) {
    ctx->mask_starts[i] = ctx->nterms;
  }

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "DestroyContext"
PetscErrorCode DestroyContext(Mat A)
{
  PetscErrorCode ierr;
  PetscInt i;
  shell_context *ctx;

  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);

  ierr = PetscFree(ctx->masks);CHKERRQ(ierr);
  ierr = PetscFree(ctx->signs);CHKERRQ(ierr);
  ierr = PetscFree(ctx->coeffs);CHKERRQ(ierr);

  for (i=0;i<LKP_SIZE;++i) {
    ierr = PetscFree(ctx->lookup[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->lookup);CHKERRQ(ierr);

  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return ierr;
}

#undef  __FUNCT__
#define __FUNCT__ "ReducedDensityMatrix"
PetscErrorCode ReducedDensityMatrix(PetscInt L,
                                    Vec x,
                                    PetscInt cut_size,
                                    PetscBool fillall,
                                    PetscScalar* m)
{
  const PetscScalar *x_array;
  PetscInt i,j,k,jmax;
  PetscInt cut_N,tr_N,tr_size;
  PetscScalar a,b;
  PetscErrorCode ierr;

  /* compute sizes of things */
  tr_size = L - cut_size; /* in case L is odd */
  cut_N = 1 << cut_size;
  tr_N = 1 << tr_size;

  /*
    Fill the reduced density matrix!

    We only need to fill the lower triangle for the
    purposes of eigensolving for entanglement entropy.
    This behavior is controlled by "fillall".
  */
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);

  for (i=0;i<cut_N;++i) {

    if (fillall) jmax = cut_N;
    else jmax = i+1;

    for (j=0;j<jmax;++j) {

      for (k=0;k<tr_N;++k) {
        a = x_array[(i<<tr_size) + k];
        b = x_array[(j<<tr_size) + k];

        m[i*cut_N + j] += a*PetscConj(b);
      }
    }
  }

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);

  return ierr;

}
