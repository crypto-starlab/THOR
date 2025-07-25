#ifndef NTT_CUDA_KERNELS_H
#define NTT_CUDA_KERNELS_H

template<typename scalar_t> __device__ __forceinline__ scalar_t
mont_mult_scalar(
    const scalar_t a, const scalar_t b,
    const scalar_t ql, const scalar_t qh,
    const scalar_t kl, const scalar_t kh) {

    // Masks.
    constexpr scalar_t one = 1;
    constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
    constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
    constexpr scalar_t fb_mask = ((one << nbits) - one);
    constexpr scalar_t lb_mask = (one << half_nbits) - one;

    const scalar_t al = a & lb_mask;
    const scalar_t ah = a >> half_nbits;
    const scalar_t bl = b & lb_mask;
    const scalar_t bh = b >> half_nbits;

    const scalar_t alpha = ah * bh;
    const scalar_t beta = ah * bl + al * bh;
    const scalar_t gamma = al * bl;

    // s = xk mod R
    const scalar_t gammal = gamma & lb_mask;
    const scalar_t gammah = gamma >> half_nbits;
    const scalar_t betal = beta & lb_mask;
    const scalar_t betah = beta >> half_nbits;

    scalar_t upper = gammal * kh;
    upper = upper + (gammah + betal) * kl;
    upper = upper << half_nbits;
    scalar_t s = upper + gammal * kl;
    s = upper + gammal * kl;
    s = s & fb_mask;

    // t = x + sq
    // u = t/R
    const scalar_t sl = s & lb_mask;
    const scalar_t sh = s >> half_nbits;
    const scalar_t sqb = sh * ql + sl * qh;
    const scalar_t sqbl = sqb & lb_mask;
    const scalar_t sqbh = sqb >> half_nbits;

    scalar_t carry = (gamma + sl * ql) >> half_nbits;
    carry = (carry + betal + sqbl) >> half_nbits;

    return alpha + betah + sqbh + carry + sh * qh;
}

template<typename scalar_t> __device__ __forceinline__ scalar_t
mont_redc_scalar(
    const scalar_t x,
    const scalar_t ql, const scalar_t qh,
    const scalar_t kl, const scalar_t kh) {

    // Masks.
    constexpr scalar_t one = 1;
    constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
    constexpr scalar_t half_nbits =  sizeof(scalar_t) * 4 - 1;
    constexpr scalar_t fb_mask = ((one << nbits) - one);
    constexpr scalar_t lb_mask = (one << half_nbits) - one;

    // Implementation.
    // s= xk mod R
    const scalar_t xl = x & lb_mask;
    const scalar_t xh = x >> half_nbits;
    const scalar_t xkb = xh * kl + xl * kh;
    scalar_t s = (xkb << half_nbits) + xl * kl;
    s = s & fb_mask;

    // t = x + sq
    // u = t/R
    // Note that x gets erased in t/R operation if x < R.
    const scalar_t sl = s & lb_mask;
    const scalar_t sh = s >> half_nbits;
    const scalar_t sqb = sh * ql + sl * qh;
    const scalar_t sqbl = sqb & lb_mask;
    const scalar_t sqbh = sqb >> half_nbits;
    scalar_t carry = (x + sl * ql) >> half_nbits;
    carry = (carry + sqbl) >> half_nbits;

    // Assume we have satisfied the condition 4*q < R.
    // Return the calculated value directly without conditional subtraction.
    return sqbh + carry + sh * qh;
}


template<typename scalar_t>
__global__ void ntt_cuda_kernel(
    scalar_t* a,
    int N,
    const torch::PackedTensorAccessor32<int, 2>even_acc,
    const torch::PackedTensorAccessor32<int, 2>odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3>psi_acc,
    const scalar_t* _2q_ptr,
    const scalar_t* ql_ptr,
    const scalar_t* qh_ptr,
    const scalar_t* kl_ptr,
    const scalar_t* kh_ptr,
    const int level
){

    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // Montgomery inputs.
    const scalar_t _2q = _2q_ptr[i];
    const scalar_t ql = ql_ptr[i];
    const scalar_t qh = qh_ptr[i];
    const scalar_t kl = kl_ptr[i];
    const scalar_t kh = kh_ptr[i];

    // Butterfly.
    const int even_j = even_acc[level][j];
    const int odd_j = odd_acc[level][j];

    // const scalar_t U = a_acc[i][even_j];
    const int even_ind = i*N + even_j;
    const scalar_t U = a[even_ind];

    const scalar_t S = psi_acc[i][level][j];

    // const scalar_t O = a_acc[i][odd_j];
    const int odd_ind = i*N + odd_j;
    const scalar_t O = a[odd_ind];

    const scalar_t V = mont_mult_scalar(S, O, ql, qh, kl, kh);

    // Store back.
    const scalar_t UplusV = U + V;
    const scalar_t UminusV = U + _2q - V;

    a[even_ind] = (UplusV < _2q)? UplusV : UplusV - _2q;
    a[odd_ind] = (UminusV < _2q)? UminusV : UminusV - _2q;
}


template<typename scalar_t>
__global__ void intt_cuda_kernel(
    scalar_t* a,
    int N,
    const torch::PackedTensorAccessor32<int, 2>even_acc,
    const torch::PackedTensorAccessor32<int, 2>odd_acc,
    const torch::PackedTensorAccessor32<scalar_t, 3>psi_acc,
    const scalar_t* _2q_ptr,
    const scalar_t* ql_ptr,
    const scalar_t* qh_ptr,
    const scalar_t* kl_ptr,
    const scalar_t* kh_ptr,
    const int level
){

    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // Montgomery inputs.
    const scalar_t _2q = _2q_ptr[i];
    const scalar_t ql = ql_ptr[i];
    const scalar_t qh = qh_ptr[i];
    const scalar_t kl = kl_ptr[i];
    const scalar_t kh = kh_ptr[i];

    // Butterfly.
    const int even_j = even_acc[level][j];
    const int odd_j = odd_acc[level][j];

    // Indexing.
    const int even_ind = i*N + even_j;
    const int odd_ind = i*N + odd_j;

    // const scalar_t U = a_acc[i][even_j];
    const scalar_t U = a[even_ind];

    const scalar_t S = psi_acc[i][level][j];

    // const scalar_t V = a_acc[i][odd_j];
    const scalar_t V = a[odd_ind];

    const scalar_t UminusV = U + _2q - V;
    const scalar_t O = (UminusV < _2q)? UminusV : UminusV - _2q;

    const scalar_t W = mont_mult_scalar(S, O, ql, qh, kl, kh);
    a[odd_ind] = W;

    const scalar_t UplusV = U + V;
    a[even_ind] = (UplusV < _2q)? UplusV : UplusV - _2q;
}



template<typename scalar_t>
__global__ void normalize_exit_reduce_cuda_kernel
(
    scalar_t* a_ptr,
    int N,
    scalar_t* Rs_ptr,
    scalar_t* _2q_ptr,
    scalar_t* ql_ptr,
    scalar_t* qh_ptr,
    scalar_t* kl_ptr,
    scalar_t* kh_ptr)
{

    // Where am I?
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // 1D index.
    const int ind = i*N + j;

    // Inputs.
    const scalar_t a = a_ptr[ind];
    const scalar_t Rs = Rs_ptr[i];
    const scalar_t _2q = _2q_ptr[i];
    const scalar_t ql = ql_ptr[i];
    const scalar_t qh = qh_ptr[i];
    const scalar_t kl = kl_ptr[i];
    const scalar_t kh = kh_ptr[i];

    // Normalize.
    scalar_t normalized = mont_mult_scalar(a, Rs, ql, qh, kl, kh);

    // Exit.
    normalized = mont_redc_scalar(normalized, ql, qh, kl, kh);

    // Reduce and store.
    const scalar_t _q = _2q >> 1;
    a_ptr[ind] = (normalized < _q)? normalized : normalized - _q;
}

#endif /* NTT_CUDA_KERNELS_H */
