
fndef = lambda expr: f"""

__device__ __forceinline__ float combinator(float r1, float c1){{

    return {expr};

}}


"""


kernel_template = f"""


// =============================================================================
#define fetch(A, m, n, bound) offs_d##A[min(static_cast<int>(n * LD##A + m), static_cast<int>(bound))]

#define add(A, B) (A + B)
#define mul(A, B) (A * B)
#define div(A, B) (A / B)
#define fma(A, B, C) C += (A * B)
#define make_FloatingPoint(x, y) (x)

#define ceildiv(A, B) (((A + B - 1) / (B)))
#define sA(i, j) sA[(j) * slda + (i)]
#define sB(i, j) sB[(j) * sldb + (i)]
#define sC(i, j) sC[(j) * sldc + (i)]

template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
          const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static __device__ void gemm_template_device_prefetch_tn(
    int M, int N, int K,
    const T *__restrict__ A, int LDA,
    const T *__restrict__ B, int LDB,
    T *__restrict__ C, int LDC,
    T alpha, T beta,
    T *sA, int slda,
    T *sB, int sldb,
    T *sC, int sldc)
{{
    int idx = threadIdx.x; // thread's m dimension
    int idy = threadIdx.y; // thread's n dimension

    int idt = DIM_X * idy + idx; // thread's global number

    int idxA = idt % DIM_XA; // idx within A
    int idyA = idt / DIM_XA; // idy within A

    int idxB = idt % DIM_XB; // idx within B
    int idyB = idt / DIM_XB; // idy within B

    int blx = blockIdx.x; // block's m dimension
    int bly = blockIdx.y; // block's n dimension

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    // Registers for the dev->shmem copy
    T ra[BLK_M / DIM_YA][BLK_K / DIM_XA];
    T rb[BLK_N / DIM_YB][BLK_K / DIM_XB];

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dA = A + blx * BLK_M * LDA + idyA * LDA + idxA;
    ptrdiff_t boundA = (LDA * (M - 1) + K) - (blx * BLK_M * LDA + idyA * LDA + idxA) - 1;

    const T *offs_dB = B + bly * BLK_N * LDB + idyB * LDB + idxB;
    ptrdiff_t boundB = (LDB * (N - 1) + K) - (bly * BLK_N * LDB + idyB * LDB + idxB) - 1;

    int m, n, k, kk;

// Zero C
#pragma unroll
    for (n = 0; n < THR_N; n++)
#pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

    if (K > 0)
    {{
// Load A dev->shmem
#pragma unroll
        for (n = 0; n < BLK_M; n += DIM_YA)
#pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XA)
                sA(n + idyA, m + idxA) = fetch(A, m, n, boundA);

// Load B dev->shmem
#pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
#pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB(m + idxB, n + idyB) = fetch(B, m, n, boundB);
    }}
    __syncthreads();

    for (kk = 0; kk < K - BLK_K; kk += BLK_K)
    {{
        offs_dA += BLK_K;
        boundA -= BLK_K;

        offs_dB += BLK_K;
        boundB -= BLK_K;

// Load A dev->regs
#pragma unroll
        for (n = 0; n < BLK_M / DIM_YA; n++)
#pragma unroll
            for (m = 0; m < BLK_K / DIM_XA; m++)
                ra[n][m] = fetch(A, m * DIM_XA, n * DIM_YA, boundA);

// Load B dev->regs
#pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++)
#pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++)
                rb[n][m] = fetch(B, m * DIM_XB, n * DIM_YB, boundB);

// Multiply
#pragma unroll
        for (k = 0; k < BLK_K; k++)
        {{
// Load A shmem->regs
#pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA(m * DIM_X + idx, k);

// Load B shmem->regs
#pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB(k, n * DIM_Y + idy);

// Compute
#pragma unroll
            for (n = 0; n < THR_N; n++)
            {{
#pragma unroll
                for (m = 0; m < THR_M; m++)
                {{
                    rC[n][m] += combinator(rA[m], rB[n]);
                }}
            }}
        }}

        __syncthreads();

// Load A regs->shmem
#pragma unroll
        for (n = 0; n < BLK_M / DIM_YA; n++)
#pragma unroll
            for (m = 0; m < BLK_K / DIM_XA; m++)
                sA(n * DIM_YA + idyA, m * DIM_XA + idxA) = ra[n][m];

// Load B regs->shmem
#pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++)
#pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++)
                sB(m * DIM_XB + idxB, n * DIM_YB + idyB) = rb[n][m];

        __syncthreads();
    }}

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
#pragma unroll
    for (k = 0; k < kk; k++)
    {{
// Load A shmem->regs
#pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA(m * DIM_X + idx, k);

// Load B shmem->regs
#pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB(k, n * DIM_Y + idy);

// Compute
#pragma unroll
        for (n = 0; n < THR_N; n++)
        {{
#pragma unroll
            for (m = 0; m < THR_M; m++)
            {{
                rC[n][m] += combinator(rA[m], rB[n]);
            }}
        }}
    }}

// Store C regs->dev
#pragma unroll
    for (n = 0; n < THR_N; n++)
    {{
        int coord_dCn = bly * BLK_N + n * DIM_Y + idy;
#pragma unroll
        for (m = 0; m < THR_M; m++)
        {{
            int coord_dCm = blx * BLK_M + m * DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N)
            {{
                ptrdiff_t offsC = (ptrdiff_t)coord_dCn * (ptrdiff_t)LDC + (ptrdiff_t)coord_dCm;

                T &regC = rC[n][m];
                T &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }}
        }}
    }}
}}
// #############################################################
// ######################### kernel ############################
// #############################################################

template <typename T, const int DIM_X, const int DIM_Y,
          const int BLK_M, const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
          const int CONJA, const int CONJB>
static __global__ void gemm_template_tn_kernel(
    int M, int N, int K,
    T const *A, int LDA,
    T const *B, int LDB,
    T *C, int LDC,
    T alpha, T beta)
{{
    extern __shared__ T *sdata_tn[];

    const int slda = BLK_M + 1; // +1 only required if A is transposed
    const int sldb = BLK_K + 1; // +1 always required
    T *sA = (T *)sdata_tn;      // sA is (BLK_M+1) x (BLK_K)
    T *sB = sA + slda * BLK_K;  // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_prefetch_tn<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M / DIM_X), (BLK_N / DIM_Y), CONJA, CONJB>(M, N, K, A, LDA, B, LDB, C, LDC, alpha, beta, sA, slda, sB, sldb, NULL, 0);
}}

// #############################################################
// ######################### caller ############################
// #############################################################

// TN, CN
template <typename T, const int DIM_X, const int DIM_Y,
          const int BLK_M, const int BLK_N, const int BLK_K,
          const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
          const int CONJA, const int CONJB>
void gemm_template_tn(
    int m, int n, int k,
    T const *dA, int ldda,
    T const *dB, int lddb,
    T *dC, int lddc,
    T alpha, T beta)
{{
    size_t shmem = 0;
    shmem += (BLK_M + 1) * BLK_K * sizeof(T); // sA
    shmem += (BLK_K + 1) * BLK_N * sizeof(T); // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid(ceildiv(m, BLK_M), ceildiv(n, BLK_N), 1);
    gemm_template_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<<dimGrid, dimBlock, shmem>>>(m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}}

#define DIM_X_tn 16
#define DIM_Y_tn 16

#define BLK_M_tn 96
#define BLK_N_tn 96
#define BLK_K_tn 16
#define DIM_XA_tn 16
#define DIM_YA_tn 16
#define DIM_XB_tn 16
#define DIM_YB_tn 16

void gemm_caller_tn(int m, int n, int k,
                    float const *dA, int ldda,
                    float const *dB, int lddb,
                    float *dC, int lddc,
                    float alpha, float beta)
{{
    gemm_template_tn<float, DIM_X_tn, DIM_Y_tn,
                     BLK_M_tn, BLK_N_tn, BLK_K_tn,
                     DIM_XA_tn, DIM_YA_tn, DIM_XB_tn, DIM_YB_tn,
                     0, 0>(m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta);
}}



"""


def generate_cuda(expr):
    return fndef(expr) + kernel_template