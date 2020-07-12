#include <torch/extension.h>

#include <vector>


#define TILE_M 64
#define TILE_K 32
#define TILE_N 64
#define PT_M 16
#define PT_N 2
#define TP_M (TILE_M / PT_M) 
#define TP_N (TILE_N / PT_N) 
#define BSZ 128
#if (TILE_M % PT_M != 0) || (TILE_N % PT_N != 0) || ((TP_M * TP_N) != BSZ)
#error "Invalid distribution size: Num threads" 
#endif

#define WARP_CNT (BSZ/32)
#define DIV8_CNT (BSZ/8)
#define GLOAD4
#define SAFE_LOAD(MAT, SY, SX, y, x) (((x) < (SX)) ? (((y) < (SY)) ? ((MAT)[(y) * (SX) + (x)]) : 0) : 0)
#define SAFE_LOAD_TRANS(MAT, SX, SY, x, y) (((x) < (SX)) ? (((y) < (SY)) ? ((MAT)[(y) * (SX) + (x)]) : 0) : 0)
#define SAFE_STORE(MAT, SY, SX, y, x, v) if(((x) < (SX)) && ((y) < (SY))) ((MAT)[(y) * (SX) + (x)] = (v));
#define SAFE_STORE_TRANS(MAT, SX, SY, x, y, v) if(((x) < (SX)) && ((y) < (SY))) ((MAT)[(y) * (SX) + (x)] = (v));
#define SAFE_LOAD4(MAT, SY, SX, y, x) ((((x)) < (SX)) ? (((y) < (SY)) ? (((float4 *)(MAT))[((y) * (SX) + (x))/4]) : zero) : zero)
#define PREFETCH

//Highly-optimzed gemm kernel
extern "C" __global__ void X_indep_gemm_opt(
    const float* __restrict__ X_A, // X instance of MxK mat 
    const float* __restrict__ X_B, // X instance of KxN mat
    float *X_C,
    int X, int M, int N, int K) // X instance of MxN mat
{
  //For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (blockIdx.x);
  int lid = (threadIdx.x); // 0 <= lid < BSZ
  int div8q = lid / 8;
  int div8r = lid % 8;
  int m, n, k;

  //How many output pixels per therad?
  //PT_M x PT_N
  float reg_C[PT_M][PT_N]={0.0f};
  //per block?
  //TILE_M x TILE_N
  //How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;
  
  //How many blocks per instance?
  //Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;
  
  if (xid > X) return;
  
  //TILE_M x TILE_K x TILE_N
  __shared__ float local_A[TILE_M * TILE_K];
  __shared__ float local_B[TILE_K * TILE_N];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  //N / TILE_N
  const float * A = X_A + xid * M * K; 
  const float * B = X_B + xid * K * N; 
  float * C = X_C + xid * M * N; 
  float4 zero = {0.0, 0.0, 0.0, 0.0};

  float4 prefetch_A[TILE_M / DIV8_CNT][TILE_K/32];
  float4 prefetch_B[TILE_K / DIV8_CNT][TILE_N/32];
  #pragma unroll
  for (m = 0; m < TILE_M; m += DIV8_CNT) {
  #pragma unroll
    for (k = 0; k < TILE_K; k += 32) {
      prefetch_A[m / DIV8_CNT][k / 32]  = SAFE_LOAD4(A, M, K, (toff_m + m + div8q), (toff_k + k + div8r * 4)); 
    }                 
  }           
  #pragma unroll
  for (k = 0; k < TILE_K; k += DIV8_CNT) {
  #pragma unroll
    for (n = 0; n < TILE_N; n += 32) {
      prefetch_B[k / DIV8_CNT][n / 32] = SAFE_LOAD4(B, K, N, (toff_k + k + div8q), (toff_n + n + div8r * 4));
    }                 
  } 

  __syncthreads();
  for (toff_k = 0; toff_k < K; toff_k += TILE_K) { 
      {
      #pragma unroll
      for (m = 0; m < TILE_M; m += DIV8_CNT) {
      #pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          ((float4 *)local_A)[((m + div8q) * TILE_K + k + div8r* 4)/ 4] = prefetch_A[m / DIV8_CNT][k / 32]; 
        }                 
      }           

      #pragma unroll
      for (k = 0; k < TILE_K; k += DIV8_CNT) {
      #pragma unroll
        for (n = 0; n < TILE_N; n += 32) {
          ((float4 *)(local_B))[((k + div8q) * TILE_N + n + div8r * 4)/4] = prefetch_B[k / DIV8_CNT][n / 32];
        }                 
      } 
    }
    __syncthreads();
    if(toff_k + TILE_K * 2 - 1 < K)
    {
#pragma unroll
      for (m = 0; m < TILE_M; m += DIV8_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          prefetch_A[m / DIV8_CNT][k / 32]  = SAFE_LOAD4(A, M, K, (toff_m + m + div8q), (toff_k + TILE_K + k + div8r*4)); 
        }                 
      }           
#pragma unroll
      for (k = 0; k < TILE_K; k += DIV8_CNT) {
#pragma unroll
        for (n = 0; n < TILE_N; n += 32) {
          prefetch_B[k / DIV8_CNT][n / 32] = SAFE_LOAD4(B, K, N, (toff_k + TILE_K + k + div8q), (toff_n + n + div8r*4));
        }                 
      } 
    }

    // Block multiplication
    #pragma unroll
    for (m = 0; m < PT_M; m ++) {
    #pragma unroll
      for (n = 0; n < PT_N; n++) { 
      #pragma unroll
        for (k = 0; k < TILE_K; k++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] * local_B[k * TILE_N + n * TP_N + nid];
        }                       
      }                 
    }           
    __syncthreads();
  }     

  #pragma unroll
  for (m = 0; m < PT_M; m ++) { 
    #pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid), reg_C[m][n]);
    }           
  }  
}

//Highly-optimzed gemm kernel
extern "C" __global__ void X_indep_gemm_opt_nt(
    const float* __restrict__ X_A, // X instance of MxK mat 
    const float* __restrict__ X_B, // X instance of NxK mat
    float *X_C,
    int X, int M, int N, int K) // X instance of MxN mat
{
  //For simplicity, assume 1D-block
  int NB_M = ((M + TILE_M - 1) / TILE_M);
  int NB_N = ((N + TILE_N - 1) / TILE_N);
  int NB_P_INST = (NB_M * NB_N);
  int bid = (blockIdx.x);
  int lid = (threadIdx.x); // 0 <= lid < BSZ
  int div8q = lid / 8;
  int div8r = lid % 8;
  int m, n, k;

  //How many output pixels per therad?
  //PT_M x PT_N
  float reg_C[PT_M][PT_N]={0.0f};
  //per block?
  //TILE_M x TILE_N
  //How many Threads per instance?
  //(TILE_M / PT_M) / (TILE_N / PT_N)

  int nid = lid % TP_N;
  int mid = lid / TP_N;
  
  //How many blocks per instance?
  //Different X instance per block
  int xid = bid / NB_P_INST;
  int nbid = bid % NB_P_INST;
  
  if (xid > X) return;
  
  //TILE_M x TILE_K x TILE_N
  __shared__ float local_A[TILE_M * TILE_K];
  __shared__ float local_B[TILE_N * TILE_K];
  int toff_k = 0;
  int toff_m = (nbid / NB_N) * TILE_M;
  int toff_n = (nbid % NB_N) * TILE_N;
  //N / TILE_N
  const float * A = X_A + xid * M * K; 
  const float * B = X_B + xid * N * K; 
  float * C = X_C + xid * M * N; 
  float4 zero = {0.0, 0.0, 0.0, 0.0};

  float4 prefetch_A[TILE_M / DIV8_CNT][TILE_K/32];
  float4 prefetch_B[TILE_N / DIV8_CNT][TILE_K/32];
  #pragma unroll
  for (m = 0; m < TILE_M; m += DIV8_CNT) {
  #pragma unroll
    for (k = 0; k < TILE_K; k += 32) {
      prefetch_A[m / DIV8_CNT][k / 32] = SAFE_LOAD4(A, M, K, (toff_m + m + div8q), (toff_k + k + div8r * 4)); 
    }                 
  }           
  #pragma unroll
  for (n = 0; n < TILE_N; n += DIV8_CNT) {
  #pragma unroll
    for (k = 0; k < TILE_K; k += 32) {
      prefetch_B[n / DIV8_CNT][k / 32] = SAFE_LOAD4(B, N, K, (toff_n + n + div8q), (toff_k + k + div8r * 4)); 
    }                 
  }
  __syncthreads();
  for (toff_k = 0; toff_k < K; toff_k += TILE_K) { 
    {
      #pragma unroll
      for (m = 0; m < TILE_M; m += DIV8_CNT) {
      #pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          ((float4 *)local_A)[((m + div8q) * TILE_K + k + div8r * 4)/4] = prefetch_A[m / DIV8_CNT][k / 32]; 
        }                 
      }           

      #pragma unroll
      for (n = 0; n < TILE_N; n += DIV8_CNT) {
      #pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          ((float4 *)local_B)[((n + div8q) * TILE_K + k + div8r * 4)/4] = prefetch_B[n / DIV8_CNT][k / 32]; 
        }                 
      }               
    }
    __syncthreads();
    if(toff_k + TILE_K * 2 - 1 < K)
    {
#pragma unroll
      for (m = 0; m < TILE_M; m += DIV8_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          prefetch_A[m / DIV8_CNT][k / 32]  = SAFE_LOAD4(A, M, K, (toff_m + m + div8q), (toff_k + TILE_K + k + div8r*4)); 
        }                 
      }           
#pragma unroll
      for (n = 0; n < TILE_N; n += DIV8_CNT) {
#pragma unroll
        for (k = 0; k < TILE_K; k += 32) {
          prefetch_B[n / DIV8_CNT][k / 32]  = SAFE_LOAD4(B, N, K, (toff_n + n + div8q), (toff_k + TILE_K + k + div8r*4)); 
        }                 
      }    
    }

    // Block multiplication
    #pragma unroll
    for (m = 0; m < PT_M; m ++) {
    #pragma unroll
      for (n = 0; n < PT_N; n++) { 
      #pragma unroll
        for (k = 0; k < TILE_K; k++) {
          reg_C[m][n] += local_A[(m * TP_M + mid) * TILE_K + k] * local_B[(n * TP_N + nid) * TILE_K + k];
        }                       
      }                 
    }           
    __syncthreads();
  }     

  #pragma unroll
  for (m = 0; m < PT_M; m ++) { 
    #pragma unroll
    for (n = 0; n < PT_N; n++) {
      SAFE_STORE(C, M, N, (toff_m + m * TP_M + mid), (toff_n + n * TP_N + nid), reg_C[m][n]);
    }           
  }  
}

__global__ void baseline_winograd4x4_gemm_k(
    float *input_tile,
    float *filter_tile,
    float *output_tile,
    int N, int C, int K, int H, int W)
{
  int P = (H + 3)/4;
  int Q = (W + 3)/4;
  int gi = (blockIdx.y * blockDim.y + threadIdx.y);
  int i = gi % K;
  int ti = gi / K;
  int j = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= K || ti >= 36 || j >= N * P * Q) return;
  float * matA = filter_tile + ti * K * C; // K * C Matrix
  float * matB = input_tile + ti * C * N * P * Q; // C * NPQ Matrix
  float * matC = output_tile + ti * K * N * P * Q; // K * NPQ Matrix
  float sum = 0;
  for(int c = 0; c < C; c++)
  {
    sum += matA[i * C + c] * matB[c * (N*P*Q) + j];
  }
  matC[i * N * P * Q + j] = sum;
}
__global__ void baseline_winograd_wgrad_gemm_k(
    float *input_tile,
    float *output_tile,
    float *filter_tile,
    int N, int C, int K, int H, int W, int P, int Q, int TS)
{
  int gi = (blockIdx.y * blockDim.y + threadIdx.y);
  int i = gi % K;
  int ti = gi / K;
  int j = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= K || ti >= TS || j >= C) return;
  float * matA = input_tile + ti * C * N * P * Q; // C * NPQ Matrix
  float * matB = output_tile + ti * K * N * P * Q; // K * NPQ Matrix
  float * matC = filter_tile + ti * K * C; // K * C Matrix
  float sum = 0;
  for(int x = 0; x < N * P * Q; x++)
  {
    sum += matA[j * N * P * Q + x] * matB[i * N * P * Q + x];
  }
  matC[i * C + j] = sum;
}

__global__ void baseline_winograd4x4_gemm_shm_k(
    float *input_tile,
    float *filter_tile,
    float *output_tile,
    int N, int C, int K, int H, int W)
{
  int P = (H + 3)/4;
  int Q = (W + 3)/4;
  int gi = (blockIdx.y * blockDim.y * 4 + threadIdx.y);
  int i = gi % K;
  int ti = gi / K;
  int j = (blockIdx.x * blockDim.x * 4 + threadIdx.x);
  int k, l, m, n;
  if (i + 48 >= K || ti >= 36 || j + 48 >= N * P * Q) return;

  __shared__ float local_A[64][64];
  __shared__ float local_B[64][64];

  float * matA = filter_tile + ti * K * C; // K * C Matrix
  float * matB = input_tile + ti * C * N * P * Q; // C * NPQ Matrix
  float * matC = output_tile + ti * K * N * P * Q; // K * NPQ Matrix
  float ans[4][4] = {0.0f};
  for (k = 0; k < C; k += 64) { 
    // Fetch data into shard memory
    #pragma unroll
    for (m = 0; m < 64; m += 16) {
    #pragma unroll
      for (n = 0; n < 64; n += 16) {
        local_A[threadIdx.y + m][threadIdx.x + n]  = matA[(m + i) * C + k + n + threadIdx.x]; 
      }                 
    }           

    #pragma unroll
    for (m = 0; m < 64; m += 16) {
    #pragma unroll
      for (n = 0; n < 64; n += 16) {
        local_B[threadIdx.y + m][threadIdx.x + n]  = matB[(k + m + threadIdx.y) * N * P * Q + n + j];
      }                 
    }           
    __syncthreads();

    // Block multiplication
    #pragma unroll
    for (m = 0; m < 4; m ++) {
    #pragma unroll
      for (n = 0; n < 4; n ++) { 
      #pragma unroll
        for (l = 0; l < 64; l++) {
          ans[m][n] += local_A[threadIdx.y + m*16][l]  * local_B[l][threadIdx.x + n*16];
        }                       
      }                 
    }           
    __syncthreads();
  }     

    #pragma unroll
  for (m = 0; m < 4; m ++) { 
    #pragma unroll
    for (n = 0; n < 4; n ++) {
      matC[(i + m*16) * N * P * Q + n * 16 + j] = ans[m][n];
    }           
  }  
}
__global__ void baseline_winograd4x4_data_tile_k( float * inputs, float * outputs, int N, int C, int H, int W, int cy)
  {
    int PH = H;
    int PW = W;
    int TP = (H + 3) / 4;
    int TQ = (W + 3) / 4;
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int nctptq = gid;
    int n = nctptq / (C * TP * TQ);
    if (n >= N) return;
    int ctptq = nctptq - n * (C * TP * TQ);
    int c = ctptq / (TP * TQ);
    int tptq = ctptq - c * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int h = tp * 4 - 1, w = tq * 4 - 1;
    float v[6][6], TV[6][6], V[6][6];

    inputs += ((n * C + c) * PH + h) * PW + w;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if(!cy){
              v[i][j] = 0 <= h + i && h + i < PH && 0 <= w + j && w + j < PW ? inputs[i * PW + j] : 0;
            }
            else {
              if(v[i][j] = 0 <= h + i && h + i < PH && 0 <= w + j && w + j < PW) v[i][j] = inputs[i * PW + j];
              else if(w + j < 0 || w + j >= PW) v[i][j] = 0;
              else if(h + i == -1) v[i][j] = inputs[(i + PH) * PW + j];
              else if(h + i == PH) v[i][j] = inputs[(i - PH) * PW + j];
            }
        }
    }

    #pragma unroll 
    for(int i = 0; i < 6; i++)
    {
      TV[0][i] =  4 * v[0][i] - 5 * v[2][i] + 1 * v[4][i];
      TV[1][i] = -4 * v[1][i] - 4 * v[2][i] + 1 * v[3][i] + 1 * v[4][i];
      TV[2][i] =  4 * v[1][i] - 4 * v[2][i] - 1 * v[3][i] + 1 * v[4][i];
      TV[3][i] = -2 * v[1][i] - 1 * v[2][i] + 2 * v[3][i] + 1 * v[4][i];
      TV[4][i] =  2 * v[1][i] - 1 * v[2][i] - 2 * v[3][i] + 1 * v[4][i];
      TV[5][i] =  4 * v[1][i] - 5 * v[3][i] + 1 * v[5][i];
    }
    #pragma unroll 
    for(int i = 0; i < 6; i++)
    {
      V[i][0] =  4 * TV[i][0] - 5 * TV[i][2] + 1 * TV[i][4];
      V[i][1] = -4 * TV[i][1] - 4 * TV[i][2] + 1 * TV[i][3] + 1 * TV[i][4];
      V[i][2] =  4 * TV[i][1] - 4 * TV[i][2] - 1 * TV[i][3] + 1 * TV[i][4];
      V[i][3] = -2 * TV[i][1] - 1 * TV[i][2] + 2 * TV[i][3] + 1 * TV[i][4];
      V[i][4] =  2 * TV[i][1] - 1 * TV[i][2] - 2 * TV[i][3] + 1 * TV[i][4];
      V[i][5] =  4 * TV[i][1] - 5 * TV[i][3] + 1 * TV[i][5];
    }

    outputs += ((c * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            outputs[0] = V[i][j];
            outputs += C * N * TP * TQ;
        }
    }
}

__global__ void flip_filter(                                              
    float * inputs,                                            
    float * outputs,                                           
    int K, int C, int R, int S                                          
    ) {                                                                 
  int kcrs = blockIdx.x * blockDim.x + threadIdx.x;                                        
  int k = kcrs / (C * R * S);                                         
  if (k >= K) return;                                                 
  int crs = kcrs - k * (C * R * S);                                   
  int c = crs / (R * S);                                              
  int rs = crs - c * (R * S);                                         
  int r = rs / (S);                                                   
  int s = rs - r * (S);                                               

  outputs[((c * K + k) * R + (R - r - 1)) * S + (S - s - 1)] = inputs[kcrs];                                                                  
  }
__global__ void X_transpose(                                              
    float * inputs,                                            
    float * outputs,                                           
    int X, int H, int W                                        
    ) {                                                                 
  int gid = blockIdx.x * blockDim.x + threadIdx.x;                                        
  int x = gid / (H * W);
  int h = (gid / W) % H ;
  int w = (gid % W) ;
  if (x >= X || h >= H || w >= W) return;                                                 
  float * input = inputs + x * H * W;
  float * output = outputs + x * H * W;
  output[w * H + h] = input[h * W + w];
  }


__global__ void baseline_winograd3x3_wgrad_input_tile( // TODO: 4x4 transform
    float *inputs,
    float *outputs,
    int N, int C, int H, int W, int cy
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int nctptq = gid;
    int TP = (H + 1) / 2;
    int TQ = (W + 1) / 2;
    int n = nctptq / (C * TP * TQ);
    if (n >= N) return;
    int ctptq = nctptq - n * (C * TP * TQ);
    int c = ctptq / (TP * TQ);
    int tptq = ctptq - c * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int h = tp * 2 - 1, w = tq * 2 - 1;
    float v[4][4], TV[4][4], V[4][4];

    inputs += ((n * C + c) * H + h) * W + w;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if(!cy){
              v[i][j] = 0 <= h + i && h + i < H && 0 <= w + j && w + j < W ? inputs[i * W + j] : 0;
            }
            else {
              if(v[i][j] = 0 <= h + i && h + i < H && 0 <= w + j && w + j < W) v[i][j] = inputs[i * W + j];
              else if(w + j < 0 || w + j >= W) v[i][j] = 0;
              else if(h + i == -1) v[i][j] = inputs[(i + H) * W + j];
              else if(h + i == H) v[i][j] = inputs[(i - H) * W + j];
            }
        }
    }

    TV[0][0] = v[0][0] - v[2][0];
    TV[0][1] = v[0][1] - v[2][1];
    TV[0][2] = v[0][2] - v[2][2];
    TV[0][3] = v[0][3] - v[2][3];
    TV[1][0] = v[1][0] + v[2][0];
    TV[1][1] = v[1][1] + v[2][1];
    TV[1][2] = v[1][2] + v[2][2];
    TV[1][3] = v[1][3] + v[2][3];
    TV[2][0] = v[2][0] - v[1][0];
    TV[2][1] = v[2][1] - v[1][1];
    TV[2][2] = v[2][2] - v[1][2];
    TV[2][3] = v[2][3] - v[1][3];
    TV[3][0] = v[3][0] - v[1][0];
    TV[3][1] = v[3][1] - v[1][1];
    TV[3][2] = v[3][2] - v[1][2];
    TV[3][3] = v[3][3] - v[1][3];

    V[0][0] = TV[0][0] - TV[0][2];
    V[1][0] = TV[1][0] - TV[1][2];
    V[2][0] = TV[2][0] - TV[2][2];
    V[3][0] = TV[3][0] - TV[3][2];
    V[0][1] = TV[0][1] + TV[0][2];
    V[1][1] = TV[1][1] + TV[1][2];
    V[2][1] = TV[2][1] + TV[2][2];
    V[3][1] = TV[3][1] + TV[3][2];
    V[0][2] = TV[0][2] - TV[0][1];
    V[1][2] = TV[1][2] - TV[1][1];
    V[2][2] = TV[2][2] - TV[2][1];
    V[3][2] = TV[3][2] - TV[3][1];
    V[0][3] = TV[0][3] - TV[0][1];
    V[1][3] = TV[1][3] - TV[1][1];
    V[2][3] = TV[2][3] - TV[2][1];
    V[3][3] = TV[3][3] - TV[3][1];

    outputs += ((c * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            outputs[0] = V[i][j];
            outputs += C * N * TP * TQ;
        }
    }
}

/*
 * inputs dim = (N, K, P, Q)
 * outputs dim = (16, K, N, TP, TQ)
 * global_work_size = {_ceil(N * K * TP * TQ, 256)}
 * local_work_size = {256}
 */
__global__ void baseline_winograd3x3_wgrad_outGrad_tile( // TODO: 4x4 transform
    float *inputs,
    float *outputs,
    int N, int K, int P, int Q 
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int nktptq = gid;
    int TP = (P + 1) / 2;
    int TQ = (Q + 1) / 2;
    int n = nktptq / (K * TP * TQ);
    if (n >= N) return;
    int ktptq = nktptq - n * (K * TP * TQ);
    int k = ktptq / (TP * TQ);
    int tptq = ktptq - k * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int p = tp * 2, q = tq * 2;
    float u0[2][2], u1[4][2], u2[4][4];

    inputs += ((n * K + k) * P + p) * Q + q;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            u0[i][j] = 0 <= p + i && p + i < P && 0 <= q + j && q + j < Q ? inputs[i * Q + j] : 0;
        }
    }

    u1[0][0] = u0[0][0];
    u1[0][1] = u0[0][1];
    u1[1][0] = (u0[0][0] + u0[1][0]) * 0.5;
    u1[1][1] = (u0[0][1] + u0[1][1]) * 0.5;
    u1[2][0] = (u0[0][0] - u0[1][0]) * 0.5;
    u1[2][1] = (u0[0][1] - u0[1][1]) * 0.5;
    u1[3][0] = u0[1][0];
    u1[3][1] = u0[1][1];

    u2[0][0] = u1[0][0];
    u2[1][0] = u1[1][0];
    u2[2][0] = u1[2][0];
    u2[3][0] = u1[3][0];
    u2[0][1] = (u1[0][0] + u1[0][1]) * 0.5;
    u2[1][1] = (u1[1][0] + u1[1][1]) * 0.5;
    u2[2][1] = (u1[2][0] + u1[2][1]) * 0.5;
    u2[3][1] = (u1[3][0] + u1[3][1]) * 0.5;
    u2[0][2] = (u1[0][0] - u1[0][1]) * 0.5;
    u2[1][2] = (u1[1][0] - u1[1][1]) * 0.5;
    u2[2][2] = (u1[2][0] - u1[2][1]) * 0.5;
    u2[3][2] = (u1[3][0] - u1[3][1]) * 0.5;
    u2[0][3] = u1[0][1];
    u2[1][3] = u1[1][1];
    u2[2][3] = u1[2][1];
    u2[3][3] = u1[3][1];

    outputs += ((k * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            outputs[0] = u2[i][j];
            outputs += K * N * TP * TQ;
        }
    }
}

/*
 * inputs dim = (16, K, C)
 * outputs dim = (K, C, 3, 3)
 * global_work_size = {_ceil(K * C, 256)}
 * local_work_size = {256}
 */
__global__ void baseline_winograd3x3_wgrad_inverse(
    float *inputs,
    float *filters,
    int N, int C, int K 
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int kc = gid;
    int k = kc / (C);
    if (k >= K) return;
    int c = kc - k * (C);
    float m0[4][4], m1[3][4], m2[3][3], mt[4];

    inputs += k * C + c;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            m0[i][j] = inputs[0];
            inputs += K * C;
        }
    }

    mt[0] = m0[1][0] + m0[2][0];
    mt[1] = m0[1][1] + m0[2][1];
    mt[2] = m0[1][2] + m0[2][2];
    mt[3] = m0[1][3] + m0[2][3];
    m1[0][0] = mt[0] + m0[0][0];
    m1[0][1] = mt[1] + m0[0][1];
    m1[0][2] = mt[2] + m0[0][2];
    m1[0][3] = mt[3] + m0[0][3];
    m1[1][0] = m0[1][0] - m0[2][0];
    m1[1][1] = m0[1][1] - m0[2][1];
    m1[1][2] = m0[1][2] - m0[2][2];
    m1[1][3] = m0[1][3] - m0[2][3];
    m1[2][0] = mt[0] + m0[3][0];
    m1[2][1] = mt[1] + m0[3][1];
    m1[2][2] = mt[2] + m0[3][2];
    m1[2][3] = mt[3] + m0[3][3];

    mt[0] = m1[0][1] + m1[0][2];
    mt[1] = m1[1][1] + m1[1][2];
    mt[2] = m1[2][1] + m1[2][2];
    m2[0][0] = mt[0] + m1[0][0];
    m2[1][0] = mt[1] + m1[1][0];
    m2[2][0] = mt[2] + m1[2][0];
    m2[0][1] = m1[0][1] - m1[0][2];
    m2[1][1] = m1[1][1] - m1[1][2];
    m2[2][1] = m1[2][1] - m1[2][2];
    m2[0][2] = mt[0] + m1[0][3];
    m2[1][2] = mt[1] + m1[1][3];
    m2[2][2] = mt[2] + m1[2][3];

    filters += (k * C + c) * 3 * 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            filters[i * 3 + j] = m2[i][j];
        }
    }
}


__global__ void baseline_winograd4x4_wgrad_outGrad_tile(
    float *inputs,
    float *outputs,
    int N, int K, int P, int Q 
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int nktptq = gid;
    int TP = (P + 3) / 4;
    int TQ = (Q + 3) / 4;
    int n = nktptq / (K * TP * TQ);
    if (n >= N) return;
    int ktptq = nktptq - n * (K * TP * TQ);
    int k = ktptq / (TP * TQ);
    int tptq = ktptq - k * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int p = tp * 4, q = tq * 4;
    float u0[4][4], u1[6][4], u2[6][6];

    inputs += ((n * K + k) * P + p) * Q + q;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            u0[i][j] = 0 <= p + i && p + i < P && 0 <= q + j && q + j < Q ? inputs[i * Q + j] : 0;
        }
    }

    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
      u1[0][i] = u0[0][i] / 4;
      u1[1][i] = -(1/6.0f) * u0[0][i] -(1/6.0f) * u0[1][i] -(1/6.0f) * u0[2][i] -(1/6.0f) * u0[3][i];
      u1[2][i] = -(1/6.0f) * u0[0][i] +(1/6.0f) * u0[1][i] -(1/6.0f) * u0[2][i] +(1/6.0f) * u0[3][i];
      u1[3][i] = (1/24.0f) * u0[0][i] +(1/12.0f) * u0[1][i] +(1/6.0f) * u0[2][i] +(1/3.0f) * u0[3][i];
      u1[4][i] = (1/24.0f) * u0[0][i] -(1/12.0f) * u0[1][i] +(1/6.0f) * u0[2][i] -(1/3.0f) * u0[3][i];
      u1[5][i] = u0[3][i];
    }
    #pragma unroll
    for(int i = 0; i < 6; i++)
    {
      u2[i][0] = u1[i][0] / 4;
      u2[i][1] = -(1/6.0f) * u1[i][0] -(1/6.0f) * u1[i][1] -(1/6.0f) * u1[i][2] -(1/6.0f) * u1[i][3];
      u2[i][2] = -(1/6.0f) * u1[i][0] +(1/6.0f) * u1[i][1] -(1/6.0f) * u1[i][2] +(1/6.0f) * u1[i][3];
      u2[i][3] = (1/24.0f) * u1[i][0] +(1/12.0f) * u1[i][1] +(1/6.0f) * u1[i][2] +(1/3.0f) * u1[i][3];
      u2[i][4] = (1/24.0f) * u1[i][0] -(1/12.0f) * u1[i][1] +(1/6.0f) * u1[i][2] -(1/3.0f) * u1[i][3];
      u2[i][5] = u1[i][3];    
    }


    outputs += ((k * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            outputs[0] = u2[i][j];
            outputs += K * N * TP * TQ;
        }
    }
}

/*
 * inputs dim = (16, K, C)
 * outputs dim = (K, C, 3, 3)
 * global_work_size = {_ceil(K * C, 256)}
 * local_work_size = {256}
 */
__global__ void baseline_winograd4x4_wgrad_inverse(
    float *inputs,
    float *filters,
    int N, int C, int K 
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int kc = gid;
    int k = kc / (C);
    if (k >= K) return;
    int c = kc - k * (C);
    float m0[6][6], m1[3][6], m2[3][3];

    inputs += k * C + c;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            m0[i][j] = inputs[0];
            inputs += K * C;
        }
    }

    #pragma unroll
    for(int i = 0; i < 6; i++)
    {
      m1[0][i] = m0[0][i] + m0[1][i] + m0[2][i] + m0[3][i] + m0[4][i];
      m1[1][i] = m0[1][i] - m0[2][i] + 2 * m0[3][i] - 2 * m0[4][i];
      m1[2][i] = m0[1][i] + m0[2][i] + 4 * m0[3][i] + 4 * m0[4][i] + m0[5][i];
    }
    #pragma unroll
    for(int i = 0; i < 3; i++)
    {
      m2[i][0] = m1[i][0] + m1[i][1] + m1[i][2] + m1[i][3] + m1[i][4];
      m2[i][1] = m1[i][1] - m1[i][2] + 2 * m1[i][3] - 2 * m1[i][4];
      m2[i][2] = m1[i][1] + m1[i][2] + 4 * m1[i][3] + 4 * m1[i][4] + m1[i][5];
    }

    filters += (k * C + c) * 3 * 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            filters[i * 3 + j] = m2[i][j];
        }
    }
}

__global__ void baseline_winograd4x4_data_untile_k(
    float *inputs,
    float *outputs,
    int N, int K, int H, int W)
   {
    int TP = (H + 3) / 4;
    int TQ = (W + 3) / 4;
    int P = H;
    int Q = W;
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int kntptq = gid;
    int k = kntptq / (N * TP * TQ);
    if (k >= K) return;
    int ntptq = kntptq - k * (N * TP * TQ);
    int n = ntptq / (TP * TQ);
    int tptq = ntptq - n * (TP * TQ);
    int tp = tptq / (TQ);
    int tq = tptq - tp * (TQ);
    int p = tp * 4, q = tq * 4;
    float m[6][6], TM[4][6], M[4][4];

    inputs += ((k * N + n) * TP + tp) * TQ + tq;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            m[i][j] = inputs[0];
            inputs += K * N * TP * TQ;
        }
    }
    
    #pragma unroll
    for(int i = 0; i < 6; i++)
    {
      TM[0][i] =  1 * m[0][i] + 1 * m[1][i] + 1 * m[2][i] + 1 * m[3][i] + 1 * m[4][i];
      TM[1][i] =  1 * m[1][i] - 1 * m[2][i] + 2 * m[3][i] - 2 * m[4][i];
      TM[2][i] =  1 * m[1][i] + 1 * m[2][i] + 4 * m[3][i] + 4 * m[4][i];
      TM[3][i] =  1 * m[1][i] - 1 * m[2][i] + 8 * m[3][i] - 8 * m[4][i] + 1 * m[5][i];
    }
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
      M[i][0] =  1 * TM[i][0] + 1 * TM[i][1] + 1 * TM[i][2] + 1 * TM[i][3] + 1 * TM[i][4];
      M[i][1] =  1 * TM[i][1] - 1 * TM[i][2] + 2 * TM[i][3] - 2 * TM[i][4];
      M[i][2] =  1 * TM[i][1] + 1 * TM[i][2] + 4 * TM[i][3] + 4 * TM[i][4];
      M[i][3] =  1 * TM[i][1] - 1 * TM[i][2] + 8 * TM[i][3] - 8 * TM[i][4] + 1 * TM[i][5];
    }
    outputs += ((n * K + k) * P + p) * Q + q;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (p + i < P && q + j < Q) {
                outputs[i * Q + j] = M[i][j];
            }
        }
    }
}

__global__ void baseline_winograd4x4_filter_tile_k(
    float * inputs,
    float *outputs,
    int K, int C
    ) {
    int lid = threadIdx.x;
    int lsz = blockDim.x;
    int bid = blockIdx.x;
    int gid = bid * lsz + lid;
    int kc = gid;
    int k = kc / (C);
    if (k >= K) return;
    int c = kc - k * (C);
    float u[3][3], TU[3][6], U[6][6];

    inputs += (k * C + c) * 3 * 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            u[i][j] = inputs[i * 3 + j];
        }
    }

    for(int i = 0; i < 3; i++)
    {
      TU[i][0] =  (1/4.0f)  * u[i][0]; 
      TU[i][1] = -(1/6.0f)  * u[i][0] - (1/6.0f)  * u[i][1] - (1/6.0f) * u[i][2]; 
      TU[i][2] = -(1/6.0f)  * u[i][0] + (1/6.0f)  * u[i][1] - (1/6.0f) * u[i][2]; 
      TU[i][3] =  (1/24.0f) * u[i][0] + (1/12.0f) * u[i][1] + (1/6.0f) * u[i][2]; 
      TU[i][4] =  (1/24.0f) * u[i][0] - (1/12.0f) * u[i][1] + (1/6.0f) * u[i][2]; 
      TU[i][5] =  1         * u[i][2]; 
    }

    for(int i = 0; i < 6; i++)
    {
      U[0][i] =  (1/4.0f)  * TU[0][i]; 
      U[1][i] = -(1/6.0f)  * TU[0][i] - (1/6.0f)  * TU[1][i] - (1/6.0f) * TU[2][i]; 
      U[2][i] = -(1/6.0f)  * TU[0][i] + (1/6.0f)  * TU[1][i] - (1/6.0f) * TU[2][i]; 
      U[3][i] =  (1/24.0f) * TU[0][i] + (1/12.0f) * TU[1][i] + (1/6.0f) * TU[2][i]; 
      U[4][i] =  (1/24.0f) * TU[0][i] - (1/12.0f) * TU[1][i] + (1/6.0f) * TU[2][i]; 
      U[5][i] =  1         * TU[2][i]; 
    }

    outputs += k * C + c;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            outputs[0] = U[i][j];
            outputs += K * C;
        }
    }
}


#define CEIL(x, y) ( ((x) + (y) - 1) / (y) )

template<typename scalar_t>
__global__ void cyconv2d_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int out_channels,
    int in_channels,
    int out_height,
    int out_width,
    int in_height,
    int in_width,
    int filter_height,
    int filter_width,
    int stride,
    int padding,
    int dilation)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= batch_size * out_channels) return;
  if (y >= out_height * out_width) return;

  int n = x / out_channels;
  int k = x % out_channels;

  int i = y / out_width;
  int j = y % out_width;

  scalar_t sum = 0.0;

  for (int c = 0; c < in_channels; c++) {
    for (int r = 0; r < filter_height; r++) {
      int p = i * stride + r * dilation - padding;
      if (p < 0) p += in_height;
      if (p >= in_height) p -= in_height;

      for (int s = 0; s < filter_width; s++) {
        int q = j * stride + s * dilation - padding;
        if (q < 0 || q >= in_width) continue;

        size_t a = n * in_channels * in_height * in_width + 
                   c * in_height * in_width +
                   p * in_width + q;

        size_t b = k * in_channels * filter_height * filter_width +
                   c * filter_height * filter_width +
                   r * filter_width + s;

        sum += input[a] * weight[b];
      }
    }
  }

  output[(size_t)x * out_height * out_width + y] = sum;
}

template<typename scalar_t>
__global__ void cyconv2d_cuda_backward_data_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    int batch_size,
    int out_channels,
    int in_channels,
    int out_height,
    int out_width,
    int in_height,
    int in_width,
    int filter_height,
    int filter_width,
    int stride,
    int padding,
    int dilation)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= batch_size * in_channels) return;
  if (y >= in_height * in_width) return;

  int n = x / in_channels;
  int c = x % in_channels;

  int i = y / in_width;
  int j = y % in_width;

  scalar_t sum = 0.0;

  for (int k = 0; k < out_channels; k++) {
    for (int r = 0; r < filter_height; r++) {
      int p = (i - r * dilation) / stride + padding;
      if ((i - r * dilation) % stride != 0) continue;
      if (p < 0) p += out_height;
      if (p >= out_height) p -= out_height;

      for (int s = 0; s < filter_width; s++) {
        int q = (j - s * dilation) / stride + padding;
        if ((j - s * dilation) % stride != 0) continue;
        if (q < 0 || q >= out_width) continue;

        size_t a = n * out_channels * out_height * out_width +
                   k * out_height * out_width +
                   p * out_width + q;

        size_t b = k * in_channels * filter_height * filter_width +
                   c * filter_height * filter_width +
                   r * filter_width + s;

        sum += grad_output[a] * weight[b];
      }
    }
  }

  grad_input[(size_t)x * in_height * in_width + y] = sum;
}

template<typename scalar_t>
__global__ void cyconv2d_cuda_backward_filter_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    int batch_size,
    int out_channels,
    int in_channels,
    int out_height,
    int out_width,
    int in_height,
    int in_width,
    int filter_height,
    int filter_width,
    int stride,
    int padding,
    int dilation)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= out_channels * in_channels) return;
  if (y >= filter_height * filter_width) return;

  int k = x / in_channels;
  int c = x % in_channels;

  int r = y / filter_width;
  int s = y % filter_width;

  scalar_t sum = 0.0;

  for (int n = 0; n < batch_size; n++) {
    for (int i = 0; i < out_height; i++) {
      int p = i * stride + r * dilation - padding;
      if (p < 0) p += in_height;
      if (p >= in_height) p -= in_height;

      for (int j = 0; j < out_width; j++) {
        int q = j * stride + s * dilation - padding;
        if (q < 0 || q >= in_width) continue;

        size_t a = n * in_channels * in_height * in_width +
                   c * in_height * in_width +
                   p * in_width + q;

        size_t b = n * out_channels * out_height * out_width +
                   k * out_height * out_width +
                   i * out_width + j;

        sum += input[a] * grad_output[b];
      }
    }
  }

  grad_weight[(size_t)x * filter_height * filter_width + y] = sum;
}


torch::Tensor cyconv2d_cuda_forward_typical(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor workspace,
  int stride,
  int padding,
  int dilation)
{
  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto in_height = input.size(2);
  const auto in_width = input.size(3);

  const auto out_channels = weight.size(0);
  const auto filter_height = weight.size(2);
  const auto filter_width = weight.size(3);

  const auto out_height =
    (in_height + 2 * padding - (dilation * (filter_height - 1) + 1)) / stride + 1;
  const auto out_width =
    (in_width + 2 * padding - (dilation * (filter_width - 1) + 1)) / stride + 1;

  int64_t y_shape[] = { batch_size, out_channels, out_height, out_width };
  auto options = torch::TensorOptions().device(input.device().type())
                                       .requires_grad(input.requires_grad());
  auto output = torch::zeros(torch::IntArrayRef(y_shape), options);

  const dim3 block_fwd(8, 8);
  const dim3 grid_fwd(
      CEIL(batch_size * out_channels, block_fwd.x),
      CEIL(out_height * out_width, block_fwd.y));

  AT_DISPATCH_FLOATING_TYPES(output.type(), "cyconv2d_forward_cuda",
      ([&]() {
        cyconv2d_cuda_forward_kernel<scalar_t>
          <<< grid_fwd, block_fwd >>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            out_channels,
            in_channels,
            out_height,
            out_width,
            in_height,
            in_width,
            filter_height,
            filter_width,
            stride,
            padding,
            dilation);
       }));

  return output;
}


torch::Tensor cyconv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor filter,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation)
{
  const auto N = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  const auto K = filter.size(0);
  const auto filter_height = filter.size(2);
  const auto filter_width = filter.size(3);


  if ( stride != 1 || dilation != 1 || filter_height != 3 || filter_width != 3 ) {
    return cyconv2d_cuda_forward_typical(input, filter, workspace, stride, padding, dilation);
  }

  int P = (H+3)/4;
  int Q = (W+3)/4;
  int inputTileSize = ((36 * C * N * P * Q + 15) / 16) * 16;
  int outputTileSize = ((36 * K * N * P * Q + 15) / 16) * 16;
  int filterTileSize = ((36 * K * C + 15) / 16 * 16);

  int64_t y_shape[] = { N, K, H, W };
  auto options = torch::TensorOptions().device(input.device().type())
                                       .requires_grad(input.requires_grad());
  auto output = torch::zeros(torch::IntArrayRef(y_shape), options);

  float * inputTile = workspace.data<float>();
  float * filterTile = workspace.data<float>() + inputTileSize;
  float * outputTile = workspace.data<float>() + inputTileSize + filterTileSize;

  dim3 tpb(256, 1, 1);
  dim3 bpg((N * C * P * Q + 255) / 256, 1, 1);

  AT_DISPATCH_FLOATING_TYPES(output.type(), "baseline_winograd4x4_data_tile_k",
    ( 
      [&] () {
        baseline_winograd4x4_data_tile_k<<<bpg, tpb>>>(input.data<float>(), inputTile, N, C, H, W, 1);
      }
    )
  );

  bpg.x = ((C * K + 255)/ 256); 
  AT_DISPATCH_FLOATING_TYPES(output.type(), "baseline_winograd4x4_filter_tile_k",
    ( 
      [&] () {
        baseline_winograd4x4_filter_tile_k<<<bpg, tpb>>>(filter.data<float>(), filterTile, C, K);
      }
    )
  );

  if(N >= 16 && C >= 16 && K >= 16)
  {
    bpg.x = 36 * ((N*P*Q + TILE_N - 1)/ TILE_N) * ((K + TILE_M - 1) / TILE_M);
    bpg.y = 1;
    tpb.x = 128;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(output.type(), "X_indep_gemm_opt",
      ( 
        [&] () {
          X_indep_gemm_opt<<<bpg, tpb>>>(filterTile, inputTile, outputTile, 36, K, N * P * Q, C);
        }
      )
    );

  }
  else
  {
    bpg.x = (N * P * Q + 255) / 256;
    bpg.y = 36 * K;
    tpb.x = 256;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(output.type(), "baseline_winograd4x4_gemm_k",
      ( 
        [&] () {
          baseline_winograd4x4_gemm_k<<<bpg, tpb>>>(inputTile, filterTile, outputTile, N, C, K, H, W); 
        }
      )
    );
    
  }

  bpg.x = ((N * K * P * Q + 255) / 256);
  bpg.y = 1;
  tpb.x = 256;
  tpb.y = 1;
  AT_DISPATCH_FLOATING_TYPES(output.type(), "baseline_winograd4x4_gemm_k",
    ( 
      [&] () {
        baseline_winograd4x4_data_untile_k<<<bpg, tpb>>>(outputTile, output.data<float>(), N, K, H, W);
      }
    )
  );

  return output;
}

std::vector<torch::Tensor> cyconv2d_cuda_backward_typical(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor weight,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation)
{
  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto in_height = input.size(2);
  const auto in_width = input.size(3);

  const auto out_channels = weight.size(0);
  const auto filter_height = weight.size(2);
  const auto filter_width = weight.size(3);

  const auto out_height =
    (in_height + 2 * padding - (dilation * (filter_height - 1) + 1)) / stride + 1;
  const auto out_width =
    (in_width + 2 * padding - (dilation * (filter_width - 1) + 1)) / stride + 1;

  auto grad_input = torch::zeros_like(input);

  const dim3 block_bwd_data(8, 8);
  const dim3 grid_bwd_data(
      CEIL(batch_size * in_channels, block_bwd_data.x),
      CEIL(in_height * in_width, block_bwd_data.y));

  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "cyconv2d_backward_data_cuda",
      ([&]() {
        cyconv2d_cuda_backward_data_kernel<scalar_t>
          <<< grid_bwd_data, block_bwd_data >>>(
            grad_output.data<scalar_t>(),
            weight.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            batch_size,
            out_channels,
            in_channels,
            out_height,
            out_width,
            in_height,
            in_width,
            filter_height,
            filter_width,
            stride,
            padding,
            dilation);
       }));

  auto grad_weight = torch::zeros_like(weight);

  const dim3 block_bwd_filter(8, 8);
  const dim3 grid_bwd_filter(
      CEIL(out_channels * in_channels, block_bwd_filter.x),
      CEIL(filter_height * filter_width, block_bwd_filter.y));

  AT_DISPATCH_FLOATING_TYPES(grad_weight.type(), "cyconv2d_backward_filter_cuda",
      ([&]() {
        cyconv2d_cuda_backward_filter_kernel<scalar_t>
          <<< grid_bwd_filter, block_bwd_filter >>>(
            input.data<scalar_t>(),
            grad_output.data<scalar_t>(),
            grad_weight.data<scalar_t>(),
            batch_size,
            out_channels,
            in_channels,
            out_height,
            out_width,
            in_height,
            in_width,
            filter_height,
            filter_width,
            stride,
            padding,
            dilation);
       }));

  return { grad_input, grad_weight };
}

std::vector<torch::Tensor> cyconv2d_cuda_backward(
  torch::Tensor input,
  torch::Tensor grad_output,
  torch::Tensor filter,
  torch::Tensor workspace,
  int stride,
  int padding,
  int dilation)
{
  const auto N = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  const auto K = filter.size(0);
  const auto filter_height = filter.size(2);
  const auto filter_width = filter.size(3);



  if ( stride != 1 || dilation != 1 || filter_height != 3 || filter_width != 3 ) {
    return cyconv2d_cuda_backward_typical(input, grad_output, filter, workspace, stride, padding, dilation);
  }

  int P = (H+3)/4;
  int Q = (W+3)/4;
  int flipFilterSize = K * C * 3 * 3;
  int inputTileSize = ((36 * C * N * P * Q + 15) / 16) * 16;
  int outputTileSize = ((36 * K * N * P * Q + 15) / 16) * 16;
  int filterTileSize = ((36 * K * C + 15) / 16 * 16);

  float * flipFilter = workspace.data<float>();
  float * inputGradTile = workspace.data<float>() + flipFilterSize;
  float * filterTile = workspace.data<float>() + flipFilterSize + inputTileSize;
  float * outputGradTile = workspace.data<float>() + flipFilterSize + inputTileSize + filterTileSize;

  auto grad_input = torch::zeros_like(input);

  dim3 tpb(256, 1, 1);
  dim3 bpg((N * K * P * Q + 255) / 256, 1, 1);
  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "baseline_winograd4x4_data_tile_k",
    ( 
      [&] () {  
        baseline_winograd4x4_data_tile_k<<<bpg, tpb>>>(grad_output.data<float>(), outputGradTile, N, K, H, W, 1);
      }
    )
  );

  bpg.x = ((C * K * 3 * 3 + 255)/ 256); 
  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "flip_filter",
    ( 
      [&] () {  
        flip_filter<<<bpg, tpb>>>(filter.data<float>(), flipFilter, K, C, 3, 3);
      }
    )
  );

  bpg.x = ((C * K + 255)/ 256); 
  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "baseline_winograd4x4_filter_tile_k",
    ( 
      [&] () {  
        baseline_winograd4x4_filter_tile_k<<<bpg, tpb>>>(flipFilter, filterTile, C, K);
      }
    )
  );

  if(N >= 16 && C >= 16 && K >= 16)
  {
    bpg.x = 36 * ((N*P*Q + TILE_N - 1)/ TILE_N) * ((C + TILE_M - 1) / TILE_M);
    bpg.y = 1;
    tpb.x = 128;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "X_indep_gemm_opt",
      ( 
        [&] () {  
          X_indep_gemm_opt<<<bpg, tpb>>>(filterTile, outputGradTile, inputGradTile, 36, C, N * P * Q, K);  
        }
      )
    );
  }
  else
  {
    bpg.x = (N * P * Q + 255) / 256;
    bpg.y = 36 * C;
    tpb.x = 256;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "baseline_winograd4x4_gemm_k",
      ( 
        [&] () {  
          baseline_winograd4x4_gemm_k<<<bpg, tpb>>>(outputGradTile, filterTile, inputGradTile, N, K, C, H, W);   
        }
      )
    );
  }

  bpg.x = ((N * C * P * Q + 255) / 256);
  bpg.y = 1;
  tpb.x = 256;
  tpb.y = 1;
  AT_DISPATCH_FLOATING_TYPES(grad_input.type(), "baseline_winograd4x4_data_untile_k",
    ( 
      [&] () {  
        baseline_winograd4x4_data_untile_k<<<bpg, tpb>>>(inputGradTile, grad_input.data<float>(), N, C, H, W);
      }
    )
  );

  // ???
  
  auto grad_filter = torch::zeros_like(filter);
  int USE_33 = ((((H+1)/2) * ((W+1)/2)) == (((H+3)/4) * ((W+3)/4)));
  int TP;
  int TQ;
  int TS;
  if(USE_33) // use F(2x2, 3x3)
  {
    TP = (H+1)/2;
    TQ = (H+1)/2;
    TS = 16;
  }
  else // use F(3x3, 4x4)
  {
    TP = (H+3)/4;
    TQ = (H+3)/4;
    TS = 36;
  }

  inputTileSize = ((TS * C * N * TP * TQ + 15) / 16) * 16;
  outputTileSize = ((TS * K * N * TP * TQ + 15) / 16) * 16;
  filterTileSize = ((TS * K * C + 15) / 16 * 16);

  float * inputTile = workspace.data<float>();
  float * filterGradTile = workspace.data<float>() + inputTileSize;
  outputGradTile = workspace.data<float>() + inputTileSize + filterTileSize;
  float * inputTileTrans = workspace.data<float>() + inputTileSize + filterTileSize + outputTileSize;

  tpb.x = 256; tpb.y = tpb.z = 1;
  bpg.x = (N * C * TP * TQ + 255) / 256; bpg.y = bpg.z = 1;
  if(USE_33)
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd3x3_wgrad_input_tile",
      ( 
        [&] () {  
          baseline_winograd3x3_wgrad_input_tile<<<bpg, tpb>>>(input.data<float>(), inputTile, N, C, H, W, 1);
        }
      )
    );
  else
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd4x4_data_tile_k",
      ( 
        [&] () {  
          baseline_winograd4x4_data_tile_k<<<bpg, tpb>>>(input.data<float>(), inputTile, N, C, H, W, 1);
        }
      )
    );


  bpg.x = ((N * K * TP * TQ + 255) / 256);
  if(USE_33)
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd3x3_wgrad_outGrad_tile",
      ( 
        [&] () {  
          baseline_winograd3x3_wgrad_outGrad_tile<<<bpg, tpb>>>(grad_output.data<float>(), outputGradTile, N, K, H, W);
        }
      )
    );
    
  else
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd4x4_wgrad_outGrad_tile",
      ( 
        [&] () {  
          baseline_winograd4x4_wgrad_outGrad_tile<<<bpg, tpb>>>(grad_output.data<float>(), outputGradTile, N, K, H, W);
        }
      )
    );

  if(N * TP * TQ >= TILE_K && C >= TILE_N && K >= TILE_M)
  {
    bpg.x = TS * ((C + TILE_N - 1)/ TILE_N) * ((K + TILE_M - 1) / TILE_M);
    bpg.y = 1;
    tpb.x = 128;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "X_indep_gemm_opt_nt",
      ( 
        [&] () {  
          X_indep_gemm_opt_nt<<<bpg, tpb>>>(outputGradTile, inputTile, filterGradTile, TS, K, C, N * TP * TQ);  
        }
      )
    );
   
  }
  else
  {
    bpg.x = (C + 255) / 256;
    bpg.y = TS * K;
    tpb.x = 256;
    tpb.y = 1;
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd_wgrad_gemm_k",
      ( 
        [&] () {  
          baseline_winograd_wgrad_gemm_k<<<bpg, tpb>>>(inputTile, outputGradTile, filterGradTile, N, C, K, H, W, TP, TQ, TS); 
        }
      )
    );
  }
  bpg.x = ((C * K + 255)/ 256); 
  bpg.y = 1;
  tpb.x = 256;
  tpb.y = 1;
  if(USE_33)
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd3x3_wgrad_inverse",
      ( 
        [&] () {  
          baseline_winograd3x3_wgrad_inverse<<<bpg, tpb>>>(filterGradTile, grad_filter.data<float>(), N, C, K);
        }
      )
    );
    
  else
    AT_DISPATCH_FLOATING_TYPES(grad_filter.type(), "baseline_winograd4x4_wgrad_inverse",
      ( 
        [&] () {  
          baseline_winograd4x4_wgrad_inverse<<<bpg, tpb>>>(filterGradTile, grad_filter.data<float>(), N, C, K);
        }
      )
    );

  return { grad_input, grad_filter };
}
