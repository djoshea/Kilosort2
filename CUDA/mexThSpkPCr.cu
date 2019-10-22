/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
using namespace std;

const int  Nthreads = 1024, maxFR = 10000, NrankMax = 3, nt0max=81, NchanMax = 17;

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  volatile __shared__ float  sW[81*NrankMax], sdata[Nthreads+81]; 
  float x, y;
  int tid, tid0, bid, i, nid, Nrank, NT, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  nt0       = (int) Params[3];
  Nrank     = (int) Params[6];  
   
  if(tid<nt0*Nrank)
      sW[tid]= W[tid];
  __syncthreads();
  
  tid0 = 0;
  while (tid0<NT-Nthreads-nt0+1){
	  if (tid<nt0) 
          sdata[tid] = data[tid0 + tid+ NT*bid];
	  
      sdata[tid + nt0] = data[tid0 + tid + nt0 + NT*bid];	  
	  __syncthreads();
      
	  x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
          y = 0.0f;
		  #pragma unroll 4
          for(i=0;i<nt0;i++)
              y    += sW[i + nid*nt0] * sdata[i+tid];

           x += y*y;
      }
      conv_sig[tid0  + tid + NT*bid]   = sqrt(x);
      
      tid0+=Nthreads;
      __syncthreads();
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  computeProjections(const double *Params, const float *dataraw,
        const int *iC, const int *st, const int *id, const float *W, float *feat){
    
    float x;
    int tidx, nt0min, tidy, my_chan, this_chan, tid, bid, nt0, NchanNear, j, t, NT, NrankPC;
    volatile __shared__ float sW[nt0max*NrankMax], sD[nt0max*NchanMax];
    
    NT 		= (int) Params[0];    
    NchanNear = (int) Params[2];
    nt0       = (int) Params[3];        
    NrankPC  = (int) Params[6];
    nt0min    = (int) Params[4];
    
    tidx = threadIdx.x;
    tidy = threadIdx.y;
    bid = blockIdx.x;
    
    // move wPCA to shared memory
    while (tidx<nt0){
        sW[tidx + tidy*nt0] = W[tidx + tidy*nt0];
        tidx+=blockDim.x;
    }
    tidx = threadIdx.x;
    
    tid = tidx + tidy*blockDim.x;
    // move raw data to shared memory    
    while (tid<nt0){
        my_chan = id[bid];
        for (j=0;j<NchanNear;j++){
            this_chan = iC[j + NchanNear*my_chan];
            sD[tid + nt0*j] = dataraw[tid + st[bid]+nt0min-1 + NT * this_chan];
        }
        tid+=blockDim.x*blockDim.y;
    }
    __syncthreads();
    
    x = 0.0f;
    for (t=0;t<nt0;t++)
        x += sD[t + nt0*tidx] * sW[t + nt0*tidy];
                
    feat[tidy + tidx*NrankPC + NrankPC*NchanNear*bid] = x;
    
    }

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC, int *st, int *id, int *counter){
    
  int nt0, indx, tid, tid0, i, bid, NT, Nchan, NchanNear,j,iChan, nt0min;
  double Cf, d;
  float spkTh;
  bool flag;
 
  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];  
  NchanNear = (int) Params[2];      
  nt0       = (int) Params[3];    
  nt0min    = (int) Params[4];
  spkTh    = (float) Params[5];  
  
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0-nt0min){
      for (i=0; i<Nchan;i++){        
          // get the indices of nearby channels for channel i
          iChan = iC[0 + NchanNear * i];
          Cf    = (double) data[tid0 + NT * iChan];
          flag = true;
            
          // loop over nearby channels, keep flag true only if this channel i
          // has the biggest value of the nearby channels
          for(j=1; j<NchanNear; j++){
              iChan = iC[j+ NchanNear * i]; // nearby channel index to check            
              if (data[tid0 + NT * iChan] > Cf){                
                flag = false;
                break;
              }                
          }
          
          if (flag){
              // channel i had the biggest value, is it big enough to 
              iChan = iC[NchanNear * i];
              if (Cf>spkTh){ // check that it's big enough to be a spike
                  d = (double) dataraw[tid0+nt0min-1 + NT*iChan];
                  if (d > Cf-1e-6){
                      // this is a hit, atomicAdd to counter and return spikes
                      indx = atomicAdd(&counter[0], 1);
                      if (indx<maxFR){
                          st[indx] = tid0;
                          id[indx] = iChan;
                      }
                  }
              }
          }          
      }
      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	max1D(const double *Params, const float *data, float *conv_sig){
    
    volatile __shared__ float  sdata[Nthreads+81];
    float y, spkTh;
    int tid, tid0, bid, i, NT, nt0;
    
    NT 		= (int) Params[0];        
    nt0       = (int) Params[3];    
    spkTh    = (float) Params[5];    
    tid 		= threadIdx.x;
    bid 		= blockIdx.x;
  
    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid]   = data[tid0 + tid + NT*bid];
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();

        y = 0.0f;
        #pragma unroll 4
        for(i=0;i<nt0;i++)
            y    = max(y, sdata[tid+i]);
        
        if (y>spkTh)
            conv_sig[tid0  + tid + NT*bid]   = y;

        tid0+=Nthreads;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* Declare input variables*/
  double *Params, *d_Params;
  unsigned int NT, Nchan, NchanNear, NrankPC;
  
  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT 		= (unsigned int) Params[0]; // total timepoints in batch
  Nchan     = (unsigned int) Params[1]; // number of total channels being sorted
  NchanNear = (unsigned int) Params[2]; // number of nearby channels to filter with templates
  NrankPC     = (unsigned int) Params[6]; // rank of template decomposition
        
  // copy params vec into d_params
  cudaMalloc(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0]));
  cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice);

   /* collect input GPU variables*/
  mxGPUArray const  *W,  *data, *iC;
  mxGPUArray *featPC, *id;
  float *d_featPC;
  int *d_id2;
  const float     *d_W, *d_data;
  const int       *d_iC;
  
  data       = mxGPUCreateFromMxArray(prhs[1]);
  d_data     = (float const *)(mxGPUGetDataReadOnly(data)); // input data
  W             = mxGPUCreateFromMxArray(prhs[2]);
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W)); // input wPCA
  iC       = mxGPUCopyFromMxArray(prhs[3]);
  d_iC     = (int const *)(mxGPUGetDataReadOnly(iC)); // input iC (nChanNear x nChan zero-based indices of nearest channels)
  
  /* allocate new GPU variables*/  
  float *d_dmax, *d_dout;
  int *d_st,  *d_id, *d_counter;
  
  cudaMalloc(&d_dout,   NT * Nchan* sizeof(float));
  cudaMalloc(&d_dmax,  NT * Nchan* sizeof(float));
  cudaMalloc(&d_st,     maxFR * sizeof(int));
  cudaMalloc(&d_id,     maxFR * sizeof(int));
  cudaMalloc(&d_counter,   2*sizeof(int));

   cudaMemset(d_dout,   0, NT * Nchan* sizeof(float)); 
  cudaMemset(d_dmax,   0, NT * Nchan * sizeof(float));
  cudaMemset(d_st,      0, maxFR *   sizeof(int));
  cudaMemset(d_id,      0, maxFR *   sizeof(int));
   cudaMemset(d_counter, 0, 2*sizeof(int));
     
  int *counter;
  counter = (int*) calloc(1,sizeof(int));
  
  // filter the data with the temporal templates
  // @djoshea d_W is wPCA is nt0 x nRank, each channel in d_data is independently
  // filtered in time, by each column of d_W individually, the result squared, and the these squared filters summed over nRank columns
  // d_out is the same size as d_data (NT * nChan)
  Conv1D<<<Nchan, Nthreads>>>(d_Params, d_data, d_W, d_dout);
  
  // get the max of the data
  // @djoshea independently for each channel, takes the local max over the next nt0 timepoint
  // and keeps that value if it is bigger than spkTh (Params[5]) so you end up with mostly zeros
  // and little nonzero blocks where a spike happens
  max1D<<<Nchan, Nthreads>>>(d_Params, d_dout, d_dmax);
  
  // take max across nearby channels
  // @djoshea this finds spikes in d_dmax on the channel where they had the largest filtered amplitude
  // outputs are d_st (the time index of each spike) and d_id (the channel number with the max value)
  maxChannels<<<NT/Nthreads,Nthreads>>>(d_Params, d_dout, d_dmax, d_iC, d_st, d_id, d_counter);
 
  cudaMemcpy(counter,     d_counter, sizeof(int), cudaMemcpyDeviceToHost);
  
  // move d_x to the CPU
  unsigned int minSize=1;
  minSize = min(maxFR, counter[0]);

  const mwSize ddF[] 	= {NrankPC * NchanNear, minSize};
  featPC 		= mxGPUCreateGPUArray(2, ddF, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_featPC 		= (float *)(mxGPUGetData(featPC));
  cudaMemset(d_featPC, 0, NrankPC*NchanNear*minSize*sizeof(float));
      
  const mwSize did[] 	= {minSize, 1};
  id 		= mxGPUCreateGPUArray(2, did, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  d_id2 		= (int *)(mxGPUGetData(id));
  
  dim3 tpP(NchanNear, NrankPC);
  if (minSize>0)      
      computeProjections<<<minSize, tpP>>>(d_Params, d_data, d_iC, d_st, d_id, d_W, d_featPC);  
  
  cudaMemcpy(d_id2, d_id, minSize * sizeof(int),   cudaMemcpyDeviceToDevice);
  
  // dWU stays a GPU array
  plhs[0] 	= mxGPUCreateMxArrayOnGPU(featPC);  
  plhs[1] 	= mxGPUCreateMxArrayOnGPU(id);  
  
  // @djoshea send st output to enable sorting and reproducible results
  // the spikes found is always the same, but the order is not, so we include st to sort it
  mxGPUArray *spikeTimes;
  int *d_spikeTimes;
  const mwSize dspikeTimes[]  = {minSize, 1}; // same size as id
  if (nlhs > 2) {
      spikeTimes 		= mxGPUCreateGPUArray(2, dspikeTimes, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
      d_spikeTimes		= (int *)(mxGPUGetData(spikeTimes));
      cudaMemcpy(d_spikeTimes, d_st, minSize * sizeof(int), cudaMemcpyDeviceToDevice);
      plhs[2]   = mxGPUCreateMxArrayOnGPU(spikeTimes);
  }

  cudaFree(d_st);
  cudaFree(d_id);  
  cudaFree(d_counter);
  cudaFree(d_Params); 
  cudaFree(d_dmax);
  cudaFree(d_dout);  
  
  mxGPUDestroyGPUArray(featPC);  
  mxGPUDestroyGPUArray(id);  
  mxGPUDestroyGPUArray(W);  
  mxGPUDestroyGPUArray(iC);
  mxGPUDestroyGPUArray(data);  
  if (nlhs > 2)
      mxGPUDestroyGPUArray(spikeTimes);
}