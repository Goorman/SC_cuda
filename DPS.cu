#include "DPS.h"

__device__ double cudakernel_F(double x, double y) {
    return (x*x + y*y) / ((1.0 + 1.0*x*y)*(1.0 + 1.0*x*y));
}

//=================================CUDA_COMPUTE_R==================================

__global__ void cudakernel_compute_r(double* r, double* dp, double* x_array, double* y_array, int cellsize_x, int cellpos_x, int cellsize_y, int cellpos_y, int bot, int top, int left, int right) {
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadind < cellsize_x * cellsize_y) {
    	int x = threadind % cellsize_x;
    	int y = threadind / cellsize_x;
    	if (x >= left && x < cellsize_x - right && y >= top && y < cellsize_y - bot){
  			r[threadind] = (dp[threadind] - cudakernel_F(x_array[cellpos_x + x], y_array[cellpos_y + y]));
  		}
    }
}

void DPS::cuda_compute_r(double* r, double* dp) {
	cudakernel_compute_r<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock>>>(r, dp, c_x_array, c_y_array, pcinfo.x_cells_num, pcinfo.x_cell_pos, pcinfo.y_cells_num, pcinfo.y_cell_pos, pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right);
}

//=================================CUDA_COMPUTE_G==================================

__global__ void cudakernel_compute_g(double *g, double *r, double alpha, int cellsize_x, int cellsize_y, int bot, int top, int left, int right) {
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadind < cellsize_x * cellsize_y) {
    	int x = threadind % cellsize_x;
    	int y = threadind / cellsize_x;
    	if ((x >= left) && (x < (cellsize_x - right)) && (y >= top) && (y < (cellsize_y - bot))){
  			g[threadind] = r[threadind] - alpha * g[threadind];
  		}
    }
}

void DPS::cuda_compute_g(double* g, double* r, double alpha) {
	cudakernel_compute_g<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock>>> (g, r, alpha, pcinfo.x_cells_num, pcinfo.y_cells_num, pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right);
	
}

//=================================CUDA_COMPUTE_P==================================

__global__ void cudakernel_compute_p(double *p, double *p_prev, double* g, double tau, int cellsize_x, int cellsize_y, int bot, int top, int left, int right) {
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadind < cellsize_x * cellsize_y) {
    	int x = threadind % cellsize_x;
    	int y = threadind / cellsize_x;
    	if (x >= left && x < cellsize_x - right && y >= top && y < cellsize_y - bot){
  			p[threadind] = p_prev[threadind] - tau * g[threadind];
  		}
    }
}

void DPS::cuda_compute_p(double tau, double* g, double* p_prev) {
	cudakernel_compute_p<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock>>>(c_p, p_prev, g, tau, pcinfo.x_cells_num, pcinfo.y_cells_num, pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right);
}

//=================================CUDA_SCALAR_PRODUCT==============================

__global__ void cudakernel_scalar_product(double *f1, double *f2, double* hhx, double* hhy, double *blockresults, int cellsize_x, int cellsize_y, int cellpos_x, int cellpos_y, int bot, int top, int left, int right) {
	extern __shared__ double shared_data[];
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;
	int threadx = threadIdx.x;
	double val = 0;
	if(threadind < cellsize_x * cellsize_y) {
		int x = threadind % cellsize_x;
    		int y = threadind / cellsize_x; 
    		if ((x >= left) && (x < (cellsize_x - right)) && (y >= top) && (y < (cellsize_y - bot))){
			val = hhx[cellpos_x + x] * hhy[cellpos_y + y] * f1[threadind] * f2[threadind];
		}
	}

	shared_data[threadx] = val;
	__syncthreads(); 

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(threadx < offset) {
			shared_data[threadx] += shared_data[threadx + offset];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		blockresults[blockIdx.x] = shared_data[0];
	}
}


__global__ void cudakernel_reduce_sum(double *input, double *results, int blocknum) {
	extern __shared__ double shared_data[];
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;
	int threadx = threadIdx.x;
	double val = 0;
	if(threadind < blocknum) {
		val = input[threadind];
	}

	shared_data[threadx] = val;
	__syncthreads(); 
	for (int offset = blocknum / 2; offset > 0; offset >>= 1) {
		if(threadx < offset) {
			shared_data[threadx] += shared_data[threadx + offset];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		*(results) = shared_data[0];
	}
}

double DPS::cuda_compute_sprod(double* f1, double* f2) {
	double local_scalar_product = 0;

	cudakernel_scalar_product<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock, pcinfo.cuda_threadsPerBlock * sizeof(double)>>>(f1, f2, c_hhx_array, c_hhy_array, product, pcinfo.x_cells_num, pcinfo.y_cells_num, pcinfo.x_cell_pos, pcinfo.y_cell_pos, pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right);
	//cudakernel_reduce_sum<<<1, pcinfo.cuda_blocksNum, pcinfo.cuda_blocksNum * sizeof(double)>>>(product, answer, pcinfo.cuda_blocksNum);
	//cudaMemcpy(&local_scalar_product, answer, sizeof(double), cudaMemcpyDeviceToHost);
	double* answerhost = new double[pcinfo.cuda_blocksNum];
	cudaMemcpy(answerhost, product, sizeof(double) * pcinfo.cuda_blocksNum, cudaMemcpyDeviceToHost);
	for (int i = 0; i < pcinfo.cuda_blocksNum;++i) {
		local_scalar_product += answerhost[i];
	}
	delete[] answerhost;
	
	double global_scalar_product = 0;

	MPI_Barrier(pmp.comm);

    int ret = MPI_Allreduce(
        &local_scalar_product,      
        &global_scalar_product,     
        1,                          
        MPI_DOUBLE,                
        MPI_SUM,                   
        pmp.comm
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error reducing scalar_product.");
    return global_scalar_product;
}

//=================================CUDA_MAX_NORM====================================

__global__ void cudakernel_maxnorm(double *f1, double *f2, double *blockresults, int cellsize_x, int cellsize_y, int cellpos_x, int cellpos_y, int bot, int top, int left, int right) {
	extern __shared__ double shared_data[];
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;
	int threadx = threadIdx.x;
	double val = 0;
	if(threadind < cellsize_x * cellsize_y) {
			val = fabs(f1[threadind] - f2[threadind]);
	}

	shared_data[threadx] = val;
	__syncthreads(); 

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(threadx < offset) {
			shared_data[threadx] = fmax(shared_data[threadx], shared_data[threadx + offset]);
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		blockresults[blockIdx.x] = shared_data[0];
	}
}

__global__ void cudakernel_reduce_max(double *input, double *results, int blocknum) {
	extern __shared__ double shared_data[];
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;
	int threadx = threadIdx.x;
	double val = 0;
	if(threadind < blocknum) {
		val = input[threadind];
	}

	shared_data[threadx] = val;
	__syncthreads(); 
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(threadx < offset) {
			shared_data[threadx] = fmax(shared_data[threadx], shared_data[threadx + offset]);
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		results[0] = shared_data[0];
	}
}

double DPS::cuda_compute_maxnorm(double* f1, double* f2) {
	double local_norm = 0;

	cudakernel_maxnorm<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock, pcinfo.cuda_threadsPerBlock * sizeof(double)>>>(f1, f2, norm, pcinfo.x_cells_num, pcinfo.y_cells_num, pcinfo.x_cell_pos, pcinfo.y_cell_pos, pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right);

	//cudakernel_reduce_max<<<1, pcinfo.cuda_blocksNum, pcinfo.cuda_blocksNum * sizeof(double)>>>(norm, norm, pcinfo.cuda_blocksNum);

	//cudaMemcpy(&local_norm, norm, sizeof(double), cudaMemcpyDeviceToHost);
	
	double* normhost = new double[pcinfo.cuda_blocksNum];
	cudaMemcpy(normhost, norm, sizeof(double) * pcinfo.cuda_blocksNum, cudaMemcpyDeviceToHost);
	for (int i = 0; i < pcinfo.cuda_blocksNum;++i) {
		local_norm = fmax(local_norm, normhost[i]);
	}
	delete[] normhost;

	double global_norm = 0;

    int ret = MPI_Allreduce(
        &local_norm,                     
        &global_norm,               
        1,                         
        MPI_DOUBLE,               
        MPI_MAX,                
        pmp.comm            
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error computing function norm difference.");
    return global_norm;
}

bool DPS::cuda_stop_condition(double* f1, double* f2) {
	double global_norm = cuda_compute_maxnorm(f1, f2); 
	return global_norm < eps;
}

//=================================CUDA_COMPUTE_LOA====================================

__global__ void cudakernel_compute_loa(double* df, double* f, double* hx, double* hhx, double* hy, double* hhy, int cellsize_x, int cellpos_x, int cellsize_y, int cellpos_y, int bottom, int top, int left, int right, double* recv_td, double* recv_dt, double* recv_lr, double* recv_rl) {
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadind < cellsize_x * cellsize_y) {
    	int x = threadind % cellsize_x;
    	int y = threadind / cellsize_x;
    	if (x >= 1 && x < cellsize_x - 1 && y >= 1 && y < cellsize_y - 1) {
  			df[y * cellsize_x + x] = 
		(   (f[y * cellsize_x + x    ] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                    (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x    ]) / hx[cellpos_x + x]
                ) / hhx[cellpos_x + x] + (
                    (f[ y      * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                    (f[(y + 1) * cellsize_x + x] - f[ y      * cellsize_x + x]) / hy[cellpos_y + y]
                ) / hhy[cellpos_y + y];
  		} else if (x == 0 && y >= 1 && y < cellsize_y - 1) {
  			if (not left) {
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x  ] - recv_lr[y]) / hx[cellpos_x + x - 1] -
                        (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[ y      * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                        (f[(y + 1) * cellsize_x + x] - f[ y      * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x == cellsize_x - 1 && y >= 1 && y < cellsize_y - 1) {
  			if (not right) {
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                        (recv_rl[y]            - f[y * cellsize_x + x    ]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[ y      * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                        (f[(y + 1) * cellsize_x + x] - f[ y      * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x >= 1 && x < cellsize_x - 1 && y == 0) {
  			if (not top) {
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x    ] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                        (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x    ]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[ y      * cellsize_x + x] - recv_td[x]) / hy[cellpos_y + y - 1] -
                        (f[(y + 1) * cellsize_x + x] - f[y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x >= 1 && x < cellsize_x - 1 && y == cellsize_y - 1) {
  			if (not bottom) {
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x    ] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                        (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x    ]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[y * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                        (recv_dt[x] - f[ y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x == 0 && y == 0) {
  			if (not top and not left) {
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x    ] - recv_lr[0]) / hx[cellpos_x + x - 1] -
                        (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[ y      * cellsize_x + x] - recv_td [0]) / hy[cellpos_y + y - 1] -
                        (f[(y + 1) * cellsize_x + x] - f[y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }

  		} else if (x == 0 && y == cellsize_y - 1) {
  			if (not bottom and not left){
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x    ] - recv_lr[cellsize_y - 1]) / hx[cellpos_x + x - 1] -
                        (f[y * cellsize_x + x + 1] - f[y * cellsize_x + x]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[y * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                        (recv_dt [0] - f[ y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x == cellsize_x - 1 && y == 0) {
  			if (not top and not right){
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x  ] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                        (recv_rl[0]              - f[y * cellsize_x + x    ]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[ y      * cellsize_x + x] - recv_td [cellsize_x - 1]) / hy[cellpos_y + y - 1] -
                        (f[(y + 1) * cellsize_x + x] - f[ y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		} else if (x == cellsize_x - 1 && y == cellsize_y - 1) {
  			if (not bottom and not right){
                df[y * cellsize_x + x] = (
                        (f[y * cellsize_x + x] - f[y * cellsize_x + x - 1]) / hx[cellpos_x + x - 1] -
                        (recv_rl [cellsize_y - 1] - f[y * cellsize_x + x]) / hx[cellpos_x + x]
                    ) / hhx[cellpos_x + x] + (
                        (f[y * cellsize_x + x] - f[(y - 1) * cellsize_x + x]) / hy[cellpos_y + y - 1] -
                        (recv_dt [cellsize_x - 1] - f[y * cellsize_x + x]) / hy[cellpos_y + y]
                    ) / hhy[cellpos_y + y];
            }
  		}
    }
}

void DPS::cuda_compute_loa(double* df, double* f) {
	cudakernel_compute_loa<<< pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock >>> 
        (df, f, c_hx_array, c_hhx_array, c_hy_array, c_hhy_array, 
            pcinfo.x_cells_num, pcinfo.x_cell_pos, pcinfo.y_cells_num, pcinfo.y_cell_pos, 
            pcinfo.bottom, pcinfo.top, pcinfo.left, pcinfo.right, 
            c_recv_td, c_recv_bu, c_recv_lr, c_recv_rl);
}

//=================================CUDA_FORM_MPI_MESSAGES====================================

__global__ void form_messages (double* f, double* c_send_lr, double* c_send_rl, double* c_send_td, double* c_send_bu, int cellsize_x, int cellsize_y) {
	int threadind = threadIdx.x + blockDim.x * blockIdx.x;

    if (threadind < cellsize_x * cellsize_y) {
    	int x = threadind % cellsize_x;
    	int y = threadind / cellsize_x;

    	if (x == 0 && y >= 1 && y < cellsize_y - 1) {
    		c_send_rl[y] = f[ y * cellsize_x + 0];
  		} else if (x == cellsize_x - 1 && y >= 1 && y < cellsize_y - 1) {
  			c_send_lr[y] = f[ (y + 1) * cellsize_x - 1];
  		} else if (x >= 1 && x < cellsize_x - 1 && y == 0) {
  			c_send_bu[x] = f[x];
  		} else if (x >= 1 && x < cellsize_x - 1 && y == cellsize_y - 1) {
  			c_send_td[x] = f[ (cellsize_y - 1) * cellsize_x + x];
  		} else if (x == 0 && y == 0) {
  			c_send_rl[y] = f[ y * cellsize_x + 0];
  			c_send_bu[x] = f[x];
  		} else if (x == 0 && y == cellsize_y - 1) {
  			c_send_rl[y] = f[ y * cellsize_x + 0];
  			c_send_td[x] = f[ (cellsize_y - 1) * cellsize_x + x];
  		} else if (x == cellsize_x - 1 && y == 0) {
  			c_send_lr[y] = f[ (y + 1) * cellsize_x - 1];
  			c_send_bu[x] = f[x];
  		} else if (x == cellsize_x - 1 && y == cellsize_y - 1) {
  			c_send_lr[y] = f[ (y + 1) * cellsize_x - 1];
  			c_send_td[x] = f[ (cellsize_y - 1) * cellsize_x + x];
  		}
  	}
}

void DPS::cuda_form_messages(double* f) {
	form_messages<<<pcinfo.cuda_blocksNum, pcinfo.cuda_threadsPerBlock>>>(f, c_send_lr, c_send_rl, c_send_td, c_send_bu, pcinfo.x_cells_num, pcinfo.y_cells_num);
}
