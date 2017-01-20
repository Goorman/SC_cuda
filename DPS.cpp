#include "DPS.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

ProcessorCellsInfo::ProcessorCellsInfo ():
x_proc_num (0),
y_proc_num (0),
x_cells_num (0),
x_cell_pos (0),
y_cells_num (0),
y_cell_pos (0),
top (false),
bottom (false),
left (false),
right (false),
cuda_threadsPerBlock(0),
cuda_blocksNum(0)
{}

ProcessorCellsInfo::ProcessorCellsInfo (int rank, 
										int grid_size_x, int grid_size_y, 
										int x_proc_num_, int y_proc_num_){
    x_proc_num = x_proc_num_;
    y_proc_num = y_proc_num_;

    int x_cells_per_proc = (grid_size_x + 1) / x_proc_num;
    int x_extra_cells_num = (grid_size_x + 1) % x_proc_num;
    int x_normal_tasks_num = x_proc_num - x_extra_cells_num;

    if (rank % x_proc_num < x_normal_tasks_num) {
        x_cells_num = x_cells_per_proc;
        x_cell_pos = rank % x_proc_num * x_cells_per_proc;
    } else {
        x_cells_num = x_cells_per_proc + 1;
        x_cell_pos = rank % x_proc_num * x_cells_per_proc + (rank % x_proc_num - x_normal_tasks_num);
    }

    int y_cells_per_proc = (grid_size_y + 1) / y_proc_num;
    int y_extra_cells_num = (grid_size_y + 1) % y_proc_num;
    int y_normal_tasks_num = y_proc_num - y_extra_cells_num;

    if (rank / x_proc_num < y_normal_tasks_num) {
        y_cells_num = y_cells_per_proc;
        y_cell_pos = rank / x_proc_num * y_cells_per_proc;
    } else {
        y_cells_num = y_cells_per_proc + 1;
        y_cell_pos = rank / x_proc_num * y_cells_per_proc + (rank / x_proc_num - y_normal_tasks_num);
    }

    top = (rank < x_proc_num);
    bottom = (rank >= x_proc_num * (y_proc_num - 1));
    left = (rank % x_proc_num == 0);
    right = (rank % x_proc_num == x_proc_num - 1);

    cudaGetDeviceProperties(&devProp, 0);


    cuda_threadsPerBlock = fmin(devProp.maxThreadsPerBlock, x_cells_num);
    cuda_blocksNum = fmin(devProp.maxThreadsPerMultiProcessor, (x_cells_num * y_cells_num + cuda_threadsPerBlock - 1) / cuda_threadsPerBlock);
}

// --------------------------------------------------------------------------------

DPS::DPS(int gridsize_): 
gridsize(gridsize_),
eps(0.0001),
q(1.5),
x0(0),
xn(2),
y0(0),
yn(2),
iterations_counter(0),
final_error(0),

p (NULL),
c_p (NULL),

send_message_lr (NULL),
send_message_rl (NULL),
send_message_td (NULL),
send_message_bu (NULL),
recv_message_lr (NULL),
recv_message_rl (NULL),
recv_message_td (NULL),
recv_message_bu (NULL),
c_send_lr (NULL),
c_send_rl (NULL),
c_send_td (NULL),
c_send_bu (NULL),
c_recv_lr (NULL),
c_recv_rl (NULL),
c_recv_td (NULL),
c_recv_bu (NULL),
recv_loa_reqs   (NULL),
send_loa_reqs   (NULL),
x_array         (NULL),
y_array         (NULL),
hx_array        (NULL),
hy_array        (NULL),
hhx_array       (NULL),
hhy_array       (NULL),
c_x_array         (NULL),
c_y_array         (NULL),
c_hx_array        (NULL),
c_hy_array        (NULL),
c_hhx_array       (NULL),
c_hhy_array       (NULL),
product           (NULL),
norm              (NULL),
answer            (NULL)
{
	send_loa_reqs = new MPI_Request [4];
    recv_loa_reqs = new MPI_Request [4];
}

DPS::~DPS(){
    if (p != NULL){
        delete [] p;
    }

    if (send_message_lr != NULL){
        delete [] send_message_lr;
    }
    if (send_message_rl != NULL){
        delete [] send_message_rl;
    }
    if (send_message_td != NULL){
        delete [] send_message_td;
    }
    if (send_message_bu != NULL){
        delete [] send_message_bu;
    }
    if (recv_message_lr != NULL){
        delete [] recv_message_lr;
    }
    if (recv_message_rl != NULL){
        delete [] recv_message_rl;
    }
    if (recv_message_td != NULL){
        delete [] recv_message_td;
    }
    if (recv_message_bu != NULL){
        delete [] recv_message_bu;
    }
    if (recv_loa_reqs != NULL){
        delete [] recv_loa_reqs;
    }
    if (send_loa_reqs != NULL){
        delete [] send_loa_reqs;
    }

    cudaFree(c_recv_bu);
    cudaFree(c_recv_td);
    cudaFree(c_recv_rl);
    cudaFree(c_recv_lr);
    cudaFree(c_send_bu);
    cudaFree(c_send_td);
    cudaFree(c_send_rl);
    cudaFree(c_send_lr);

    if (pmp.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&pmp.comm);
    }
}

void DPS::compute_grid(ProcessorMPIParameters& pmp_, int& x_proc_num, int& y_proc_num){
    x_proc_num = int(ceil(sqrt(pmp_.size)));
    while(pmp_.size % x_proc_num != 0) {
        x_proc_num--;
    }
    y_proc_num = pmp_.size / x_proc_num;
}

double DPS::getFinalError() {
    return final_error;
}

int DPS::getProcessorNumber() {
    return pmp.size;
}

int DPS::getIterationNumber() {
    return iterations_counter;
}

MPI_Comm DPS::PrepareMPIComm(ProcessorMPIParameters& pmp, 
							 int x_proc_num, int y_proc_num) {
    MPI_Comm rank_comm;
    if (pmp.rank < x_proc_num * y_proc_num){
        MPI_Comm_split(MPI_COMM_WORLD, 1, pmp.rank, &rank_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, pmp.rank, &rank_comm);
    }

    return rank_comm;
}

void DPS::ComputeApproximation(ProcessorMPIParameters& pmp_) {
    int x_proc_num = 0;
    int y_proc_num = 0;
    compute_grid(pmp_, x_proc_num, y_proc_num);

    MPI_Comm algComm = PrepareMPIComm(pmp_, x_proc_num, y_proc_num);
    if (algComm == MPI_COMM_NULL)
        return;

    if (pmp.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&pmp.comm);
    }
    pmp = ProcessorMPIParameters(algComm);
    pcinfo = ProcessorCellsInfo(pmp.rank, gridsize, gridsize, x_proc_num, y_proc_num);

    if (p != NULL) delete [] p;

    p = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* p_prev = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* g = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* r = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dp = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dg = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];
    double* dr = new double [pcinfo.x_cells_num * pcinfo.y_cells_num];

    x_array = new double[gridsize + 1];
    y_array = new double[gridsize + 1];
    hx_array = new double[gridsize + 1];
    hy_array = new double[gridsize + 1];
    hhx_array = new double[gridsize + 1];
    hhy_array = new double[gridsize + 1];

    double sprod_dg_and_g = 1;
    double sprod_dr_and_g = 1;
    double sprod_r_and_g = 1;
    double alpha = 0;
    double tau = 0;

    allocate_message_arrays();
    initialize_net_arrays();
    initialize_border_with_zero(g);
    initialize_border_with_zero(r);
    initialize_border_with_zero(dp);
    initialize_border_with_zero(dg);
    initialize_border_with_zero(dr);
    initialize_with_border_function(p);
    initialize_with_border_function(p_prev);
   
    int cudamallocsize = pcinfo.x_cells_num * pcinfo.y_cells_num * sizeof(double);
    double* c_p_prev, *c_g, *c_r, *c_dp, *c_dg, *c_dr;
    cudaMalloc(&c_p, cudamallocsize); cudaMemcpy(c_p, p, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_p_prev, cudamallocsize); cudaMemcpy(c_p_prev, p_prev, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_g, cudamallocsize); cudaMemcpy(c_g, g, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_r, cudamallocsize); cudaMemcpy(c_r, r, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_dp, cudamallocsize); cudaMemcpy(c_dp, dp, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_dg, cudamallocsize); cudaMemcpy(c_dg, dg, cudamallocsize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_dr, cudamallocsize); cudaMemcpy(c_dr, dr, cudamallocsize, cudaMemcpyHostToDevice);

    cudaMalloc(&product, (pcinfo.cuda_blocksNum + 1) * sizeof(double));
    cudaMalloc(&norm, (pcinfo.cuda_blocksNum + 1) * sizeof(double));
    cudaMalloc(&answer, (sizeof(double)));

    int cudamallocarraysize = (gridsize + 1) * sizeof(double);
    cudaMalloc(&c_x_array, cudamallocarraysize); cudaMemcpy(c_x_array, x_array, cudamallocarraysize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_y_array, cudamallocarraysize); cudaMemcpy(c_y_array, y_array, cudamallocarraysize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_hx_array, cudamallocarraysize); cudaMemcpy(c_hx_array, hx_array, cudamallocarraysize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_hy_array, cudamallocarraysize); cudaMemcpy(c_hy_array, hy_array, cudamallocarraysize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_hhx_array, cudamallocarraysize); cudaMemcpy(c_hhx_array, hhx_array, cudamallocarraysize, cudaMemcpyHostToDevice);
    cudaMalloc(&c_hhy_array, cudamallocarraysize); cudaMemcpy(c_hhy_array, hhy_array, cudamallocarraysize, cudaMemcpyHostToDevice);
	
    iterations_counter = 0;
	// ALGORITHM ITERATION 1
    {
	compute_loa(c_dp, c_p_prev);
	cuda_compute_r(c_r, c_dp);
	std::swap(c_g, c_r);
    	compute_loa(c_dg, c_g);
//	std::cout << iterations_counter <<" c_p_prev: ";
//	printcuda(c_p_prev,cudamallocsize);
//	std::cout << iterations_counter <<" c_dp: ";
//	printcuda(c_dp, cudamallocsize);
//	std::cout << iterations_counter << " c_g: ";
//	printcuda(c_g, cudamallocsize);
//	std::cout << iterations_counter << " c_dg: ";
//	printcuda(c_dg, cudamallocsize);

	sprod_r_and_g = cuda_compute_sprod(c_g, c_g);
    	sprod_dg_and_g = cuda_compute_sprod(c_dg, c_g);
//	if (pmp.rank == 0) std::cout << iterations_counter << " " << sprod_r_and_g << " " << sprod_dg_and_g << std::endl;
	if (sprod_dg_and_g != 0){
            tau = sprod_r_and_g / sprod_dg_and_g;
        } else {
            throw DPS_exception( "Division by 0 in tau computation, iteration 1.");
        }
        cuda_compute_p (tau, c_g, c_p_prev);
//        if (pmp.rank == 0) std::cout << iterations_counter<< " tau: " <<  tau << std::endl;
//	std::cout << "c_p: ";
//	printcuda(c_p, cudamallocsize);

	std::swap(c_p, c_p_prev);
        iterations_counter++;
    }
	// ALGORITHM ITERATION 2+
    while(true){
        compute_loa(c_dp, c_p_prev);
        cuda_compute_r (c_r, c_dp);
        compute_loa(c_dr, c_r);
        sprod_dr_and_g = cuda_compute_sprod(c_dr, c_g);
        alpha = sprod_dr_and_g / sprod_dg_and_g;
        cuda_compute_g (c_g, c_r, alpha);
        compute_loa(c_dg, c_g);
// 	std::cout << iterations_counter <<" c_p_prev: ";
//	printcuda(c_p_prev,cudamallocsize);
//	std::cout << iterations_counter <<" c_dp: ";
//	printcuda(c_dp, cudamallocsize);
//	std::cout << "c_g: ";
//	printcuda(c_g, cudamallocsize);
//	std::cout << "c_dg: ";
//	printcuda(c_dg, cudamallocsize);

	sprod_r_and_g = cuda_compute_sprod(c_r, c_g);
        sprod_dg_and_g = cuda_compute_sprod(c_dg, c_g);
//        if (pmp.rank == 0) std::cout << iterations_counter << " " << sprod_r_and_g << " " << sprod_dg_and_g << std::endl;
	if (sprod_dg_and_g != 0){
            tau = sprod_r_and_g / sprod_dg_and_g;
        } else {
            throw DPS_exception( "Division by 0 in tau computation.");
        }
//        if (pmp.rank == 0) std::cout << iterations_counter << " tau: " <<  tau << std::endl;
	cuda_compute_p (tau, c_g, c_p_prev);
//	std::cout << "c_p: ";
//	printcuda(c_p, cudamallocsize);
        if (cuda_stop_condition (c_p, c_p_prev))
            break;

        std::swap(c_p, c_p_prev);
        iterations_counter++;
    }

    cudaMemcpy(p, c_p, cudamallocsize, cudaMemcpyDeviceToHost);
    final_error = compute_error();

    delete [] dp;
    delete [] dg;
    delete [] dr;
    delete [] g;
    delete [] r;
    cudaFree(c_dp);
    cudaFree(c_dg);
    cudaFree(c_dr);
    cudaFree(c_g);
    cudaFree(c_r);
}

double DPS::F(double x, double y) {
    return (x * x + y * y) / ((1 + x * y) * (1 + x * y));
}

double DPS::phi(double x, double y) {
    return log(1 + x * y);
}

double DPS::x_(int i) {
    return x_array[i];  
}

double DPS::y_(int i) {
    return y_array[i];  
}

double DPS::hx_(int i) {
    return hx_array[i];
}

double DPS::hy_(int i) {
    return hy_array[i];
}

double DPS::hhx_(int i) {
    return hhx_array[i];
}

double DPS::hhy_(int i) {
    return hhy_array[i];
}

double DPS::compute_error() {
    double local_error = 0;

    #pragma omp parallel for schedule(static) reduction(+:local_error)
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            local_error += fabs(phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + j)) - p[j * pcinfo.x_cells_num + i]);
        }
    }

    double global_error = 0;

    int ret = MPI_Allreduce(
        &local_error,      
        &global_error,     
        1,                          
        MPI_DOUBLE,                
        MPI_SUM,                   
        pmp.comm
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error computing error.");

    return global_error;
}

double DPS::compute_maxnorm(double* f1, double* f2) {
	double norm = 0;
    double priv_norm = 0;
    #pragma omp parallel firstprivate (priv_norm)
    {
        #pragma omp for schedule (static)
        for (int i = 0; i < pcinfo.x_cells_num * pcinfo.y_cells_num; i++){
            priv_norm = fmax(priv_norm, fabs(f1[i] - f2[i]));
        }

        #pragma omp critical
        {
            norm = fmax(priv_norm, norm);
        }
    }

    double global_norm = 0;

    int ret = MPI_Allreduce(
        &norm,                     
        &global_norm,               
        1,                         
        MPI_DOUBLE,               
        MPI_MAX,                
        pmp.comm            
    );
    if (ret != MPI_SUCCESS) throw DPS_exception("Error computing function norm difference.");
    return global_norm;
}

bool DPS::stop_condition(double* f1, double* f2) {
	double global_norm = compute_maxnorm(f1, f2); 
    return global_norm < eps;
}

int DPS::receive_loa_messages(){
    int recv_amount = 0;
    int ret = MPI_SUCCESS;

    if (not pcinfo.left){
        ret = MPI_Irecv(
            recv_message_lr,                            
            pcinfo.y_cells_num,                    
            MPI_DOUBLE,                            
            pmp.rank - 1,                 
            DPS::LoaLeftRight,                    
            pmp.comm,                         
            &(recv_loa_reqs[recv_amount])            
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message from left to right.");
    }
    if (not pcinfo.right){
        ret = MPI_Irecv(
            recv_message_rl,                            
            pcinfo.y_cells_num,                  
            MPI_DOUBLE,                              
            pmp.rank + 1,                  
            DPS::LoaRightLeft,                     
            pmp.comm,                        
            &(recv_loa_reqs[recv_amount])           
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message from right to left.");
    }
    if (not pcinfo.top){
        ret = MPI_Irecv(
            recv_message_td,                         
            pcinfo.x_cells_num,               
            MPI_DOUBLE,                                
            pmp.rank - pcinfo.x_proc_num,   
            DPS::LoaTopDown,                      
            pmp.comm,                          
            &(recv_loa_reqs[recv_amount])        
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message top to down.");
    }
    if (not pcinfo.bottom){
        ret = MPI_Irecv(
            recv_message_bu,                          
            pcinfo.x_cells_num,                 
            MPI_DOUBLE,                           
            pmp.rank + pcinfo.x_proc_num, 
            DPS::LoaBottomUp,                     
            pmp.comm,                          
            &(recv_loa_reqs[recv_amount])           
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error receiving message bottom to up.");
    }

    return recv_amount;
}

int DPS::send_loa_messages(){
    int send_amount = 0;
    int ret = MPI_SUCCESS;

    if (not pcinfo.right){
        ret = MPI_Isend(
            send_message_lr,                   
            pcinfo.y_cells_num,                
            MPI_DOUBLE,                              
            pmp.rank + 1,            
            DPS::LoaLeftRight,                
            pmp.comm,                       
            &(send_loa_reqs[send_amount])          
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message from left to right.");
    }
    if (not pcinfo.left){
        ret = MPI_Isend(
            send_message_rl,                           
            pcinfo.y_cells_num,                 
            MPI_DOUBLE,                                
            pmp.rank - 1,                      
            DPS::LoaRightLeft,                     
            pmp.comm,                       
            &(send_loa_reqs[send_amount])            
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message from right to left.");
    }
    if (not pcinfo.bottom){
        ret = MPI_Isend(
            send_message_td,                       
            pcinfo.x_cells_num,                     
            MPI_DOUBLE,                               
            pmp.rank + pcinfo.x_proc_num,   
            DPS::LoaTopDown,                       
            pmp.comm,                          
            &(send_loa_reqs[send_amount])          
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message top to down.");
    }
    if (not pcinfo.top){
        ret = MPI_Isend(
            send_message_bu,                        
            pcinfo.x_cells_num,                     
            MPI_DOUBLE,                             
            pmp.rank - pcinfo.x_proc_num,  
            DPS::LoaBottomUp,                      
            pmp.comm,                           
            &(send_loa_reqs[send_amount])          
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DPS_exception("Error sending message bottom to up.");
    }

    return send_amount;
}

void DPS::compute_loa(double* df, double* f){
    int i = 0;
    int j = 0;
    int ret = MPI_SUCCESS;

    cuda_form_messages(f);

    cudaMemcpy(send_message_lr, c_send_lr, pcinfo.y_cells_num*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_message_rl, c_send_rl, pcinfo.y_cells_num*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_message_td, c_send_td, pcinfo.x_cells_num*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(send_message_bu, c_send_bu, pcinfo.x_cells_num*sizeof(double), cudaMemcpyDeviceToHost);

    int send_amount = send_loa_messages();

    int recv_amount = receive_loa_messages();

    ret = MPI_Waitall(
        recv_amount,
        recv_loa_reqs,
        MPI_STATUS_IGNORE 
    );

    if (ret != MPI_SUCCESS) throw DPS_exception("Error waiting for recv's in compute_loa.");

    cudaMemcpy(c_recv_lr, recv_message_lr, pcinfo.y_cells_num*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_recv_rl, recv_message_rl, pcinfo.y_cells_num*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_recv_td, recv_message_td, pcinfo.x_cells_num*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(c_recv_bu, recv_message_bu, pcinfo.x_cells_num*sizeof(double), cudaMemcpyHostToDevice);

    cuda_compute_loa(df, f);

    ret = MPI_Waitall(
        send_amount, 
        send_loa_reqs,
        MPI_STATUS_IGNORE
    );

    if (ret != MPI_SUCCESS) throw DPS_exception("Error waiting for sends after last compute_loa.");
}

void DPS::compute_r(double* r, double* dp) {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            r[j * pcinfo.x_cells_num + i] =
                dp[j * pcinfo.x_cells_num + i] -
                F(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + j));
        }
    }
}

void DPS::compute_g(double* g, double* r, double alpha) {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            g[j * pcinfo.x_cells_num + i] = r[j * pcinfo.x_cells_num + i] - alpha * g[j * pcinfo.x_cells_num + i];
        }
    }
}

void DPS::compute_p(double tau, double* g, double* p_prev) {
    #pragma omp parallel for schedule(static) 
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            p[j * pcinfo.x_cells_num + i] = p_prev[j * pcinfo.x_cells_num + i] - tau * g[j * pcinfo.x_cells_num + i];
        }
    }
}

double DPS::compute_sprod(double* f1, double* f2) {
    double local_scalar_product = 0;

    #pragma omp parallel for schedule(static) reduction(+:local_scalar_product)
    for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
        for (int i = static_cast<int>(pcinfo.left); i < pcinfo.x_cells_num - static_cast<int>(pcinfo.right); i++){
            double hhx = hhx_(pcinfo.x_cell_pos + i);
            double hhy = hhx_(pcinfo.y_cell_pos + j);
            local_scalar_product += hhx * hhy * f1[j * pcinfo.x_cells_num + i] * f2[j * pcinfo.x_cells_num + i];
        }
    }

    double global_scalar_product = 0;

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

void DPS::allocate_message_arrays() {
    send_message_lr = new double [pcinfo.y_cells_num];
    send_message_rl = new double [pcinfo.y_cells_num];
    send_message_td = new double [pcinfo.x_cells_num];
    send_message_bu = new double [pcinfo.x_cells_num];

    recv_message_lr = new double [pcinfo.y_cells_num];
    recv_message_rl = new double [pcinfo.y_cells_num];
    recv_message_td = new double [pcinfo.x_cells_num];
    recv_message_bu = new double [pcinfo.x_cells_num];

    cudaMalloc(&c_send_lr, pcinfo.y_cells_num*sizeof(double));
    cudaMalloc(&c_send_rl, pcinfo.y_cells_num*sizeof(double));
    cudaMalloc(&c_send_td, pcinfo.x_cells_num*sizeof(double));
    cudaMalloc(&c_send_bu, pcinfo.x_cells_num*sizeof(double));

    cudaMalloc(&c_recv_lr, pcinfo.y_cells_num*sizeof(double));
    cudaMalloc(&c_recv_rl, pcinfo.y_cells_num*sizeof(double));
    cudaMalloc(&c_recv_td, pcinfo.x_cells_num*sizeof(double));
    cudaMalloc(&c_recv_bu, pcinfo.x_cells_num*sizeof(double));

}

void DPS::initialize_net_arrays(){
    for (int i = 0; i < gridsize + 1; ++i) {
        double c = (pow(1 + i * 1.0 / gridsize, q) - 1) / (pow(2, q) - 1);
        x_array[i] = xn * c + x0 * (1 - c);
        y_array[i] = yn * c + y0 * (1 - c);
    }

    for (int i = 0; i < gridsize; ++i) {
        hx_array[i] = x_array[i+1] - x_array[i]; 
        hy_array[i] = y_array[i+1] - y_array[i];
    }
    hx_array[gridsize] = 0;
    hy_array[gridsize] = 0;

    for (int i = 1; i < gridsize; ++i) {
        hhx_array[i] = 0.5 * (hx_array[i] + hx_array[i-1]); 
        hhy_array[i] = 0.5 * (hy_array[i] + hy_array[i-1]);
    } 
    hhx_array[0] = 0;
    hhy_array[0] = 0;
    hhx_array[gridsize] = 0;
    hhy_array[gridsize] = 0;
}

void DPS::initialize_border_with_zero(double* f){
    if (pcinfo.left){
        #pragma omp parallel for schedule(static) 
        for (int j = 0; j < pcinfo.y_cells_num; j++){
            f[j * pcinfo.x_cells_num + 0] = 0;
        }
    }
    if (pcinfo.right){
        #pragma omp parallel for schedule(static) 
        for (int j = 0; j < pcinfo.y_cells_num; j++){
            f[j * pcinfo.x_cells_num + (pcinfo.x_cells_num - 1)] = 0;
        }
    }
    if (pcinfo.top){
        #pragma omp parallel for schedule(static) 
        for (int i = 0; i < pcinfo.x_cells_num; i++){
            f[0 * pcinfo.x_cells_num + i] = 0;
        }
    }
    if (pcinfo.bottom){
        #pragma omp parallel for schedule(static) 
        for (int i = 0; i < pcinfo.x_cells_num; i++){
            f[(pcinfo.y_cells_num - 1) * pcinfo.x_cells_num + i] = 0;
        }
    }
}

void DPS::initialize_with_border_function(double* f){
    #pragma omp parallel
    {
        if (pcinfo.left){
            #pragma omp for schedule (static)
            for (int j = 0; j < pcinfo.y_cells_num; j++){
                double value = phi(x_(pcinfo.x_cell_pos + 0), y_(pcinfo.y_cell_pos + j));
                f[j * pcinfo.x_cells_num + 0] = value;
            }
        }
        if (pcinfo.right){
            #pragma omp for schedule (static)
            for (int j = 0; j < pcinfo.y_cells_num; j++){
                double value = phi(x_(pcinfo.x_cell_pos + (pcinfo.x_cells_num - 1)), y_(pcinfo.y_cell_pos + j));
                f[j * pcinfo.x_cells_num + (pcinfo.x_cells_num - 1)] = value;
            }
        }
        if (pcinfo.top){
            #pragma omp for schedule (static)
            for (int i = 0; i < pcinfo.x_cells_num; i++){
                double value = phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + 0));
                f[0 * pcinfo.x_cells_num + i] = value;
            }
        }
        if (pcinfo.bottom){
            #pragma omp for schedule (static)
            for (int i = 0; i < pcinfo.x_cells_num; i++){
                double value = phi(x_(pcinfo.x_cell_pos + i), y_(pcinfo.y_cell_pos + (pcinfo.y_cells_num - 1)));
                f[(pcinfo.y_cells_num - 1) * pcinfo.x_cells_num + i] = value;
            }
        }

        #pragma omp for schedule (static)
        for (int j = static_cast<int>(pcinfo.top); j < pcinfo.y_cells_num - static_cast<int>(pcinfo.bottom); j++){
            memset(&(f[j * pcinfo.x_cells_num + static_cast<int>(pcinfo.left)]), 0,
                (pcinfo.x_cells_num - static_cast<int>(pcinfo.right) - static_cast<int>(pcinfo.left)) * sizeof(*f));
        }
    }
}

void DPS::printcuda(double* array, int size) {
    double* newarray = new double[size/sizeof(double)];
    cudaMemcpy(newarray, array, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size/sizeof(double); ++i) {
        std::cout << newarray[i] << " ";
    }
    delete newarray;
    std::cout << std::endl;
}

void DPS::PrintP(string& dir_name) {
    std::stringstream ss;
    ss << "./" << dir_name << "/fa" << pmp.rank << ".txt";
    std::fstream fout(ss.str().c_str(), std::fstream::out);

    for (int j = 0; j < pcinfo.y_cells_num; ++j) {
        for (int i = 0; i < pcinfo.x_cells_num; ++i) {
            fout << x_(pcinfo.x_cell_pos + i) << " " << y_(pcinfo.y_cell_pos + j) 
                 << " " << p[j * pcinfo.x_cells_num + i] << std::endl;
        }
    }

    fout.close();
}

