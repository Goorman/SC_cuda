#ifndef DPS_h
#define DPS_h

#include <stdio.h>
#include <mpi.h>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

using std::string;

class DPS_exception: public std::exception
{
private:
    string error;
public:
    DPS_exception(const string& error_): 
        error (error_) {}

    virtual ~DPS_exception() throw() {}

    virtual const char* what() const throw()
    {
        return error.c_str();
    }
};

struct ProcessorMPIParameters {
    int rank;
    int size;
    MPI_Comm comm;

        public:
    ProcessorMPIParameters(MPI_Comm comm_ = MPI_COMM_WORLD){
        comm = comm_;
        MPI_Comm_rank (comm, &rank); 
        MPI_Comm_size (comm, &size);
    }
};

struct ProcessorCellsInfo {
    int x_proc_num;
    int y_proc_num;

    int x_cells_num;
    int x_cell_pos;
    int y_cells_num;
    int y_cell_pos;

    // These parameters are True if processor's cell rectangle touches the border.
    bool top;
    bool bottom;
    bool left;
    bool right;

    int cuda_threadsPerBlock;
    int cuda_blocksNum;

    ProcessorCellsInfo ();
    ProcessorCellsInfo (int rank,
                         int grid_size_x, int grid_size_y, 
                         int x_proc_num_, int y_proc_num_);

    cudaDeviceProp devProp;
};

// DPS - Dirichlet Problem Solver
class DPS {
public:
    DPS (int gridsize_);
    ~DPS ();
    void ComputeApproximation(ProcessorMPIParameters& pmp);

    void PrintP(string& out_dir) ;

    double getFinalError() ;
    int getProcessorNumber() ;
    int getIterationNumber() ;

private:
    ProcessorMPIParameters pmp;
    ProcessorCellsInfo pcinfo;

    double* p;
    //cuda counterpart
    double* c_p;

    double* x_array;
    double* y_array;
    double* hx_array;
    double* hy_array;
    double* hhx_array;
    double* hhy_array;
    //cuda counterparts
    double* c_x_array;
    double* c_y_array;
    double* c_hx_array;
    double* c_hy_array;
    double* c_hhx_array;
    double* c_hhy_array;

    int gridsize;
    double eps;
    double q;
    int x0;
    int xn;
    int y0;
    int yn;
    int iterations_counter;
    double final_error;

    MPI_Comm PrepareMPIComm (ProcessorMPIParameters& pmp, 
                             int x_proc_num, 
                             int y_proc_num) ;

    double F(double x, double y) ;
    double phi(double x, double y) ;
    double x_(int i) ;   // i from 0 to gridsize
    double y_(int i) ;   // i from 0 to gridsize
    double hx_(int i) ;  // i from 0 to gridsize - 1
    double hy_(int i) ;  // i from 0 to gridsize - 1
    double hhx_(int i) ; // i from 1 to gridsize - 1
    double hhy_(int i) ; // i from 1 to gridsize - 1

    double compute_error() ;
    double compute_maxnorm(double* f1, double* f2) ;
    bool stop_condition(double* f1, double* f2) ;
    void compute_grid(ProcessorMPIParameters& pmp, 
                      int& x_proc_num, int& y_proc_num);

    int receive_loa_messages();
    int send_loa_messages();
    void compute_loa(double* d_f, double* f);
    double compute_sprod(double* f1, double* f2) ;
    void compute_r(double* r, double* d_p) ;
    void compute_g(double* g, double* r, double alpha) ;
    void compute_p(double tau, double* g, double* p_prev);

    void cuda_compute_r(double* r, double* dp);
    void cuda_compute_g(double* g, double* r, double alpha);
    void cuda_compute_p(double tau, double* g, double* p_prev);
    double cuda_compute_sprod(double* f1, double* f2);
    double cuda_compute_maxnorm(double* f1, double* f2);
    bool cuda_stop_condition(double* f1, double* f2);
    void cuda_compute_loa(double* d_f, double* f);
    void cuda_form_messages(double* f);

    void allocate_message_arrays();
    void initialize_net_arrays();
    void initialize_border_with_zero(double* f);
    void initialize_with_border_function(double* f);

    void printcuda(double* array, int size);

    double* send_message_lr;
    double* send_message_rl;
    double* send_message_td;
    double* send_message_bu;
    double* recv_message_lr;
    double* recv_message_rl;
    double* recv_message_td;
    double* recv_message_bu;
    //cuda counterparts
    double* c_send_lr;
    double* c_send_rl;
    double* c_send_td;
    double* c_send_bu;
    double* c_recv_lr;
    double* c_recv_rl;
    double* c_recv_td;
    double* c_recv_bu;

    double* product;
    double* norm;
    double* answer;

    MPI_Request* recv_loa_reqs;
    MPI_Request* send_loa_reqs;

    enum MPI_MessageTypes {
        LoaLeftRight,
        LoaRightLeft,
        LoaTopDown,
        LoaBottomUp,
        DumpSync
    };
};

#endif /* DPS_h */
