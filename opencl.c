#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include "mtx_sparse.h"
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 256
#define REPEAT 1000

#define MAX_S 10
#define MIN_S -10
#define PRECISION 1e-10

#define JAGPADD 1
// finite number of iterations, which is not larger than the size of the matrix
// gcc SpMV_cl.c mtx_sparse.c -fopenmp -O2 -I/usr/include/cuda -L/usr/lib64 -l:"libOpenCL.so.1" -o ou


// TODO------------- Spremeni vector v rand()!!!!
int generate_vector_s(double *vector, int vec_len){
    
    for(int i = 0; i < vec_len; i++){
        double new_num = rand() % (MAX_S + 1 - MIN_S) + MIN_S;
        vector[i] = i;
    }
    return 0;
}
int matrix_vector_product(struct mtx_ELL mELL, double *vector_s, double *vector_b){
    int el_in_row = mELL.num_elementsinrow;
    for(int i = 0; i < mELL.num_rows; i++){ // Gre se enkrat cez za "brezveze". Mogoce bols z calloc nardit vsakic
        vector_b[i] = 0;
    }
    for(int i = 0; i < mELL.num_rows; i++){
        for(int j = 0; j < el_in_row; j++){
            int col_idx = mELL.col[j*mELL.num_rows + i];
            // printf("data-> %lf   s-> %lf\n",mELL.data[i*el_in_row + j], vector_s[col_idx]);
            vector_b[i] += mELL.data[j*mELL.num_rows + i] * vector_s[col_idx];
        }
    }
    return 0;
}

int vector_vector_minus(double *vector1, double *vector2, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = vector1[i] - vector2[i];
    }
    return 0;
}

double vectorT_vector_product(double *vector_1, double *vector_2, int len){
    double product = 0;
    for(int i = 0; i < len; i++){
        product += vector_1[i] *vector_2[i];
    }
    return product;
}

int copy_vector(double *vector1, double *vector2, int len){
    for(int i = 0; i < len; i++){
        vector2[i] = vector1[i];
    }
    return 0;
}

int scalar_vector_product(double coef, double *vector, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = coef * vector[i];
    }
    return 0;
}

int vector_vector_plus(double *vector1, double *vector2, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = vector1[i] + vector2[i];
    }
    return 0;
}

int print_vector(double *vector, int len){
    for(int i = 0; i < len; i++){
        printf("%ld\n", vector[i]);
    }
    return 0;
}




// OPENCL functions
int matrix_vector_product_CL(struct mtx_JDS mJDS, double *vector_x, double *vector_temp, cl_context context, cl_command_queue command_queue, cl_kernel kernelELL){
    cl_int clStatus;

    cl_mem vecIn = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                mJDS.num_cols * sizeof(cl_double), vector_x, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, vecIn, CL_TRUE, 0,						
                                    mJDS.num_cols*sizeof(cl_double), vector_x, 0, NULL, NULL);	

    cl_mem vecOut = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                    mJDS.num_cols * sizeof(cl_double), NULL, &clStatus);
    clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void *)&vecIn);
    // printf("cl stat7-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_mem), (void *)&vecOut);
    // printf("cl stat8-> %d\n", clStatus);

    int curr_jag_size = 0;
    int low_limit = 0;
    int upper_limit = 0;
    int curr_jag_rows = 0;
    int count_row_permute = 0;
    int count_len = mJDS.max_elementsinrow; // count for every iteration, max len = 4, count_len = 4,3,2,1
    for(int i = 0; i < mJDS.size_of_jaggged_ptr; i++){
        // printf("count_len--: %d\n", count_len);
        low_limit = mJDS.jagged_ptr[i];
        upper_limit = mJDS.jagged_ptr[i+1];
        curr_jag_size = upper_limit - low_limit; // 8, 4
        if(curr_jag_size == 0){
            count_len--;
            continue;
        }
        // printf("curr_jag_size: %d\n",curr_jag_size);
        curr_jag_rows = curr_jag_size / count_len; // 2 vsakic v mojem primeru.
        // printf("curr_jag_rows: %d\n", curr_jag_rows);
        
        clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void *)&low_limit);
        // printf("cl stat9-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 6, sizeof(cl_int), (void *)&count_row_permute);
        // printf("cl stat10-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 7, sizeof(cl_int), (void *)&curr_jag_rows); // mELL.num_rows
        // printf("cl stat11-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 8, sizeof(cl_int), (void *)&count_len); // mELL.num_elementsinrow
        // printf("cl stat12-> %d\n", clStatus);
        size_t local_item_size = WORKGROUP_SIZE; // 1024
        int num_groups = ((curr_jag_rows - 1) / local_item_size + 1);
        size_t global_item_size_ELL = num_groups * local_item_size; // 1024

        clStatus = clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL,						
                                        &global_item_size_ELL, &local_item_size, 0, NULL, NULL);	
        // printf("cl stat12-> %d\n", clStatus);
        count_row_permute += curr_jag_rows;
        count_len--;
    }

    clStatus = clEnqueueReadBuffer(command_queue, vecOut, CL_TRUE, 0,				
                                    mJDS.num_rows*sizeof(cl_double), vector_temp, 0, NULL, NULL);
    // printf("cl stat6-> %d\n", clStatus);

    clStatus = clReleaseMemObject(vecIn);
    clStatus = clReleaseMemObject(vecOut);
    
}

double vectorT_vector_product_CL(double *vector_1, double *vector_2, int len, cl_context context, cl_command_queue command_queue, cl_kernel kernel_dot_product){
    cl_int clStatus;
    size_t local_item_size = WORKGROUP_SIZE;
	int num_groups = ((len-1)/local_item_size+1);		
    size_t global_item_size = num_groups*local_item_size;

    cl_mem vec1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                len*sizeof(double), vector_1, &clStatus);
    
    cl_mem vec2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                len*sizeof(double), vector_2, &clStatus);
    
    cl_mem p_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
								num_groups*sizeof(double), NULL, &clStatus);

    clStatus  = clSetKernelArg(kernel_dot_product, 0, sizeof(cl_mem), (void *)&vec1);
    clStatus |= clSetKernelArg(kernel_dot_product, 1, sizeof(cl_mem), (void *)&vec2);
    clStatus |= clSetKernelArg(kernel_dot_product, 2, sizeof(cl_mem), (void *)&p_d);
    clStatus |= clSetKernelArg(kernel_dot_product, 3, sizeof(cl_int), (void *)&len);
	clStatus |= clSetKernelArg(kernel_dot_product, 4, local_item_size*sizeof(double), NULL);

    clStatus = clEnqueueNDRangeKernel(command_queue, kernel_dot_product, 1, NULL,						
								      &global_item_size, &local_item_size, 0, NULL, NULL);

    double *p = (double*)malloc(num_groups*sizeof(double));
    clStatus = clEnqueueReadBuffer(command_queue, p_d, CL_TRUE, 0,						
							       num_groups*sizeof(double), p, 0, NULL, NULL);	
    
	double dotProductOpenCL = 0.0;
    for(int i = 0; i < num_groups; i++){
		dotProductOpenCL += p[i];
    }

    return dotProductOpenCL;
}


int main(int argc, char *argv[]) // argv -> 0: matrix, 1: kernel, 3: precision
{
    FILE *f;
    struct mtx_COO mCOO;
    struct mtx_CSR mCSR;
    struct mtx_ELL mELL;
    struct mtx_JDS mJDS;
	cl_int clStatus;
    int repeat;
    printf("Start\n");
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        
        if ((f = fopen(argv[1], "r")) == NULL){ 
            printf("Unable to open file\n");
            exit(1);
        }
    }
    // create sparse matrices
    if (mtx_COO_create_from_file(&mCOO, f) != 0){
        printf("Error reading file.");
        exit(1);
    }
    
    mtx_CSR_create_from_mtx_COO(&mCSR, &mCOO);
    mtx_ELL_create_from_mtx_CSR(&mELL, &mCSR);
    mtx_JDS_create_from_mtx_CSR(&mJDS, &mCSR, JAGPADD);


    // START OPENCL:
    printf("Start OPENCL\n");

    FILE *fp;
    char fileName[100];
    char *source_str;
    size_t source_size;
    sprintf(fileName,"%s.cl", argv[2]);
    fp = fopen(fileName, "r");
    if (!fp) 
	{
		fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose( fp );
    
    // Get platforms
    cl_uint num_platforms;
    clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    
    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    // Context
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);
 
    // Command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

    // Create and build a program
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char *build_log;
	clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return 1;
    }

    // number of iterations must be size of matrix
    // int iter = mJDS.num_cols * mJDS.num_rows;
    int iter = 1;
    double *vector_s = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_b = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_temp= (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_r = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_Ap = (double *)calloc(mELL.num_cols, sizeof(double)); // A*p(k)
    double *vector_beta_num = (double *)calloc(mELL.num_cols, sizeof(double));

    double *vector_alpha_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_beta_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_alpha_A_p = (double *)calloc(mELL.num_cols, sizeof(double));

    double *vecOutJDS = (double *)calloc(mJDS.num_rows, sizeof(double));
    double *vector_zeros = (double *)calloc(mELL.num_rows, sizeof(double));

    generate_vector_s(vector_s, mELL.num_cols);
    matrix_vector_product(mELL, vector_s, vector_b); //A*s, generate vector b.
    
    // 2. Inicialize starting vector of ones X0
    double *vector_x = (double *)calloc(mELL.num_cols, sizeof(double));
    for(int i = 0; i < mELL.num_cols; i++){
        vector_x[i] = 1;
    }

    double precision_curr = 0;
    double coef_alpha = 0;
    double coef_alpha_denom = 0;
    double coef_beta = 0;
    double coef_beta_num = 0;
    // transfer data, col, row_permute and create kernel mELLxVec
    cl_mem mJDS_col = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                      mJDS.data_arr_length * sizeof(cl_int), NULL, &clStatus);
    cl_mem mJDS_data = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                       mJDS.data_arr_length * sizeof(cl_double), NULL, &clStatus);

    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_col, CL_TRUE, 0,						
                                    mJDS.data_arr_length * sizeof(cl_int), mJDS.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_data, CL_TRUE, 0,						
                                    mJDS.data_arr_length * sizeof(cl_double), mJDS.data, 0, NULL, NULL);				
    cl_mem mJDS_row_permute = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mJDS.jds_rows * sizeof(cl_int), NULL, &clStatus); 
    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_row_permute, CL_TRUE, 0,						
                                    mJDS.jds_rows * sizeof(cl_int), mJDS.row_permute, 0, NULL, NULL);

    cl_kernel kernelELL = clCreateKernel(program, "mELLxVec", &clStatus);
    clStatus |= clSetKernelArg(kernelELL, 0, sizeof(cl_mem), (void *)&mJDS_col);
    // printf("cl stat4-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void *)&mJDS_data);
    // printf("cl stat5-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void *)&mJDS_row_permute);
    // printf("cl stat6-> %d\n", clStatus);

    //kernel for dot product
    cl_kernel kernel_dot_product = clCreateKernel(program, "dotProduct", &clStatus);

    // matrix_vector_product(mELL, vector_x, vector_temp); // A*x
    matrix_vector_product_CL(mJDS, vector_x, vector_temp, context, command_queue, kernelELL); // dela.
    
    
    vector_vector_minus(vector_b, vector_temp, mJDS.num_cols, vector_r); // r = b- A*x
    
    copy_vector(vector_r, vector_p, mJDS.num_cols); // r -> p

    int k = 0;
    iter = mJDS.num_cols * mJDS.num_rows;
    while(k < iter){
        precision_curr = vectorT_vector_product_CL(vector_r, vector_r, mJDS.num_cols, context, command_queue, kernel_dot_product); // dela
        if(precision_curr <= PRECISION){
            break;
        }

        matrix_vector_product_CL(mJDS, vector_p, vector_Ap, context, command_queue, kernelELL); // dela
        /* for(int i = 0; i < mJDS.num_cols; i++){
            printf("AP-> %lf\n", vector_Ap[i]);
        } */
        coef_alpha_denom = vectorT_vector_product_CL(vector_p, vector_Ap, mJDS.num_cols, context, command_queue, kernel_dot_product); // dela
        coef_alpha = precision_curr / coef_alpha_denom;

        scalar_vector_product(coef_alpha, vector_p, mJDS.num_cols, vector_alpha_p);
        vector_vector_plus(vector_x, vector_alpha_p, mJDS.num_cols, vector_x);

        scalar_vector_product(coef_alpha, vector_Ap, mJDS.num_cols, vector_alpha_A_p);
        vector_vector_minus(vector_r, vector_alpha_A_p, mELL.num_cols, vector_r);


        coef_beta_num = vectorT_vector_product_CL(vector_r, vector_r, mJDS.num_cols, context, command_queue, kernel_dot_product);
        coef_beta = coef_beta_num / precision_curr;

        scalar_vector_product(coef_beta, vector_p, mJDS.num_cols, vector_beta_p);
        vector_vector_plus(vector_r, vector_beta_p, mJDS.num_cols, vector_p);

        k++;
    }

    for(int i = 0; i < mELL.num_cols; i++){
        printf("%lf -- %lf\n",vector_s[i],vector_x[i]);
    }


    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    clStatus = clReleaseKernel(kernelELL);
    clStatus = clReleaseProgram(program);
    
    clStatus = clReleaseMemObject(mJDS_col);
    clStatus = clReleaseMemObject(mJDS_data);
    clStatus = clReleaseMemObject(mJDS_row_permute);

    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
	free(devices);
    free(platforms);


    mtx_COO_free(&mCOO);
    mtx_CSR_free(&mCSR);
    mtx_ELL_free(&mELL);

	return 0;
}
