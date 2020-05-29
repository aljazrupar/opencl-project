#ifndef MTX_SPARSE
#define MTX_SPARSE

#include <stdio.h>


struct mtx_COO  // COOrdinates
{
    int *row;
    int *col;
    double *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_CSR  // Compressed Sparse Row
{
    int *rowptr;
    int *col;
    double *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_ELL      // ELLiptic (developed by authors of ellipctic package)
{
    int *col;
    double *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int num_elements;
    int num_elementsinrow;    
};

struct mtx_JDS
{
    int *col;
    double *data;
    int *jagged_ptr;
    int *row_permute;
    int max_elementsinrow;
    int data_arr_length;
    int num_of_jags_nonzero; // num of jags
    int size_of_jaggged_ptr; // size of jagged_ptr -1
    int jds_rows; // number of rows stored in JDS.
    int jag_padd;
    int num_nonzeros; // size of data and col.
    int num_rows; // all row
    int num_cols; // all col
};

int mtx_COO_create_from_file(struct mtx_COO *mCOO, FILE *f);
int mtx_COO_free(struct mtx_COO *mCOO);

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO);
int mtx_CSR_free(struct mtx_CSR *mCSR);

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR);
int mtx_ELL_free(struct mtx_ELL *mELL);

int mtx_JDS_create_from_mtx_CSR(struct mtx_JDS *mJDS, struct mtx_CSR *mCSR, int JAGPADD);
int mtx_JDS_free(struct mtx_JDS *mJDS);

#endif