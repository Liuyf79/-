// ######################################################
// ##  姓名: 刘羽丰
// ##  文件说明: 初始版本,没有优化,作为后续对比
// ######################################################
#include <stdio.h>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h> 
#include <cmath>

// ######################################################
// ##  INIT函数,将主机端的两数组进行初始化
// ##  A_host: 输入的矩阵,初始化时随机0-15
// ##  B_host: 输出的矩阵,初始化时初始为0
// ##  ROWS,COLS分别为矩阵的高和宽
// ######################################################
void INIT(int* A_host,float* B_host,int ROWS,int COLS){
    // srand(time(NULL));
    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            A_host[i*COLS+j] = rand() % 16;
            B_host[i*COLS+j] = 0;
        }
    }
}

// ######################################################
// ##  核函数,计算二维数组中以每个元素为中心的熵
// ##  A: 输入的矩阵
// ##  B: 输出的矩阵
// ##  rows,cols分别为矩阵的高和宽
// ######################################################
__global__ void cal_entropy(int *A, float *B, int rows, int cols){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < rows*cols){
        // 变换得到坐标
        int row = idx / cols;
        int col = idx % cols; 
        // dight数组,统计
        int digit[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
        // 该元素周围元素数量
        int count  = 0;
        // 记录熵
        float result = 0;
        // 中间值
        float temp = 0;
        // 遍历周围元素并统计
        for(int i = row - 2; i <= row + 2; i++){
            for(int j = col - 2; j <= col + 2; j++){
                if(i < 0 || i >= rows || j < 0 || j >= cols)
                    continue;
                ++digit[A[i*cols+j]];
                ++count;
            }
        }
        // 计算熵
        for(int i = 0; i < 16; i++)
            if(digit[i] != 0){
                temp = float(digit[i])/count;
                result += temp*logf(temp);
            }
        // 读入到输出矩阵
        B[idx] = -result;
        // printf("c:%d idx:%d r:%f\n",count,idx,B[idx]);
    }
}

//主函数
int main(int argc,char *argv[])
{   
    int i;
    // 矩阵高度
    int ROWS = 5;
    // 矩阵宽度
    int COLS = 5;
    // 一个块的线程数
    int block = 256;

    for(i = 1; i < argc; i++)
    {
        if(i == 1){
            ROWS = atoi(argv[i]);
        }
        else if(i == 2){
            COLS = atoi(argv[i]);
        }
        else if(i == 3){
            block = atoi(argv[i]);
        }
    }
    // 输入的二维数组,值为0-15
    int Bytes = ROWS*COLS*sizeof(int);
    // 输出的二维数组,元素类型为float
    int FBytes = ROWS*COLS*sizeof(float);
    //开辟主机内存
    int* A_host = (int*)malloc(Bytes);
    float* B_host = (float*)malloc(FBytes);
    // 初始化
    INIT(A_host,B_host,ROWS,COLS);

    //开辟设备内存
    int* A_dev = NULL;
    float* B_dev = NULL;

    cudaMalloc((void**)&A_dev, Bytes);
    cudaMalloc((void**)&B_dev, FBytes);
    //输入数据从主机内存拷贝到设备内存
    cudaMemcpy(A_dev, A_host, Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, FBytes, cudaMemcpyHostToDevice);

    //GPU计时
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //运行程序
    cal_entropy<<<(COLS*ROWS-1)/block+1,block>>>(A_dev, B_dev, ROWS, COLS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //输出GPU执行时间
    printf("gpu_time:%fms\n",elapsedTime);


    //释放内存
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(B_host);
    free(A_host);
    return 0;
}