// ######################################################
// ##  姓名: 刘羽丰
// ##  文件说明: 第四版，这版将log数组放到device全局存储。
// ######################################################
#include <stdio.h>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h> 
#include <cmath>

// 声明全局内存
__device__ float mylog[25];

// ######################################################
// ##  INIT函数,将主机端的两数组进行初始化
// ##  A: 输入的矩阵,初始化时随机0-15
// ##  B: 输出的矩阵,初始化时初始为0
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
// ##  A_host: 输入的矩阵
// ##  B_host: 输出的矩阵
// ##  rows,cols分别为矩阵的高和宽
// ######################################################
__global__ void cal_entropy(int *A, float *B, int rows, int cols){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < rows*cols){
        // 变换得到坐标
        int row = idx / cols;
        int col = idx - row*cols;
        // printf("%d %f\n",idx,mylog[idx]);
        // 避免大量重复if，计算窗口四边界
        int up = max(row-2,0);
        int down = min(row+2,rows-1);
        int left = max(col-2,0);
        int right = min(col+2,cols-1);
        // 利用char存储降低寄存器压力
        char digit[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // printf("(%d,%d) l:%d,r:%d,u:%d,d:%d\n",row,col,left,right,up,down);
        // 根据窗口四边界可直接得到窗口大小
        int count = (right-left+1)*(down-up+1);

        // 记录熵
        float result = 0;
        // 遍历周围元素并统计
        for(int i = up; i <= down; i++){
            for(int j = left; j <= right; j++){
                ++digit[A[i*cols+j]];
            }
        }
        // 计算熵
        for(int i = 0; i < 16; i++){
            if(digit[i] != 0){
                result += (((float)digit[i])/count)*(mylog[digit[i]-1]-mylog[count-1]);
            }
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
    // log数组
    const float my_log[25] = {
        0.000000,
        0.693147,
        1.098612,
        1.386294,
        1.609438,
        1.791759,
        1.945910,
        2.079442,
        2.197225,
        2.302585,
        2.397895,
        2.564949,
        2.639057,
        2.708050,
        2.772589,
        2.833213,
        2.890372,
        2.944439,
        2.995732,
        3.044522,
        3.091042,
        3.135494,
        3.178054};
    // 拷贝log数组到全局内存中
    cudaMemcpyToSymbol(mylog, (const float*)my_log, sizeof(my_log));
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