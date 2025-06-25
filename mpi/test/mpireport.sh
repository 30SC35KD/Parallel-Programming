#!/bin/bash
# 完整可执行的 mpi 分析脚本

# 配置日志文件路径
LOG_FILE="mpi_profile.log"
PROGRAM="./ntt_mpi"  # 替换为你的程序路径

# 检查程序是否存在
if [ ! -f "$PROGRAM" ]; then
    echo "错误：程序 $PROGRAM 不存在！"
    exit 1
fi

# 设置环境变量
export MPI_PI_LOGFILE="$LOG_FILE"
export LD_PRELOAD=$(mpicc --showme:libdirs)/libmpi.so
export MPI_PI_MODE=collect

# 运行程序
echo "正在运行 MPI 程序，日志将保存至 $LOG_FILE..."
mpirun -n 4 "$PROGRAM"

# 检查日志是否生成
if [ -f "$LOG_FILE" ]; then
    echo "日志文件已生成："
    cat "$LOG_FILE"
else
    echo "错误：未能生成日志文件 $LOG_FILE！"
    echo "请检查 MPI 程序是否正常运行，或尝试手动指定 LD_PRELOAD 路径。"
fi

# 清理环境变量
unset LD_PRELOAD
unset MPI_PI_MODE