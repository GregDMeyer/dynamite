#pragma once

#define CONCAT_U(a, b) a ## _ ## b
#define C(a, b) CONCAT_U(a, b)

#define GPU_BLOCK_SIZE 1024
#define GPU_BLOCK_NUM 1024
