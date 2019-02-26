#define MAX_CPU_NUMBER 4

// ARM assembly functions use -mfloat-abi=hard calling convention, Android does
// not, unless armeabi-v7a-hard is specified in Application.mk, but this is not
// supported anymore
#if defined ANDROID && defined __NEON__
#define FCALL __attribute__((pcs("aapcs-vfp")))
#else
#define FCALL
#endif

#define ZERO 0

int sgemm_tcopy_4(long m, long n, float *a, long lda, float *b);
int sgemm_ncopy_4(long m, long n, float *a, long lda, float *b);

int FCALL sgemm_kernel(long m, long n, long k, float alpha, float *sa,
                       float *sb, float *c, long ldc);
