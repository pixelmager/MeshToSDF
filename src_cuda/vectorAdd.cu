/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <assert.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
///__global__ void
///vectorAdd(const float *A, const float *B, float *C, int numElements)
///{
///    int i = blockDim.x * blockIdx.x + threadIdx.x;
///
///    if (i < numElements)
///    {
///        C[i] = A[i] + B[i];
///    }
///}

// ==================================================================
//note: reference https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//                https://docs.nvidia.com/nsight-visual-studio-edition/cuda-debugger/
//                https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

typedef float float32_t;

//note: from https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool AlmostEqualRelative(float A, float B, float maxRelDiff = FLT_EPSILON)
{
	// Calculate the difference.
	float diff = fabs(A - B);
	A = fabs(A);
	B = fabs(B);
	// Find the largest
	float largest = (B > A) ? B : A;

	if (diff <= largest * maxRelDiff)
		return true;
	return false;
}


LARGE_INTEGER currentTime;
LARGE_INTEGER m_timer_frequency;
void init_timers()
{
	QueryPerformanceFrequency( &m_timer_frequency );
}
uint64_t gettime_ms()
{
	QueryPerformanceCounter(&currentTime);
	return static_cast<uint64_t>( static_cast<double>(currentTime.QuadPart) / static_cast<double>(m_timer_frequency.QuadPart) * 1000.0 );
}

// =================================================

struct vec3_t
{
	float32_t x, y, z;
	//vec3_t(){}
	__host__ __device__ vec3_t( float32_t in_x, float32_t in_y, float32_t in_z ) : x(in_x), y(in_y), z(in_z) {}
	__host__ __device__ vec3_t( const vec3_t &v ) : x(v.x), y(v.y), z(v.z) {}
	//explicit vec3_t( __m128 v ) : x(v.m128_f32[0]), y(v.m128_f32[1]), z(v.m128_f32[2] ) {}
};

__host__ __device__
float32_t dot( const vec3_t &a, const vec3_t &b )
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__
vec3_t operator+( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.x + b.x,
				   a.y + b.y,
				   a.z + b.z );
}
__host__ __device__
vec3_t sub( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.x - b.x,
				a.y - b.y,
				a.z - b.z );
}
__host__ __device__
vec3_t operator-( const vec3_t &a, const vec3_t &b )
{
	return sub(a, b);
}
__host__ __device__
vec3_t mul( const vec3_t &a, const vec3_t &b )
{
	return vec3_t(  a.x * b.x,
					a.y * b.y,
					a.z * b.z );
}
__host__ __device__
vec3_t operator*( const vec3_t &a, const vec3_t &b )
{
	return mul(a, b);
}
__host__ __device__
vec3_t operator*( const vec3_t &v, const float32_t c )
{
	return vec3_t( v.x * c,
				   v.y * c,
				   v.z * c );
}
__host__ __device__
vec3_t operator*( const float32_t c, const vec3_t &v )
{
	return vec3_t( v.x * c,
				   v.y * c,
				   v.z * c );
}
__host__ __device__
vec3_t cross( const vec3_t &a, const vec3_t &b )
{
	return vec3_t( a.y*b.z - a.z*b.y,
				   a.z*b.x - a.x*b.z,
				   a.x*b.y - a.y*b.x );
}
__host__ __device__
vec3_t min(const vec3_t& v0, const vec3_t& v1)
{
	return vec3_t(min(v0.x, v1.x),
		min(v0.y, v1.y),
		min(v0.z, v1.z));
}
__host__ __device__
vec3_t max(const vec3_t& v0, const vec3_t& v1)
{
	return vec3_t(max(v0.x, v1.x),
		max(v0.y, v1.y),
		max(v0.z, v1.z));
}


// =================================================

__host__ __device__
float32_t sign( const float32_t v )
{
	//TODO: return sign bit?
	//return (v>=0.0f) ? 1.0f : -1.0f;

	// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/sign.xhtml
	if ( v == 0.0f )
		return 0.0f;
	return (v>0.0f) ? 1.0f : -1.0f;
}

__host__ __device__
float32_t clamp( float32_t v, float32_t mn, float32_t mx )
{
	return max(mn, min(mx, v));
}

// =================================================

struct sdf_t
{
	struct sdfheader_t
	{
			int32_t dim_x, dim_y, dim_z;
			float32_t bb_mn_x, bb_mn_y, bb_mn_z;
			float32_t bb_mx_x, bb_mx_y, bb_mx_z;
	} header;
	
    float32_t *d_data;
	float32_t *h_data;
};
struct aabb_t
{
	vec3_t mn;
	vec3_t mx;
	aabb_t() : mn(FLT_MAX,FLT_MAX,FLT_MAX), mx(-FLT_MAX,-FLT_MAX,-FLT_MAX) {}
};

void init_sdf( sdf_t *sdf, aabb_t bb, int32_t siz_x, int32_t siz_y, int32_t siz_z )
{
	//enum { SIMD_SIZ = 16, SIMD_ALIGN=4*SIMD_SIZ };
	//const int32_t simd_align = sizeof(float32_t) * simd_siz;

	sdf->header.dim_x = siz_x;
	sdf->header.dim_y = siz_y;
	sdf->header.dim_z = siz_z;

	//note: extend bb by a border
	const int32_t border_siz = sdf->header.dim_x/4;
	const vec3_t bb_gridcell_siz = (bb.mx - bb.mn) * vec3_t( 1.0f/sdf->header.dim_x, 1.0f/sdf->header.dim_y, 1.0f/sdf->header.dim_z );
	bb.mn = bb.mn - (float32_t)border_siz * bb_gridcell_siz;
	bb.mx = bb.mx + (float32_t)border_siz * bb_gridcell_siz;

	sdf->header.bb_mn_x = bb.mn.x;
	sdf->header.bb_mn_y = bb.mn.y;
	sdf->header.bb_mn_z = bb.mn.z;

	sdf->header.bb_mx_x = bb.mx.x;
	sdf->header.bb_mx_y = bb.mx.y;
	sdf->header.bb_mx_z = bb.mx.z;

	sdf->h_data = (float32_t*)_aligned_malloc( sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z, 16 );

    sdf->d_data = NULL;
    cudaError err = cudaMalloc((void **)&sdf->d_data, sizeof(float32_t) * sdf->header.dim_x * sdf->header.dim_y * sdf->header.dim_z );
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector sdf-data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void deinit_sdf( sdf_t *sdf )
{
	_aligned_free( sdf->h_data );
    cudaFree(sdf->d_data );
}


// =================================================


//note: host, fileread
struct itm_header_t
{
	uint64_t num_indices;
	uint64_t num_positions;
};
//uint32_t indices[num_indices];
//float32_t positions[num_positions];

struct itm_t
{
	itm_header_t header;
	uint32_t *indices;
	float32_t *positions;
};

itm_t* itm_readgeom( char const * const fn )
{
	FILE *infile;
	errno_t err = fopen_s( &infile, fn, "rb" );
	if ( infile == nullptr ) { printf("wtf2\n%i\n", err); return nullptr; }

	itm_t * const ret = (itm_t*)malloc( sizeof(itm_t) );

	fread( &ret->header, sizeof(itm_header_t), 1, infile );

	ret->indices = (uint32_t*)malloc( sizeof(uint32_t) * ret->header.num_indices );
	ret->positions = (float32_t*)malloc( sizeof(float32_t) * ret->header.num_positions );

	fread( ret->indices, sizeof(uint32_t), ret->header.num_indices, infile );
	fread( ret->positions, sizeof(float32_t), ret->header.num_positions, infile );
	fclose( infile );

	return ret;
}

//note: from http://iquilezles.org/www/articles/triangledistance/triangledistance.htm
__host__ __device__
float32_t dot2( const vec3_t &v ) { return dot(v,v); }

__host__ __device__
float32_t udTriangle_sq( const vec3_t &v1, const vec3_t &v2, const vec3_t &v3, const vec3_t &p )
{
    vec3_t v21 = v2 - v1; vec3_t p1 = p - v1;
    vec3_t v32 = v3 - v2; vec3_t p2 = p - v2;
    vec3_t v13 = v1 - v3; vec3_t p3 = p - v3;
    vec3_t nor = cross( v21, v13 );

    //return sqrtf((sign(dot(cross(v21,nor),p1)) + 
	return ((sign(dot(cross(v21,nor),p1)) + 
             sign(dot(cross(v32,nor),p2)) + 
             sign(dot(cross(v13,nor),p3))<2.0f) 
             ?
             min( min( 
             dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0f,1.0f)-p1),
             dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0f,1.0f)-p2) ),
             dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0f,1.0f)-p3) )
             :
             dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}

// =================================================

//note: naive, map every thread to a gridcell, run through all triangles per cell

__host__ __device__
float32_t calc_gridcell( const int32_t num_indices, uint32_t const * const tri_indices, const vec3_t *positions,
						 int32_t x, int32_t y, int32_t z, int32_t xn, int32_t yn, int32_t zn,
						 const vec3_t &bb_mn, const vec3_t &bb_range )
{
	vec3_t p_nm = vec3_t( (static_cast<float32_t>(x)+0.5f) / static_cast<float32_t>(xn),
						  (static_cast<float32_t>(y)+0.5f) / static_cast<float32_t>(yn),
						  (static_cast<float32_t>(z)+0.5f) / static_cast<float32_t>(zn) );
	vec3_t p = bb_mn + bb_range * p_nm;

	float d_min = FLT_MAX;
	for ( size_t idx_tri=0,num_tris=num_indices/3; idx_tri<num_tris; ++idx_tri )
	{
		const uint32_t idx0 = tri_indices[ 3*idx_tri+0 ];
		const uint32_t idx1 = tri_indices[ 3*idx_tri+1 ];
		const uint32_t idx2 = tri_indices[ 3*idx_tri+2 ];
		const vec3_t &p0 = positions[ idx0 ];
		const vec3_t &p1 = positions[ idx1 ];
		const vec3_t &p2 = positions[ idx2 ];

		float32_t ud = udTriangle_sq( p0, p1, p2, p );
		if ( ud < d_min )
			d_min = ud;
	}

	return d_min;
}

__global__
void sdf_naive( sdf_t::sdfheader_t sdfheader, const itm_header_t itmheader, const uint32_t *tri_indices, const float32_t *mesh_positions, float32_t *out_sdf_data)
{
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh_positions[0] );

	const int32_t x = threadIdx.x;
	const int32_t y = threadIdx.y;
	const int32_t z = threadIdx.z;
	
	const int32_t xn = sdfheader.dim_x;
	const int32_t yn = sdfheader.dim_y;
	const int32_t zn = sdfheader.dim_z;

	const vec3_t bb_min = vec3_t( sdfheader.bb_mn_x, sdfheader.bb_mn_y, sdfheader.bb_mn_z );
	const vec3_t bb_max = vec3_t( sdfheader.bb_mx_x, sdfheader.bb_mx_y, sdfheader.bb_mx_z );
	const vec3_t bb_range = bb_max-bb_min;

	float32_t d_min = calc_gridcell( itmheader.num_indices, tri_indices, positions,
									 x, y, z, xn, yn, zn,
		                             bb_min, bb_range);
		
    int idx = x + y*xn + z*xn*yn;
	out_sdf_data[idx] = sqrtf( d_min );
}


void invoke_naive( const itm_t &mesh, sdf_t &sdf )
{
    const int numBlocks = 1;
    dim3 threadsPerBlock( sdf.header.dim_x, sdf.header.dim_y, sdf.header.dim_z );

    uint32_t *d_indices = NULL;
    checkCudaErrors( cudaMalloc( &d_indices, sizeof(float32_t) * mesh.header.num_indices) );
    checkCudaErrors( cudaMemcpy(d_indices, mesh.indices, sizeof(float32_t)*mesh.header.num_indices, cudaMemcpyKind::cudaMemcpyHostToDevice ) );

    float32_t *d_positions = NULL;
    checkCudaErrors( cudaMalloc( &d_positions, sizeof(float32_t) * mesh.header.num_positions) );
    checkCudaErrors( cudaMemcpy(d_positions, mesh.positions, sizeof(float32_t)*mesh.header.num_positions, cudaMemcpyKind::cudaMemcpyHostToDevice ) );

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    sdf_naive<<<numBlocks, threadsPerBlock>>>(sdf.header, mesh.header, d_indices, d_positions, sdf.d_data);
	cudaEventRecord(stop);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	checkCudaErrors( cudaDeviceSynchronize() );
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf( "kernel-time %fms\n", milliseconds );
}

// =================================================


// ====

void eval_sdf__bruteforce( sdf_t &sdf, itm_t const * const mesh, float32_t *out_sdf_data )
{
	//printf("%s\n", __FUNCTION__);
	assert( mesh->header.num_positions % 3 == 0 );
	uint32_t const * const indices = reinterpret_cast<uint32_t const * const>( &mesh->indices[0] );
	vec3_t const * const positions = reinterpret_cast<vec3_t const * const>( &mesh->positions[0] );

	const vec3_t bb_min = vec3_t( sdf.header.bb_mn_x, sdf.header.bb_mn_y, sdf.header.bb_mn_z );
	const vec3_t bb_max = vec3_t( sdf.header.bb_mx_x, sdf.header.bb_mx_y, sdf.header.bb_mx_z );
	const vec3_t bb_range = bb_max-bb_min;

	for ( int z=0,zn=sdf.header.dim_z; z<zn; ++z ) {
	printf( "z=%d\n", z );
	for ( int y=0,yn=sdf.header.dim_y; y<yn; ++y ) {
	for ( int x=0,xn=sdf.header.dim_x; x<xn; ++x )
	{
		float32_t d_min = calc_gridcell( mesh->header.num_indices, indices, positions,
										 x, y, z, xn, yn, zn,
										 bb_min, bb_range );

		int idx = x + y*xn + z*xn*yn;
		out_sdf_data[idx] = sqrtf( d_min );
	}}}
}


// ==================================================================

/**
 * Host main routine
 */
int main(void)
{

    // =====================================================
    itm_t * mesh = itm_readgeom( "data/bunny.itm" );

	aabb_t bb;
	for ( size_t i=0, in=mesh->header.num_positions/3; i<in; ++i )
	{
		vec3_t *p = reinterpret_cast<vec3_t*>( &mesh->positions[ i ] );
		bb.mn = min( bb.mn, *p );
		bb.mx = max( bb.mx, *p );
	}

	//TODO: max 1024 all together...
    enum { GRID_SIZ_X=10,
           GRID_SIZ_Y=10,
           GRID_SIZ_Z=10
    };
    sdf_t sdf;    
    init_sdf(&sdf, bb, GRID_SIZ_X, GRID_SIZ_Y, GRID_SIZ_Z );


    // =====================================================

    invoke_naive( *mesh, sdf );


	float32_t *h_sdf = (float32_t*)malloc( sizeof(float32_t) * sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z );
    checkCudaErrors( cudaMemcpy(h_sdf, sdf.d_data, sizeof(float32_t) * sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z, cudaMemcpyDeviceToHost) );

	//for ( int i=0, n=sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z; i<n; ++i )
	//{
	//	printf( "%f\n", h_sdf[i] );
	//}

	init_timers();
	uint64_t t0_ms = gettime_ms();
	eval_sdf__bruteforce( sdf, mesh, sdf.h_data );
	uint64_t t1_ms = gettime_ms();
	printf( "%dms\n", (int)(t1_ms-t0_ms) );

	bool sane = true;
	float32_t mindiff =  FLT_MAX;
	float32_t maxdiff = -FLT_MAX;
	for (int i = 0, n = sdf.header.dim_x * sdf.header.dim_y * sdf.header.dim_z; i < n; ++i)
	{
		const float32_t d0 = sdf.h_data[i];
		const float32_t d1 = h_sdf[i];
		if (!AlmostEqualRelative(d0, d1))
		{
			float32_t diff = abs(d1 - d0);
			if (diff < mindiff) mindiff = diff;
			if (diff > maxdiff) maxdiff = diff;
			sane = false;
		}
	}

	if( mindiff < maxdiff )
		printf("delta: [min;max]=[%f;%f]", mindiff, maxdiff);

	printf( "\nsane: %s\n", sane?"true":"false" );

    {
        free( mesh->indices );
        free( mesh->positions );
        free( mesh );

		//TODO: free device memory...
    }

    printf("Done\n");
    return 0;
}

